from sae_variants import *
from ioi_utils import *
from tqdm import tqdm
from typing import Dict, Any, Optional

from torch.optim.lr_scheduler import _LRScheduler

class MidTrainingWarmupScheduler(_LRScheduler):
    """
    A learning rate scheduler that performs a linear warmup for the first
    `num_warmup_steps`, otherwise keeping the learning rate constant.
    """
    def __init__(self, optimizer, num_warmup_steps, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        self.in_warmup = False
        self.warmup_start_lrs = [0.0] * len(self.initial_lrs)
        super().__init__(optimizer, last_epoch)

    def start_warmup(self):
        self.in_warmup = True
        self.current_step = 0
        self.warmup_start_lrs = [group['lr'] for group in self.optimizer.param_groups]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = 0.1 * self.warmup_start_lrs[i]

    def get_lr(self):
        if not self.in_warmup:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        if self.current_step >= self.num_warmup_steps:
            self.in_warmup = False
            return self.initial_lrs
        
        warmup_factor = (self.current_step + 1) / self.num_warmup_steps
        return [
            base_lr * (0.1 + 0.9 * warmup_factor)
            for base_lr in self.initial_lrs
        ]

    def step(self, epoch=None):
        if self.in_warmup:
            self.current_step += 1
        super().step(epoch)


BETA_1 = 0.0 # following https://transformer-circuits.pub/2024/feb-update/index.html
BETA_2 = 0.999
L1_COEFF = 5.0
LR = 0.0003 # following https://arxiv.org/pdf/2404.16014
BATCH_SIZE = 512 # a somewhat lower batch size because our dataset is very small


################################################################################
### vanilla SAEs
################################################################################
@op
def train_vanilla(
    A: Tensor,
    d_hidden: int,
    start_epoch: int,
    end_epoch: int,
    encoder_state_dict: Optional[Dict[str, Tensor]],
    optimizer_state_dict: Optional[Any],
    scheduler_state_dict: Optional[Any],
    l1_coeff: float = L1_COEFF,
    lr: float=LR,
    beta1: float=BETA_1,
    beta2: float=BETA_2,
    batch_size: int = BATCH_SIZE,
    resample_every: Optional[int] = None,
    resample_warmup_steps: int = 100, # following https://arxiv.org/pdf/2404.16014 etc. 
    torch_random_seed: int = 42,
    freeze_decoder: bool = False,
    ) -> Tuple[Dict[str, Tensor], dict, dict, List[Dict[str, float]]]:
    torch.manual_seed(torch_random_seed)
    d_activation = A.shape[1]
    encoder = VanillaAutoEncoder(d_activation=d_activation, 
                            d_hidden=d_hidden, freeze_decoder=freeze_decoder).cuda()
    encoder.load_state_dict(encoder_state_dict, strict=True)
    optim = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2))
    optim.load_state_dict(optimizer_state_dict)
    scheduler = MidTrainingWarmupScheduler(optim, resample_warmup_steps)
    scheduler.load_state_dict(scheduler_state_dict)
    pbar = tqdm(range(start_epoch, end_epoch))
    n = A.shape[0]
    metrics = []
    for epoch in pbar:
        perm = torch.randperm(n)
        epoch_metrics = {
            "l2_loss": 0,
            "l1_loss": 0,
            "l0_loss": 0,
            "dead_mask": Tensor([True for _ in range(d_hidden)]).cuda().bool(), # dead neurons throughout the entire epoch
        }
        feature_counts = 0
        for i in range(0, n, batch_size):
            A_batch = A[perm[i:i+batch_size]]
            optim.zero_grad()
            A_hat, acts, l2_loss, l1_loss = encoder(A_batch)
            loss = l2_loss + l1_loss * l1_coeff
            loss.backward()
            encoder.make_decoder_weights_and_grad_unit_norm()
            optim.step()
            scheduler.step()
            actual_batch_size = A_batch.shape[0]
            epoch_metrics["l2_loss"] += l2_loss.item() * actual_batch_size
            epoch_metrics["l1_loss"] += l1_loss.item() * actual_batch_size
            epoch_metrics["l0_loss"] += (acts > 0).sum(dim=-1).float().mean().item() * actual_batch_size
            feature_counts += (acts > 0).float().sum(dim=0)
            dead_features_batch = (acts > 0).sum(dim=0) == 0
            epoch_metrics["dead_mask"] = epoch_metrics["dead_mask"] & dead_features_batch # take AND w/ False to indicate alive neurons
        epoch_metrics["l2_loss"] /= n
        epoch_metrics["l1_loss"] /= n
        epoch_metrics["l0_loss"] /= n
        epoch_metrics['frac_dead'] = epoch_metrics["dead_mask"].float().mean().item()
        metrics.append(epoch_metrics)
        if epoch != 0 and resample_every is not None and epoch % resample_every == 0:
            dead_indices = (feature_counts < 1).nonzero().squeeze()
            if len(dead_indices) > 0:
                resample_vanilla(encoder, dead_indices, A, l1_coeff, optim)
                scheduler.start_warmup()
        pbar.set_description(f"l2_loss: {epoch_metrics['l2_loss']:.4f}, l1_loss: {epoch_metrics['l1_loss']:.4f}, \
                             l0_loss: {epoch_metrics['l0_loss']:.4f}, frac_dead: {epoch_metrics['frac_dead']:.4f}")
    return encoder.state_dict(), optim.state_dict(), scheduler.state_dict(), metrics


@torch.no_grad()
def resample_vanilla(encoder: VanillaAutoEncoder, dead_indices: Tensor, A: Tensor, l1_coeff: float,
                   optimizer: torch.optim.Adam, W_enc_reinit_scale: float = 0.2):
    """
    Re-initializes the weights of the encoder for the given indices, following
    the re-initialization strategy from Anthropic
    """
    ### collect losses of the encoder on the activations
    batch_size = 64
    n = A.shape[0]
    l2_parts, l1_parts = [], []
    for i in range(0, n, batch_size):
        A_batch = A[i:i+batch_size]
        _, _, l2_losses, l1_losses = encoder.forward_detailed(A_batch)
        l2_parts.append(l2_losses)
        l1_parts.append(l1_losses)
    l2_losses = torch.cat(l2_parts)
    l1_losses = torch.cat(l1_parts)
    total_losses = l2_losses + l1_losses * l1_coeff
    squared_losses = total_losses ** 2

    ### sample indices of examples to use for re-initialization
    sampling_probabilities = squared_losses / squared_losses.sum()
    sample_indices = torch.multinomial(
        sampling_probabilities,
        len(dead_indices),
        replacement=True, # in case there are more dead indices than examples
    )

    ### re-initialize decoder and encoder weights
    encoder.W_dec.data[dead_indices, :] = A[sample_indices] / A[sample_indices].norm(dim=-1, keepdim=True)
    encoder.W_enc.data[:, dead_indices] = encoder.W_dec.data[dead_indices, :].T.clone()
    # now, figure out the average norm of W_enc over alive neurons
    alive_indices = torch.ones(encoder.W_enc.shape[1], dtype=torch.bool)
    alive_indices[dead_indices] = False
    avg_alive_enc_norm = encoder.W_enc[:, alive_indices].norm(dim=-1).mean()
    encoder.W_enc.data[:, dead_indices] *= W_enc_reinit_scale * avg_alive_enc_norm
    encoder.b_enc.data[dead_indices] = 0.0

    ### reset the optimizer weights for the changed parameters
    # we must reset only the encoder bias, and the encoder and decoder weights
    reset_adam_optimizer_params(optimizer, encoder.b_enc, dead_indices)
    reset_adam_optimizer_params(optimizer, encoder.W_enc, (slice(None), dead_indices))
    reset_adam_optimizer_params(optimizer, encoder.W_dec, (dead_indices, slice(None)))


@torch.no_grad()
def reset_adam_optimizer_params(optimizer: torch.optim.Adam, param: torch.nn.Parameter, param_idx: Tensor):
    """
    Reset the Adam optimizer parameters for a specific weight.

    Args:
        optimizer (torch.optim.Optimizer): The Adam optimizer.
        param (torch.nn.Parameter): The parameter containing the weights.
        param_idx (tuple): The index of the specific weight to reset.
    """
    for state in optimizer.state[param]:
        if state in ['exp_avg', 'exp_avg_sq']:
            optimizer.state[param][state][param_idx] = torch.zeros_like(optimizer.state[param][state][param_idx])



################################################################################
### Gated SAEs
################################################################################
def train_gated(
    A: Tensor,
    d_activation: int, 
    d_hidden: int,
    start_epoch: int,
    end_epoch: int,
    encoder_state_dict: Optional[Dict[str, Tensor]],
    optimizer_state_dict: Optional[Any],
    scheduler_state_dict: Optional[Any],
    l1_coeff: float = L1_COEFF,
    lr: float=LR,
    beta1: float=BETA_1,
    beta2: float=BETA_2,
    batch_size: int = BATCH_SIZE,
    resample_every: Optional[int] = None,
    resample_warmup_steps: int = 100, # following https://arxiv.org/pdf/2404.16014 etc. 
    torch_random_seed: int = 42,
    ):
    """
    Training following the Gated SAE architecture from https://arxiv.org/pdf/2404.16014
    """ 
    torch.manual_seed(torch_random_seed)
    encoder = GatedAutoEncoder(d_activation=d_activation, d_hidden=d_hidden).cuda()
    encoder.load_state_dict(encoder_state_dict, strict=True)
    optim = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2))
    optim.load_state_dict(optimizer_state_dict)
    scheduler = MidTrainingWarmupScheduler(optim, resample_warmup_steps)
    scheduler.load_state_dict(scheduler_state_dict)
    pbar = tqdm(range(start_epoch, end_epoch))
    n = A.shape[0]
    metrics = []
    for epoch in pbar:
        perm = torch.randperm(n)
        epoch_metrics = {
            "l2_loss": 0,
            "l1_loss": 0,
            "aux_loss": 0,
            "l0_loss": 0,
            "total_loss": 0,
            "epoch": epoch,
        }
        feature_counts = 0
        for i in range(0, n, batch_size):
            A_batch = A[perm[i:i+batch_size]]
            optim.zero_grad()
            f_tilde, pi_gate, L_sparsity = encoder.encode(A_batch)
            A_hat, L_reconstruct, L_aux = encoder.decode(f_tilde, pi_gate, A_batch)
            loss = L_reconstruct + L_sparsity * l1_coeff + L_aux # important part
            loss.backward()
            encoder.make_decoder_weights_and_grad_unit_norm()
            optim.step()
            scheduler.step()
            actual_batch_size = A_batch.shape[0]
            epoch_metrics["l2_loss"] += L_reconstruct.item() * actual_batch_size
            epoch_metrics["l1_loss"] += L_sparsity.item() * actual_batch_size
            epoch_metrics["l0_loss"] += (pi_gate > 0).sum(dim=-1).float().mean().item() * actual_batch_size
            epoch_metrics["aux_loss"] += L_aux.item() * actual_batch_size
            epoch_metrics["total_loss"] += loss.item() * actual_batch_size
            feature_counts += (pi_gate > 0).float().sum(dim=0)
        epoch_metrics["l2_loss"] /= n
        epoch_metrics["l1_loss"] /= n
        epoch_metrics["l0_loss"] /= n
        epoch_metrics["aux_loss"] /= n
        epoch_metrics["total_loss"] /= n
        metrics.append(epoch_metrics)
        if epoch != 0 and resample_every is not None and epoch % resample_every == 0:
            dead_indices = (feature_counts < 1).nonzero().squeeze()
            if len(dead_indices) > 0:
                resample_gated(encoder, dead_indices, A, l1_coeff, optim)
                scheduler.start_warmup()
        pbar.set_description(f"l2_loss: {epoch_metrics['l2_loss']:.4f}, l1_loss: {epoch_metrics['l1_loss']:.4f}, l0_loss: {epoch_metrics['l0_loss']:.4f}")
    return encoder.state_dict(), optim.state_dict(), scheduler.state_dict(), metrics


@torch.no_grad()
def resample_gated(encoder: GatedAutoEncoder, dead_indices: Tensor, A: Tensor, l1_coeff: float,
                   optimizer: torch.optim.Adam, W_enc_reinit_scale: float = 0.2):
    """
    Re-initializes the weights of the encoder for the given indices, following
    the re-initialization strategy from Anthropic
    """
    ### collect losses of the encoder on the activations
    batch_size = 64
    n = A.shape[0]
    L_reconstruct_parts, L_sparsity_parts, L_aux_parts = [], [], []
    for i in range(0, n, batch_size):
        A_batch = A[i:i+batch_size]
        f_tilde, pi_gate, L_sparsity = encoder.encode(A_batch)
        A_hat, L_reconstruct, L_aux = encoder.decode(f_tilde, pi_gate, A_batch)
        L_reconstruct_parts.append(L_reconstruct)
        L_sparsity_parts.append(L_sparsity)
        L_aux_parts.append(L_aux)
    L_reconstruct = torch.cat(L_reconstruct_parts)
    L_sparsity = torch.cat(L_sparsity_parts)
    L_aux = torch.cat(L_aux_parts)
    total_losses = L_reconstruct + L_sparsity * l1_coeff + L_aux
    squared_losses = total_losses ** 2

    ### sample indices of examples to use for re-initialization
    sampling_probabilities = squared_losses / squared_losses.sum()
    sample_indices = torch.multinomial(
        sampling_probabilities,
        len(dead_indices),
        replacement=True, # in case there are more dead indices than examples
    )

    ### re-initialize decoder and encoder weights
    encoder.W_dec.data[dead_indices, :] = A[sample_indices] / A[sample_indices].norm(dim=-1, keepdim=True)
    encoder.W_gate.data[:, dead_indices] = encoder.W_gate.data[dead_indices, :].T.clone()
    # ? not specified in the paper, defaulting to 0 (so, factor of 1 after exp)
    encoder.r_mag.data[dead_indices] = 0.0 
    # now, figure out the average norm of W_gate over alive neurons
    alive_indices = torch.ones(encoder.W_gate.shape[1], dtype=torch.bool)
    alive_indices[dead_indices] = False
    avg_alive_enc_norm = encoder.W_gate[:, alive_indices].norm(dim=-1).mean()
    encoder.W_gate.data[:, dead_indices] *= W_enc_reinit_scale * avg_alive_enc_norm
    encoder.b_gate.data[dead_indices] = 0.0
    encoder.b_mag.data[dead_indices] = 0.0

    ### reset the optimizer weights for the changed parameters
    # we must reset only the encoder bias, and the encoder and decoder weights
    reset_adam_optimizer_params(optimizer, encoder.b_gate, dead_indices)
    reset_adam_optimizer_params(optimizer, encoder.b_mag, dead_indices)
    reset_adam_optimizer_params(optimizer, encoder.W_gate, (slice(None), dead_indices))
    reset_adam_optimizer_params(optimizer, encoder.W_dec, (dead_indices, slice(None)))
    reset_adam_optimizer_params(optimizer, encoder.r_mag, dead_indices)