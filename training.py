from sae_variants import *
from ioi_utils import *
from tqdm import tqdm
from typing import Dict, Any, Optional

from torch.optim.lr_scheduler import _LRScheduler

class DefaultConfig:
    # following https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    BETA_1 = 0.9 
    BETA_2 = 0.999
    WEIGHT_DECAY = 0.0
    L1_COEFF = 5.0
    # slightly larger
    LR = 3e-4 
    RESAMPLE_WARMUP_STEPS = 100 # following https://arxiv.org/pdf/2404.16014 etc. 
    # a somewhat lower batch size because our dataset is very small
    BATCH_SIZE = 512 


class SAELRScheduler(_LRScheduler):
    """
    A learning rate scheduler that performs a linear warmup for the first
    `num_warmup_steps`, otherwise keeping the learning rate constant.
    """
    def __init__(self, optimizer, 
                 num_warmup_steps: int,
                 final_decay_start: int = 0,
                 final_decay_end: int = 0,
                 last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.final_decay_start = final_decay_start
        self.final_decay_end = final_decay_end
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        self.current_warmup_start = None
        self.in_warmup = False
        self.warmup_start_lrs = [0.0] * len(self.initial_lrs)
        super().__init__(optimizer, last_epoch)

    def start_warmup(self):
        self.in_warmup = True
        # self.current_step = 0
        self.current_warmup_start = self.current_step
        self.warmup_start_lrs = [group['lr'] for group in self.optimizer.param_groups]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = 0.1 * self.warmup_start_lrs[i]

    def get_lr(self):
        if self.in_warmup and self.current_step >= self.final_decay_start:
            raise ValueError("Cannot be in warmup and final decay at the same time") 
        if self.current_step >= self.final_decay_start:
            # decay lr linearly to 0 
            decay_progress = (self.current_step - self.final_decay_start) / (self.final_decay_end - self.final_decay_start)
            return [group['lr'] * (1 - decay_progress) for group in self.optimizer.param_groups]
        elif not self.in_warmup:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            warmup_progress = (self.current_step - self.current_warmup_start) / self.num_warmup_steps
            return [
                base_lr * (0.1 + 0.9 * warmup_progress)
                for base_lr in self.warmup_start_lrs
            ]

    def step(self, epoch=None):
        self.current_step += 1
        if self.current_warmup_start is not None:
            if self.current_step - self.current_warmup_start >= self.num_warmup_steps:
                self.in_warmup = False
                self.current_warmup_start = None
        super().step(epoch)

################################################################################
### computing the logitdiff loss
################################################################################
@op
def get_dataset_mean(A: Tensor) -> Tensor:
    return A.mean(dim=0)

def mean_ablate_hook(activation: Tensor, hook: HookPoint, node: Node, mean: Tensor, idx: Tensor) -> Tensor:
    activation[idx] = mean
    return activation

@torch.no_grad()
def encoder_hook(activation: Tensor, hook: HookPoint, node: Node, encoder: Union[VanillaAutoEncoder, GatedAutoEncoder],
                 idx: Tensor, ) -> Tensor:
    A = activation[idx]
    reconstruction = encoder.get_reconstruction(A)
    activation[idx] = reconstruction
    return activation

@op
def compute_mean_ablated_lds(
    node: Node,
    prompts: Any,
    A_mean: Tensor,
    batch_size: int,
) -> float:
    mean_ablated_logits = run_with_hooks.f(
        prompts=prompts,
        hooks=None,
        semantic_nodes=[node],
        semantic_hooks=[(node.activation_name, partial(mean_ablate_hook, node=node, mean=A_mean))],
        batch_size=batch_size,
    )
    mean_ablated_ld = (mean_ablated_logits[:, 0] - mean_ablated_logits[:, 1]).mean().item()
    return mean_ablated_ld

@torch.no_grad()
def get_logitdiff_loss(
    encoder: Any, node: Node, prompts: List[Prompt], batch_size: int,
    normalization_scale: Optional[float] = None,
    clean_ld: Optional[float] = None, mean_ablated_ld: Optional[float] = None,
    ) -> float:
    mean_ablated_ld = mean_ablated_ld
    encoder_logits = run_with_hooks(
        prompts=prompts,
        hooks=None,
        semantic_nodes=[node],
        semantic_hooks=[(node.activation_name, partial(encoder_hook, node=node, encoder=encoder, normalization_scale=normalization_scale))],
        batch_size=batch_size,
    )
    encoder_ld = (encoder_logits[:, 0] - encoder_logits[:, 1]).mean().item()
    # score = (clean_ld - encoder_ld) / (clean_ld - mean_ablated_ld)
    score = (encoder_ld - mean_ablated_ld).abs().item() / (clean_ld - mean_ablated_ld).abs().item()
    return score

@op
def eval_ld_loss(
    encoder: VanillaAutoEncoder,
    A_mean: Tensor,
    prompts: Any,
    node: Node,
    ld_subsample_size: Optional[int] = None,
    normalize: bool = False,
    A: Optional[Tensor] = None,
    return_additional_metrics: bool = False,
    random_seed: int = 42,
    mean_clean_ld: Optional[float] = None,
    mean_ablated_ld: Optional[float] = None,
    ) -> float:
    if normalize:
        assert A is not None
        _, normalization_scale = normalize_sae_inputs(A)
    else:
        normalization_scale = None
    if ld_subsample_size is not None:
        random.seed(random_seed)
        ld_loss_prompts = random.sample(prompts, ld_subsample_size)
    else:
        ld_loss_prompts = prompts
    ld_loss, clean_ld_mean, ablated_ld_mean, encoder_ld_mean = get_logitdiff_loss(encoder=encoder, node=node, prompts=ld_loss_prompts,
                                                                                  batch_size=200, activation_mean=A_mean, normalization_scale=normalization_scale,
                                                                                  mean_clean_ld=mean_clean_ld, mean_ablated_ld=mean_ablated_ld)
    if return_additional_metrics:
        return (ld_loss, clean_ld_mean, ablated_ld_mean, encoder_ld_mean)
    else:
        return ld_loss

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
    l1_coeff: float = DefaultConfig.L1_COEFF,
    lr: float=DefaultConfig.LR,
    beta1: float=DefaultConfig.BETA_1,
    beta2: float=DefaultConfig.BETA_2,
    weight_decay: float = DefaultConfig.WEIGHT_DECAY,
    batch_size: int = DefaultConfig.BATCH_SIZE,
    resample_epochs: Optional[List[int]] = None,
    resample_warmup_steps: int = DefaultConfig.RESAMPLE_WARMUP_STEPS, 
    torch_random_seed: int = 42,
    freeze_decoder: bool = False,
    final_decay_start: int = 1500,
    final_decay_end: int = 2000,
    enc_dtype: str = "fp32",
    ) -> Tuple[Dict[str, Tensor], dict, dict, List[Dict[str, float]]]:
    torch.manual_seed(torch_random_seed)
    d_activation = A.shape[1]
    if encoder_state_dict is None:
        assert optimizer_state_dict is None and scheduler_state_dict is None
    encoder = VanillaAutoEncoder(d_activation=d_activation, enc_dtype=enc_dtype,
                            d_hidden=d_hidden, freeze_decoder=freeze_decoder).cuda()
    if encoder_state_dict is not None:
        encoder.load_state_dict(encoder_state_dict, strict=True)
    optim = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    if optimizer_state_dict is not None:
        optim.load_state_dict(optimizer_state_dict)
    scheduler = SAELRScheduler(optim, num_warmup_steps=resample_warmup_steps, 
                              final_decay_start=final_decay_start, final_decay_end=final_decay_end)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
    pbar = tqdm(range(start_epoch, end_epoch))
    n = A.shape[0]
    metrics = []
    for epoch in pbar:
        perm = torch.randperm(n)
        epoch_metrics = {
            'epoch': epoch, # to keep track of the epoch
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
            optim.step()
            encoder.make_decoder_weights_and_grad_unit_norm()
            actual_batch_size = A_batch.shape[0]
            epoch_metrics["l2_loss"] += l2_loss.item() * actual_batch_size
            epoch_metrics["l1_loss"] += l1_loss.item() * actual_batch_size
            epoch_metrics["l0_loss"] += (acts > 0).sum(dim=-1).float().mean().item() * actual_batch_size
            feature_counts += (acts > 0).float().sum(dim=0)
            dead_features_batch = (acts > 0).sum(dim=0) == 0
            epoch_metrics["dead_mask"] = epoch_metrics["dead_mask"] & dead_features_batch # take AND w/ False to indicate alive neurons
        scheduler.step()
        epoch_metrics["l2_loss"] /= n
        epoch_metrics["l1_loss"] /= n
        epoch_metrics["l0_loss"] /= n
        epoch_metrics['frac_dead'] = epoch_metrics["dead_mask"].float().mean().item()
        num_alive = (1-epoch_metrics['frac_dead'])*d_hidden
        del epoch_metrics["dead_mask"]
        metrics.append(epoch_metrics)
        if resample_epochs is not None and epoch in resample_epochs:
            dead_indices = (feature_counts < 1).nonzero().squeeze()
            # make sure dead_indices is not 0-dimensional
            if len(dead_indices.shape) == 0:
                dead_indices = dead_indices.unsqueeze(0)
            if len(dead_indices) > 0:
                resample_vanilla(encoder, dead_indices, A, l1_coeff, optim)
                scheduler.start_warmup()
        pbar.set_description(f"l2_loss: {epoch_metrics['l2_loss']:.4f}, l1_loss: {epoch_metrics['l1_loss']:.4f}, " + 
                             f"l0_loss: {epoch_metrics['l0_loss']:.4f}, frac_dead: {epoch_metrics['frac_dead']:.4f}")
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
@op
def train_gated(
    A: Tensor,
    d_hidden: int,
    start_epoch: int,
    end_epoch: int,
    encoder_state_dict: Optional[Dict[str, Tensor]],
    optimizer_state_dict: Optional[Any],
    scheduler_state_dict: Optional[Any],
    l1_coeff: float = DefaultConfig.L1_COEFF,
    lr: float=DefaultConfig.LR,
    beta1: float=DefaultConfig.BETA_1,
    beta2: float=DefaultConfig.BETA_2,
    weight_decay: float = DefaultConfig.WEIGHT_DECAY,
    batch_size: int = DefaultConfig.BATCH_SIZE,
    resample_epochs: Optional[List[int]] = None,
    resample_warmup_steps: int = DefaultConfig.RESAMPLE_WARMUP_STEPS, 
    torch_random_seed: int = 42,
    freeze_decoder: bool = False,
    final_decay_start: int = 1500,
    final_decay_end: int = 2000,
    enc_dtype: str = "fp32",
    ) -> Tuple[Dict[str, Tensor], dict, dict, List[Dict[str, float]]]:
    torch.manual_seed(torch_random_seed)
    d_activation = A.shape[1]
    if encoder_state_dict is None:
        assert optimizer_state_dict is None and scheduler_state_dict is None
    encoder = GatedAutoEncoder(d_activation=d_activation, d_hidden=d_hidden).cuda()
    if encoder_state_dict is not None:
        encoder.load_state_dict(encoder_state_dict, strict=True)
    optim = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    if optimizer_state_dict is not None:
        optim.load_state_dict(optimizer_state_dict)
    scheduler = SAELRScheduler(optim, num_warmup_steps=resample_warmup_steps, 
                              final_decay_start=final_decay_start, final_decay_end=final_decay_end)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
    pbar = tqdm(range(start_epoch, end_epoch))
    n = A.shape[0]
    metrics = []
    for epoch in pbar:
        perm = torch.randperm(n)
        epoch_metrics = {
            'epoch': epoch, # to keep track of the epoch
            "l2_loss": 0,
            "l1_loss": 0,
            "l0_loss": 0,
            "l_aux": 0,
            "dead_mask": Tensor([True for _ in range(d_hidden)]).cuda().bool(), # dead neurons throughout the entire epoch
        }
        feature_counts = 0
        for i in range(0, n, batch_size):
            A_batch = A[perm[i:i+batch_size]]
            optim.zero_grad()

            f_tilde, pi_gate, L_sparsity = encoder.encode(A_batch)
            A_hat, L_reconstruct, L_aux = encoder.decode(f_tilde, pi_gate, A_batch)
            loss = L_reconstruct + L_sparsity * l1_coeff + L_aux # important part
            loss.backward()

            optim.step()
            encoder.make_decoder_weights_and_grad_unit_norm()
            actual_batch_size = A_batch.shape[0]
            epoch_metrics["l2_loss"] += L_reconstruct.item() * actual_batch_size
            epoch_metrics["l1_loss"] += L_sparsity.item() * actual_batch_size
            epoch_metrics["l0_loss"] += (pi_gate > 0).sum(dim=-1).float().mean().item() * actual_batch_size
            epoch_metrics["l_aux"] += L_aux.item() * actual_batch_size
            feature_counts += (pi_gate > 0).float().sum(dim=0)
            dead_features_batch = (pi_gate > 0).sum(dim=0) == 0
            epoch_metrics["dead_mask"] = epoch_metrics["dead_mask"] & dead_features_batch # take AND w/ False to indicate alive neurons
        scheduler.step()
        epoch_metrics["l2_loss"] /= n
        epoch_metrics["l1_loss"] /= n
        epoch_metrics["l0_loss"] /= n
        epoch_metrics["l_aux"] /= n
        epoch_metrics['frac_dead'] = epoch_metrics["dead_mask"].float().mean().item()
        del epoch_metrics["dead_mask"]
        metrics.append(epoch_metrics)
        if resample_epochs is not None and epoch in resample_epochs:
            dead_indices = (feature_counts < 1).nonzero().squeeze()
            # make sure dead_indices is not 0-dimensional
            if len(dead_indices.shape) == 0:
                dead_indices = dead_indices.unsqueeze(0)
            if len(dead_indices) > 0:
                resample_gated(encoder, dead_indices, A, l1_coeff, optim)
                scheduler.start_warmup()
        pbar.set_description(f"l2_loss: {epoch_metrics['l2_loss']:.4f}, l1_loss: {epoch_metrics['l1_loss']:.4f}, " + 
                             f"l0_loss: {epoch_metrics['l0_loss']:.4f}, l_aux: {epoch_metrics['l_aux']:.4f}, frac_dead: {epoch_metrics['frac_dead']:.4f}")
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