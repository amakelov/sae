import torch
from torch import nn, Tensor
from fancy_einsum import einsum
from torch.nn import functional as F

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from transformer_lens import utils


from mandala._next.imports import op, NewArgDefault, Storage
from typing import Tuple

from ioi_utils import *


@op
def normalize_activations(A: Tensor, scale: Optional[float] = NewArgDefault()) -> Tuple[Tensor, float]:
    """
    Normalize activations following
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes:
    multiply by a scalar so that the average norm is sqrt(dimension)
    """
    assert len(A.shape) == 2
    if isinstance(scale, NewArgDefault):
        d_activation = A.shape[1]
        avg_norm = A.norm(p=2, dim=1).mean()
        normalization_scale = avg_norm / d_activation ** 0.5
        return A / normalization_scale, normalization_scale
    else:
        return A / scale, scale

@op
def normalize_grad(A_grad: Tensor, scale: float) -> Tensor:
    """
    To be used w/ attribution SAEs
    """
    return A_grad / scale


class VanillaAutoEncoder(nn.Module):
    def __init__(self, 
                 d_activation: int,
                 d_hidden: int,
                 enc_dtype: str = "fp32",
                 freeze_decoder: bool = False,
                 random_seed: int = 0,
                 ):
        super().__init__()
        self.d_activation = d_activation
        self.d_hidden = d_hidden
        self.freeze_decoder = freeze_decoder
        dtype = torch.float32 if enc_dtype == "fp32" else torch.float16
        # set the random seed before initializing the weights
        torch.manual_seed(random_seed)
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.d_hidden, d_activation, dtype=dtype)))
        if freeze_decoder:
            self.W_dec.requires_grad = False
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_activation, dtype=dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.W_enc = nn.Parameter(torch.empty(d_activation, d_hidden, dtype=dtype))
        # initialize W_enc from W_dec, following https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        self.W_enc.data = self.W_dec.data.T.detach().clone()
    
    def forward_detailed(self, A: Tensor):
        x_cent = A - self.b_dec 
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec

        l2_losses = (x_reconstruct.float() - A.float()).pow(2).sum(-1)
        l1_losses = acts.float().abs().sum(-1)
        return x_reconstruct, acts, l2_losses, l1_losses
    
    def forward(self, A: Tensor):
        x_reconstruct, acts, l2_losses, l1_losses = self.forward_detailed(A)
        
        l2_loss = l2_losses.mean()
        l1_loss = l1_losses.mean()
        loss = l2_loss + l1_loss
        return x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        if self.freeze_decoder:
            return
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed

    ### encapsulate some pieces of logic here w/ no_grad to avoid bugs
    @torch.no_grad()
    def get_activation_pattern(self, A: Tensor) -> Tensor:
        _, acts, _, _ = self.forward_detailed(A)
        return (acts > 0).bool()
    
    @torch.no_grad()
    def get_feature_magnitudes(self, A: Tensor) -> Tensor:
        _, acts, _, _ = self.forward_detailed(A)
        return acts
    
    @torch.no_grad()
    def get_reconstructions(self, A: Tensor) -> Tensor:
        x_reconstruct, _, _, _ = self.forward_detailed(A)
        return x_reconstruct


class GatedAutoEncoder(nn.Module):
    """
    Following the Gated SAE paper https://arxiv.org/pdf/2404.16014
    """
    def __init__(self, d_activation: int, d_hidden: int):
        super().__init__()
        self.d_activation = d_activation
        self.d_hidden = d_hidden

        # the decoder matrix
        W_dec = torch.randn(d_hidden, d_activation)
        # normalize so that each row (i.e., decoder vector) has unit norm
        W_dec = W_dec / torch.norm(W_dec, dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)

        # the gating matrix (also denoted W_enc in the paper figure 3)
        self.W_gate = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_activation, d_hidden)))

        self.r_mag = nn.Parameter(torch.zeros(d_hidden))
        self.b_gate = nn.Parameter(torch.zeros(d_hidden))
        self.b_mag = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_activation))

        self.relu = nn.ReLU()
    
    def encode(self, X: Tensor):
        # using the paper's notation
        X_centered = X - self.b_dec
        pi_gate = einsum( 'dim hidden, batch dim -> batch hidden', self.W_gate, X_centered,) + self.b_gate
        # f_gate gives the activation pattern for the hidden layer
        f_gate = (pi_gate > 0).float()
        W_mag = einsum( 'hidden, dim hidden -> dim hidden', torch.exp(self.r_mag), self.W_gate,)
        f_mag = einsum('dim hidden, batch dim -> batch hidden', W_mag, X_centered, ) + self.b_mag
        f_tilde = f_gate * f_mag
        L_sparsity = nn.ReLU()(pi_gate).norm(dim=1, p=1).mean()
        return f_tilde, pi_gate, L_sparsity
    
    def decode(self, f_tilde: Tensor, pi_gate: Tensor, X: Tensor):
        X_hat = einsum('batch hidden, hidden dim -> batch dim', f_tilde, self.W_dec, ) + self.b_dec
        L_reconstruct = (X_hat - X).pow(2).sum(dim=1).mean()
        
        # compute the auxiliary loss
        W_dec_clone = self.W_dec.clone().detach()
        b_dec_clone = self.b_dec.clone().detach()
        x_hat_frozen = einsum(
            'batch hidden, hidden dim -> batch dim', nn.ReLU()(pi_gate), W_dec_clone, 
        ) + b_dec_clone
        L_aux = (X - x_hat_frozen).pow(2).sum(dim=1).mean()
        return X_hat, L_reconstruct, L_aux

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    ### encapsulate some pieces of logic here w/ no_grad to avoid bugs
    @torch.no_grad()
    def get_activation_pattern(self, A: Tensor):
        _, pi_gate, _ = self.encode(A)
        return (pi_gate > 0).bool()
    
    @torch.no_grad()
    def get_feature_magnitudes(self, X: Tensor) -> Tensor:
        """
        This returns the actual magnitudes in the decomposition x_hat = sum_j
        f_j W_dec_j + b_dec
        """
        # use f_tilde, because X_hat = f_tilde @ W_dec + b_dec
        f_tilde, _, _ = self.encode(X)
        return f_tilde
    
    @torch.no_grad()
    def get_reconstructions(self, X: Tensor) -> Tensor:
        f_tilde, pi_gate, _ = self.encode(X)
        X_hat, _, _ = self.decode(f_tilde, pi_gate, X)
        return X_hat
    



################################################################################
### Attribution SAEs
################################################################################
# attribution SAEs are basically vanilla SAEs, but with a different loss
# function

class AttributionAutoEncoder(VanillaAutoEncoder):
    """
    Following https://transformer-circuits.pub/2024/april-update/index.html#attr-dl
    """
    def forward_detailed(self, A: Tensor, A_grad: Optional[Tensor] = None):
        A_centered = A - self.b_dec 
        acts = F.relu(A_centered @ self.W_enc + self.b_enc)
        A_hat = acts @ self.W_dec + self.b_dec

        l2_losses = (A_hat.float() - A.float()).pow(2).sum(-1)
        l1_losses = acts.float().abs().sum(-1)

        # and now, two new losses
        # in the notation of the blog post,
        # hidden activations are y (d_hidden,)
        # the reconstructions are x_hat = y @ W_dec + b_dec
        # the gradient w.r.t. the *hidden* activations is
        # grad_y (metric) = grad_x_hat (metric) @ W_dec.T
        # and b/c we don't want to compute all the gradients for reconstructions
        # we just use grad_x (metric), lol

        if A_grad is not None:
            attribution_sparsity_losses = (acts * (einsum('batch act_dim, hidden act_dim -> batch hidden', A_grad, self.W_dec))).abs().sum(-1)
            unexplained_attribution_losses = einsum("batch act_dim, batch act_dim -> batch", A - A_hat, A_grad).abs()
        else:
            attribution_sparsity_losses = None
            unexplained_attribution_losses = None

        return A_hat, acts, l2_losses, l1_losses, attribution_sparsity_losses, unexplained_attribution_losses
    
    # def get_reconstruction(self, A: Tensor) -> Tensor:
    #     x_reconstruct, _, _, _ = self.forward_detailed(A)
    #     return x_reconstruct
    
    def forward(self, A: Tensor, A_grad: Optional[Tensor] = None):
        x_reconstruct, acts, l2_losses, l1_losses, attribution_sparsity_losses, unexplained_attribution_losses = self.forward_detailed(A, A_grad)
        
        l2_loss = l2_losses.mean()
        l1_loss = l1_losses.mean()
        if A_grad is not None:
            attribution_sparsity_loss = attribution_sparsity_losses.mean()
            unexplained_attribution_loss = unexplained_attribution_losses.mean()
        else:
            attribution_sparsity_loss = None
            unexplained_attribution_loss = None
        return x_reconstruct, acts, l2_loss, l1_loss, attribution_sparsity_loss, unexplained_attribution_loss

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        if self.freeze_decoder:
            return
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed

    ### encapsulate some pieces of logic here w/ no_grad to avoid bugs
    @torch.no_grad()
    def get_activation_pattern(self, A: Tensor) -> Tensor:
        _, acts, _, _, _, _ = self.forward_detailed(A)
        return (acts > 0).bool()
    
    @torch.no_grad()
    def get_feature_magnitudes(self, A: Tensor) -> Tensor:
        _, acts, _, _, _, _ = self.forward_detailed(A)
        return acts
    
    @torch.no_grad()
    def get_reconstructions(self, A: Tensor) -> Tensor:
        x_reconstruct, _, _, _, _, _ = self.forward_detailed(A)
        return x_reconstruct


@op
@batched(args=['prompts'], n_outputs=1, reducer='cat')
def collect_gradients(
        prompts: List[Prompt],
        # layers_and_activations: List[Tuple[int, Literal['z', 'q', 'k']]],
        nodes: List[Node],
        batch_size: Optional[int] = None,
        ) -> Dict[Node, Tensor]:
    """
    Given a list of prompts, collect the gradients of the logit difference with
    respect to the given activation in the given layer. 

    This will return a tensor of shape (n_prompts, seq_len, n_heads, head_dim)
    from which you can then select gradients for desired heads and positions.
    """
    model: HookedTransformer = MODELS[MODEL_ID]
    model.requires_grad_(True)
    prompt_dataset = PromptDataset(prompts, model=model)

    layers_and_activations = list({(node.layer, node.component_name) for node in nodes})
    activations = {}
    grads = {}

    def get_forward_hook(location: Tuple[int, str]):
        def hook(model, input, output):
            activations[location] = output
            output.retain_grad()
        return hook
    
    def get_backward_hook(location: Tuple[int, str]):
        def hook(grad):
            grads[location] = grad
        return hook
    
    activation_attrs = {
        'z': 'hook_z',
        'q': 'hook_q',
        'k': 'hook_k',
        'v': 'hook_v',
    }

    forward_handles = {}

    for layer, activation_name in layers_and_activations:
        location = (layer, activation_name)
        forward_hook_handle = getattr(model.blocks[layer].attn, activation_attrs[activation_name]).register_forward_hook(get_forward_hook(location))
        forward_handles[location] = forward_hook_handle
    
    input_tensor = prompt_dataset.tokens
    # forward all the inputs ONCE
    output = model(input_tensor)[:, -1, :]
    # compute the tensor of logit differences that we want to take the gradient of
    answer_logits = torch.gather(output, dim=1, index=prompt_dataset.answer_tokens.cuda())
    ld = answer_logits[:, 0] - answer_logits[:, 1]

    backward_handles = {}
    for layer, activation_name in layers_and_activations:
        activation = activations[(layer, activation_name)]
        backward_hook_handle = activation.register_hook(get_backward_hook((layer, activation_name)))
        backward_handles[(layer, activation_name)] = backward_hook_handle


    individual_gradients = {}
    for i in range(len(prompt_dataset)): # iterate over the batch
        # lol why isn't there (?) a way to get the batch of gradients in one go :(
        # Backward pass
        model.zero_grad() # make sure the model is zeroed out
        ld[i].backward(retain_graph=True)
        # The gradient of the loss with respect to the internal activation
        for layer, activation_name in layers_and_activations:
            if (layer, activation_name) not in individual_gradients:
                individual_gradients[(layer, activation_name)] = []
            internal_activation_grad = grads[(layer, activation_name)][i].detach().clone()
            individual_gradients[(layer, activation_name)].append(internal_activation_grad)
    
    #! undo all the things we've done to the model
    for layer, activation_name in layers_and_activations:
        forward_hook_handle = forward_handles[(layer, activation_name)]
        forward_hook_handle.remove()
        backward_hook_handle = backward_handles[(layer, activation_name)]
        backward_hook_handle.remove()
    model.requires_grad_(False)

    for key in individual_gradients:
        individual_gradients[key] = torch.stack(individual_gradients[key], dim=0).cpu()
    
    # now, turn into a dict keyed by the nodes
    node = nodes[0]

    nodes_result = {}
    for node in nodes:
        layer, component_name = node.layer, node.component_name
        full_grad = individual_gradients[(layer, component_name)]
        nodes_result[node] = full_grad[node.idx(prompts=prompts)]

    return nodes_result

def get_gradients(storage: Storage, nodes: List[Node], prompts: Any, computing: bool, n_batches: int) -> Dict[Node, Tensor]:
    with storage:
        prompts_raw = storage.unwrap(prompts)
        n_total = len(prompts_raw)
        grads_parts = []
        result = {node: [] for node in nodes}

        for i in tqdm(list(range(n_batches))):
            # print(f'Batch {i}/{n_batches}')
            start = i * (n_total // n_batches)
            end = (i + 1) * (n_total // n_batches)
            prompts = prompts_raw[start:end]
            grads_part = collect_gradients(prompts=prompts, nodes=nodes, batch_size=20,)
            if computing:
                storage.commit()
                storage.atoms.clear()
            else:
                grads_part = storage.unwrap(grads_part)
                for node in nodes:
                    result[node].append(grads_part[node])
    if computing:
        return None
    else:
        return {k: torch.cat(v, dim=0).cuda() for k, v in result.items()}

################################################################################
### top-k SAEs
################################################################################
class TopKAutoEncoder(nn.Module):
    def __init__(self, d_activation: int, d_hidden: int, k: int):
        super().__init__()
        self.d_activation = d_activation
        self.d_hidden = d_hidden
        self.k = k

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_activation, self.d_hidden, ).cuda(),
                nonlinearity="relu",
            )
        )
        self.W_dec = nn.Parameter(self.W_enc.data.mT.contiguous())

        self.b_pre = nn.Parameter(torch.zeros(self.d_activation).cuda())
    
    def forward(self, A: Tensor):
        acts = einsum('batch d_activation, d_activation d_hidden -> batch d_hidden', A - self.b_pre, self.W_enc)
        top_acts, top_indices = acts.topk(self.k, dim=-1, 
                                          sorted=False # we don't care about the order
                                          )
        # top_indices is (batch, k)
        # we want to get a tensor of shape (batch, d_hidden) out of this
        # by putting the top_k indices in the right places
        # top_indices is (batch, k)
        z = torch.zeros_like(acts)
        z.scatter_(dim=-1, index=top_indices, src=top_acts)
        A_reconstruct = z @ self.W_dec + self.b_pre
        return A_reconstruct, acts, z
    
    @property
    def b_dec(self): # a trick for compatibility with the other SAEs
        return self.b_pre
    
    @torch.no_grad()
    def get_activation_pattern(self, A: Tensor) -> Tensor:
        _, acts, z = self.forward(A)
        return (z != 0).bool()
    
    @torch.no_grad()
    def get_feature_magnitudes(self, A: Tensor) -> Tensor:
        _, _, z = self.forward(A)
        return z
    
    @torch.no_grad()
    def get_reconstructions(self, A: Tensor) -> Tensor:
        A_reconstruct, _, _ = self.forward(A)
        return A_reconstruct
    


