import torch
from torch import nn, Tensor
from fancy_einsum import einsum
from torch.nn import functional as F

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from transformer_lens import utils


from mandala._next.imports import op
from typing import Tuple


@op
def normalize_activations(A: Tensor) -> Tuple[Tensor, float]:
    """
    Normalize activations following
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes:
    multiply by a scalar so that the average norm is sqrt(dimension)
    """
    assert len(A.shape) == 2
    d_activation = A.shape[1]
    avg_norm = A.norm(p=2, dim=1).mean()
    normalization_scale = avg_norm / d_activation ** 0.5
    return A / normalization_scale, normalization_scale


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
    
    def get_reconstruction(self, A: Tensor) -> Tensor:
        x_reconstruct, _, _, _ = self.forward_detailed(A)
        return x_reconstruct
    
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
    def get_activation_pattern(self, X: Tensor):
        _, pi_gate, _ = self.encode(X)
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
    def get_reconstructions(self, X: Tensor):
        f_tilde, pi_gate, _ = self.encode(X)
        X_hat, _, _ = self.decode(f_tilde, pi_gate, X)
        return X_hat
    