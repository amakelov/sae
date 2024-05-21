import torch
from torch import nn, Tensor
from fancy_einsum import einsum
from torch.nn import functional as F

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from transformer_lens import utils

class VanillaAutoEncoder(nn.Module):
    def __init__(self, 
                 d_activation: int,
                 d_hidden: int,
                 enc_dtype: str = "fp32",
                 freeze_decoder: bool = False,
                 ):
        super().__init__()
        self.d_activation = d_activation
        self.d_hidden = d_hidden
        self.freeze_decoder = freeze_decoder
        dtype = torch.float32 if enc_dtype == "fp32" else torch.float16
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_activation, self.d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.d_hidden, d_activation, dtype=dtype)))
        if freeze_decoder:
            self.W_dec.requires_grad = False
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_activation, dtype=dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
    
    def forward_detailed(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec

        l2_losses = (x_reconstruct.float() - x.float()).pow(2).sum(-1)
        l1_losses = acts.float().abs().sum(-1)
        return x_reconstruct, acts, l2_losses, l1_losses
    
    def forward(self, x):
        # x_cent = x - self.b_dec
        # acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        # x_reconstruct = acts @ self.W_dec + self.b_dec

        # l2_losses = (x_reconstruct.float() - x.float()).pow(2).sum(-1)
        # l1_losses = acts.float().abs().sum(-1)
        x_reconstruct, acts, l2_losses, l1_losses = self.forward_detailed(x)
        
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


class GatedAutoEncoder(nn.Module):
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
        pi_gate = einsum(self.W_gate, X_centered, 'dim hidden, batch dim -> batch hidden') + self.b_gate
        f_gate = (pi_gate > 0).float()
        W_mag = einsum(torch.exp(self.r_mag), self.W_gate, 'hidden, dim hidden -> dim hidden')
        f_mag = einsum(W_mag, X_centered, 'dim hidden, batch dim -> batch hidden') + self.b_mag
        f_tilde = f_gate * f_mag
        L_sparsity = nn.ReLU()(pi_gate).norm(dim=1, p=1).mean()
        return f_tilde, pi_gate, L_sparsity
    
    def decode(self, f_tilde: Tensor, pi_gate: Tensor, X: Tensor):
        x_hat = einsum(f_tilde, self.W_dec, 'batch hidden, hidden dim -> batch dim') + self.b_dec
        L_reconstruct = (x_hat - f_tilde).norm(dim=1, p=2).mean()
        
        # compute the auxiliary loss
        W_dec_clone = self.W_dec.clone().detach()
        b_dec_clone = self.b_dec.clone().detach()
        x_hat_frozen = einsum(
            nn.ReLU()(pi_gate), W_dec_clone, 'batch hidden, hidden dim -> batch dim'
        ) + b_dec_clone
        L_aux = (X - x_hat_frozen).norm(dim=1, p=2).mean()
        return x_hat, L_reconstruct, L_aux

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        if self.freeze_decoder:
            return
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed
    