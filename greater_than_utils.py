from torch.optim.lr_scheduler import _LRScheduler
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from functools import partial
import altair as alt
alt.data_transformers.disable_max_rows()
from torch.utils.data import Dataset, DataLoader
import math
import inspect
from tqdm import tqdm
from collections import defaultdict
import functools
from collections import OrderedDict
from abc import ABC, abstractmethod
import json
from pathlib import Path
import random
from typing import Tuple, List, Sequence, Union, Any, Optional, Literal, Iterable, Callable, Dict
import typing

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('talk')
sns.set_style('darkgrid')
sns.set_palette('muted')

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Parameter
from torch import nn
from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Float as JaxFloat
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.parametrizations import orthogonal
from torch.nn import functional as F
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint
from fancy_einsum import einsum


from ioi_utils import Node, HookedTransformer, batched
import torch
from typing import List, Union, Sequence
import numpy as np
import random

from torch import Tensor
from mandala._next.imports import *
from mandala._next.common_imports import *

MODELS = {}
MODEL_ID = 'gpt2-small'

### reproduce the main aspects of the dataset of https://arxiv.org/pdf/2305.00586
NOUNS = joblib.load('gt_nouns.joblib')

# figure out which years are tokenized the way we want
prompt = "The war lasted from the year 1732 to the year 17"

HEAD_LOCATIONS = [(5, 1), (5, 5), (6, 1), (6, 9), (7, 10), (8, 8), (8, 11)]
NODES = [Node(component_name='z', layer=layer, head=head, seq_pos=-1) for layer, head in HEAD_LOCATIONS]
YYS = [str(x)[2:] for x in range(1701, 1720)]

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


class Prompt:
    def __init__(self,
                 noun: str, 
                 yy: str,
                 xx: str = '17', # for simplicity, everything is set in the 18th century
                 ):
        """
        - xx: the century
        - yy: the last two digits of the year
        """
        self.noun = noun
        self.xx = xx
        self.yy = yy
    
    def __repr__(self) -> str:
        return f"Prompt({self.sentence})"

    @property
    def sentence(self) -> str:
        return f"The {self.noun} lasted from the year {self.xx}{self.yy} to the year {self.xx}"
    
    def with_changed_yy(self, yy: str) -> 'Prompt':
        return Prompt(noun=self.noun, xx=self.xx, yy=yy)

class PromptDistribution:
    """
    A class to represent a distribution over prompts.
    """

    def __init__(
        self,
        yys: List[str],
        nouns: List[str],
    ):
        self.yys = yys
        self.nouns = nouns

    def sample_one(self,) -> Prompt:
        """
        Sample a single prompt from the distribution.
        """
        yy = random.choice(self.yys)
        noun = random.choice(self.nouns)
        return Prompt(noun=noun, yy=yy)


FULL_DISTRIBUTION = PromptDistribution(yys=YYS, nouns=NOUNS)


class PromptDataset:
    def __init__(self, prompts: List[Prompt], model: HookedTransformer):
        # assert len(prompts) > 0
        self.prompts = np.array(prompts)
        self.model = model
        ls = self.lengths
        sess.d()
        if not all(x == ls[0] for x in ls):
            raise ValueError("Prompts must all have the same length")

    def __getitem__(self, idx: Union[int, Sequence, slice]) -> "PromptDataset":
        if isinstance(idx, int):
            prompts = [self.prompts[idx]]
        else:
            prompts = self.prompts[idx]
            if isinstance(prompts, Prompt):
                prompts = [prompts]
        assert all(isinstance(x, Prompt) for x in prompts)
        return PromptDataset(prompts=prompts, model=self.model)

    def __len__(self) -> int:
        return len(self.prompts)

    def __repr__(self) -> str:
        return f"{[x for x in self.prompts]}"

    @property
    def lengths(self) -> List[int]:
        return [self.model.to_tokens(x.sentence).shape[1] for x in self.prompts]

    @property
    def tokens(self) -> Tensor:
        return self.model.to_tokens([x.sentence for x in self.prompts])

@op
@batched(args=['prompts'], n_outputs=1, reducer='cat')
def run_with_cache(
    prompts: Any, 
    nodes: MList[Node],
    batch_size: int,
    model_id: str = MODEL_ID,
    verbose: bool = True,
    return_logits: bool = False,
    offload_to_cpu: bool = False,
    clear_cache: bool = False,
) -> MList[Tensor]:
    """
    Run the model on the given prompts, and return the activations for the
    given nodes.
    """
    model = MODELS[model_id]
    if len(prompts) % batch_size != 0:
        raise ValueError(f"Number of prompts ({len(prompts)}) must be a multiple of batch_size ({batch_size})")
    prompt_dataset = PromptDataset(prompts=prompts, model=model)
    logits, cache = model.run_with_cache(prompt_dataset.tokens, names_filter=Node.get_names_filter(nodes))
    # model.reset_hooks() ---> this is potentially confusing
    # return {node: node.get_value(cache, prompts=prompts) for node in nodes}
    acts = [node.get_value(cache, prompts=prompts) for node in nodes]
    if return_logits:
        res = acts + [logits]
    else:
        res = acts
    if offload_to_cpu:
        res = [x.cpu() for x in res]
    if clear_cache:
        torch.cuda.empty_cache()
    return res




FEATURE_SUBSETS = [
    ('yy',),
]
FEATURE_SUBSETS = [tuple(sorted(x)) for x in FEATURE_SUBSETS]

def get_yy_to_idx(distribution: PromptDistribution) -> Dict[str, int]:
    # return {name: i for i, name in enumerate(distribution.names)}
    return {yy: i for i, yy in enumerate(distribution.yys)}

# setup to work with features
YY_TO_IDX = get_yy_to_idx(FULL_DISTRIBUTION)

FEATURE_SIZES = {
    'bias': 1,
    'yy': len(YY_TO_IDX),
}

# this collects possible ways to parametrize activations of the model
FEATURE_CONFIGURATIONS = {
    'independent': [('yy', ), ],
}
# add bias to each
FEATURE_CONFIGURATIONS = {k: [('bias', )] + v for k, v in FEATURE_CONFIGURATIONS.items()}

@op
def generate_yy_samples(n_samples, yys, random_seed: int = 0) -> Any:
    np.random.seed(random_seed)
    return np.random.choice(yys, n_samples, replace=True)

@op
def get_cf_prompts(
    prompts: List[Prompt],
    features: Tuple[str, ...],
    yy_targets: List[str],
) -> Any:
    assert features == ('yy',)
    return [p.with_changed_yy(yy=yy) for p, yy in zip(prompts, yy_targets)]

@op
def generate_prompts(distribution: PromptDistribution, random_seed: int, n_prompts: int,
                     ) -> Any:
    random.seed(random_seed)
    return [distribution.sample_one() for _ in range(n_prompts)]

################################################################################
### working with features
################################################################################
def get_prompt_representation(p: Prompt) -> Dict[str, int]:
    # extracts feature values from a prompt
    return {'yy': YY_TO_IDX[p.yy], 'bias': 0} 

def get_prompt_feature_vals(p: Prompt) -> Dict[str, Any]:
    return { 'yy': p.yy , 'bias': 0}

def get_feature_shape(feature: Tuple[str,...]) -> Tuple[int, ...]:
    return tuple(FEATURE_SIZES[f] for f in feature)

def get_feature_deep_idx(feature: Tuple[str,...], prompt_rep: Dict[str, int]) -> Tuple[int,...]:
    # given the feature values for a prompt, returns the index we should use to
    # index into a code representing that feature (without the last dimension,
    # which is the dimension of the code vectors and may vary)
    return tuple([prompt_rep[f] for f in feature])

@op
def get_prompt_feature_idxs(prompts: Optional[Sequence[Prompt]],
                            features: List[Tuple[str,...]],
                            prompt_reps: Optional[List[dict]] = None,
                            ) -> Dict[Tuple[str,...], Tensor]:
    """
    Return a dictionary mapping each feature to a batch of indices into the
    code for this feature over the prompts (indices don't take into account the
    last dimension in the codes, which is the dimension of the code vectors and
    may vary).
    """
    if prompt_reps is None:
        assert prompts is not None
        prompt_reps = [get_prompt_representation(p) for p in prompts]
    prompt_feature_idxs = {f: torch.tensor([get_feature_deep_idx(f, prompt_rep)
                                            for prompt_rep in prompt_reps])
                            for f in features}
    return prompt_feature_idxs

def get_reconstructions(
    codes: Any, # Dict[tuple, Tensor],
    prompt_feature_idxs: Optional[Dict[tuple, Tensor]] = None,
    prompts: Optional[Sequence[Prompt]] = None,
    decomposed: bool = False,
    ) -> Union[Tensor, Dict[tuple, Tensor]]:
    """
    Reconstruct prompts according to the given codes. if prompt_feature_idxs is
    not given, it will be computed from prompts.
    """
    prompt_vectors = {}
    if prompt_feature_idxs is None:
        assert prompts is not None
        prompt_feature_idxs = get_prompt_feature_idxs(prompts, codes.keys())
    for f, idx in prompt_feature_idxs.items():
        # add the last dimension
        full_idx = tuple([idx[:, i].cpu().numpy() for i in range(idx.shape[1])] + [slice(None, None, None)])
        prompt_vectors[f] = codes[f][full_idx]
    if not decomposed:
        return sum(prompt_vectors.values())
    else:
        return prompt_vectors

################################################################################
### computing codes
################################################################################
@op
def get_mean_codes(
    features: List[Tuple[str,...]],
    A: Tensor,
    prompts: Any, # List[Prompt]
) -> Tuple[Any, Any]:
    """
    Compute codes using the mean of the (centered) activations for a given
    feature value.
    """
    # get the shape of the code for each feature
    feature_shapes = {f: get_feature_shape(f) for f in features}
    dim = A.shape[1]
    feature_shapes = {f: tuple(list(feature_shapes[f]) + [dim]) for f in features}
    # get the attributes of the prompts
    prompt_feature_idxs = get_prompt_feature_idxs.f(prompts=prompts, features=features)
    # group the prompt feature indices by feature value:
    prompt_feature_groups = {
        f: {} # will be {value in feature_idxs: [indices where this value appears in feature_idxs]}
        for f in features
    }
    for f, feature_idxs in prompt_feature_idxs.items():
        # populate prompt_feature_groups[f] = {value in feature_idxs: [indices
        # where this value appears in feature_idxs]}
        for idx, feature_idx in enumerate(feature_idxs):
            value = tuple([x.item() for x in feature_idx])
            if value not in prompt_feature_groups[f]:
                prompt_feature_groups[f][value] = []
            prompt_feature_groups[f][value].append(idx)
        # convert to tensors
        for value, indices in prompt_feature_groups[f].items():
            prompt_feature_groups[f][value] = torch.tensor(indices)
    codes = {
        f: torch.zeros(feature_shape).cuda() for f, feature_shape in feature_shapes.items()
    }
    A_mean = A.mean(dim=0)
    if ('bias',) in features:
        codes[('bias',)] = A_mean.unsqueeze(0)
    A_centered = A - A_mean
    for f, groups in prompt_feature_groups.items():
        if f != ('bias',):
            for value, indices in groups.items():
                codes[f][value] = A_centered[indices].mean(dim=0)
    reconstructions = get_reconstructions(
        codes=codes, prompt_feature_idxs=prompt_feature_idxs,
    )
    return codes, reconstructions


################################################################################
### feature editing
################################################################################
def get_edited_act(
    val: Tensor,
    method: str,
    feature_idxs_to_delete: Dict[Tuple[str,...], List[Tuple[int,...]]],
    feature_idxs_to_insert: Dict[Tuple[str,...], List[Tuple[int,...]]],
    codes: Dict[Tuple[str,...], Tensor],
    A_reference: Optional[Tensor] = None,
):
    """
    The core editing function: perform one or several edits on an activation
    using the given codes and method.

    Note that the methods based on subspace ablations are not commutative with
    respect to the order of feature insertion/deletion. 
    """
    val = val.clone()
    if method == 'arithmetic':
        for f, idx_to_delete in feature_idxs_to_delete.items():
            val = val - torch.stack([codes[f][i] for i in idx_to_delete])
        for f, idx_to_insert in feature_idxs_to_insert.items():
            val = val + torch.stack([codes[f][i] for i in idx_to_insert])
    elif method == 'zero_ablate_subspace':
        for f, idx_to_delete in feature_idxs_to_delete.items():
            code_to_delete = torch.stack([codes[f][i] for i in idx_to_delete])
            code_to_delete = code_to_delete / code_to_delete.norm(dim=-1, keepdim=True)
            projections = einsum('batch dim, batch dim -> batch', val, code_to_delete)
            val = val + einsum('batch, batch dim -> batch dim', - projections, code_to_delete)
        for f, idx_to_insert in feature_idxs_to_insert.items():
            val = val + torch.stack([codes[f][i] for i in idx_to_insert])
    elif method == 'mean_ablate_subspace':
        assert A_reference is not None
        for f, idx_to_delete in feature_idxs_to_delete.items():
            code_to_delete = torch.stack([codes[f][i] for i in idx_to_delete])
            code_to_delete = code_to_delete / code_to_delete.norm(dim=-1, keepdim=True)
            # mean_projection = (A_reference @ code_to_delete).mean(dim=0)
            reference_projections = einsum('dim, batch dim -> batch', A_reference.mean(dim=0), code_to_delete)
            projections = einsum('batch dim, batch dim -> batch', (reference_projections.unsqueeze(1) - val), code_to_delete)
            val = val + einsum('batch, batch dim -> batch dim', projections, code_to_delete)
        for f, idx_to_insert in feature_idxs_to_insert.items():
            val = val + torch.stack([codes[f][i] for i in idx_to_insert])
    else:
        raise ValueError(f'unknown method {method}')
    return val


@op
def get_cf_edited_act(
    val: Tensor,
    features_to_edit: Tuple[str,...],
    base_prompts: Any, # List[Prompt],
    cf_prompts: Any, # List[Prompt],
    codes: Any, # Dict[Tuple[str,...], Tensor],
    method: Literal['mean_ablate_subspace', 'zero_ablate_subspace', 'arithmetic'],
    A_ref: Optional[Tensor] = None,
) -> Tensor:
    """
    Edit an activation using the given counterfactual prompts in order to infer
    the new values for the features being edited.
    
    - features_to_edit is a tuple representing the features we want to change,
    e.g. ('s', 'io_pos',). The new values for these features will be inferred
    from the counterfactual prompts. Then, we use the features in the codes to
    figure out how to edit the activation.
    """
    code_features = list(codes.keys())
    base_feature_idxs = get_prompt_feature_idxs(prompts=base_prompts, 
                                                features=code_features, )
    cf_feature_idxs = get_prompt_feature_idxs(prompts=cf_prompts,
                                              features=code_features, )

    def turn_tensor_to_tuples(t: Tensor) -> List[Tuple[int,...]]:
        return [tuple(x.cpu().tolist()) for x in t]
    
    base_feature_idxs = {k: turn_tensor_to_tuples(v) for k, v in base_feature_idxs.items()}
    cf_feature_idxs = {k: turn_tensor_to_tuples(v) for k, v in cf_feature_idxs.items()}

    edited_act = get_edited_act(
        val=val,
        codes=codes,
        feature_idxs_to_delete=base_feature_idxs,
        feature_idxs_to_insert=cf_feature_idxs,
        A_reference=A_ref,
        method=method,
    )
    return edited_act

def get_forced_hook(
    prompts: List[Prompt],
    node: Node, 
    A: Tensor,
) -> Tuple[str, Callable]:
    """
    Get a hook that forces the activation of the given node to be the given value.
    """
    def hook_fn(activation: Tensor, hook: HookPoint) -> Tensor:
        idx = node.idx(prompts=prompts)
        activation[idx] = A
        return activation
    return (node.activation_name, hook_fn)

def get_model_obj(model_id: str) -> HookedTransformer:
    return MODELS[model_id]

@op
def run_activation_patch(
    base_prompts: Any, # List[Prompt],
    cf_prompts: Any, # List[Prompt],
    nodes: List[Node],
    activations: List[Tensor],
    batch_size: int,
    model_id: str = MODEL_ID,
    return_predictions: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Run a standard activation patch in a batched way
    """
    model = get_model_obj(model_id)
    assert all([len(base_prompts) == v.shape[0] for v in activations])
    n = len(base_prompts)
    n_batches = (n + batch_size - 1) // batch_size
    base_logits_list = []
    cf_logits_list = []
    predictions_list = []
    for i in tqdm(range(n_batches)):
        batch_indices = slice(i * batch_size, (i + 1) * batch_size)
        prompts_batch = base_prompts[batch_indices]
        cf_batch = cf_prompts[batch_indices]
        base_dataset = PromptDataset(prompts_batch, model=model)
        cf_dataset = PromptDataset(cf_batch, model=model)
        hooks = [get_forced_hook(prompts=prompts_batch, node=node, A=act[batch_indices]) for node, act in zip(nodes, activations)]
        changed_logits = model.run_with_hooks(base_dataset.tokens, fwd_hooks=hooks)[:, -1, :]
        # base_answer_logits = changed_logits.gather(dim=-1, index=base_dataset.answer_tokens.cuda())
        # cf_answer_logits = changed_logits.gather(dim=-1, index=cf_dataset.answer_tokens.cuda())
        # base_logits_list.append(base_answer_logits)
        # cf_logits_list.append(cf_answer_logits)
        predictions = changed_logits.argmax(dim=-1)
        predictions_list.append(predictions)
    # base_logits = torch.cat(base_logits_list, dim=0)
    # cf_logits = torch.cat(cf_logits_list, dim=0)
    predictions = torch.cat(predictions_list, dim=0)
    if return_predictions:
        return None, (None, predictions) #! lol, lmfaoooo
    else:
        return None, (None, None)


def get_probability_difference(
        logits: Tensor, # of shape (batch, vocab), last token logits
        yy_values: List[str],
        yy_token_idxs: List[int],
        yys: List[str], # of shape (batch, )
        ):
    """
    The metric used to discover the circuit in the paper is the *probability*
    difference, and we adopt it here as well.
    """
    yy_idx_in_list = yy_values.index(yy)
    tokens_gt_yy = Tensor(yy_token_idxs[yy_idx_in_list + 1:]).to(logits.device)
    tokens_lte_yy = Tensor(yy_token_idxs[:yy_idx_in_list + 1]).to(logits.device)
    logits_gt_yy = logits[:, tokens_gt_yy]



################################################################################
### everything else
################################################################################
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

    activations = {}
    grads = {}

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

    activations = {}
    grads = {}

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
### some utils for faster interp computations
################################################################################
def get_feature_mask(feature: Tuple[str,...],
                     prompt_feature_idxs: Dict[str, List[Tuple[int,...]]],
                     ) -> Tensor:
    if feature == ('name_anywhere',):
        assert ('io',) in prompt_feature_idxs
        assert ('s',) in prompt_feature_idxs
        num_examples = prompt_feature_idxs[('io',)].shape[0]
        mask = torch.zeros(num_examples, FEATURE_SIZES[feature[0]]).cuda()
        mask[range(num_examples), prompt_feature_idxs[('io',)][:, 0]] = 1
        mask[range(num_examples), prompt_feature_idxs[('s',)][:, 0]] = 1
        sess.d()
    else:
        feature_idxs = prompt_feature_idxs[feature]
        feature_shape = tuple([FEATURE_SIZES[f] for f in feature])
        num_examples = len(feature_idxs)
        mask = torch.zeros((num_examples, *feature_shape)).cuda()
        # create an indexing object into the mask that uses range(num_examples) for the 
        # first dimension and the feature_idxs for the rest of the dimensions
        if len(feature) == 1:
            mask[range(num_examples), feature_idxs[:, 0]] = 1
        elif len(feature) == 2:
            mask[range(num_examples), feature_idxs[:, 0], feature_idxs[:, 1]] = 1
        else:
            raise NotImplementedError
    return mask

def get_feature_scores(activation_pattern: Tensor, feature_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    intersections = einsum("batch hidden, batch ... -> hidden ...", activation_pattern, feature_mask)
    feature_supports = feature_mask.sum(dim=0)
    activation_supports = activation_pattern.sum(dim=0)
    num_feature_dimensions = len(feature_mask.shape) - 1
    # insert extra dimensions in `activation_supports`
    for i in range(num_feature_dimensions):
        activation_supports = activation_supports.unsqueeze(-1)
    precision = intersections / activation_supports
    recall = intersections / feature_supports
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def multidim_argmax(t: Tensor, ) -> Tuple[Tuple[Tensor, ...], Tensor]:
    # compute argmax over all dimensions but the first
    t = t.clone()
    # set the nan values to -inf
    t[torch.isnan(t)] = float('-inf')
    t_reshaped = t.view(t.shape[0], -1)
    argmax_indices = t_reshaped.argmax(dim=1)
    argmax_values = t_reshaped[range(t_reshaped.shape[0]), argmax_indices]
    # go back to original shape
    if len(t.shape) == 2:
        return (argmax_indices,), argmax_values
    elif len(t.shape) == 3:
        return (argmax_indices // t.shape[2], argmax_indices % t.shape[2]), argmax_values
    else:
        raise NotImplementedError

def multidim_topk(t: Tensor, k: int) -> Tuple[Tensor, ...]:
    # compute topk over all dimensions but the first
    t = t.clone()
    # set the nan values to -inf
    t[torch.isnan(t)] = float('-inf')
    t_reshaped = t.view(t.shape[0], -1)
    topk_indices = t_reshaped.topk(k, dim=1).indices
    # go back to original shape
    if len(t.shape) == 2:
        return topk_indices,
    elif len(t.shape) == 3:
        return topk_indices // t.shape[2], topk_indices % t.shape[2]
    else:
        raise NotImplementedError


################################################################################
### main ops
################################################################################
@op(__allow_side_effects__=True) # again, because of non-deterministic nn.Module hashes
def get_high_f1_features(
    encoder: Union[VanillaAutoEncoder, GatedAutoEncoder, AttributionAutoEncoder],
    attributes: List[Tuple[str,...]],
    prompt_feature_idxs: Any,
    A_normalized: Tensor, # must be normalized for the encoder
    topk: int,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Given some attributes, find the top features wrt these attributes based on
    the F1 score; return the features together with their scores.

    Returns: 
    - {attr: tensor of shape (*attr_shape, topk) of the top features (ordered)}
    - {attr: tensor of shape (*attr_shape, topk) of the top F1 scores (same order)}
    """
    activation_pattern = encoder.get_activation_pattern(A=A_normalized).float()
    masks = {attr: get_feature_mask(attr, prompt_feature_idxs=prompt_feature_idxs) for attr in attributes} # f -> (num_examples, *feature_shape)
    # attr -> (_, _, f1 scores) where f1 scores has shape (num_features, num_attr_values)
    f1_scores = {attr: get_feature_scores(activation_pattern, masks[attr])[2] for attr in attributes}
    for attr in f1_scores.keys():
        t = f1_scores[attr].clone()
        t[torch.isnan(t)] = float('-inf')
        f1_scores[attr] = t
    features_and_scores = {attr: torch.topk(v, k=topk, dim=0) for attr, v in f1_scores.items()}
    features = {attr: v.indices.T for attr, v in features_and_scores.items()}
    f1_scores = {attr: v.values.T for attr, v in features_and_scores.items()}
    return features, f1_scores


@op(__allow_side_effects__=True) # again, because of non-deterministic nn.Module hashes
def autointerp_fast(
    encoder: Union[VanillaAutoEncoder, GatedAutoEncoder, AttributionAutoEncoder],
    features: List[Tuple[str,...]],
    prompt_feature_idxs: Any, 
    A_normalized: Tensor,
    features_to_group: List[Tuple[str,...]],
    feature_batch_size: Optional[int] = None,
    max_group_size: int = 10,
    ) -> Tuple[Any, Any, Any, Any]:
    """
    Score features according to the F1 score, with greedy search over subsets of
    up to `max_group_size` attribute values. Return:
    - the best F1 score and corresponding index for each feature
    - the best F1 score and corresponding indices for each group of features
    """
    activation_pattern = encoder.get_activation_pattern(A=A_normalized).float()
    n_examples = A_normalized.shape[0]
    n_features = activation_pattern.shape[1]
    masks = {f: get_feature_mask(f, prompt_feature_idxs=prompt_feature_idxs) for f in features} # f -> (num_examples, *feature_shape)

    if feature_batch_size is None:
        activation_pattern_batches = [activation_pattern]
    else:
        activation_pattern_batches = [activation_pattern[:, i:i+feature_batch_size] for i in range(0, n_features, feature_batch_size)]
    
    top_features_list = []
    top_scores_list = []
    group_features_list = []
    group_scores_list = []

    for activation_pattern in activation_pattern_batches:
        scores = {f: get_feature_scores(activation_pattern, masks[f]) for f in features}
        # collect the best features according to F1 score
        top_features_and_scores = {f: multidim_argmax(scores[f][2]) for f in features}
        top_features = {f: v[0] for f, v in top_features_and_scores.items()}
        top_scores = {f: v[1] for f, v in top_features_and_scores.items()}
        # now, for the features where we look for groups, compute the scores for the
        # top 1, 2, ..., `max_group_size` elements
        group_scores = {}
        group_features = {}
        for f in features_to_group:
            f1 = scores[f][2]
            topk_indices = multidim_topk(f1, max_group_size)
            group_features[f] = topk_indices
            # compute the masks for each group
            if len(topk_indices) == 1:
                subgroup_mask = torch.cumsum(masks[f][:, topk_indices[0]], dim=2)
            elif len(topk_indices) == 2:
                subgroup_mask = torch.cumsum(masks[f][:, topk_indices[0], topk_indices[1]], dim=2)
            else:
                raise NotImplementedError
            # compute scores wrt grouped mask
            group_intersections = einsum("batch hidden, batch hidden group_size -> hidden group_size", activation_pattern, subgroup_mask)
            recall = group_intersections / subgroup_mask.sum(dim=0)
            precision = group_intersections / activation_pattern.sum(dim=0).unsqueeze(-1)
            group_f1 = 2 * precision * recall / (precision + recall + 1e-8)
            group_scores[f] = group_f1
        top_features_list.append(top_features)
        top_scores_list.append(top_scores)
        group_features_list.append(group_features)
        group_scores_list.append(group_scores)

    if feature_batch_size is not None:
        for f in features:
            num_elts_in_tuple = len(top_features_list[0][f])
            if num_elts_in_tuple == 1:
                top_features[f] = (torch.cat([d[f][0] for d in top_features_list], dim=0),)
                top_scores[f] = (torch.cat([d[f] for d in top_scores_list], dim=0),)
            elif num_elts_in_tuple == 2:
                top_features[f] = (torch.cat([d[f][0] for d in top_features_list], dim=0), torch.cat([d[f][1] for d in top_features_list], dim=0))
                top_scores[f] = (torch.cat([d[f] for d in top_scores_list], dim=0), torch.cat([d[f] for d in top_scores_list], dim=0))
        for f in features_to_group:
            num_elts_in_tuple = len(group_features_list[0][f])
            if num_elts_in_tuple == 1:
                group_features[f] = (torch.cat([d[f][0] for d in group_features_list], dim=0),)
                group_scores[f] = (torch.cat([d[f] for d in group_scores_list], dim=0),)
            elif num_elts_in_tuple == 2:
                group_features[f] = (torch.cat([d[f][0] for d in group_features_list], dim=0), torch.cat([d[f][1] for d in group_features_list], dim=0))
                group_scores[f] = (torch.cat([d[f] for d in group_scores_list], dim=0), torch.cat([d[f] for d in group_scores_list], dim=0))
    else:
        top_features = top_features_list[0]
        top_scores = top_scores_list[0]
        group_features = group_features_list[0]
        group_scores = group_scores_list[0]
    return top_features, top_scores, group_features, group_scores


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

################################################################################
### lol what is this doing here?
################################################################################
@op
def precompute_activations(X: Any, node: Node, batch_size: int, with_timeout: Optional[float] = None) -> torch.Tensor:
    if with_timeout is not None:
        A = run_custom(prompts=X, batch_size=batch_size, node=node, timeout=with_timeout)
    else:
        A = run_with_cache(prompts=X, nodes=[node], batch_size=batch_size, hook_specs=[],)[0]
    return A

@op
def get_dataset_mean(A: Tensor) -> Tensor:
    return A.mean(dim=0)

def mean_ablate_hook(activation: Tensor, hook: HookPoint, node: Node, mean: Tensor, idx: Tensor) -> Tensor:
    activation[idx] = mean
    return activation

# def encoder_hook(activation: Tensor, hook: HookPoint, node: Node, encoder: Union[VanillaAutoEncoder, GatedAutoEncoder], normalization_scale: Optional[float] = None, idx: Tensor) -> Tensor:
#                  idx: Tensor, normalization_scale: Optional[float] = None,
#                  ) -> Tensor:
#     with torch.no_grad():
#         if normalization_scale is not None:
#             A = activation[idx] / normalization_scale
#         else:
#             A = activation[idx]
#         reconstructions = encoder(A)[1]
#         if normalization_scale is not None:
#             reconstructions = reconstructions * normalization_scale
#         activation[idx] = reconstructions
#     return activation


################################################################################
### editing interventions
################################################################################
def get_feature_weights(
    encoder: Union[VanillaAutoEncoder, GatedAutoEncoder],
    A: Tensor, 
    batch_size: int,
    ) -> Tuple[Tensor, Tensor]:
    """
    Given reconstruction x_hat = sum_j f_j d_j + b_dec, the weight of a feature
    j is w_j = f_j d_j^T (x_hat - b_dec) / ||x_hat - b_dec||^2.
    """
    recons = encoder.get_reconstruction(A)
    feature_magnitudes = encoder.get_feature_magnitudes(A)
    num_examples = A.shape[0]
    num_batches = num_examples // batch_size
    feature_weights_batches = []
    for i in range(num_batches): # to avoid OOM
        magnitudes_batch = feature_magnitudes[i*batch_size:(i+1)*batch_size]
        # this is x_hat - b_dec
        centered_recons_batch = recons[i*batch_size:(i+1)*batch_size] - encoder.b_dec.detach().unsqueeze(0)
        centered_recons_norms = centered_recons_batch.norm(dim=-1, keepdim=True)
        feature_weights = einsum('batch hidden, hidden dim, batch dim -> batch hidden', magnitudes_batch, encoder.W_dec.detach(), centered_recons_batch) / centered_recons_norms**2
        feature_weights_batches.append(feature_weights)
    feature_weights = torch.cat(feature_weights_batches, dim=0)
    sums = feature_weights.sum(dim=1)
    nonzero_sums = sums[sums != 0]
    ones = torch.ones_like(nonzero_sums)
    #! a sanity check
    assert torch.allclose(nonzero_sums, ones, atol=0.05), sums
    return feature_weights, feature_magnitudes

@op
def compute_total_feature_weight(
    encoder: Union[VanillaAutoEncoder, GatedAutoEncoder],
    A: Tensor,
    batch_size: int,
    feature_idxs: Tensor,  # of shape (batch, num_chosen_features)
    ) -> Tensor:
    """
    Compute the total weight of chosen features (could be different for each
    example). We use this as a proxy for the "magnitude" of an edit.
    """
    weights, _ = get_feature_weights(encoder, A, batch_size)
    selected_weights = torch.stack([weights[range(A.shape[0]), feature_idxs[:, i]] for i in range(feature_idxs.shape[1])], dim=1)
    return selected_weights.sum(dim=1) # of shape (batch,)


def get_top_k_features_per_prompt(
        attr_idxs_clean: Tensor, # shape (batch,)
        clean_active: Tensor, # boolean mask of shape (batch, n_features)
        high_f1_features: Tensor, # index tensor, shape (attribute_size, n_features)
        k: int,
) -> Tensor:
    n_examples = clean_active.shape[0]
    # of shape (batch, n_features)
    # provides the sorted indices of features, from highest to lowest F1, for the attribute
    prompt_high_f1_features = high_f1_features[attr_idxs_clean]

    # now, we permute the mask so that the entries are sorted according to the F1 score
    batch_indices = torch.arange(n_examples).cuda().unsqueeze(1).expand_as(clean_active)
    # now, we obtain a boolean mask where M[i, j] = True if the j-th highest F1 feature is active in the i-th prompt 
    clean_active_in_decreasing_f1_order = clean_active[batch_indices, prompt_high_f1_features]

    def get_indices_of_first_k_nonzeros(X: Tensor, k: int) -> Tensor:
        idx = torch.arange(X.shape[1], 0, -1).cuda()
        return torch.topk(X * idx, k=k, dim=1).indices

    indices_of_top_features = get_indices_of_first_k_nonzeros(clean_active_in_decreasing_f1_order, k=k)
    batch_indices = torch.arange(n_examples).cuda().unsqueeze(1).expand_as(indices_of_top_features)
    top_k_features_per_prompt = prompt_high_f1_features[batch_indices, indices_of_top_features]
    return top_k_features_per_prompt

@op(__allow_side_effects__=True) 
@batched(args=['A_clean', 'A_cf', 'clean_prompts', 'cf_prompts',], n_outputs=3, reducer='cat', verbose=False)
def get_edit_using_f1_scores(
    encoder: Union[VanillaAutoEncoder, GatedAutoEncoder, AttributionAutoEncoder],
    A_clean_normalized: Tensor,
    A_cf_normalized: Tensor,
    clean_prompts: Any, 
    cf_prompts: Any,
    clean_feature_idxs: Dict[Tuple[str,...], Tensor],
    cf_feature_idxs: Dict[Tuple[str,...], Tensor],
    attribute: Tuple[str,...],
    high_f1_features_dict: Dict[Tuple[str,...], Tensor],
    normalization_scale: float,
    num_exchange: int,
    batch_size: int = 100,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform an activation edit of a single attribute, using the features with
    high F1 score for the attribute as a guide. Specifically, 
    - `high_f1_features` is of shape (*attribute_shape, num_sae_features), and
    for each index into a value of the attribute contains the ordered list of
    the topk features with the highest F1 score for this value of the attribute
    - we subtract the `num_exchange` top-F1-score features from the clean
    activation
    - and add the `num_exchange` top-F1-score features from the counterfactual
    activation

    UPDATE in version 1:
    - we now only change the top `num_exchange` features PRESENT in the activations.
    UPDATE in version 2:
    - fix bug when determining the top features!

    NOTE: this returns the *unnormalized* edited activations, in their original
    scale.
    """
    high_f1_features = high_f1_features_dict[attribute]
    n_examples = A_clean_normalized.shape[0]
    magnitudes_clean = encoder.get_feature_magnitudes(A_clean_normalized)
    magnitudes_cf = encoder.get_feature_magnitudes(A_cf_normalized)

    # clean_feature_idxs = get_prompt_feature_idxs(
    #     prompts=clean_prompts,
    #     features=[attribute],
    # )
    # cf_feature_idxs = get_prompt_feature_idxs(
    #     prompts=cf_prompts, 
    #     features=[attribute],
    # )
    # now, figure out which features to add/remove
    attr_idxs_clean = clean_feature_idxs[attribute].squeeze()
    attr_idxs_cf = cf_feature_idxs[attribute].squeeze()
    assert len(attr_idxs_clean.shape) == 1
    ### take the first num_exchange features present in the activations
    def pad_2d_boolean_mask(mask: Tensor, desired_count: int) -> Tensor:
        res = mask.clone()
        for i in range(res.shape[0]):
            current_count = res[i].sum().item()
            num_to_set_true = desired_count - current_count
            if num_to_set_true <= 0:
                continue
            else:
                res[i, torch.where(~res[i])[0][:num_to_set_true]] = True
        return res
    clean_active = (magnitudes_clean > 0).bool()
    cf_active = (magnitudes_cf > 0).bool()
    # unnecessary w/ new vectorized implementation
    # clean_active = pad_2d_boolean_mask(clean_active, num_exchange)
    # cf_active = pad_2d_boolean_mask(cf_active, num_exchange)




    ### slow way to get the features to add/remove
    # features_to_remove = []
    # for i in range(n_examples):
    #     t = high_f1_features[attr_idxs_clean[i]]
    #     features_to_remove.append(t[torch.isin(t, torch.where(clean_active[i])[0])][:num_exchange])
    # features_to_remove = torch.stack(features_to_remove, dim=0).long()
    # features_to_add = []
    # for i in range(n_examples):
    #     t = high_f1_features[attr_idxs_cf[i]]
    #     features_to_add.append(t[torch.isin(t, torch.where(cf_active[i])[0])][:num_exchange])
    # features_to_add = torch.stack(features_to_add, dim=0).long()

    ### vectorized way to get the features to add/remove
    features_to_remove = get_top_k_features_per_prompt(attr_idxs_clean, clean_active, high_f1_features, num_exchange)
    features_to_add = get_top_k_features_per_prompt(attr_idxs_cf, cf_active, high_f1_features, num_exchange)

    ### now, perform the edits
    W_dec = encoder.W_dec.detach() # (hidden, dim)

    # shape (batch, num_exchange)
    coeffs_to_remove = torch.stack([magnitudes_clean[range(n_examples), features_to_remove[:, i]] for i in range(num_exchange)], dim=1)
    coeffs_to_add = torch.stack([magnitudes_cf[range(n_examples), features_to_add[:, i]] for i in range(num_exchange)], dim=1)

    # shape (batch, num_exchange, dim)
    decoders_to_remove = W_dec[features_to_remove, :]
    decoders_to_add = W_dec[features_to_add, :]

    to_remove = einsum("batch num_exchange, batch num_exchange dim -> batch dim", coeffs_to_remove, decoders_to_remove)
    to_add = einsum("batch num_exchange, batch num_exchange dim -> batch dim", coeffs_to_add, decoders_to_add)

    A_edited_normalized = A_clean_normalized - to_remove + to_add
    # A_edited of shape (batch, dim)
    # features_to_remove: indices of the features to remove, of shape (batch, num_exchange)
    # features_to_add: indices of the features to add, of shape (batch, num_exchange)
    return A_edited_normalized * normalization_scale, features_to_remove, features_to_add


@op(__allow_side_effects__=True)
@batched(args=['A_clean_normalized', 'A_cf_normalized'], n_outputs=5, reducer='cat', verbose=False)
def get_edit_using_sae_opt(
    A_clean_normalized: Tensor,
    A_cf_normalized: Tensor,
    encoder: Union[VanillaAutoEncoder, GatedAutoEncoder, AttributionAutoEncoder],
    num_exchange: int,
    normalization_scale: float,
    diff_to_use: Literal['reconstruction', 'activation'] = 'activation', # gives better results
    batch_size: int = 100,
    ) -> Tuple[Tensor, Any, Any, Tensor, Tensor]:
    """
    Greedily solve the optimization problem of subtracting/adding the fewest 
    features to minimize the norm. 
    """
    n_examples = A_clean_normalized.shape[0]
    recons_clean = encoder.get_reconstructions(A_clean_normalized)
    recons_cf = encoder.get_reconstructions(A_cf_normalized)
    magnitudes_clean = encoder.get_feature_magnitudes(A_clean_normalized)
    magnitudes_cf = encoder.get_feature_magnitudes(A_cf_normalized)

    if diff_to_use == 'reconstruction':
        diff = recons_cf - recons_clean # shape (batch, dim)
    elif diff_to_use == 'activation':
        diff = A_cf_normalized - A_clean_normalized
    else:
        raise ValueError(f"Invalid value for `diff_to_use`: {diff_to_use}")
    
    W_dec = encoder.W_dec.detach().clone()

    def optimize_vectorized(num_exchange:int):
        current_sums = torch.zeros_like(diff) # shape (batch, dim)
        best_features_list = []
        best_scores_list = []
        # initialize the *differences* between each respective feature's
        # contribution in the cf and clean reconstructions
        feature_diffs = einsum('batch hidden, hidden dim -> batch hidden dim', magnitudes_cf-magnitudes_clean, W_dec)
        for i in range(num_exchange):
            a = current_sums.unsqueeze(1) + feature_diffs - diff.unsqueeze(1) # shape (batch, hidden, dim)
            scores = a.norm(dim=-1) # (batch, hidden)
            best_features = scores.argmin(dim=1) # (batch,)
            best_scores = scores[torch.arange(n_examples), best_features] # (batch,)
            best_scores_list.append(best_scores)
            best_features_list.append(best_features)
            current_sums += feature_diffs[torch.arange(n_examples), best_features, :] # (batch, dim)
            # set the contributions of the features we edited to zero to avoid them in the next round
            feature_diffs[torch.arange(n_examples), best_features, :] = 0.0
        # the features we changed during opt
        best_features = torch.stack(best_features_list, dim=1) # of shape (num_examples, n_exchange)
        best_scores = torch.stack(best_scores_list, dim=1)
        # the hidden activations of the edited features on the clean side
        edited_clean = torch.stack([magnitudes_clean[range(n_examples), best_features[:, i]] for i in range(num_exchange)], dim=1)
        # the hidden activations of the edited features on the cf side
        edited_cf = torch.stack([magnitudes_cf[range(n_examples), best_features[:, i]] for i in range(num_exchange)], dim=1)
        return best_features, best_scores, current_sums, edited_clean, edited_cf
    best_features, best_scores, deltas, edited_clean, edited_cf = optimize_vectorized(num_exchange)
    A_edited_normalized = A_clean_normalized + deltas
    return A_edited_normalized * normalization_scale, best_features, best_scores, edited_clean, edited_cf


# @op
# def get_sae_reconstructions(
#     encoder: Union[AutoEncoder, SparseAutoencoder],
#     A: torch.Tensor,
#     normalization_scale: Optional[torch.Tensor],
# ) -> Tensor:
#     is_webtext_sae = get_is_webtext_sae(encoder=encoder)
#     use_normalization = (normalization_scale is not None and (not is_webtext_sae))
#     if use_normalization:
#         A = A / normalization_scale
#     with torch.no_grad():
#         if is_webtext_sae:
#             encoder = encoder.to(A.device)
#             recons, _ = encoder(A)
#         else:
#             recons = encoder(A)[1]
#     if use_normalization:
#         recons = recons * normalization_scale
#     return recons

def get_forced_hook(
    prompts: List[Prompt],
    node: Node, 
    A: Tensor,
) -> Tuple[str, Callable]:
    """
    Get a hook that forces the activation of the given node to be the given value.
    """
    def hook_fn(activation: Tensor, hook: HookPoint) -> Tensor:
        idx = node.idx(prompts=prompts)
        activation[idx] = A
        return activation
    return (node.activation_name, hook_fn)

def remove_features(
    A: Tensor,
    encoder: Union[VanillaAutoEncoder, GatedAutoEncoder],
    feature_magnitudes: Tensor, # (batch, hidden)
    feature_idxs: Tensor, # (num_remove)
):
    """
    Edit activation by removing the given feature indices
    """
    W_dec = encoder.W_dec.detach() # (hidden, dim)
    feature_contribution = einsum('batch hidden, hidden dim -> batch dim', feature_magnitudes[:, feature_idxs], W_dec[feature_idxs, :])
    A = A - feature_contribution
    return A

def keep_only_features(
    A: Tensor,
    encoder: Union[VanillaAutoEncoder, GatedAutoEncoder],
    feature_magnitudes: Tensor, # (batch, hidden)
    feature_idxs: Tensor, # (num_remove)
):
    """
    Edit activation by removing all features except the given feature indices
    """
    W_dec = encoder.W_dec.detach() # (hidden, dim)
    n_features = W_dec.shape[0]
    feature_idxs_to_remove = torch.tensor([i for i in range(n_features) if i not in feature_idxs], device=feature_idxs.device).long()
    feature_contribution = einsum('batch hidden, hidden dim -> batch dim', feature_magnitudes[:, feature_idxs_to_remove], W_dec[feature_idxs_to_remove, :])
    A = A - feature_contribution
    return A

@op
def get_interp_approximation_intervention(
    prompts: Any, # List[Prompt]
    nodes: List[Node],
    As: List[Tensor],
    batch_size: int,
    encoders: List[Union[VanillaAutoEncoder, GatedAutoEncoder]],
    features: List[Tensor], # List[Tensor]
    keep_or_remove: Literal['keep', 'remove'],
    model_id: str = 'gpt2small',
) -> Tensor:
    """
    Run activation patch that either removes or keeps only the chosen features
    from the activation.  This saves memory by avoiding the storage of the
    edited activations.
    """
    model = get_model_obj(model_id)
    n = len(prompts)
    n_batches = (n + batch_size - 1) // batch_size
    answer_logits_list = []
    is_webtext_sae = get_is_webtext_sae(encoder=encoders[0])
    if is_webtext_sae:
        assert all(x is None for x in normalization_scales)
    for i in tqdm(range(n_batches)):
        batch_indices = slice(i * batch_size, (i + 1) * batch_size)
        prompts_batch = prompts[batch_indices]
        batch_dataset = PromptDataset(prompts_batch, model=model)
        As_batches = [A[batch_indices] for A in As]

        As_edited = []
        for A, encoder, normalization_scale, node, feature_idxs in zip(As_batches, encoders, normalization_scales, nodes, features):
            if normalization_scale is None:
                normalization_scale = 1.0
            with torch.no_grad():
                if is_webtext_sae:
                    encoder = encoder.to(A.device)
                    encoder_acts = encoder(A)[1]
                else:
                    encoder_acts = encoder(A / normalization_scale)[2]
            if keep_or_remove == 'remove':
                A_edited = remove_features(A / normalization_scale, encoder, encoder_acts, feature_idxs)
            elif keep_or_remove == 'keep':
                A_edited = keep_only_features(A / normalization_scale, encoder, encoder_acts, feature_idxs)
            A_edited = A_edited * normalization_scale
            As_edited.append(A_edited)

        hooks = [get_forced_hook(prompts=prompts_batch, node=node, A=A_edited) for node, A_edited in zip(nodes, As_edited)]
        changed_logits = model.run_with_hooks(batch_dataset.tokens, fwd_hooks=hooks)[:, -1, :]
        answer_logits = changed_logits.gather(dim=-1, index=batch_dataset.answer_tokens.cuda())
        answer_logits_list.append(answer_logits)
    answer_logits = torch.cat(answer_logits_list, dim=0)
    return answer_logits


# ### what
# @torch.no_grad()
# def get_freqs(encoder: AutoEncoder, A: Tensor, batch_size: Optional[int] = None) -> Tuple[Tensor, float]:
#     """
#     Get the feature frequencies for the given activations, and the fraction of
#     dead neurons.
#     """
#     act_freq_scores = torch.zeros(encoder.d_hidden, dtype=torch.float32).cuda()
#     total = 0
#     if batch_size is None:
#         num_batches = 1
#         batch_size = A.shape[0]
#     else:
#         num_batches = A.shape[0] // batch_size
#     with torch.no_grad():
#         for i in range(num_batches):
#             A_batch = A[i*batch_size:(i+1)*batch_size]
#             acts = encoder(A_batch)[2]
#             act_freq_scores += (acts > 0).sum(0)
#             total += acts.shape[0]
#     act_freq_scores /= total
#     frac_dead = (act_freq_scores==0).float().mean().item()
#     return act_freq_scores, frac_dead



@op
def get_activation_distance(
    A_target: Tensor,
    A_edited: Tensor,
    A_target_grad: Optional[Tensor],
    method: Literal['l2', 'attribution'],
) -> float:
    """
    Compute a number measuring the distance between the target and edited
    activations. Note that "distances" computed using different methods cannot
    be compared meaningfully.
    """
    if method == 'l2':
        return (A_target - A_edited).norm(dim=1).mean()
    elif method == 'attribution':
        return ((A_target-A_edited)*A_target_grad).abs().mean()
    else:
        raise ValueError()



def get_feature_weights(
    encoder: Union[VanillaAutoEncoder, GatedAutoEncoder, AttributionAutoEncoder],
    A_normalized: Tensor, 
    batch_size: int,
    ) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        recons = encoder.get_reconstructions(A_normalized)
        acts = encoder.get_feature_magnitudes(A_normalized)
    num_examples = A_normalized.shape[0]
    num_batches = num_examples // batch_size
    feature_weights_batches = []
    for i in tqdm(range(num_batches), disable=False):
        acts_batch = acts[i*batch_size:(i+1)*batch_size]
        centered_recons_batch = recons[i*batch_size:(i+1)*batch_size] - encoder.b_dec.detach().unsqueeze(0)
        centered_recons_norms = centered_recons_batch.norm(dim=-1, keepdim=True)
        feature_weights = einsum('batch hidden, hidden dim, batch dim -> batch hidden', acts_batch, encoder.W_dec.detach(), centered_recons_batch) / centered_recons_norms**2
        feature_weights_batches.append(feature_weights)
    feature_weights = torch.cat(feature_weights_batches, dim=0)
    sums = feature_weights.sum(dim=1)
    nonzero_sums = sums[sums != 0]
    # set the nan values to 1 in `nonzero_sums`
    nonzero_sums[torch.isnan(nonzero_sums)] = 1
    if nonzero_sums[(nonzero_sums - 1).abs() > 0.05].shape[0] > 10:
        print(f'Found > 10 nonzero_sums that are not 1: {nonzero_sums[(nonzero_sums - 1).abs() > 0.05]}')
    return feature_weights, acts

@op(__allow_side_effects__=True)
def compute_removed_weight(
    encoder: Union[VanillaAutoEncoder, GatedAutoEncoder, AttributionAutoEncoder],
    A_normalized: Tensor,
    batch_size: int,
    best_features: Tensor, 
    ) -> Tensor:
    weights, _ = get_feature_weights(encoder, A_normalized, batch_size,)
    best_weights = torch.stack([weights[range(A_normalized.shape[0]), best_features[:, i]] for i in range(best_features.shape[1])], dim=1)
    return best_weights.sum(dim=1)


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
    """
    lol, why is this a thing
    """
    activation[idx] = mean
    return activation

@op
def compute_mean_ablated_lds(
    node: Node,
    prompts: Any,
    A_mean: Tensor,
    batch_size: int,
) -> float:
    """
    Get the mean-ablated logitdiff
    """
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
def encoder_hook(activation: Tensor, 
                 hook: HookPoint,
                 node: Node,
                 encoder: Union[VanillaAutoEncoder, GatedAutoEncoder],
                 idx: Tensor, 
                 encoder_normalization_scale: float,
                 ) -> Tensor:
    """
    Replace the activations at the given index with the reconstruction from the
    encoder; computes reconstructions on the fly
    """
    A = activation[idx]
    #! very important to normalize the activations before passing them to the encoder
    reconstruction = (encoder.get_reconstructions(A / encoder_normalization_scale) * encoder_normalization_scale).detach().clone()
    activation[idx] = reconstruction
    return activation

@op(__allow_side_effects__=True) # we allow this here because nn.Modules have non-deterministic content hash :(
@torch.no_grad()
def get_logitdiff_loss(
    encoder: Union[VanillaAutoEncoder, GatedAutoEncoder, AttributionAutoEncoder], 
    encoder_normalization_scale: float,
    node: Node,
    prompts: List[Prompt],
    batch_size: int,
    clean_ld: Optional[float] = None,
    mean_ablated_ld: Optional[float] = None,
    ) -> float:
    """
    DESPITE THE NAME, this actually computes something which is more like an
    accuracy score. The closer this is to 1, the better the encoder is at
    reconstructing the activations of the given node w.r.t. model predictions.
    """
    mean_ablated_ld = mean_ablated_ld
    encoder_logits = run_with_hooks.f(
        prompts=prompts,
        hooks=None,
        semantic_nodes=[node],
        semantic_hooks=[(node.activation_name, partial(encoder_hook, node=node, encoder=encoder, encoder_normalization_scale=encoder_normalization_scale))],
        batch_size=batch_size,
    )
    encoder_ld = (encoder_logits[:, 0] - encoder_logits[:, 1]).mean().item()
    # score = (clean_ld - encoder_ld) / (clean_ld - mean_ablated_ld)
    score = abs(encoder_ld - mean_ablated_ld) / abs(clean_ld - mean_ablated_ld)
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
def get_vanilla(encoder_state_dict: Dict[str, Tensor], 
                d_activation: int,
                d_hidden: int,
                enc_dtype: str = "fp32",
                freeze_decoder: bool = False,
                random_seed: int = 0) -> VanillaAutoEncoder:
    """
    Get a vanilla SAE with the given parameters
    """
    encoder = VanillaAutoEncoder(d_activation=d_activation, d_hidden=d_hidden, enc_dtype=enc_dtype, freeze_decoder=freeze_decoder, random_seed=random_seed).cuda()
    encoder.load_state_dict(encoder_state_dict, strict=True)
    return encoder


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
def get_gated(encoder_state_dict: Dict[str, Tensor], 
             d_activation: int,
             d_hidden: int,
             ) -> GatedAutoEncoder:
    """
    Get a vanilla SAE with the given parameters
    """
    encoder = GatedAutoEncoder(d_activation=d_activation, d_hidden=d_hidden).cuda()
    encoder.load_state_dict(encoder_state_dict, strict=True)
    return encoder

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
        L_reconstruct_parts.extend([L_reconstruct.item()] * A_batch.shape[0])
        L_sparsity_parts.extend([L_sparsity.item()] * A_batch.shape[0])
        L_aux_parts.extend([L_aux.item()] * A_batch.shape[0])
    L_reconstruct = torch.Tensor(L_reconstruct_parts).cuda()
    L_sparsity = torch.Tensor(L_sparsity_parts).cuda()
    L_aux = torch.Tensor(L_aux_parts).cuda()
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
    encoder.W_gate.data[:, dead_indices] = encoder.W_dec.data[dead_indices, :].T.detach().clone()
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





################################################################################
### attribution SAEs
################################################################################
@op
def get_attribution(encoder_state_dict: Dict[str, Tensor], 
                    d_activation: int,
                    d_hidden: int,
                    enc_dtype: str = "fp32",
                    freeze_decoder: bool = False,
                    ) -> AttributionAutoEncoder:
    """
    Get a vanilla SAE with the given parameters
    """
    encoder = AttributionAutoEncoder(d_activation=d_activation, d_hidden=d_hidden, enc_dtype=enc_dtype, freeze_decoder=freeze_decoder).cuda()
    encoder.load_state_dict(encoder_state_dict, strict=True)
    return encoder



@op
def train_attribution(
    A: Tensor,
    A_grad: Tensor,
    d_hidden: int,
    start_epoch: int,
    end_epoch: int,
    encoder_state_dict: Optional[Dict[str, Tensor]],
    optimizer_state_dict: Optional[Any],
    scheduler_state_dict: Optional[Any],
    attribution_sparsity_penalty: float,
    unexplained_attribution_penalty: float,
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
    encoder = AttributionAutoEncoder(d_activation=d_activation, enc_dtype=enc_dtype,
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
            "attribution_sparsity_loss": 0,
            "unexplained_attribution_loss": 0,
            "dead_mask": Tensor([True for _ in range(d_hidden)]).cuda().bool(), # dead neurons throughout the entire epoch
        }
        feature_counts = 0
        for i in range(0, n, batch_size):
            A_batch = A[perm[i:i+batch_size]]
            A_grad_batch = A_grad[perm[i:i+batch_size]]
            optim.zero_grad()
            # A_hat, acts, l2_loss, l1_loss = encoder(A_batch)
            # loss = l2_loss + l1_loss * l1_coeff
            # loss.backward()
            A_hat, acts, l2_loss, l1_loss, attribution_sparsity_loss, unexplained_attribution_loss = encoder.forward(A=A_batch, A_grad=A_grad_batch)
            loss = l2_loss + l1_loss * l1_coeff + attribution_sparsity_loss * attribution_sparsity_penalty + unexplained_attribution_loss * unexplained_attribution_penalty
            loss.backward()
            optim.step()
            encoder.make_decoder_weights_and_grad_unit_norm()
            actual_batch_size = A_batch.shape[0]
            epoch_metrics["l2_loss"] += l2_loss.item() * actual_batch_size
            epoch_metrics["l1_loss"] += l1_loss.item() * actual_batch_size
            epoch_metrics["l0_loss"] += (acts > 0).sum(dim=-1).float().mean().item() * actual_batch_size
            epoch_metrics["attribution_sparsity_loss"] += attribution_sparsity_loss.item() * actual_batch_size
            epoch_metrics["unexplained_attribution_loss"] += unexplained_attribution_loss.item() * actual_batch_size

            feature_counts += (acts > 0).float().sum(dim=0)
            dead_features_batch = (acts > 0).sum(dim=0) == 0
            epoch_metrics["dead_mask"] = epoch_metrics["dead_mask"] & dead_features_batch # take AND w/ False to indicate alive neurons
        scheduler.step()
        epoch_metrics["l2_loss"] /= n
        epoch_metrics["l1_loss"] /= n
        epoch_metrics["l0_loss"] /= n
        epoch_metrics["attribution_sparsity_loss"] /= n
        epoch_metrics["unexplained_attribution_loss"] /= n
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
                resample_attribution(encoder=encoder, 
                                        dead_indices=dead_indices, 
                                        A=A, A_grad=A_grad,
                                        l1_coeff=l1_coeff, 
                                        attribution_sparsity_penalty=attribution_sparsity_penalty, 
                                        unexplained_attribution_penalty=unexplained_attribution_penalty,
                                        optimizer=optim)
                scheduler.start_warmup()
        pbar.set_description(f"l2_loss: {epoch_metrics['l2_loss']:.4f}, l1_loss: {epoch_metrics['l1_loss']:.4f}, " + 
                             f"l0_loss: {epoch_metrics['l0_loss']:.4f}, frac_dead: {epoch_metrics['frac_dead']:.4f}, " +
                             f"attribution_sparsity_loss: {epoch_metrics['attribution_sparsity_loss']:.4f}, " + 
                             f"unexplained_attribution_loss: {epoch_metrics['unexplained_attribution_loss']:.4f}")
    return encoder.state_dict(), optim.state_dict(), scheduler.state_dict(), metrics


@torch.no_grad()
def resample_attribution(encoder: VanillaAutoEncoder, dead_indices: Tensor, A: Tensor, 
                         A_grad: Tensor,
                         l1_coeff: float, attribution_sparsity_penalty: float, unexplained_attribution_penalty: float,
                           optimizer: torch.optim.Adam, W_enc_reinit_scale: float = 0.2):
    """
    Re-initializes the weights of the encoder for the given indices, following
    the re-initialization strategy from Anthropic
    """
    ### collect losses of the encoder on the activations
    batch_size = 64
    n = A.shape[0]
    l2_parts, l1_parts = [], []
    attribution_sparsity_parts, unexplained_attribution_parts = [], []
    for i in range(0, n, batch_size):
        A_batch = A[i:i+batch_size]
        A_grad_batch = A_grad[i:i+batch_size]
        _, _, l2_losses, l1_losses, attribution_sparsity_losses, unexplained_attribution_losses = encoder.forward_detailed(A_batch, A_grad_batch)
        l2_parts.append(l2_losses)
        l1_parts.append(l1_losses)
        attribution_sparsity_parts.append(attribution_sparsity_losses)
        unexplained_attribution_parts.append(unexplained_attribution_losses)
    l2_losses = torch.cat(l2_parts)
    l1_losses = torch.cat(l1_parts)
    attribution_sparsity_losses = torch.cat(attribution_sparsity_parts)
    unexplained_attribution_losses = torch.cat(unexplained_attribution_parts)
    total_losses = l2_losses + l1_losses * l1_coeff + attribution_sparsity_losses * attribution_sparsity_penalty + unexplained_attribution_losses * unexplained_attribution_penalty
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