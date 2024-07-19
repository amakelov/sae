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


from mandala._next.imports import op, sess, MList, MDict

MODEL_ID = 'gpt2small'
MODELS = {}

ROOT = Path(__file__).parent
NAMES_PATH = ROOT / "data" / "names.json"
OBJECTS_PATH = ROOT / "data" / "objects.json"
PLACES_PATH = ROOT / "data" / "places.json"
PREFIXES_PATH = ROOT / "data" / "prefixes.json"
TEMPLATES_PATH = ROOT / "data" / "templates.json"
GENDERS_TRAIN_PATH = ROOT / "data" / "genders_train.txt"
GENDERS_TEST_PATH = ROOT / "data" / "genders_test.txt"

NAMES = json.load(open(NAMES_PATH))
OBJECTS = json.load(open(OBJECTS_PATH))
PLACES = json.load(open(PLACES_PATH))
PREFIXES = json.load(open(PREFIXES_PATH))
TEMPLATES = json.load(open(TEMPLATES_PATH))

PREVIOUS_TOKEN_HEADS = [(2, 2), (4, 11)]
DUPLICATE_TOKEN_HEADS = [(0, 1), (3, 0), (0, 10)]
INDUCTION_HEADS = [(5, 5), (6, 9), (5, 8), (5, 9)]
S_INHIBITION_HEADS = [(7, 3), (7, 9), (8, 6), (8, 10)]
NAME_MOVERS = [(9, 9), (9, 6), (10, 0)]
NEGATIVE_NAME_MOVERS = [(10, 7), (11, 10)]
BACKUP_NAME_MOVERS = [(9, 0), (9, 7), (10, 1), (10, 2), (10, 10), (11, 2), (11, 9)]

def load_genders_dict() -> Dict[str, str]:
    with open(GENDERS_TRAIN_PATH, 'r') as f:
        lines = f.readlines()
    with open(GENDERS_TEST_PATH, 'r') as f:
        lines += f.readlines()
    res = {}
    for l in lines:
        name, gender = l.split(', ')
        name, gender = name.replace("'", ""), gender.replace("'", "").replace('\n', '')
        res[name] = gender
    return res

GENDERS_DICT = load_genders_dict()


def get_model_obj(model_id: str) -> HookedTransformer:
    return MODELS[model_id]

def get_model(model_name: str = "gpt2-small",
               config: Literal['default', 'webtext'] = 'default') -> HookedTransformer:
    if config == 'default':
        model = HookedTransformer.from_pretrained(
            model_name=model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=True,
        )
    elif config == 'webtext':
        model = HookedTransformer.from_pretrained(model_name=model_name,)
    else:
        raise ValueError(f"Invalid config: {config}")
    model.requires_grad_(False)
    return model


def is_single_token(s: str, model: HookedTransformer) -> bool:
    """
    Check if a string is a single token in the vocabulary of a model.
    """
    try:
        model.to_single_token(s)
        return True
    except Exception as e:
        return False

class Prompt:
    """
    Represent a general ABC prompt using a template, and operations on it that
    are useful for generating datasets.
    """

    def __init__(
        self,
        names: Tuple[str, str, str],
        prefix: str,
        template: str,
        obj: str,
        place: str,
    ):
        self.names = names
        self.prefix = prefix
        self.template = template
        self.obj = obj
        self.place = place
        if self.is_ioi:
            self.s_name = self.names[2] # subject always appears in third position
            self.io_name = [x for x in self.names[:2] if x != self.s_name][0]
        else:
            self.io_name = None
            self.s_name = None
        self.str_tokens = MODELS[MODEL_ID].to_str_tokens(self.sentence)
    
    @property
    def pattern(self) -> Literal['ABB', 'BAB']:
        assert self.is_ioi
        if self.names[1] == self.names[2]:
            return 'ABB'
        else:
            return 'BAB'
        
    @property
    def flipped_pattern(self) -> Literal['ABB', 'BAB']:
        if self.pattern == 'ABB':
            return 'BAB'
        else:
            return 'ABB'
    
    def with_changed_sname(self, new_sname: str) -> 'Prompt':
        assert new_sname not in self.names
        new_names = [new_sname if x == self.s_name else x for x in self.names]
        return Prompt(
            names=tuple(new_names),
            template=self.template,
            obj=self.obj,
            place=self.place,
            prefix=self.prefix,
        )
    
    def with_changed_ioname(self, new_ioname: str) -> 'Prompt':
        assert new_ioname not in self.names
        new_names = [new_ioname if x == self.io_name else x for x in self.names]
        return Prompt(
            names=tuple(new_names),
            template=self.template,
            obj=self.obj,
            place=self.place,
            prefix=self.prefix,
        )
    
    @property
    def semantic_pos(self) -> Dict[str, int]:
        if self.is_ioi:
            return {
                'io': self.io_token,
                's1': self.s1_token,
                's1+1': self.s1_plus1_token,
                's2': self.s2_token,
                'end': self.end_token,
            }
        else:
            io_token =  [i for i in range(len(self.str_tokens)) if self.str_tokens[i] == f' {self.names[0]}'][0]
            s1_token = [i for i in range(len(self.str_tokens)) if self.str_tokens[i] == f' {self.names[1]}'][0]
            s1_plus1_token = s1_token + 1
            s2_token =  [i for i in range(len(self.str_tokens)) if self.str_tokens[i] == f' {self.names[2]}'][0]
            end_token = len(self.str_tokens) - 1
            return {
                'io': io_token,
                's1': s1_token,
                's1+1': s1_plus1_token,
                's2': s2_token,
                'end': end_token,
            }

    @property
    def io_token(self) -> int:
        idxs = [i for i in range(len(self.str_tokens)) if self.str_tokens[i] == f' {self.io_name}']
        return idxs[0]
    
    @property
    def s1_token(self) -> int:
        idxs = [i for i in range(len(self.str_tokens)) if self.str_tokens[i] == f' {self.s_name}']
        return idxs[0]
    
    @property
    def s1_plus1_token(self) -> int:
        return self.s1_token + 1
    
    @property
    def s2_token(self) -> int:
        idxs = [i for i in range(len(self.str_tokens)) if self.str_tokens[i] == f' {self.s_name}']
        return idxs[1]
    
    @property
    def end_token(self) -> int:
        return len(self.str_tokens) - 1

    @property
    def is_ioi(self) -> bool:
        return self.names[2] in self.names[:2] and len(set(self.names)) == 2

    def __repr__(self) -> str:
        return f"<===PROMPT=== {self.sentence}>"

    @property
    def sentence(self) -> str:
        return self.prefix + self.template.format(
            name_A=self.names[0],
            name_B=self.names[1],
            name_C=self.names[2],
            object=self.obj,
            place=self.place,
        )

    @staticmethod
    def canonicalize(things: Tuple[str, str, str]) -> Tuple[str, str, str]:
        # the unique elements of the tuple, in the order they appear
        ordered_uniques = list(OrderedDict.fromkeys(things).keys())
        canonical_elts = ['A', 'B', 'C']
        uniques_to_canonical = {x: y for x, y in zip(ordered_uniques, canonical_elts[:len(ordered_uniques)])}
        return tuple([uniques_to_canonical[x] for x in things])

    @staticmethod
    def matches_pattern(names: Tuple[str, str, str], pattern: str) -> bool:
        return Prompt.canonicalize(names) == Prompt.canonicalize(tuple(pattern))
    
    def resample_pattern(self, orig_pattern: str, new_pattern: str,
                         name_distribution: Sequence[str]) -> "Prompt":
        """
        Change the pattern of the prompt, while keeping the names that are
        mapped to the same symbols in the original and new patterns the same.

        Args:
            orig_pattern (str): _description_
            new_pattern (str): _description_
            name_distribution (Sequence[str]): _description_

        Example:
            prompt = train_distribution.sample_one(pattern='ABB')
            (prompt.sentence, 
            prompt.resample_pattern(orig_pattern='ABB', new_pattern='BAA', 
                                    name_distribution=train_distribution.names,).sentence,
            prompt.resample_pattern(orig_pattern='ABB', new_pattern='CDD', 
                                    name_distribution=train_distribution.names,).sentence,
            prompt.resample_pattern(orig_pattern='ABB', new_pattern='ACC', 
                                    name_distribution=train_distribution.names,).sentence,
        
        >>> ('Then, Olivia and Anna had a long and really crazy argument. Afterwards, Anna said to',
        >>> 'Then, Anna and Olivia had a long and really crazy argument. Afterwards, Olivia said to',
        >>> 'Then, Joe and Kelly had a long and really crazy argument. Afterwards, Kelly said to',
        >>> 'Then, Olivia and Carl had a long and really crazy argument. Afterwards, Carl said to')
        )
        """
        assert len(orig_pattern) == 3
        assert len(new_pattern) == 3
        assert len(set(orig_pattern)) == len(set(new_pattern)) == 2
        assert self.matches_pattern(names=self.names, pattern=orig_pattern)
        orig_to_name = {orig_pattern[i]: self.names[i] for i in range(3)}
        new_names = [None for _ in range(3)]
        new_pos_to_symbol = {}
        for i, symbol in enumerate(new_pattern):
            if symbol in orig_to_name.keys():
                new_names[i] = orig_to_name[symbol]
            else:
                new_pos_to_symbol[i] = symbol
        new_symbols = new_pos_to_symbol.values()
        if len(new_symbols) > 0:
            new_symbol_to_name = {}
            # must sample some *new* names
            available_names = [x for x in name_distribution if x not in self.names]
            for symbol in new_symbols:
                new_symbol_to_name[symbol] = random.choice(available_names)
                available_names.remove(new_symbol_to_name[symbol])
            # populate new_names with new symbols
            for i, symbol in new_pos_to_symbol.items():
                new_names[i] = new_symbol_to_name[symbol]
        return Prompt(
            names=tuple(new_names),
            template=self.template,
            obj=self.obj,
            place=self.place,
            prefix=self.prefix,
        )



def load_data(data: Union[List[str], str, Path]) -> List[str]:
    if isinstance(data, (str, Path)):
        with open(data) as f:
            data: List[str] = json.load(f)
    return data


class PromptDataset(Dataset):
    def __init__(self, prompts: List[Prompt], model: HookedTransformer):
        # assert len(prompts) > 0
        self.prompts: Sequence[Prompt] = np.array(prompts)
        self.model = model
        ls = self.lengths
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

    def __add__(self, other: "PromptDataset") -> "PromptDataset":
        return PromptDataset(
            prompts=list(self.prompts) + list(other.prompts), model=self.model
        )

    @property
    def lengths(self) -> List[int]:
        return [self.model.to_tokens(x.sentence).shape[1] for x in self.prompts]

    @property
    def tokens(self) -> Tensor:
        return self.model.to_tokens([x.sentence for x in self.prompts])

    @property
    def io_tokens(self) -> Tensor:
        return torch.tensor(
            [self.model.to_single_token(f" {x.io_name}") for x in self.prompts]
        )

    @property
    def s_tokens(self) -> Tensor:
        return torch.tensor(
            [self.model.to_single_token(f" {x.s_name}") for x in self.prompts]
        )

    @property
    def answer_tokens(self) -> JaxFloat[Tensor, "batch 2"]:
        # return a tensor with two columns: self.io_tokens and self.s_tokens
        return torch.tensor(
            [
                [
                    self.model.to_single_token(f" {x.io_name}"),
                    self.model.to_single_token(f" {x.s_name}"),
                ]
                for x in self.prompts
            ]
        )



class PromptDistribution:
    """
    A class to represent a distribution over prompts.

    It uses a combination of names, places, objects, prefixes, and templates
    loaded from JSON files or provided lists.

    Each prompt is constructed using a selected template and a randomly selected
    name, object, and place.

    Attributes
    ----------
    prefix_len : int
        The length of the prefix to use when creating the prompts.
    """

    def __init__(
        self,
        names: Union[List[str], str, Path],
        places: Union[List[str], str, Path],
        objects: Union[List[str], str, Path],
        prefixes: Union[List[str], str, Path],
        templates: Union[List[str], str, Path],
        prefix_len: int = 2,
    ):
        self.prefix_len = prefix_len
        self.names = load_data(names)
        self.places = load_data(places)
        self.objects = load_data(objects)
        self.prefixes = load_data(prefixes)
        self.templates = load_data(templates)

    def sample_one(self,
                   pattern: str, 
                   ) -> Prompt:
        """
        Sample a single prompt from the distribution.
        """
        template = random.choice(self.templates)
        unique_ids = list(set(pattern))
        unique_names = random.sample(self.names, len(unique_ids))
        assert len(set(unique_names)) == len(unique_names)
        prompt_names = tuple([unique_names[unique_ids.index(i)] for i in pattern])
        obj = random.choice(self.objects)
        place = random.choice(self.places)
        prefix = self.prefixes[self.prefix_len]
        return Prompt(
            names=prompt_names, template=template, obj=obj, place=place, prefix=prefix
        )

train_distribution = PromptDistribution(
    names=NAMES[:len(NAMES) // 2],
    objects=OBJECTS[:len(OBJECTS) // 2],
    places=PLACES[:len(PLACES) // 2],
    prefix_len=2,
    prefixes=PREFIXES,
    templates=TEMPLATES[:2]
)

test_distribution = PromptDistribution(
    names=NAMES[len(NAMES) // 2:],
    objects=OBJECTS[len(OBJECTS) // 2:],
    places=PLACES[len(PLACES) // 2:],
    prefix_len=2,
    prefixes=PREFIXES,
    templates=TEMPLATES[2:]
)

full_distribution = PromptDistribution(
    names=NAMES,
    objects=OBJECTS,
    places=PLACES,
    prefix_len=2,
    prefixes=PREFIXES,
    templates=TEMPLATES,
)


class Node:
    """
    Mostly a copy of the one in path_patching.py, we'll see if it diverges
    """

    def __init__(
        self,
        component_name: Literal[
            "z",
            "attn_out",
            "pre",
            "post",
            "mlp_out",
            "resid_pre",
            "resid_post",
            "resid_mid",
            "q",
            "k",
            "v",
            "pattern",
            "attn_scores",
            "result",
            "q_input",
            "k_input",
            "v_input",
            'scale_ln1',
            'scale_ln2',
            'scale_final',
            "ln_final",
        ],
        layer: Optional[int] = None,
        head: Optional[int] = None,
        neuron: Optional[int] = None,
        seq_pos: Optional[Union[int, str]] = None, # string used for semantic indexing
    ):
        assert isinstance(component_name, str)
        self.component_name = component_name
        if layer is not None:
            assert isinstance(layer, int)
        self.layer = layer
        if head is not None:
            assert isinstance(head, int)
        self.head = head
        if neuron is not None:
            assert isinstance(neuron, int)
        self.neuron = neuron
        if seq_pos is not None:
            assert isinstance(seq_pos, (int, str))
        self.seq_pos = seq_pos

    def with_resolved_position(self, prompt: Prompt) -> 'Node':
        """
        Return a new node with the seq_pos resolved to an integer.
        """
        if isinstance(self.seq_pos, str):
            return Node(
                component_name=self.component_name,
                layer=self.layer,
                head=self.head,
                neuron=self.neuron,
                seq_pos=prompt.semantic_pos[self.seq_pos],
            )
        else:
            return self
    
    def __hash__(self) -> int:
        return hash((self.component_name, self.layer, self.head, self.neuron, self.seq_pos))
    
    def __lt__(self, other: 'Node') -> bool:
        return hash(self) < hash(other)
    
    def __eq__(self, other: 'Node') -> bool:
        return hash(self) == hash(other)
    
    def __le__(self, other: 'Node') -> bool:
        return hash(self) <= hash(other)

    @property
    def activation_name(self) -> str:
        if self.component_name == 'scale_ln1':
            return utils.get_act_name('scale', layer=self.layer, layer_type='ln1')
        elif self.component_name == 'scale_ln2':
            return utils.get_act_name('scale', layer=self.layer, layer_type='ln2')
        elif self.component_name == 'scale_final':
             return utils.get_act_name('scale', layer=None)
        else:
            return utils.get_act_name(self.component_name, layer=self.layer)

    @property
    def shape_type(self) -> List[str]:
        """
        List of the meaning of each dimension of the full activation for this
        node (i.e., what you'd get if you did `cache[self.activation_name]`).
        
        This is just for reference
        """
        if self.component_name in [
            "resid_pre",
            "resid_post",
            "resid_mid",
            "q_input",
            "k_input",
            "v_input",
        ]:
            return ["batch", "seq", "d_model"]
        elif self.component_name == 'pattern':
            return ["batch", "head", "query_pos", "key_pos"]
        elif self.component_name in ["q", "k", "v", "z"]:
            return ["batch", "seq", "head", "d_head"]
        elif self.component_name in ["result"]:
            return ["batch", "seq", "head", "d_model"]
        elif self.component_name == 'scale':
            return ['batch', 'seq']
        elif self.component_name == 'post':
            return ['batch', 'seq', 'd_mlp']
        else:
            raise NotImplementedError

    def idx(self, prompts: Optional[List[Prompt]] = None) -> Tuple[Union[int, slice, Tensor, None], ...]:
        """
        Index into the full activation to restrict to layer / head / neuron /
        seq_pos
        """
        if isinstance(self.seq_pos, str):
            assert prompts is not None
            seq_pos_idx = torch.Tensor([p.semantic_pos[self.seq_pos] for p in prompts]).long()
            batch_idx = torch.arange(len(prompts)).long()
        elif isinstance(self.seq_pos, int):
            seq_pos_idx = self.seq_pos
            batch_idx = slice(None)
        elif self.seq_pos is None:
            seq_pos_idx = slice(None)
            batch_idx = slice(None)
        else:
            raise NotImplementedError

        if self.neuron is not None:
            raise NotImplementedError

        elif self.component_name in ['pattern', 'attn_scores']:
            assert self.head is not None
            return tuple([slice(None), self.head, slice(None), slice(None)])
        elif self.component_name in ["q", "k", "v", "z", "result"]:
            assert self.head is not None, "head must be specified for this component"
            return tuple([batch_idx, seq_pos_idx, self.head, slice(None)])
        elif self.component_name == 'scale':
            return tuple([slice(None), slice(None)])
        elif self.component_name == 'post':
            return tuple([batch_idx, seq_pos_idx, slice(None)])
        else:
            return tuple([batch_idx, seq_pos_idx, slice(None)])
    
    @property
    def names_filter(self) -> Callable:
        return lambda x: x in [self.activation_name]
    
    @staticmethod
    def get_names_filter(nodes: List['Node']) -> Callable:
        return lambda x: any(node.names_filter(x) for node in nodes)

    @property
    def needs_head_results(self) -> bool:
        return self.component_name in ['result']
    
    def get_value(self, cache: ActivationCache, 
                  prompts: Optional[List[Prompt]] = None
                  ) -> Tensor:
        return cache[self.activation_name][self.idx(prompts=prompts)]
    
    def __repr__(self) -> str:
        properties = OrderedDict({
            "component_name": self.component_name,
            "layer": self.layer,
            "head": self.head,
            "neuron": self.neuron,
            "seq_pos": self.seq_pos,
        })
        properties = ", ".join(f"{k}={v}" for k, v in properties.items() if v is not None)
        return f"Node({properties})"
    
    @property
    def displayname(self) -> str:
        if self.component_name in ('q', 'k', 'v', 'z'):
            return f'{self.component_name}@L{self.layer}H{self.head}@{self.seq_pos}'
        else:
            raise NotImplementedError
    

################################################################################
### batched decorator
################################################################################
class batched:
    """
    A decorator to run a function in batches over given arguments. The results
    from each batch are aggregated using a reducer function, e.g. sum, mean, or
    concatenation.
    
    Things that came up during use:
    - sometimes, you return a list of things, and you want to concatenate across
    respective elements of the list, instead of concatenating all the lists into
    one big list.
    - sometimes you return a variable number of outputs
    - sometimes it is more natural to concatenate over a dimension different 
    from the first one.
    - sometimes you want to concatenate dataframes instead of tensors.
    
    """

    def __init__(
        self,
        args: List[str],
        n_outputs: Union[int, Literal['var']],
        reducer: Union[Callable, str] = "cat",
        shuffle: bool = False,
        verbose: bool = True,
    ):
        self.args = args
        self.n_outputs = n_outputs
        self.reducer = reducer
        self.shuffle = shuffle
        self.verbose = verbose
        if self.shuffle:
            raise NotImplementedError
    
    T = typing.TypeVar("T", Tensor, np.ndarray, Sequence)
    @staticmethod
    def get_slice(x: T, idx: np.ndarray) -> T:
        if isinstance(x, (Tensor, np.ndarray)):
            return x[idx]
        elif isinstance(x, (list, tuple)):
            return type(x)([x[i] for i in idx])
        elif isinstance(x, dict):
            return type(x)({k: batched.get_slice(v, idx) for k, v in x.items()})
        else:
            try:
                return x[idx]
            except:
                raise NotImplementedError(f"Cannot slice {type(x)}")
    
    @staticmethod
    def get_arg_length(x: T, ) -> int:
        if isinstance(x, (Tensor, np.ndarray)):
            return x.shape[0]
        # elif isinstance(x, (list, tuple)):
        #     element_lengths = [batched.get_arg_length(x[i]) for i in range(len(x))]
        #     if len(set(element_lengths)) != 1:
        #         raise ValueError(f"Argument {x} has elements of different lengths")
        #     return element_lengths[0]
        elif isinstance(x, dict):
            value_lengths = [batched.get_arg_length(v) for v in x.values()]
            if len(set(value_lengths)) != 1:
                raise ValueError(f"Dict argument {x} has values of different lengths")
            return value_lengths[0]
        else:
            try:
                return len(x)
            except:
                raise NotImplementedError(f"Cannot get length of {type(x)}")
    
    @staticmethod
    def average_objs(xs: List[T], dim: int = 0) -> Union[T, Dict[Any, T], List[T]]:
        assert len({type(x) for x in xs}) == 1
        if isinstance(xs[0], (Tensor, np.ndarray)):
            return sum(xs) / len(xs)
        elif isinstance(xs[0], pd.DataFrame):
            return sum(xs) / len(xs)
        elif isinstance(xs[0], list):
            assert len({len(x) for x in xs}) == 1
            return [batched.average_objs([x[i] for x in xs], dim=dim) for i in range(len(xs[0]))]
        elif isinstance(xs[0], dict):
            # check all dicts have the same set of keys
            assert all(set(x.keys()) == set(xs[0].keys()) for x in xs)
            return {k: batched.average_objs([x[k] for x in xs], dim=dim) for k in xs[0].keys()}
        elif xs[0] is None:
            return None
        else:
            raise NotImplementedError
        
    @staticmethod
    def concatenate_objs(xs: Any, dim: int = 0) -> Any:
        assert len({type(x) for x in xs}) == 1
        # if isinstance(xs[0], TransientObj):
        #     return Transient(batched.concatenate_objs([x.obj for x in xs], dim=dim))
        if isinstance(xs[0], Tensor):
            return torch.cat(xs, dim=dim)
        elif isinstance(xs[0], np.ndarray):
            return np.concatenate(xs, axis=dim)
        elif isinstance(xs[0], pd.DataFrame):
            return pd.concat(xs, ignore_index=True)
        elif isinstance(xs[0], dict):
            # check all dicts have the same set of keys
            assert all(set(x.keys()) == set(xs[0].keys()) for x in xs)
            return {k: batched.concatenate_objs([x[k] for x in xs], dim=dim) for k in xs[0].keys()}
        elif isinstance(xs[0], list):
            assert len({len(x) for x in xs}) == 1
            return [batched.concatenate_objs([x[i] for x in xs], dim=dim) for i in range(len(xs[0]))]
        elif xs[0] is None:
            return None
        else:
            raise NotImplementedError

    def __call__(self, func: Callable) -> "func":
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            batch_size = kwargs.get("batch_size", None)
            verbose = kwargs.get("verbose", self.verbose)
            if batch_size is None:
                return func(*args, **kwargs)
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            named_args = dict(bound_args.arguments)
            batching_args = {k: named_args[k] for k in self.args}
            # check all the lengths are the same
            # lengths = [len(v) for v in batching_args.values()]
            lengths = [batched.get_arg_length(v) for v in batching_args.values()]
            assert (
                len(set(lengths)) == 1
            ), f"All batched arguments must have the same length. Instead got lengths {lengths}"
            length = lengths[0]
            assert length > 0
            num_batches = math.ceil(length / batch_size)
            results = []
            pbar = tqdm if verbose else lambda x: x
            for i in pbar(range(num_batches)):
                batch_idx = np.arange(
                    i * batch_size, min(lengths[0], (i + 1) * batch_size)
                )
                batched_args = {k: batched.get_slice(v, batch_idx) for k, v in batching_args.items()}
                named_args.update(batched_args)
                results.append(func(**named_args))
            # todo: refactor this logit to be uniform across reducers
            if self.reducer.startswith('cat'):
                if self.reducer == 'cat':
                    dim = 0
                else:
                    _, dim = self.reducer.split('_')
                    dim = int(dim)
                # concatenate the results per output
                if self.n_outputs == 1:
                    return batched.concatenate_objs(results, dim=dim)
                else:
                    assert len({len(r) for r in results}) == 1
                    return tuple([
                        batched.concatenate_objs([r[i] for r in results], dim=dim)
                        for i in range(len(results[0]))
                    ])
            elif self.reducer == "mean":
                if self.n_outputs == 1:
                    return batched.average_objs(results)
                else:
                    assert len({len(r) for r in results}) == 1
                    return tuple([
                        sum([r[i] for r in results]) / len(results)
                        for i in range(len(results[0]))
                    ])
            else:
                raise NotImplementedError

        return wrapper

################################################################################
### batched utils
################################################################################
@op
@batched(args=['prompts'], n_outputs=1, reducer='cat')
def estimate_resid_scales_before(
    prompts: Any,
    nodes: List[Node],
    batch_size: int,
    model_id: str = MODEL_ID,
    verbose: bool = True,
) -> List[Tensor]:
    corresponding_resid_nodes = [
        Node(component_name='resid_pre', layer=node.layer, seq_pos=node.seq_pos)
        for node in nodes
    ]
    model = MODELS[model_id]
    prompt_dataset = PromptDataset(prompts=prompts, model=model)
    _, cache = model.run_with_cache(prompt_dataset.tokens, names_filter=Node.get_names_filter(corresponding_resid_nodes))
    acts = [node.get_value(cache, prompts=prompts) for node in corresponding_resid_nodes]
    #! importantly, center the activations first
    acts = [act - act.mean(dim=0) for act in acts]
    return [act.norm(dim=-1) for act in acts]

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
    print(f'Batch size: {batch_size}')
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

@op
@batched(args=['prompts'], n_outputs=1, reducer='cat')
def run_with_hooks(
    prompts: Any, 
    hooks: Optional[List[Tuple[str, Callable]]],
    batch_size: int,
    return_predictions: bool = False,
    semantic_nodes: Optional[List[Node]] = None,
    semantic_hooks: Optional[List[Tuple[str, Callable]]] = None,
    model_id: str = MODEL_ID,
    return_full_last_logits: bool = False,
) -> Tensor:
    model = MODELS[model_id]
    prompt_dataset = PromptDataset(prompts=prompts, model=model)
    assert (semantic_hooks is None) == (semantic_nodes is None)
    if semantic_nodes is not None:
        assert hooks is None
        assert semantic_hooks is not None
        hooks = []
        idxs_by_semantic_pos = {k: [p.semantic_pos[k] for p in prompts] for k in prompts[0].semantic_pos.keys()}
        for node, hook in zip(semantic_nodes, semantic_hooks):
            hooks.append((hook[0], partial(hook[1], idx=node.idx(prompts=prompts))))
    model.reset_hooks()
    logits = model.run_with_hooks(prompt_dataset.tokens, fwd_hooks=hooks)
    if return_full_last_logits:
        return logits[:, -1, :]
    if return_predictions:
        return logits[:, -1, :].argmax(dim=-1)
    else:
        return logits[:, -1, :].gather(1, index=prompt_dataset.answer_tokens.cuda())

def get_deletion_hooks(codes_dict: Dict[Node, Dict[tuple, Tensor]], 
                       feature: Tuple[str,...],
                       feature_value_idx: Tuple[int,...], 
                       method: str = 'zero_ablate_subspace',
                       A_reference_dict: Optional[Dict[Node, Tensor]] = None,
                       ) -> List[Tuple[str, Callable]]:
    """
    Return logit differences when we intervene by deleting the feature from the
    given node.
    """
    codes_to_delete  = {}
    for node, node_codes in codes_dict.items():
        code_vals = node_codes[feature]
        code_to_delete = code_vals[feature_value_idx] # shape (dim,)
        codes_to_delete[node] = code_to_delete

    def deletion_hook_factory(activation: Tensor, hook: HookPoint, 
                      code_to_delete: Tensor, node: Node, idx: Tensor,
                      ) -> Tensor:
        val = activation[idx] # shape (..., dim)
        # expected_proj = (A_reference[node][:10_000] @
        # direction_to_delete).mean(dim=0)
        if method == 'zero_ablate_subspace':
            code_to_delete = code_to_delete / code_to_delete.norm()
            new_val = val + einsum('batch, dim -> batch dim', - val @ code_to_delete, code_to_delete)
        elif method == 'mean_ablate_subspace':
            assert A_reference_dict is not None
            code_to_delete = code_to_delete / code_to_delete.norm()
            mean_projection = (A_reference_dict[node] @ code_to_delete).mean(dim=0)
            new_val = val + einsum('batch, dim -> batch dim', (mean_projection - val) @ code_to_delete, code_to_delete)
        elif method == 'subtract_code':
            new_val = val - code_to_delete
        else:
            raise ValueError(f'unknown method {method}')
        activation[idx] = new_val
        return activation
    semantic_hooks = [(node.activation_name, partial(deletion_hook_factory,
                                            code_to_delete=codes_to_delete[node],
                                            node=node)) for node in codes_dict.keys()]
    return semantic_hooks
    
def flip_pattern(p: Prompt) -> Prompt:
    if p.names[1] == p.names[2]: # ABB
        return Prompt( # BAB
            names=(p.names[1], p.names[0], p.names[2]),
            prefix=p.prefix,
            template=p.template,
            obj=p.obj,
            place=p.place,
        )
    elif p.names[0] == p.names[2]: # BAB
        return Prompt( # ABB
            names=(p.names[1], p.names[0], p.names[2]),
            prefix=p.prefix,
            template=p.template,
            obj=p.obj,
            place=p.place,
        )
    else:
        raise ValueError
            
