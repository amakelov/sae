from functools import partial
from ioi_utils import *


################################################################################
### setup
################################################################################

FEATURE_SUBSETS = [
    ('io_pos',),
    ('s',),
    ('io',),
    ('s', 'io_pos',),
    ('io', 'io_pos'),
    ('s', 'io',),
    ('io_pos', 's', 'io',),
]
FEATURE_SUBSETS = [tuple(sorted(x)) for x in FEATURE_SUBSETS]

def get_name_to_idx(distribution: PromptDistribution) -> Dict[str, int]:
    return {name: i for i, name in enumerate(distribution.names)}

def get_template_to_idx(distribution: PromptDistribution) -> Dict[str, int]:
    return {template: i for i, template in enumerate(distribution.templates)}

# setup to work with features
NAME_TO_IDX = get_name_to_idx(full_distribution)
TEMPLATE_TO_IDX = get_template_to_idx(full_distribution)
OBJ_TO_IDX = {obj: i for i, obj in enumerate(full_distribution.objects)}
PLACE_TO_IDX = {place: i for i, place in enumerate(full_distribution.places)}
FEATURE_SIZES = {
    'bias': 1,
    'name1': len(NAME_TO_IDX),
    'name2': len(NAME_TO_IDX),
    'name3': len(NAME_TO_IDX),
    'name_anywhere': len(NAME_TO_IDX),
    'io': len(NAME_TO_IDX),
    'io_gender': 2,
    's': len(NAME_TO_IDX),
    's_gender': 2,
    'template': len(TEMPLATE_TO_IDX),
    'io_pos': 2,
    'obj': len(OBJ_TO_IDX),
    'place': len(PLACE_TO_IDX),
}

# this collects possible ways to parametrize activations of the model
FEATURE_CONFIGURATIONS = {
    'independent': [('io', ), ('s', ), ('io_pos',)],
    'coupled': [('io', 'io_pos'), ('s', 'io_pos'), ],
    'coupled_with_gender': [('io', 'io_pos'), ('s', 'io_pos'), ('io_gender',), ('s_gender',)],
    'independent_with_gender': [('io', ), ('s', ), ('io_pos',), ('io_gender',), ('s_gender',)],
    'independent_gender_only': [('io_gender', ), ('s_gender', ), ('io_pos',),],
    'coupled_gender_only': [('io_gender', 'io_pos',), ('s_gender', 'io_pos'),],
    'io': [('io',), ('io_pos',), ('s',)],
    's': [('s',), ('io_pos',), ('io',)],
    'coupled+irrelevant': [('io', 'io_pos'), ('s', 'io_pos'), ('obj',), ('place',)],
    'independent+irrelevant': [('io', ), ('s', ), ('io_pos',), ('obj',), ('place',)],
    'names': [('name1', ), ('name2', ), ('name3', )],
    'name1and2': [('name1', ), ('name2', )],
    ### other
    'name1': [('name1', )],
    'sname': [('s', ),],
    'coupled_plus': [('io', 'io_pos'), ('s', 'io_pos'), ('io_pos',)],
    's_only': [('s', ), ('io_pos',)],
    'io_only': [('io', 'io_pos'),],
    'pos_only': [('io_pos',)],
    'coupled_meta': [('io', 'io_pos'), ('s', 'io_pos'), ('obj',), ('place',)],
    ### features we expect to see in z-activations of heads
    'ind': [('s',), ('io_pos',)],
    'si': [('s',), ('io_pos',)],
    'nm': [('io',)],
}
# add bias to each
FEATURE_CONFIGURATIONS = {k: [('bias', )] + v for k, v in FEATURE_CONFIGURATIONS.items()}

@op
def generate_name_samples(n_samples, names, random_seed: int = 0) -> Any:
    np.random.seed(random_seed)
    return np.random.choice(names, n_samples, replace=True)


@op
def get_cf_prompts(
    prompts: Any, # List[Prompt],
    features: Tuple[str, ...],
    s_targets: Any=None, # Optional[List[str]],
    io_targets: Any=None, # Optional[List[str]],
) -> Any:
    if 'io_pos' in features:
        prompts = [p.resample_pattern(orig_pattern=p.pattern,
                                      new_pattern=p.flipped_pattern, 
                                      name_distribution=NAMES)
                   for p in prompts]
    if 's' in features:
        assert s_targets is not None
        prompts = [p.with_changed_sname(new_sname=s_target) for p, s_target in zip(prompts, s_targets)]
    if 'io' in features:
        assert io_targets is not None
        prompts = [p.with_changed_ioname(new_ioname=io_target) for p, io_target in zip(prompts, io_targets)]
    return prompts

@op
def generate_prompts(distribution: PromptDistribution, patterns: List[str], 
                     prompts_per_pattern: int, random_seed: int,
                     ) -> Any:
    random.seed(random_seed)
    parts = [[distribution.sample_one(pattern=pattern) 
              for _ in range(prompts_per_pattern)] for pattern in patterns]
    prompts = [p for part in parts for p in part]
    return prompts


@op
def make_prompts_abc(
    prompts: Any, # List[Prompt],
    names: Any, # List[str]
    ) -> Any:
    res = []
    for p in prompts:
        available_names = set(names) - set(p.names)
        c_name = random.choice(list(available_names))
        res.append(Prompt(
            names=(p.names[0], p.names[1], c_name),
            prefix=p.prefix,
            obj=p.obj,
            place=p.place,
            template=p.template,
        ))
    return res
            
################################################################################
### working with features
################################################################################
def get_prompt_representation(p: Prompt) -> Dict[str, int]:
    # extracts feature values from a prompt
    return {
        'name1': NAME_TO_IDX[p.names[0]],
        'name2': NAME_TO_IDX[p.names[1]],
        'name3': NAME_TO_IDX[p.names[2]],
        'name_anywhere': 0, # this is just a placeholder value, should not be used
        'io': NAME_TO_IDX[p.io_name],
        's': NAME_TO_IDX[p.s_name],
        'template': TEMPLATE_TO_IDX[p.template],
        'io_pos': 0 if p.io_token < p.s1_token else 1,
        'bias': 0,
        'obj': OBJ_TO_IDX[p.obj],
        'place': PLACE_TO_IDX[p.place],
        's_gender': 0 if GENDERS_DICT[p.s_name] == 'M' else 1,
        'io_gender': 0 if GENDERS_DICT[p.io_name] == 'M' else 1,
    }

def get_prompt_feature_vals(p: Prompt) -> Dict[str, Any]:
    return {
        'io': p.io_name,
        's': p.s_name,
        'io_pos': 0 if p.io_token < p.s1_token else 1,
    }

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

def get_parametrization(node: Node, eventually: str, use_names: bool = False):
    # if node.seq_pos in ('s1', 's1+1', 'io',):
    #     return 'name1and2' # by this time, you only see two names
    # else:
    #     return eventually
    if node.seq_pos == 'io':
        return 'io' if not use_names else 'name1and2'
    elif node.seq_pos in ('s1', 's1+1',):
        return 's' if not use_names else 'name1and2'
    else:
        return eventually

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
    prompt_feature_idxs = get_prompt_feature_idxs(prompts=prompts, features=features)
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


@op
def train_mse_codes(
    features: List[Tuple[str,...]],
    A: Tensor, 
    prompts: Sequence[Prompt],
    n_epochs: int = 1000,
    lr: float = 0.1, 
    manual_bias: bool = False,
) -> Tuple[Any, Any]:
    """
    Return the codes, losses and (uncentered, i.e. full) reconstructions
    """
    dim = A.shape[1]
    feature_shapes = {f: get_feature_shape(f) for f in features}
    codes = {f: torch.randn(*feature_shapes[f], dim).cuda().requires_grad_() for f in features}
    prompt_feature_idxs = get_prompt_feature_idxs(prompts=prompts, features=features)

    if manual_bias:
        A_mean = A.mean(dim=0)
        bias_code = A_mean.unsqueeze(0)
        A = A - A_mean
        del codes[('bias',)]
        del prompt_feature_idxs[('bias',)]

    params = list(codes.values())
    optimizer = torch.optim.Adam(params, lr=lr)
    pbar = tqdm(range(n_epochs))
    losses = []
    for epoch in pbar:
        if GlobalContext.current is not None:
            c = GlobalContext.current
        else:
            c = Context()
        with c(mode='noop'):
            reconstructions = get_reconstructions(
                codes=codes, prompt_feature_idxs=prompt_feature_idxs,
                )
        # compute example reconstructions
        optimizer.zero_grad()
        loss = ((reconstructions - A) ** 2).mean()
        if epoch % 10 == 0:
            pbar.set_description(f'loss: {loss.item():.4f}')
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    if manual_bias:
        # add back the bias
        codes[('bias',)] = bias_code
        # add back the mean
        reconstructions = reconstructions + A_mean
    return ({k: v.detach() for k, v in codes.items()},
            reconstructions.detach())

@op
def get_mse_codes(
    features: List[Tuple[str,...]],
    A: Tensor,
    prompts: Any, # List[Prompt]
) -> Tuple[Any, Any]:
    """
    Compute codes by minimizing the MSE of reconstructions. Note that this can
    be written as a regression problem
        CX = A
    where C is the sparse matrix of code activation patterns (0s and 1s) and X
    is the matrix of code vectors. We can solve this exactly using least
    squares.
    """
    prompt_feature_idxs = get_prompt_feature_idxs(prompts=prompts, features=features)
    feature_shapes = {f: get_feature_shape(f) for f in features}
    feature_sizes = {f: np.prod(feature_shapes[f]) for f in features}
    # compute one-hot representations of the prompts
    def flatten_idx(tup: Tuple[int,...], shape: Tuple[int,...]) -> int:
        idx = 0
        for i, dim in enumerate(shape):
            idx *= dim
            idx += tup[i]
        return idx
    
    def unflatten_idx(idx: int, shape: Tuple[int,...]) -> Tuple[int,...]:
        tup = []
        for dim in reversed(shape):
            tup.append(idx % dim)
            idx //= dim
        return tuple(reversed(tup))
    
    feature_reps = {}
    for f, feature_idxs in prompt_feature_idxs.items():
        rep = torch.zeros((len(prompts), feature_sizes[f])).cuda()
        flat_idxs = [flatten_idx(tuple(idx.long()), feature_shapes[f]) for idx in feature_idxs]
        rep[torch.arange(len(prompts)), flat_idxs] = 1
        feature_reps[f] = rep
    concat_rep = torch.cat([feature_reps[f] for f in features], dim=1)
    assert concat_rep.shape[0] == len(prompts)

    # now we want to solve the regression concat_rep @ x = A
    X = torch.linalg.lstsq(concat_rep.cuda(), A).solution

    # now we need to put things back into the right shape
    codes = {}
    start = 0
    for f in features:
        end = start + feature_sizes[f]
        codes[f] = X[start:end, :].reshape(feature_shapes[f] + A.shape[1:])
        start = end
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

@op
def run_activation_patch(
    base_prompts: Any, # List[Prompt],
    cf_prompts: Any, # List[Prompt],
    nodes: List[Node],
    activations: List[Tensor],
    batch_size: int,
    model_id: str = MODEL_ID,
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
    for i in tqdm(range(n_batches)):
        batch_indices = slice(i * batch_size, (i + 1) * batch_size)
        prompts_batch = base_prompts[batch_indices]
        cf_batch = cf_prompts[batch_indices]
        base_dataset = PromptDataset(prompts_batch, model=model)
        cf_dataset = PromptDataset(cf_batch, model=model)
        hooks = [get_forced_hook(prompts=prompts_batch, node=node, A=act[batch_indices]) for node, act in zip(nodes, activations)]
        changed_logits = model.run_with_hooks(base_dataset.tokens, fwd_hooks=hooks)[:, -1, :]
        base_answer_logits = changed_logits.gather(dim=-1, index=base_dataset.answer_tokens.cuda())
        cf_answer_logits = changed_logits.gather(dim=-1, index=cf_dataset.answer_tokens.cuda())
        base_logits_list.append(base_answer_logits)
        cf_logits_list.append(cf_answer_logits)
    base_logits = torch.cat(base_logits_list, dim=0)
    cf_logits = torch.cat(cf_logits_list, dim=0)
    return base_logits, cf_logits


################################################################################
### simulated circuit
################################################################################
@op
@batched(args=['prompts'], n_outputs=1, reducer='cat')
def estimate_final_layernorm_scale(prompts: Any, # List[Prompt]
) -> Tensor:
    node = Node('scale_final')
    ln_scale_final = run_with_cache(prompts=prompts, nodes=[node], batch_size=100,)
    res = ln_scale_final[0].mean(dim=0)
    return res

################################################################################
### interaction w/ circuit 
################################################################################
class Head:
    def __init__(self, layer: int, head: int):
        self.layer = layer
        self.head = head
        self.attn_node = Node(component_name='attn_scores', layer=self.layer, head=self.head,)
    
    def __eq__(self, other):
        return self.layer == other.layer and self.head == other.head
    
    def __hash__(self):
        return hash((self.layer, self.head))
    
    def __repr__(self):
        return f'Head(L{self.layer}H{self.head})'
    
    @property
    def displayname(self) -> str:
        return f'L{self.layer}H{self.head}'


class HeadData:
    def __init__(self, 
                 ks: Dict[str, Node],
                 vs: Dict[str, Node],
                 qs: Dict[str, Node],
                 zs: Dict[str, Node],
                 scores: Optional[Tensor],
                 probs: Optional[Tensor],
                 head_class: str,
    ):
        self.ks = ks
        self.vs = vs
        self.qs = qs
        self.zs = zs
        self.scores = scores
        self.probs = probs
        self.head_class = head_class

        self.sem_probs_and_scores: pd.DataFrame = None
        self.reconstructed_attention = None
    
    @property
    def adjacent_nodes(self) -> List[Node]:
        return list(self.ks.values()) + list(self.vs.values()) + list(self.qs.values()) + list(self.zs.values())

    
class Circuit:
    def __init__(self, 
                 prompts: Optional[List[Prompt]] = None,
                 codes: Optional[Dict[str, Dict[Node, Dict[Tuple[str,...], Tensor]]]] = None,):
        if prompts is None:
            prompts = []
        self.prompts = prompts
        self.prompt_reps = [get_prompt_representation(p=p) for p in prompts]
        self.codes = codes
        try:
            model = MODELS[MODEL_ID]
            self.answer_tokens = PromptDataset(prompts, model=model).answer_tokens
        except:
            print('could not find model')
        
        ### placeholders for outputs
        self.logits: Tensor = None
        self.answer_logits: Tensor = None
        self.lds: Tensor = None

        self.nodes: Dict[Node, dict] = {}
        self.heads: Dict[Head, HeadData] = {}

        ### populate the heads
        self.pt = []
        for layer, head in PREVIOUS_TOKEN_HEADS:
            h = Head(layer=layer, head=head)
            self.heads[h] = HeadData(
                ks={'s1': Node('k', layer=layer, head=head, seq_pos='s1')},
                vs={'s1': Node('v', layer=layer, head=head, seq_pos='s1')},
                qs={'s1+1': Node('q', layer=layer, head=head, seq_pos='s1+1')},
                zs={'s1+1': Node('z', layer=layer, head=head, seq_pos='s1+1')},
                scores=None, probs=None, head_class='pt',)
            self.pt.append(h)
        self.dt = []
        for layer, head in DUPLICATE_TOKEN_HEADS:
            h = Head(layer=layer, head=head)
            self.heads[h] = HeadData(
                ks={'s1': Node('k', layer=layer, head=head, seq_pos='s1'),
                    's2': Node('k', layer=layer, head=head, seq_pos='s2'),},
                vs={'s1': Node('v', layer=layer, head=head, seq_pos='s1'),
                    's2': Node('v', layer=layer, head=head, seq_pos='s2'),},
                qs={'s2': Node('q', layer=layer, head=head, seq_pos='s2'),},
                zs={'s2': Node('z', layer=layer, head=head, seq_pos='s2'),},
                scores=None, probs=None, head_class='dt',)
            self.dt.append(h)
        self.ind = []
        for layer, head in INDUCTION_HEADS:
            h = Head(layer=layer, head=head)
            self.heads[h] = HeadData(
                ks={'s1+1': Node('k', layer=layer, head=head, seq_pos='s1+1'),},
                vs={'s1+1': Node('v', layer=layer, head=head, seq_pos='s1+1'),},
                qs={'s2': Node('q', layer=layer, head=head, seq_pos='s2'),},
                zs={'s2': Node('z', layer=layer, head=head, seq_pos='s2'),},
                scores=None, probs=None, head_class='ind',)
            self.ind.append(h)
        self.si = []
        for layer, head in S_INHIBITION_HEADS:
            h = Head(layer=layer, head=head)
            self.heads[h] = HeadData(
                ks={
                    # 'io': Node('k', layer=layer, head=head, seq_pos='io'),
                    # 's1': Node('k', layer=layer, head=head, seq_pos='s1'),
                    's2': Node('k', layer=layer, head=head, seq_pos='s2'),
                    },
                vs={
                    # 'io': Node('v', layer=layer, head=head, seq_pos='io'),
                    # 's1': Node('v', layer=layer, head=head, seq_pos='s1'),
                    's2': Node('v', layer=layer, head=head, seq_pos='s2'),
                    },
                qs={
                    # 'end': Node('q', layer=layer, head=head, seq_pos='end'),
                    },
                zs={'end': Node('z', layer=layer, head=head, seq_pos='end'),},
                scores=None, probs=None, head_class='si',)
            self.si.append(h)
        self.nm = []
        for layer, head in NAME_MOVERS:
            h = Head(layer=layer, head=head)
            self.heads[h] = HeadData(
                ks={'io': Node('k', layer=layer, head=head, seq_pos='io'),
                    's1': Node('k', layer=layer, head=head, seq_pos='s1'),},
                vs={'io': Node('v', layer=layer, head=head, seq_pos='io'),
                    's1': Node('v', layer=layer, head=head, seq_pos='s1'),},
                qs={'end': Node('q', layer=layer, head=head, seq_pos='end'),},
                zs={'end': Node('z', layer=layer, head=head, seq_pos='end'),},
                scores=None, probs=None, head_class='nm',)
            self.nm.append(h)
        self.bnm = []
        for layer, head in BACKUP_NAME_MOVERS:
            h = Head(layer=layer, head=head)
            self.heads[h] = HeadData(
                ks={'io': Node('k', layer=layer, head=head, seq_pos='io'),
                    's1': Node('k', layer=layer, head=head, seq_pos='s1'),
                    # 's2': Node('k', layer=layer, head=head, seq_pos='s2'),
                    },
                vs={'io': Node('v', layer=layer, head=head, seq_pos='io'),
                    's1': Node('v', layer=layer, head=head, seq_pos='s1'),},
                qs={'end': Node('q', layer=layer, head=head, seq_pos='end'),},
                zs={'end': Node('z', layer=layer, head=head, seq_pos='end'),},
                scores=None, probs=None, head_class='bnm',)
            self.bnm.append(h)
        self.nnm = []
        for layer, head in NEGATIVE_NAME_MOVERS:
            h = Head(layer=layer, head=head)
            self.heads[h] = HeadData(
                ks={'io': Node('k', layer=layer, head=head, seq_pos='io'),
                    's1': Node('k', layer=layer, head=head, seq_pos='s1'),
                    # 's2': Node('k', layer=layer, head=head, seq_pos='s2'),
                    },
                vs={'io': Node('v', layer=layer, head=head, seq_pos='io'),
                    's1': Node('v', layer=layer, head=head, seq_pos='s1'),},
                qs={'end': Node('q', layer=layer, head=head, seq_pos='end'),},
                zs={'end': Node('z', layer=layer, head=head, seq_pos='end'),},
                scores=None, probs=None, head_class='nnm',)
            self.nnm.append(h)
        
        ### now populate the nodes
        for head, head_data in self.heads.items():
            for node in head_data.adjacent_nodes:
                self.nodes[node] = {
                    'act': None,
                    'recons': {},
                    'head_class': head_data.head_class,
                }    
    
    def ks(self, hs: List[Head]) -> List[Node]:
        return [a for x in [list(self.heads[h].ks.values()) for h in hs] for a in x]
    
    def qs(self, hs: List[Head]) -> List[Node]:
        return [a for x in [list(self.heads[h].qs.values()) for h in hs] for a in x]
    
    def vs(self, hs: List[Head]) -> List[Node]:
        return [a for x in [list(self.heads[h].vs.values()) for h in hs] for a in x]
    
    def zs(self, hs: List[Head]) -> List[Node]:
        return [a for x in [list(self.heads[h].zs.values()) for h in hs] for a in x]
    
    def get_node_head_class(self, node: Node) -> str:
        for candidate_node, data in self.nodes.items():
            if node.layer == candidate_node.layer and node.head == candidate_node.head and node.component_name == candidate_node.component_name:
                return data['head_class']
        return 'other'
    
    def extract_sem_attn_probs_and_scores(self, head: Head,) -> pd.DataFrame:
        # it's attn[query_idx, key_idx]
        scores = self.heads[head].scores
        probs = self.heads[head].probs
        qs = self.heads[head].qs
        ks = self.heads[head].ks
        rows = []
        for i, p in enumerate(self.prompts):
            semantic_to_idx = p.semantic_pos
            for k_pos in ks.keys():
                for q_pos in qs.keys():
                    rows.append({
                        'qpos': q_pos,
                        'kpos': k_pos,
                        'prob': probs[i, semantic_to_idx[q_pos], semantic_to_idx[k_pos]].item(),
                        'score': scores[i, semantic_to_idx[q_pos], semantic_to_idx[k_pos]].item(),
                        'prompt_idx': i,
                    })
        return pd.DataFrame(rows)

    def get_freeze_hooks(self, nodes: List[Node]) -> List[Tuple[str, Callable]]:
        """
        Get hooks that freeze node values to what they are on the base prompts
        """
        def get_hook(node):
            def hook_fn(activation: Tensor, hook: HookPoint) -> Tensor:
                target_act = self.nodes[node]['act'].cuda()
                idx = node.idx(prompts=self.prompts)
                activation[idx] = target_act
                return activation
            return (node.activation_name, hook_fn)
        hooks = [get_hook(node) for node in nodes]
        return hooks
    
    def get_cf_hooks(self, nodes: List[Node], 
                     cf_circuit: 'Circuit') -> List[Tuple[str, Callable]]:
        def get_hook(node):
            def hook_fn(activation: Tensor, hook: HookPoint) -> Tensor:
                target_act = cf_circuit.nodes[node]['act'].cuda()
                idx = node.idx(prompts=self.prompts)
                activation[idx] = target_act
                return activation
            return (node.activation_name, hook_fn)
        hooks = [get_hook(node) for node in nodes]
        return hooks
    
    def get_edited_activations(self, nodes: List[Node], features: List[Tuple[str,...]],
                               method: Literal['mean_ablate_subspace', 'zero_ablate_subspace', 'arithmetic'],
                               cf_circuits: Dict[Tuple[str,...], 'Circuit'], codes_id: str, A_dict: Dict[Any, Tensor]):
        """
        Pure function that gives all the edited activations for this circuit's
        stored activations.
        """
        flat_features = tuple([f for fs in features for f in fs])
        cf_circuit = cf_circuits[flat_features]
        codes_dict = self.codes[codes_id]
        res = []
        for node in nodes:
            codes = codes_dict[node]
            features_present = codes.keys()
            # potentially, we must extend some features to match the features we
            # have in the codes
            new_features = []
            for f in features:
                if f == ('s',):
                    if f not in features_present:
                        new_features.append(('s', 'io_pos'))
                    else:
                        new_features.append(f)
                elif f == ('io',):
                    if f not in features_present:
                        new_features.append(('io', 'io_pos'))
                    else:
                        new_features.append(f)
                elif f == ('io_pos',):
                    if f not in features_present:
                        new_features.append(('s', 'io_pos'))
                    else:
                        new_features.append(f)
                else:
                    raise NotImplementedError(f'Got feature {f} but only know how to handle s, io, and io_pos')
            base_feature_idxs = get_prompt_feature_idxs(prompts=self.prompts, features=new_features)
            base_feature_idxs = {k: [tuple(v.cpu().tolist()) for v in vs] for k, vs in base_feature_idxs.items()}
            cf_feature_idxs = get_prompt_feature_idxs(prompts=cf_circuit.prompts, features=new_features)
            cf_feature_idxs = {k: [tuple(v.cpu().tolist()) for v in vs] for k, vs in cf_feature_idxs.items()}
            edited_act = get_edited_act(
                val=self.nodes[node]['act'].cuda(),
                codes=codes_dict[node],
                feature_idxs_to_delete=base_feature_idxs,
                feature_idxs_to_insert=cf_feature_idxs,
                A_reference=A_dict[(node.seq_pos, node.component_name, node.layer, node.head)],
                method=method,
            )
            res.append(edited_act)
        return res
        
    def get_simultaneous_edit_hooks(self, nodes: List[Node], features: List[Tuple[str,...]],
                       method: Literal['mean_ablate_subspace', 'zero_ablate_subspace', 'arithmetic'],
                       cf_circuits: Dict[Tuple[str,...], 'Circuit'],
                       codes_id: str, A_dict: Dict[Any, Tensor]):
        edited_activations = self.get_edited_activations(
            nodes=nodes,
            features=features,
            method=method,
            cf_circuits=cf_circuits,
            codes_id=codes_id,
            A_dict=A_dict,
        )
        def hook_fn(activation: Tensor, hook: HookPoint, i: int) -> Tensor:
            node = nodes[i]
            edited_act = edited_activations[i]
            idx = node.idx(prompts=self.prompts)
            activation[idx] = edited_act
            return activation
        hooks = [(node.activation_name, partial(hook_fn, i=i)) for i, node in enumerate(nodes)]
        return hooks
                               
    def get_edit_hooks(self, nodes: List[Node], features: List[Tuple[str,...]],
                       method: Literal['mean_ablate_subspace', 'zero_ablate_subspace', 'arithmetic'],
                       cf_circuits: Dict[Tuple[str,...], 'Circuit'],
                       codes_id: str, A_dict: Dict[Any, Tensor]):
        hooks = []
        flat_features = tuple([f for fs in features for f in fs])
        cf_circuit = cf_circuits[flat_features]
        codes_dict = self.codes[codes_id]
        for node in nodes:
            codes = codes_dict[node]
            features_present = codes.keys()
            # potentially, we must extend some features to match the features we
            # have in the codes
            new_features = []
            for f in features:
                if f == ('s',):
                    if f not in features_present:
                        new_features.append(('s', 'io_pos'))
                    else:
                        new_features.append(f)
                elif f == ('io',):
                    if f not in features_present:
                        new_features.append(('io', 'io_pos'))
                    else:
                        new_features.append(f)
                elif f == ('io_pos',):
                    if f not in features_present:
                        new_features.append(('s', 'io_pos'))
                    else:
                        new_features.append(f)
                else:
                    raise NotImplementedError(f'Got feature {f} but only know how to handle s, io, and io_pos')
            base_feature_idxs = get_prompt_feature_idxs(prompts=self.prompts, features=new_features)
            base_feature_idxs = {k: [tuple(v.cpu().tolist()) for v in vs] for k, vs in base_feature_idxs.items()}
            cf_feature_idxs = get_prompt_feature_idxs(prompts=cf_circuit.prompts, features=new_features)
            cf_feature_idxs = {k: [tuple(v.cpu().tolist()) for v in vs] for k, vs in cf_feature_idxs.items()}
            name, hook = get_edit_hook(
                node=node,
                codes=codes_dict[node],
                feature_idxs_to_delete=base_feature_idxs,
                feature_idxs_to_insert=cf_feature_idxs,
                method=method,
                A_reference=A_dict[(node.seq_pos, node.component_name, node.layer, node.head)],
            )
            hook = partial(hook, idx=node.idx(prompts=self.prompts))
            hooks.append((name, hook))
        return hooks
                
    def run(self, 
            answer_tokens: Optional[Tensor] = None,
            hooks: Optional[List[Tuple[str, Callable]]] = None, 
            return_full_last_logits: bool = False,
            batch_size: Optional[int] = None,
            verbose: bool = False,
            ):
        if answer_tokens is None:
            answer_tokens = self.answer_tokens
        if hooks is None:
            hooks = []
        if batch_size is None:
            batch_size = len(self.prompts)
        model: HookedTransformer = MODELS[MODEL_ID]
        nodes_list = list(self.nodes.keys())
        heads_list = list(self.heads.keys())
        attn_nodes = [h.attn_node for h in heads_list]
        num_nodes = len(nodes_list)
        
        if GlobalContext.current is not None:
            c = GlobalContext.current
        else:
            c = Context()
    
        with model.hooks(fwd_hooks=hooks):
            with c(mode='noop'):
                acts = run_with_cache(
                    prompts=self.prompts,
                    nodes=nodes_list + attn_nodes,
                    batch_size=batch_size,
                    return_logits=True,
                    verbose=verbose,
                    offload_to_cpu=True,
                    clear_cache=True,
                )
        for node, act in zip(nodes_list, acts[:num_nodes]):
            self.nodes[node]['act'] = act
        for head, act in zip(heads_list, acts[num_nodes:-1]):
            self.heads[head].scores = act
            self.heads[head].probs = torch.softmax(act, dim=-1)
            self.heads[head].sem_probs_and_scores = self.extract_sem_attn_probs_and_scores(head=head)
        logits = acts[-1][:, -1, :]

        self.logits = logits
        self.answer_logits = logits.gather(dim=-1, index=answer_tokens.to(logits.device))
        self.lds = self.answer_logits[:, 0] - self.answer_logits[:, 1]

        if return_full_last_logits:
            return self.logits
        else:
            return self.answer_logits
        
    def compute_reconstructions(self, decomposed=True):
        for code_type, codes_dict in self.codes.items():
            for node in self.nodes:
                node_codes = codes_dict[node]
                recons = get_reconstructions(
                    codes=node_codes,
                    prompts=self.prompts,
                    decomposed=decomposed,
                )
                recons['full'] = sum(recons.values())
                self.nodes[node]['recons'][code_type] = recons

    def reconstruct_attention(self, head: Head, code_type: str):
        self.heads[head].reconstructed_attention = reconstruct_attention(codes_dict=self.codes[code_type],
                                                 prompts=self.prompts,
                                                 head_data=self.heads[head],)
    
    def compare_attentions(self, head: Head, code_type: str):
        true_df = self.heads[head].sem_probs_and_scores
        recons_df = self.heads[head].reconstructed_attention.query('qfeature == "full" and kfeature == "full"')
        df = true_df.merge(recons_df, on=['qpos', 'kpos', 'prompt_idx'], suffixes=['_true', '_recons'])
        return df
    
    def __getitem__(self, x: Any):
        if isinstance(x, Node):
            return self.nodes[x]
        elif isinstance(x, Head):
            return self.heads[x]
        else:
            raise NotImplementedError    
    

def gather_attn_scores(head_data: HeadData,
                       activations_dict: Dict[Node, Tensor],
                       apply_biases: bool = False,
                       ):
    ks = head_data.ks
    qs = head_data.qs
    dfs = []
    for q_pos, q_node in qs.items():
        for k_pos, k_node in ks.items():
            q_act = activations_dict[q_node]
            k_act = activations_dict[k_node]
            if apply_biases:
                raise NotImplementedError()
            scores = einsum('batch dim, batch dim -> batch', q_act, k_act) / np.sqrt(64)
            dfs.append(pd.DataFrame({
                'qpos': [q_pos for _ in range(len(scores))],
                'kpos': [k_pos for _ in range(len(scores))],
                'dotprod': scores.cpu(),
                'prompt_idx': list(range(len(scores))),
            }))
    return pd.concat(dfs, ignore_index=True)
            
    
def reconstruct_attention(codes_dict: Dict[Node, Dict[Tuple[str,...], Tensor]],
                          prompts: List[Prompt],
                          head_data: HeadData,
                          ):
    qs = head_data.qs
    ks = head_data.ks
    dfs = []
    for k_pos, k_node in ks.items():
        for q_pos, q_node in qs.items():
            q_codes, k_codes = codes_dict[q_node], codes_dict[k_node]
            k_recons:dict = get_reconstructions(codes=k_codes, prompts=prompts, decomposed=True,)
            k_recons['full'] = sum(k_recons.values())
            q_recons:dict = get_reconstructions(codes=q_codes, prompts=prompts, decomposed=True,)
            q_recons['full'] = sum(q_recons.values())
            for k_feature, k_activations in k_recons.items():
                for q_feature, q_activations in q_recons.items():
                    dfs.append(pd.DataFrame({
                        'qpos': [q_pos for _ in range(len(k_activations))],
                        'kpos': [k_pos for _ in range(len(k_activations))],
                        'qfeature': [q_feature for _ in range(len(k_activations))],
                        'kfeature': [k_feature for _ in range(len(k_activations))],
                        'coords': [str((q_pos, k_pos, str(q_feature), str(k_feature))) for _ in range(len(k_activations))],
                        'reconstruction': einsum('batch dim, batch dim -> batch', q_activations, k_activations).cpu() / np.sqrt(64),
                        'prompt_idx': list(range(len(k_activations))),
                    }))
    return pd.concat(dfs, ignore_index=True)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

@op
def train_logistic_probe(
    A_train: Tensor, 
    A_test: Tensor,
    y_train: Any,
    y_test: Any,
    C: float = 1.0,
) -> Tuple[LogisticRegression, float, float]:
    if isinstance(y_train, list):
        y_train = np.array(y_train)
    elif isinstance(y_train, Tensor):
        y_train = y_train.cpu().numpy()
    elif isinstance(y_train, np.ndarray):
        pass
    else:
        raise ValueError(f'unknown type {type(y_train)}')
    if isinstance(y_test, list):
        y_test = np.array(y_test)
    elif isinstance(y_test, Tensor):
        y_test = y_test.cpu().numpy()
    elif isinstance(y_test, np.ndarray):
        pass
    else:
        raise ValueError(f'unknown type {type(y_train)}')
    scaler = StandardScaler()
    A_train = scaler.fit_transform(A_train.cpu())
    A_test = scaler.transform(A_test.cpu())
    clf = LogisticRegression(C=C, max_iter=1000).fit(A_train, y_train)
    train_acc = clf.score(A_train, y_train)
    test_acc = clf.score(A_test, y_test)
    return clf, train_acc, test_acc
