from ioi_utils import *
from circuit_utils import multidim_argmax, multidim_topk, get_feature_mask, get_feature_scores, get_prompt_representation, get_prompt_feature_idxs
from webtext_utils import *

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

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
def precompute_activations(X: Any, node: Node, batch_size: int, with_timeout: Optional[float] = None) -> torch.Tensor:
    if with_timeout is not None:
        A = run_custom(prompts=X, batch_size=batch_size, node=node, timeout=with_timeout)
    else:
        A = run_with_cache(prompts=X, nodes=[node], batch_size=batch_size, hook_specs=[],)[0]
    return A

@op
def get_dataset_mean(A: Tensor) -> Tensor:
    return A.mean(dim=0)

################################################################################
### autoencoder and supporting functions 
################################################################################
@torch.no_grad()
def get_freqs(encoder: AutoEncoder, A: Tensor, batch_size: Optional[int] = None) -> Tuple[Tensor, float]:
    """
    Get the feature frequencies for the given activations, and the fraction of
    dead neurons.
    """
    act_freq_scores = torch.zeros(encoder.d_hidden, dtype=torch.float32).cuda()
    total = 0
    if batch_size is None:
        num_batches = 1
        batch_size = A.shape[0]
    else:
        num_batches = A.shape[0] // batch_size
    with torch.no_grad():
        for i in range(num_batches):
            A_batch = A[i*batch_size:(i+1)*batch_size]
            acts = encoder(A_batch)[2]
            act_freq_scores += (acts > 0).sum(0)
            total += acts.shape[0]
    act_freq_scores /= total
    frac_dead = (act_freq_scores==0).float().mean().item()
    return act_freq_scores, frac_dead

@op(version=2)
def get_high_f1_features(
    encoder: Union[AutoEncoder, SparseAutoencoder],
    attributes: List[Tuple[str,...]],
    prompt_feature_idxs: Any,
    A: Tensor,
    topk: int,
    normalization_scale: Optional[float],
) -> Tuple[Any, Any]:
    """
    Given some attributes, find the top features wrt these attributes based on
    the F1 score; return the features together with their scores.

    Returns: 
    - {attr: tensor of shape (*attr_shape, topk) of the top features (ordered)}
    - {attr: tensor of shape (*attr_shape, topk) of the top F1 scores (same order)}
    """
    is_webtext_sae = get_is_webtext_sae(encoder)
    use_normalization = (normalization_scale is not None and (not is_webtext_sae))
    with torch.no_grad():
        if use_normalization:
            A = A / normalization_scale
        if is_webtext_sae:
            # the only thing this function cares about is the activation
            # pattern, so we don't need other adjustments
            encoder = encoder.to(A.device)
            acts = encoder.encoder(A)
        else:
            acts = encoder(A)[2]
    activation_pattern = (acts > 0).float() # (num_examples, num_features)
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

@op(version=1)
def autointerp_fast(
    encoder: Union[AutoEncoder, SparseAutoencoder],
    features: List[Tuple[str,...]],
    prompt_feature_idxs: Any, 
    A: Tensor,
    features_to_group: List[Tuple[str,...]],
    max_group_size: int,
    normalization_scale: Optional[float],
    feature_batch_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Any, Any]:
    """
    Score features according to the F1 score. Return:
    - the best F1 score and corresponding index for each feature
    - the best F1 score and corresponding indices for each group of features
    """
    is_webtext_sae = get_is_webtext_sae(encoder=encoder)
    use_normalization = (normalization_scale is not None and (not is_webtext_sae))
    with torch.no_grad():
        if use_normalization:
            A = A / normalization_scale
        if is_webtext_sae:
            # this function only cares about the activation pattern, no need for
            # other adjustments
            encoder = encoder.to(A.device)
            acts = encoder(A)[1]
        else:
            encoder = encoder.to(A.device)
            acts = encoder(A)[2]
    activation_pattern = (acts > 0).float() # (num_examples, num_features)
    n_examples = A.shape[0]
    n_features = acts.shape[1]
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


@op
def eval_ld_loss(
    encoder: AutoEncoder,
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

def mean_ablate_hook(activation: Tensor, hook: HookPoint, node: Node, mean: Tensor, idx: Tensor) -> Tensor:
    activation[idx] = mean
    return activation

@op
def compute_mean_ablated_lds(
    node: Node,
    prompts: Any,
    A_mean: Tensor,
    batch_size: int,
) -> float:
    if GlobalContext.current is not None:
        c = GlobalContext.current
    else:
        c = Context()
    with c(mode='noop'):
        mean_ablated_logits = run_with_hooks(
            prompts=prompts,
            hooks=None,
            semantic_nodes=[node],
            semantic_hooks=[(node.activation_name, partial(mean_ablate_hook, node=node, mean=A_mean))],
            batch_size=batch_size,
        )
    mean_ablated_ld = (mean_ablated_logits[:, 0] - mean_ablated_logits[:, 1]).mean().item()
    return mean_ablated_ld

def encoder_hook(activation: Tensor, hook: HookPoint, node: Node, encoder: AutoEncoder,
                 idx: Tensor, normalization_scale: Optional[float] = None,
                 ) -> Tensor:
    with torch.no_grad():
        if normalization_scale is not None:
            A = activation[idx] / normalization_scale
        else:
            A = activation[idx]
        reconstructions = encoder(A)[1]
        if normalization_scale is not None:
            reconstructions = reconstructions * normalization_scale
        activation[idx] = reconstructions
    return activation


@torch.no_grad()
def get_logitdiff_loss(
    encoder: AutoEncoder, node: Node, prompts: List[Prompt], batch_size: int,
    activation_mean: Tensor, normalization_scale: Optional[float] = None,
    mean_clean_ld: Optional[float] = None, mean_ablated_ld: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
    """
    The close this is to zero, the better the reconstruction.
    """
    if GlobalContext.current is not None:
        c = GlobalContext.current
    else:
        c = Context()
    with c(mode='noop'):
        if mean_clean_ld is None:
            clean_logits = run_with_hooks(
                prompts=prompts,
                hooks=[],
                batch_size=batch_size,
            )
            clean_ld = (clean_logits[:, 0] - clean_logits[:, 1]).mean().item()
        else:
            clean_ld = mean_clean_ld
        # zero_ablated_logits = run_with_hooks(
        #     prompts=prompts,
        #     hook_specs=[],
        #     hooks=[(node.activation_name, partial(zero_ablate_hook, node=node))],
        #     batch_size=batch_size,
        # )
        if mean_ablated_ld is None:
            mean_ablated_logits = run_with_hooks(
                prompts=prompts,
                hooks=None,
                semantic_nodes=[node],
                semantic_hooks=[(node.activation_name, partial(mean_ablate_hook, node=node, mean=activation_mean))],
                batch_size=batch_size,
            )
            mean_ablated_ld = (mean_ablated_logits[:, 0] - mean_ablated_logits[:, 1]).mean().item()
        else:
            mean_ablated_ld = mean_ablated_ld
        encoder_logits = run_with_hooks(
            prompts=prompts,
            hooks=None,
            semantic_nodes=[node],
            semantic_hooks=[(node.activation_name, partial(encoder_hook, node=node, encoder=encoder, normalization_scale=normalization_scale))],
            batch_size=batch_size,
        )
        encoder_ld = (encoder_logits[:, 0] - encoder_logits[:, 1]).mean().item()
    # clean_ld = (clean_logits[:, 0] - clean_logits[:, 1]).mean()
    # mean_ablated_ld = (mean_ablated_logits[:, 0] - mean_ablated_logits[:, 1]).mean()
    score = (clean_ld - encoder_ld) / (clean_ld - mean_ablated_ld)
    return score, clean_ld, mean_ablated_ld, encoder_ld

def normalize_sae_inputs(A: Tensor) -> Tuple[Tensor, float]:
    """
    Normalize by a scalar, so that on average, the sum of squares in each row of
    A is 1. This is optionally used to make the same hyperparameters applicable 
    across different input scales.
    """
    assert len(A.shape) == 2
    scale = A.pow(2).sum(dim=1).mean().sqrt()
    return A / scale, scale.item()

################################################################################
### editing with SAEs
################################################################################
def get_feature_weights(
    encoder: Union[AutoEncoder, SparseAutoencoder],
    A: Tensor, 
    batch_size: int,
    normalization_scale: Optional[float],
    ) -> Tuple[Tensor, Tensor]:
    is_webtext_sae = get_is_webtext_sae(encoder=encoder)
    use_normalization = (normalization_scale is not None) and (not is_webtext_sae)
    if use_normalization:
        A = A / normalization_scale
    with torch.no_grad():
        if is_webtext_sae:
            encoder = encoder.to(A.device)
            recons, acts = encoder(A)
            # convert the reconstructions back to original scale and mean
            # this is the only adjustment we need here
            recons = (recons - encoder.mean) / encoder.standard_norm
        else:
            _, recons, acts, _, _ = encoder(A)
    num_examples = A.shape[0]
    num_batches = num_examples // batch_size
    feature_weights_batches = []
    for i in range(num_batches):
        acts_batch = acts[i*batch_size:(i+1)*batch_size]
        centered_recons_batch = recons[i*batch_size:(i+1)*batch_size] - encoder.b_dec.detach().unsqueeze(0)
        centered_recons_norms = centered_recons_batch.norm(dim=-1, keepdim=True)
        feature_weights = einsum('batch hidden, hidden dim, batch dim -> batch hidden', acts_batch, encoder.W_dec.detach(), centered_recons_batch) / centered_recons_norms**2
        feature_weights_batches.append(feature_weights)
    feature_weights = torch.cat(feature_weights_batches, dim=0)
    sums = feature_weights.sum(dim=1)
    nonzero_sums = sums[sums != 0]
    ones = torch.ones_like(nonzero_sums)
    assert torch.allclose(nonzero_sums, ones, atol=0.05), sums
    return feature_weights, acts


@op
def compute_removed_weight(
    encoder: Union[AutoEncoder, SparseAutoencoder],
    A: Tensor,
    batch_size: int,
    normalization_scale: Optional[float],
    best_features: Tensor, 
    ) -> Tensor:
    weights, _ = get_feature_weights(encoder, A, batch_size, normalization_scale)
    best_weights = torch.stack([weights[range(A.shape[0]), best_features[:, i]] for i in range(best_features.shape[1])], dim=1)
    return best_weights.sum(dim=1)


@op(version=2)
@batched(args=['A_clean', 'A_cf', 'clean_prompts', 'cf_prompts',], n_outputs=3, reducer='cat', verbose=False)
def get_edit_using_f1_scores(
    A_clean: Tensor,
    A_cf: Tensor,
    clean_prompts: Any, 
    cf_prompts: Any,
    attribute: Tuple[str,...],
    high_f1_features: Any, # of shape (*attribute_shape, topk)
    encoder: Union[AutoEncoder, SparseAutoencoder],
    num_exchange: int,
    normalization_scale: Optional[float] = None,
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
    """
    n_examples = A_clean.shape[0]
    is_webtext_sae = get_is_webtext_sae(encoder=encoder)
    use_normalization = (normalization_scale is not None and (not is_webtext_sae))
    if use_normalization:
        A_clean = A_clean / normalization_scale
        A_cf = A_cf / normalization_scale
    
    with torch.no_grad():
        if is_webtext_sae:
            encoder = encoder.to(A_clean.device)
            acts_clean = encoder.encoder(A_clean)
            acts_cf = encoder.encoder(A_cf)
        else:
            _, _, acts_clean, _, _ = encoder(A_clean)
            _, _, acts_cf, _, _ = encoder(A_cf)

    clean_feature_idxs = get_prompt_feature_idxs(
        prompts=clean_prompts,
        features=[attribute],
    )
    cf_feature_idxs = get_prompt_feature_idxs(
        prompts=cf_prompts, 
        features=[attribute],
    )
    # now, figure out which features to add/remove
    attr_idxs_clean = clean_feature_idxs[attribute].squeeze()
    attr_idxs_cf = cf_feature_idxs[attribute].squeeze()
    assert len(attr_idxs_clean.shape) == 1
    # old code 
    # features_to_remove = high_f1_features[attr_idxs_clean, :num_exchange] # of shape (batch, num_exchange)
    # features_to_add = high_f1_features[attr_idxs_cf, :num_exchange] # of shape (batch, num_exchange) 
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
    clean_active = (acts_clean > 0).bool()
    cf_active = (acts_cf > 0).bool()
    clean_active = pad_2d_boolean_mask(clean_active, num_exchange)
    cf_active = pad_2d_boolean_mask(cf_active, num_exchange)

    features_to_remove = []
    for i in tqdm(range(n_examples)):
        t = high_f1_features[attr_idxs_clean[i]]
        features_to_remove.append(t[torch.isin(t, torch.where(clean_active[i])[0])][:num_exchange])
        # this is wrong, fixed in v2
        # features_to_remove.append(high_f1_features[attr_idxs_clean[i], torch.where(clean_active[i])[0][:num_exchange]])
    features_to_remove = torch.stack(features_to_remove, dim=0).long()
    features_to_add = []
    for i in tqdm(range(n_examples)):
        t = high_f1_features[attr_idxs_cf[i]]
        features_to_add.append(t[torch.isin(t, torch.where(cf_active[i])[0])][:num_exchange])
        # this is wrong, fixed in v2
        # features_to_add.append(high_f1_features[attr_idxs_cf[i], torch.where(cf_active[i])[0][:num_exchange]])
    features_to_add = torch.stack(features_to_add, dim=0).long()
    
    ### now, perform the edits
    W_dec = encoder.W_dec.detach() # (hidden, dim)

    # shape (batch, num_exchange)
    coeffs_to_remove = torch.stack([acts_clean[range(n_examples), features_to_remove[:, i]] for i in range(num_exchange)], dim=1)
    coeffs_to_add = torch.stack([acts_cf[range(n_examples), features_to_add[:, i]] for i in range(num_exchange)], dim=1)

    # shape (batch, num_exchange, dim)
    decoders_to_remove = W_dec[features_to_remove, :]
    decoders_to_add = W_dec[features_to_add, :]

    to_remove = einsum("batch num_exchange, batch num_exchange dim -> batch dim", coeffs_to_remove, decoders_to_remove)
    to_add = einsum("batch num_exchange, batch num_exchange dim -> batch dim", coeffs_to_add, decoders_to_add)

    if is_webtext_sae:
        # we must bring the vectors to the original activation scale
        to_remove = to_remove * encoder.standard_norm + encoder.mean
        to_add = to_add * encoder.standard_norm + encoder.mean

    A_edited = A_clean - to_remove + to_add
    if use_normalization:
        A_edited = A_edited * normalization_scale
    return A_edited, features_to_remove, features_to_add


@op(version=3)
@batched(args=['A_clean', 'A_cf'], n_outputs=5, reducer='cat', verbose=False)
def get_edit_using_sae_opt(
    A_clean: Tensor,
    A_cf: Tensor,
    encoder: Union[AutoEncoder, SparseAutoencoder],
    num_exchange: int,
    normalization_scale: Optional[float] = None,
    diff_to_use: Literal['reconstruction', 'activation'] = 'activation',
    batch_size: int = 100,
    ) -> Tuple[Tensor, Any, Any, Tensor, Tensor]:
    """
    Greedily solve the optimization problem of subtracting/adding the fewest 
    features to minimize the norm 
    """
    is_webtext_sae = get_is_webtext_sae(encoder=encoder)
    use_normalization = (normalization_scale is not None and (not is_webtext_sae))
    n_examples = A_clean.shape[0]
    if use_normalization:
        A_clean = A_clean / normalization_scale
        A_cf = A_cf / normalization_scale
    
    if not is_webtext_sae:
        with torch.no_grad():
            _, recons_clean, acts_clean, _, _ = encoder(A_clean)
            _, recons_cf, acts_cf, _, _ = encoder(A_cf)
    else:
        with torch.no_grad():
            encoder = encoder.to(A_clean.device)
            recons_clean, acts_clean = encoder(A_clean)
            recons_cf, acts_cf = encoder(A_cf)

    if diff_to_use == 'reconstruction':
        diff = recons_cf - recons_clean # shape (batch, dim)
    elif diff_to_use == 'activation':
        diff = A_cf - A_clean
    else:
        raise ValueError(f"Invalid value for `diff_to_use`: {diff_to_use}")
    
    W_dec = encoder.W_dec.detach().clone()

    def optimize_vectorized(num_exchange:int):
        current_sums = torch.zeros_like(diff) # shape (batch, dim)
        best_features_list = []
        best_scores_list = []
        # initialize the *differences* between each respective feature's
        # contribution in the cf and clean reconstructions
        feature_diffs = einsum('batch hidden, hidden dim -> batch hidden dim', acts_cf-acts_clean, W_dec)
        if is_webtext_sae:
            # we must bring the dot products to the original scale of activations
            feature_diffs = feature_diffs * encoder.standard_norm + encoder.mean.unsqueeze(0).unsqueeze(0)
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
        edited_clean = torch.stack([acts_clean[range(n_examples), best_features[:, i]] for i in range(num_exchange)], dim=1)
        # the hidden activations of the edited features on the cf side
        edited_cf = torch.stack([acts_cf[range(n_examples), best_features[:, i]] for i in range(num_exchange)], dim=1)
        return best_features, best_scores, current_sums, edited_clean, edited_cf

    best_features, best_scores, deltas, edited_clean, edited_cf = optimize_vectorized(num_exchange)
    A_edited = A_clean + deltas
    if use_normalization:
        A_edited = A_edited * normalization_scale
        
    return A_edited, best_features, best_scores, edited_clean, edited_cf


@op(version=1)
def get_edit_using_sae_2(
    A_clean: Tensor,
    A_cf: Tensor,
    encoder: AutoEncoder,
    threshold: float = 0.1,
    normalization_scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Edit activations using a learned SAE. Find the features that matter the most
    for the reconstructions of the clean/cf activations, and subtract/add based
    on this
    """
    with torch.no_grad():
        feature_weights_clean, acts_clean = get_feature_weights(encoder, A_clean, 100, normalization_scale=normalization_scale)
        feature_weights_cf, acts_cf = get_feature_weights(encoder, A_cf, 100, normalization_scale=normalization_scale)
    removal_mask = ((feature_weights_clean > threshold) & ~(feature_weights_cf > threshold))
    removed_feature_weight = (feature_weights_clean * removal_mask).sum(dim=1)
    acts_edited = acts_clean.clone()
    acts_edited[removal_mask] = 0
    addition_mask = (~(feature_weights_clean > threshold) & (feature_weights_cf > threshold))
    acts_edited = acts_edited + acts_cf * addition_mask
    num_active_features = (acts_clean > threshold).sum(dim=1).float()
    # num_edited_features = (acts_edited > 0).sum(dim=1).float()
    num_edited_features = (removal_mask | addition_mask).sum(dim=1).float()
    # now, build A_edited
    A_edited = encoder.b_dec.detach() + einsum('batch hidden, hidden dim -> batch dim', acts_edited, encoder.W_dec.detach())
    if normalization_scale is not None:
        A_edited = A_edited * normalization_scale
    return A_edited, num_active_features, num_edited_features, removed_feature_weight

@op
def get_sae_reconstructions(
    encoder: Union[AutoEncoder, SparseAutoencoder],
    A: torch.Tensor,
    normalization_scale: Optional[torch.Tensor],
) -> Tensor:
    is_webtext_sae = get_is_webtext_sae(encoder=encoder)
    use_normalization = (normalization_scale is not None and (not is_webtext_sae))
    if use_normalization:
        A = A / normalization_scale
    with torch.no_grad():
        if is_webtext_sae:
            encoder = encoder.to(A.device)
            recons, _ = encoder(A)
        else:
            recons = encoder(A)[1]
    if use_normalization:
        recons = recons * normalization_scale
    return recons

from circuit_utils import get_forced_hook

def remove_features(
    A: Tensor,
    encoder: Union[AutoEncoder, SparseAutoencoder],
    encoder_acts: Tensor, # (batch, hidden)
    feature_idxs: Tensor, # (num_remove)
):
    """
    Edit activation by removing the given feature indices
    """
    is_webtext_sae = get_is_webtext_sae(encoder)
    W_dec = encoder.W_dec.detach() # (hidden, dim)
    feature_contribution = einsum('batch hidden, hidden dim -> batch dim', encoder_acts[:, feature_idxs], W_dec[feature_idxs, :])
    if is_webtext_sae:
        feature_contribution = feature_contribution * encoder.standard_norm + encoder.mean
    A = A - feature_contribution
    return A

def keep_features(
    A: Tensor,
    encoder: Union[AutoEncoder, SparseAutoencoder],
    encoder_acts: Tensor, # (batch, hidden)
    feature_idxs: Tensor, # (num_remove)
):
    """
    Edit activation by removing all features except the given feature indices
    """
    is_webtext_sae = get_is_webtext_sae(encoder)
    W_dec = encoder.W_dec.detach() # (hidden, dim)
    n_features = W_dec.shape[0]
    feature_idxs_to_remove = torch.tensor([i for i in range(n_features) if i not in feature_idxs], device=feature_idxs.device).long()
    feature_contribution = einsum('batch hidden, hidden dim -> batch dim', encoder_acts[:, feature_idxs_to_remove], W_dec[feature_idxs_to_remove, :])
    if is_webtext_sae:
        feature_contribution = feature_contribution * encoder.standard_norm + encoder.mean
    A = A - feature_contribution
    return A

@op(version=2)
def get_interp_intervention(
    prompts: Any, # List[Prompt]
    nodes: List[Node],
    As: List[Tensor],
    batch_size: int,
    encoders: List[Union[AutoEncoder, SparseAutoencoder]],
    features: List[Tensor], # List[Tensor]
    keep_or_remove: Literal['keep', 'remove'],
    normalization_scales: List[Optional[float]],
    model_id: str = 'gpt2small',
) -> Tensor:
    """
    Run activation patch that either removes or keeps given features from the
    activations. This saves memory by avoiding the storage of the edited
    activations.
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
                A_edited = keep_features(A / normalization_scale, encoder, encoder_acts, feature_idxs)
            A_edited = A_edited * normalization_scale
            As_edited.append(A_edited)

        hooks = [get_forced_hook(prompts=prompts_batch, node=node, A=A_edited) for node, A_edited in zip(nodes, As_edited)]
        changed_logits = model.run_with_hooks(batch_dataset.tokens, fwd_hooks=hooks)[:, -1, :]
        answer_logits = changed_logits.gather(dim=-1, index=batch_dataset.answer_tokens.cuda())
        answer_logits_list.append(answer_logits)
    answer_logits = torch.cat(answer_logits_list, dim=0)
    return answer_logits