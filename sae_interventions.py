from ioi_utils import *
from circuit_utils import get_prompt_feature_idxs
from sae_variants import VanillaAutoEncoder, GatedAutoEncoder, AttributionAutoEncoder
# from webtext_utils import *

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

################################################################################
### lol what is this doing here?
################################################################################
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

from circuit_utils import get_forced_hook

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
    for i in tqdm(range(num_batches), disable=True):
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
    ones = torch.ones_like(nonzero_sums)
    sess.d()
    assert torch.allclose(nonzero_sums, ones, atol=0.05), sums
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