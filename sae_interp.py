from ioi_utils import *
from sae_variants import VanillaAutoEncoder, GatedAutoEncoder, AttributionAutoEncoder
from circuit_utils import FEATURE_SIZES

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

