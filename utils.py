import json
import torch
import torch.nn.functional as F
import numpy as np
import pickle






def save_pkl(ar, fp):
    with open(fp, 'wb') as f:
        pickle.dump(ar, f)
def get_pkl(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)



def load_mmap_shape(json_file):
    with open(json_file) as f:
        meta = json.load(f)
    return tuple(meta["shape"])



def marginal_nll(scores, labels):

    # Mask out invalid entries (e.g., queries with no positives)
    valid_mask = labels.sum(dim=1) > 0
    if not valid_mask.any():
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    scores = scores[valid_mask]
    labels = labels[valid_mask]

    # Exponentiate scores
    exp_scores = torch.exp(scores)

    # For each query, numerator = sum of exp(sim) over positive candidates
    numerator = (exp_scores * labels).sum(dim=1)
    denominator = exp_scores.sum(dim=1) + 1e-8  # avoid division by zero

    # Negative log-likelihood
    loss = -torch.log(numerator / denominator + 1e-8)

    return loss.mean()



def info_nce_loss(scores,targets):
    return F.cross_entropy(scores, targets, ignore_index=-100)


def get_labels(query_candidates_cuis, query_cui, loss_type):
    """
        Generate labels for a query:
        - InfoNCE: integer index of first positive, -100 if no match
        - Marginal NLL: float vector of 0.0/1.0 per candidate
    """
    if loss_type == "info_nce_loss":
        matches = np.where(query_candidates_cuis == query_cui)[0]
        if len(matches) == 0:
            return torch.tensor(-100, dtype=torch.long)
        else:
            return torch.tensor(matches[0], dtype=torch.long)
    elif loss_type == "marginal_nll":
        labels = (query_candidates_cuis == query_cui).astype(np.float32)
        return torch.tensor(labels, dtype=torch.float)


def compute_metrics(scores, targets, k=5):
    """
    Compute top-k accuracy and MRR for retrieval scores.
    Works for both info_nce_loss (int targets) and marginal_nll (float vector targets).
    """
    with torch.no_grad():
        batch_size = scores.size(0)
        topk = scores.size(1)

        # Handle info_nce style targets (one positive index or -100)
        if targets.dtype == torch.long:
            acc_count, rr_sum, valid = 0, 0.0, 0
            topk_preds = scores.topk(k, dim=-1).indices  # [B, k]

            for i in range(batch_size):
                t = targets[i].item()
                if t == -100 or t < 0 or t >= topk:
                    continue
                valid += 1
                preds = topk_preds[i].tolist()
                # Acc@k
                if t in preds:
                    acc_count += 1
                # Reciprocal rank
                rank = (scores[i].argsort(descending=True) == t).nonzero(as_tuple=True)[0].item() + 1
                rr_sum += 1.0 / rank

            acc_at_k = acc_count / max(valid, 1)
            mrr = rr_sum / max(valid, 1)

        # Handle marginal_nll style targets (float vector of 0/1)
        else:
            # Get the index of all positives
            positives = (targets > 0.5)
            acc_count, rr_sum, valid = 0, 0.0, 0
            topk_preds = scores.topk(k, dim=-1).indices

            for i in range(batch_size):
                pos_idxs = positives[i].nonzero(as_tuple=True)[0].tolist()
                if len(pos_idxs) == 0:
                    continue
                valid += 1
                preds = topk_preds[i].tolist()
                if any(p in pos_idxs for p in preds):
                    acc_count += 1
                # Compute rank of first correct one
                ranking = scores[i].argsort(descending=True)
                for r, idx in enumerate(ranking.tolist(), start=1):
                    if idx in pos_idxs:
                        rr_sum += 1.0 / r
                        break

            acc_at_k = acc_count / max(valid, 1)
            mrr = rr_sum / max(valid, 1)

    return acc_at_k, mrr


def compute_metrics_eval(scores, targets, multiple_ks=None, k=5):
    """
    Compute top-k accuracy and MRR for retrieval scores.
    Works for both info_nce_loss (int targets) and marginal_nll (float vector targets).

    Args:
        scores (Tensor): shape [batch_size, topk]
        targets (Tensor): either long (for info_nce) or float (for marginal_nll)
        multiple_ks (list[int], optional): e.g., [1, 5, 10]. If provided, returns multiple accuracies.
        k (int): default top-k if multiple_ks not provided

    Returns:
        dict: {'acc@1': ..., 'acc@5': ..., 'mrr': ...}
    """
    with torch.no_grad():
        batch_size = scores.size(0)
        topk = scores.size(1)
        ks = multiple_ks if multiple_ks is not None else [k]

        results = {}
        total_rr_sum, total_valid = 0.0, 0

        # Handle info_nce style targets (single positive index)
        if targets.dtype == torch.long:
            topk_preds = scores.argsort(dim=-1, descending=True)  # [B, topk]
            acc_counts = {kk: 0 for kk in ks}

            for i in range(batch_size):
                t = targets[i].item()
                if t == -100 or t < 0 or t >= topk:
                    continue
                total_valid += 1
                preds = topk_preds[i].tolist()

                # Reciprocal rank
                rank = preds.index(t) + 1 if t in preds else topk + 1
                total_rr_sum += 1.0 / rank

                for kk in ks:
                    if t in preds[:kk]:
                        acc_counts[kk] += 1

            for kk in ks:
                results[f"acc@{kk}"] = acc_counts[kk] / max(total_valid, 1)
            results["mrr"] = total_rr_sum / max(total_valid, 1)

        # Handle marginal_nll style targets (float vector of 0/1)
        else:
            positives = (targets > 0.5)
            topk_preds = scores.argsort(dim=-1, descending=True)
            acc_counts = {kk: 0 for kk in ks}

            for i in range(batch_size):
                pos_idxs = positives[i].nonzero(as_tuple=True)[0].tolist()
                if len(pos_idxs) == 0:
                    continue
                total_valid += 1
                preds = topk_preds[i].tolist()

                # Reciprocal rank of first correct
                for r, idx in enumerate(preds, start=1):
                    if idx in pos_idxs:
                        total_rr_sum += 1.0 / r
                        break

                for kk in ks:
                    if any(p in pos_idxs for p in preds[:kk]):
                        acc_counts[kk] += 1

            for kk in ks:
                results[f"acc@{kk}"] = acc_counts[kk] / max(total_valid, 1)
            results["mrr"] = total_rr_sum / max(total_valid, 1)

    return results
