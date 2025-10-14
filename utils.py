import json
import torch
import torch.nn.functional as F
import numpy as np


from config import loss_type


def load_mmap_shape(json_file):
    with open(json_file) as f:
        meta = json.load(f)
    return tuple(meta["shape"])


    

def marginal_nll(score, target):
    """
    sum all scores among positive samples
    """
    predict = F.softmax(score, dim=-1)
    loss = predict * target
    loss = loss.sum(dim=-1)                   # sum all positive scores
    loss = loss[loss > 0]                     # filter sets with at least one positives
    loss = torch.clamp(loss, min=1e-9, max=1) # for numerical stability
    loss = -torch.log(loss)                   # for negative log likelihood
    if len(loss) == 0:
        loss = loss.sum()                     # will return zero loss
    else:
        loss = loss.mean()
    return loss


def info_nce_loss(scores,targets):
    # logits = F.log_softmax(scores, dim=-1)
    # loss = -(logits * targets).sum(dim=1)
    # return loss.mean()
    return F.cross_entropy(scores, targets, ignore_index=-100)


def get_labels(cand_cuis, query_cui):
    """
        Generate labels for a query:
        - InfoNCE: integer index of first positive, -100 if no match
        - Marginal NLL: float vector of 0.0/1.0 per candidate
    """
    if loss_type == "info_nce_loss":
        matches = np.where(cand_cuis == query_cui)[0]
        if len(matches) == 0:
            return torch.tensor(-100, dtype=torch.long)
        else:
            return torch.tensor(matches[0], dtype=torch.long)
    elif loss_type == "marginal_nll":
        labels = (cand_cuis == query_cui).astype(np.float32)
        return torch.tensor(labels, dtype=torch.float)


from torchmetrics.functional import retrieval_reciprocal_rank, retrieval_recall

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
