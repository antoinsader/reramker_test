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
    logits = F.log_softmax(scores, dim=-1)
    loss = -(logits * targets).sum(dim=1)
    return loss.mean()


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
