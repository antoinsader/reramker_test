import json
import torch
import torch.nn.functional as F


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
