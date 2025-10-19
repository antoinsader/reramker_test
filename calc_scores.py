



import torch.nn.functional as F
from utils import get_pkl


batch_x  = get_pkl("./data/draft/batch_x.pkl")
batch_y  = get_pkl("./data/draft/batch_y.pkl")
batch_y_pred  = get_pkl("./data/draft/batch_y_pred.pkl")

batch_y = batch_y.to("cuda")
batch_y_pred = batch_y_pred.to("cuda")

query_tokens, candidate_tokens= batch_x

print(f"labels: {batch_y}")
print(f"batch_y_pred: {batch_y_pred}")
print(f"batch_y: {batch_y}")

def info_nce_loss(scores,targets, temperature=0.07):
    scores = scores / temperature
    return F.cross_entropy(scores, targets, ignore_index=-100)

loss = info_nce_loss(batch_y_pred, batch_y)
print(f"loss: {loss.item()}")