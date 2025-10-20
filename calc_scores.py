


import numpy as np
import torch.nn.functional as F
from split import load_mmap_shape
from utils import get_pkl
import config

# batch_x  = get_pkl("./data/draft/batch_x.pkl")
# batch_y  = get_pkl("./data/draft/batch_y.pkl")
# batch_y_pred  = get_pkl("./data/draft/batch_y_pred.pkl")

# batch_y = batch_y.to("cuda")
# batch_y_pred = batch_y_pred.to("cuda")

# query_tokens, candidate_tokens= batch_x

# print(f"labels: {batch_y}")
# print(f"batch_y_pred: {batch_y_pred}")
# print(f"batch_y: {batch_y}")

# def info_nce_loss(scores,targets, temperature=0.07):
#     scores = scores / temperature
#     return F.cross_entropy(scores, targets, ignore_index=-100)

# loss = info_nce_loss(batch_y_pred, batch_y)
# print(f"loss: {loss.item()}")


queries_shape =  load_mmap_shape(config.paths['queries']['meta'])
dict_shape =  load_mmap_shape(config.paths['dict']['meta'])

query_att = np.memmap(config.paths["queries"]['att'], mode="r", dtype=np.int32, shape=queries_shape)
dictionary_att = np.memmap(config.paths["dict"]['att'], mode="r", dtype=np.int32, shape=dict_shape)


# after loading query_att
print("Query attention stats:", np.mean(query_att), np.max(query_att), np.min(query_att))
# after loading dict_att
print("Dict attention stats:", np.mean(dictionary_att), np.max(dictionary_att), np.min(dictionary_att))

print(f"queries_shape: {queries_shape}")
print(f"dict_shape: {dict_shape}")

