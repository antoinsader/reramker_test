from collections import defaultdict
import random
import numpy as np
import json
from tqdm import tqdm
import faiss
import faiss.contrib.torch_utils
import os
import math
import torch

from data import load_dictionary, load_queries
from config import GlobalConfig, paths
from process import MyDataset, MyEncoder, TokensPaths
from utils import get_pkl, save_pkl



use_samples_cache = False
use_build_cache= False
use_faiss_cache= False


cfg = GlobalConfig()


use_cuda = True
device = "cuda"


tokens_paths = TokensPaths("dict", "queries")
dataset = MyDataset(tokens_paths, cfg)
cfg.model.model_name = "./output/fine_tuned_embeddings/"
encoder = MyEncoder(use_cuda, cfg)

faiss_index_name =cfg.faiss.index_name
faiss_cluster_samples_num = cfg.faiss.cluster_samples
use_amp = cfg.train.use_amp
topk = cfg.train.topk
hidden_size = cfg.model.hidden_size

N = tokens_paths.dict_shape[0]
dictionary_entries_n  = N

print("INIT SAMPLES....")


sample_size= faiss_cluster_samples_num
dictionary_inputs =   dataset.dictionary_inputs
dictionary_att = dataset.dictionary_att
queries_cuis = dataset.query_cuis
#train samples
sample_size= faiss_cluster_samples_num


torch.cuda.empty_cache()

if sample_size > N:
    sample_size = N
sample_indices = torch.randperm(N)[:sample_size] 

samples_batch_size = 8_000
samples_embeds = torch.empty((sample_size, hidden_size), dtype=torch.float32)
cursor = 0

samples_dir = "./draft"
os.makedirs(samples_dir, exist_ok=True)
samples_path = samples_dir + "/samples.pkl"
dict_embeds_path  = samples_dir + "/dicts_embs.pkl"
faiss_path = samples_dir + "/faiss_index.faiss"
if os.path.exists(samples_path) and use_samples_cache:
    samples_embeds = get_pkl(samples_path)

else:
    for start in tqdm(range(0, len(sample_indices), samples_batch_size),  desc="embed samples"):
        end = min(start+samples_batch_size, len(sample_indices))
        batch_idx = sample_indices[start:end]


        inp  = torch.as_tensor(dictionary_inputs[batch_idx], device=device)
        att = torch.as_tensor(dictionary_att[batch_idx],device=device)

        batch_embeds = encoder.get_emb(inp, att, use_amp=use_amp, use_no_grad=True)
        samples_embeds[cursor : cursor+(end-start)] = batch_embeds
        cursor += (end -start)
        del batch_embeds, inp, att

    save_pkl(samples_embeds, samples_path)


torch.cuda.empty_cache()

print("INIT INDEX")

if os.path.exists(faiss_path ) and use_faiss_cache:
    gpu_resources = faiss.StandardGpuResources()
    index = faiss.read_index(faiss_path)
    co = faiss.GpuClonerOptions()
    co.allowCpuCoarseQuantizer = True
    index = faiss.index_cpu_to_gpu(gpu_resources, 0 , index, co)

    faiss_index = index


else:
    num_threads = min(32, os.cpu_count() or 8)
    faiss.omp_set_num_threads(num_threads)


    gpu_resources = faiss.StandardGpuResources()
    num_clusters = cfg.faiss.num_clusters(N) # if N is 4m then around 4000
    num_quantizers = cfg.faiss.num_quantizers
    nbits= cfg.faiss.nbits # bits per sub quantizer
    quantizer = faiss.IndexHNSWFlat(hidden_size, 32)
    quantizer.hnsw.efConstruction = cfg.faiss.hnsw_efConstruction
    quantizer.hnsw.efSearch = cfg.faiss.hnsw_efSearch
    index = faiss.GpuIndexIVFPQ(gpu_resources, quantizer, hidden_size, num_clusters, num_quantizers, nbits)
    index.useFloat16LookupTables = use_amp
    index.nprobe = cfg.faiss.nrprobe
    faiss_index = index
    print("INIT INDEX FINISHED")

    print(f"Training on samples...")
    faiss_index.train(samples_embeds)
    del samples_embeds
    print(f"Training on samples finsihed")
    batch_size = cfg.faiss.build_batch_size
    torch.cuda.empty_cache()
    faiss.write_index(faiss.index_gpu_to_cpu(faiss_index), faiss_path)


print(f"Building with dictionary embeds")


if os.path.exists(dict_embeds_path) and use_build_cache:
    embs_all = get_pkl(dict_embeds_path)
    for start in tqdm(range(0, N, batch_size), desc="Building faiss index"):
        end = min(start + batch_size, N)
        embs_batch = torch.as_tensor(embs_all[start:end], device=device, dtype=torch.float32).contiguous()
        faiss_index.add(embs_batch)
else:
    embs_all = torch.empty((N, hidden_size), dtype=torch.float32)
    for start in tqdm(range(0, N, batch_size), desc="Building faiss index"):
        end = min(start + batch_size, N)
        inp  = torch.as_tensor(dictionary_inputs[start:end], device=device)
        att = torch.as_tensor(dictionary_att[start:end],device=device)
        embs = encoder.get_emb(inp, att, use_amp=use_amp, use_no_grad=True)
        faiss_index.add(embs.contiguous())
        embs_all[start:end] = embs.detach().cpu()
        del inp, att, embs
        torch.cuda.empty_cache()

    save_pkl(embs_all, dict_embeds_path)

torch.cuda.empty_cache()
del dictionary_inputs, dictionary_att
print(f"Building with dictionary embeds FINISHED")


print(f"Searching....")
(tokens_size, max_length ) = tokens_paths.query_shape
N = tokens_size
candidates = np.zeros((N,topk))

query_inputs = dataset.query_inputs
query_att = dataset.query_att
candidates = np.zeros((N, topk), dtype=np.int64)
for start in range(0, N,batch_size):
    end = min(start + batch_size, N)
    inp  = torch.as_tensor(query_inputs[start:end], device=device)
    att = torch.as_tensor(query_att[start:end],device=device)
    embs = encoder.get_emb(inp, att, use_amp=use_amp, use_no_grad=True)
    embs = embs.contiguous()
    _, chunk_cand_idxs = faiss_index.search(embs, topk)
    candidates[start:end] = chunk_cand_idxs.cpu().numpy()
    del inp, att, embs
del query_inputs, query_att

print(f"Searching FINISHED")
torch.cuda.empty_cache()


print(f"Calculating recall....")

correct = 0
query_cuis = dataset.query_cuis
dict_cuis = dataset.dict_cuis
num_queries = len(query_cuis)
dict_cuis = np.array(dict_cuis)

ks = [1,2,4,5,8,10,12,15,17,20]
correct = defaultdict(int)
for i in range(len(query_cuis)):
    q_cui = query_cuis[i]
    # print(f"tt : {dictionary_cuis[candidates[i, :3]]}")
    candidates_cuis = dict_cuis[candidates[i, :max(ks)] ]
    for k in ks:
        if q_cui in candidates_cuis[:k]:
            correct[k] += 1

recalls = {k: correct[k] / max(len(queries_cuis), 1) for k in ks}
print(f"Calculating recall FINISHED")
print(f"recalls: {recalls} ")

