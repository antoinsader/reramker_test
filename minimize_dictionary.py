import json
from config import GlobalConfig
from process import MyDataset, TokensPaths
from tqdm import tqdm
import numpy as np
from config import paths


cfg = GlobalConfig()
tokens_paths = TokensPaths("dict", "queries")
dataset = MyDataset(tokens_paths, cfg)
dict_cuis = np.array(dataset.dict_cuis)
N_dict = len(dict_cuis)
queries_cuis = np.array(dataset.query_cuis)
unique_query_cuis = np.unique(queries_cuis)
target_size = 1_000_000

print(f"Dictionary size before: {N_dict:,}")
print(f"Unique query CUIs: {len(unique_query_cuis):,}")

# ----------------------------
# Step 1: find dictionary indices for query CUIs
# ----------------------------
cui_to_idx = {}
for idx, cui in enumerate(dict_cuis):
    if cui not in cui_to_idx:
        cui_to_idx[cui] = []
    cui_to_idx[cui].append(idx)

must_keep_idxs = []
for q_cui in unique_query_cuis:
    if q_cui in cui_to_idx:
        must_keep_idxs.extend(cui_to_idx[q_cui])

must_keep_idxs = np.unique(must_keep_idxs)
print(f"Must-keep entries (covering all query CUIs): {len(must_keep_idxs):,}")

# ----------------------------
# Step 2: fill up with random extras until we reach target_size
# ----------------------------
if len(must_keep_idxs) >= target_size:
    selected_idxs = np.random.choice(must_keep_idxs, size=target_size, replace=False)
else:
    remaining = list(set(range(N_dict)) - set(must_keep_idxs))
    needed = target_size - len(must_keep_idxs)
    random_fill = np.random.choice(remaining, size=needed, replace=False)
    selected_idxs = np.concatenate([must_keep_idxs, random_fill])

print(f"Final dictionary size: {len(selected_idxs):,}")

# ----------------------------
# Step 3: subset dictionary arrays
# ----------------------------
dictionary_inputs_small = dataset.dictionary_inputs[selected_idxs]
dictionary_att_small = dataset.dictionary_att[selected_idxs]
dict_cuis_small = dict_cuis[selected_idxs]

# optional sanity check
coverage = np.isin(unique_query_cuis, dict_cuis_small).mean()
print(f"CUI coverage after sampling: {coverage*100:.2f}%")

paths_key = "small_dict"
np.save(paths[paths_key]['ids'] , dict_cuis_small)
names_size = len(dict_cuis_small)
max_length = tokens_paths.dict_shape[1]
print(f"Creating memmap...")
input_ids_mmap = np.memmap(
    paths[paths_key]['inp'],
    mode="w+",
    dtype=np.int32,
    shape=(names_size, max_length)
)
att_mask_mmap = np.memmap(
    paths[paths_key]['att'],
    mode="w+",
    dtype=np.int32,
    shape=(names_size, max_length)
)

meta = {"shape": (names_size, max_length)}
with open(paths[paths_key]["meta"], "w") as f:
    json.dump(meta, f)
batch_size = 64_000
for start in tqdm(range(0, names_size, batch_size), desc=f"Writing small dict"):
    end = min(batch_size+start , names_size)
    input_ids_mmap[start:end] = dictionary_inputs_small[start:end]
    att_mask_mmap[start:end] = dictionary_att_small[start:end]
    

input_ids_mmap.flush()
att_mask_mmap.flush()

print("Finished")