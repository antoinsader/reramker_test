import numpy as np
from tqdm import tqdm
import json
from datasets import Dataset
from functools import partial
from transformers import AutoTokenizer

from config import paths, max_length, tokenizer_name, dictionary_path, queries_dir, test_queries_dir

from data import load_dictionary, load_queries

def tokenize_fn(batch, tokenizer, max_length):
    return tokenizer(
        batch["names"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )



def tt(cuis, names, paths_key, tokenizer ):
    batch_size = 1048
    np.save(paths[paths_key]['ids'] , cuis)
    names_size = len(names)
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

    names_dataset = Dataset.from_dict({"names": names})
    tokenized = names_dataset.map(
        partial(tokenize_fn, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        batch_size=batch_size
    )
    
    for start in tqdm(range(0, names_size, batch_size), desc="writing to memmap", unit="batch"):
        end = min(start + batch_size , names_size)
        tokenized_batch = tokenized[start: end]

        input_ids_mmap[start:end ] = np.array(tokenized_batch['input_ids'], dtype=np.int32)
        att_mask_mmap[start:end ] = np.array(tokenized_batch['attention_mask'], dtype=np.int32)
        del tokenized_batch

    input_ids_mmap.flush()
    att_mask_mmap.flush()
    print("tokenized")
    return True


tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

train_dictionary=load_dictionary(dictionary_path)
train_queries = load_queries(queries_dir)
test_queries = load_queries(test_queries_dir)


dictionary_names, dictionary_cuis = [row[0] for row in train_dictionary], [row[1] for row in train_dictionary]
query_names, query_cuis = [row[0] for row in train_queries], [row[1] for row in train_queries]
test_names, test_cuis = [row[0] for row in test_queries], [row[1] for row in test_queries]

dictionary_cuis = [d.replace("MESH:", "") for d in dictionary_cuis]
query_cuis = [d.replace("MESH:", "") for d in query_cuis]
test_cuis = [d.replace("MESH:", "") for d in test_cuis]


# tt(query_cuis, query_names, 'queries', tokenizer)
tt(dictionary_cuis, dictionary_names, 'dict', tokenizer)
# tt(test_cuis, test_names, 'test', tokenizer)
