import argparse
import os
import numpy as np
from tqdm import tqdm
import json
from datasets import Dataset
from functools import partial
from transformers import AutoTokenizer

from config import paths
from config import GlobalConfig
from data import load_dictionary, load_queries
from utils import save_pkl

def tokenize_fn(batch, tokenizer, max_length):
    return tokenizer(
        batch["names"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )



def parse_args():
    """
    Parse input arguments
    """
    cfg = GlobalConfig()
    parser = argparse.ArgumentParser(description='ranker train')

    parser.add_argument('--dictionary_path',  type=str)
    parser.add_argument('--queries_dir',  type=str)
    
    parser.add_argument('--max_length',  type=int)

    parser.add_argument('--skip_tokenizing_dictionary',  action="store_true")
    parser.add_argument('--skip_tokenizing_queries',  action="store_true")
    
    args = parser.parse_args()


    if args.skip_tokenizing_dictionary:
        cfg.tokenize.skip_tokenize_dictionary = True
    if args.skip_tokenizing_queries:
        cfg.tokenize.skip_tokenize_queries = True
    
    if args.dictionary_path:
        assert os.path.exists(args.dictionary_path), f'Dict path: {args.dictionary_path} not exists'
        cfg.paths.dictionary_raw_path = args.dictionary_path

    if args.queries_dir:
        assert os.path.isdir(args.queries_dir), f'Queries dir: {args.queries_dir} not exists'
        cfg.paths.queries_raw_dir = args.queries_dir

    if args.max_length:
        cfg.tokenize.max_length = args.max_length


    return cfg


def tt(cuis, names, paths_key, tokenizer, cfg, semantics=None ):
    max_length = cfg.tokenize.max_length
    batch_size = 4096

    print("Saving cuis..")
    np.save(paths[paths_key]['ids'] , cuis)
    names_size = len(names)
    
    if semantics:
        save_pkl(semantics, paths[paths_key]['semantics_pkl'])

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


    for start in tqdm(range(0, names_size, batch_size), desc=f"Tokenizing"):
        end = min(start+batch_size, names_size)
        batch_texts = names[start:end]
        enc = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        input_ids_mmap[start:end] = np.asarray(enc["input_ids"], np.int32)
        att_mask_mmap[start:end] = np.asarray(enc["attention_mask"], np.int32)
        del batch_texts, enc

    input_ids_mmap.flush()
    att_mask_mmap.flush()
    print("tokenized")
    return True



if __name__=="__main__":
    cfg = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    if not cfg.tokenize.skip_tokenize_dictionary:
        print(f"Reading dictionary...")
        train_dictionary=load_dictionary(cfg.paths.dictionary_raw_path)
        
        dictionary_names, dictionary_cuis = [row[0] for row in train_dictionary], [row[1] for row in train_dictionary]
        dictionary_cuis = [d.replace("MESH:", "") for d in dictionary_cuis]
        tt(cuis=dictionary_cuis,names= dictionary_names ,  paths_key="dict", tokenizer = tokenizer, cfg=cfg)



    if not cfg.tokenize.skip_tokenize_queries:
        print(f"Reading queries...")
        train_queries = load_queries(cfg.paths.queries_raw_dir)
        query_names, query_cuis, semantic_types = [row[0] for row in train_queries], [row[1] for row in train_queries], [row[2] for row in train_queries]
        query_cuis = [d.replace("MESH:", "") for d in query_cuis]
        tt(cuis=query_cuis,
           names=query_names, 
           semantics=semantic_types,  
           paths_key="queries", 
           tokenizer = tokenizer, 
           cfg=cfg)



    # test_queries = load_queries(test_queries_dir)
    # test_names, test_cuis = [row[0] for row in test_queries], [row[1] for row in test_queries]
    # test_cuis = [d.replace("MESH:", "") for d in test_cuis]
    # tt(test_cuis, test_names, 'test', tokenizer, cfg)





