import argparse
from multiprocessing import Pool, set_start_method
import os
import random
import numpy as np
from tqdm import tqdm
import json
from datasets import Dataset
from functools import partial
from transformers import AutoTokenizer
import re

from functools import partial

from config import paths
from config import GlobalConfig
from data import load_dictionary, load_queries
from utils import save_pkl

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenize_fn(batch, tokenizer, max_length):
    return tokenizer(
        batch["names"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )



def get_context(text, mention_start, mention_end, special_tokens_dict, window=5):
    """
    Steps:
    1. Extract a region around the mention, bounded by nearest ',' or '.' on each side.
    2. Recompute mention position within that region.
    3. Tokenize region and get token indices that cover the mention span.
    4. Keep up to `window` tokens before and after the mention span.
    5. Insert [mention_start] ... [mention_end] around the mention tokens.
    """

    # -------------------------
    # 1. Extract punctuation-bounded region
    # -------------------------
    # find closest punctuation (comma or period) to the left
    left_dot = text.rfind('.', 0, mention_start)
    left_com = text.rfind(',', 0, mention_start)
    left_boundary_raw = max(left_dot, left_com)
    if left_boundary_raw == -1:
        region_start_char = 0
    else:
        region_start_char = left_boundary_raw + 1  # start AFTER the punctuation

    # find closest punctuation (comma or period) to the right
    right_dot = text.find('.', mention_end)
    right_com = text.find(',', mention_end)

    right_candidates = [p for p in (right_dot, right_com) if p != -1]
    if right_candidates:
        region_end_char = min(right_candidates)
    else:
        region_end_char = len(text)

    region = text[region_start_char:region_end_char]

    # recompute mention offsets relative to this region
    rel_start = mention_start - region_start_char
    rel_end = mention_end - region_start_char


    # -------------------------
    # 2. Tokenize region and map char spans
    # -------------------------
    # We'll do whitespace-ish tokens and keep offsets
    tokens = re.findall(r'\S+', region)

    offsets = []
    cursor = 0
    for tok in tokens:
        # find this token from cursor forward so repeated words don't all map to the first spot
        start = region.find(tok, cursor)
        end = start + len(tok)
        offsets.append((start, end))
        cursor = end

    # -------------------------
    # 3. Identify which tokens cover the mention span
    # -------------------------
    # start_token_idx = first token whose span overlaps mention_start
    start_token_idx = 0
    for i, (s, e) in enumerate(offsets):
        if s <= rel_start < e:
            start_token_idx = i
            break

    # end_token_idx = last token whose span overlaps mention_end
    end_token_idx = len(tokens) - 1
    for i, (s, e) in enumerate(offsets):
        if s < rel_end <= e:
            end_token_idx = i
            break

    # -------------------------
    # 4. Clip to `window` tokens on each side
    # -------------------------
    # indices we keep:
    #   left slice: up to `window` tokens before start_token_idx
    #   mention slice: start_token_idx .. end_token_idx
    #   right slice: up to `window` tokens after end_token_idx

    left_keep_start = max(0, start_token_idx - window)
    left_tokens = tokens[left_keep_start:start_token_idx]

    mention_tokens = tokens[start_token_idx:end_token_idx + 1]

    right_keep_end = min(len(tokens), end_token_idx + 1 + window)
    right_tokens = tokens[end_token_idx + 1:right_keep_end]

    # -------------------------
    # 5. Assemble with markers
    # -------------------------
    final_tokens = (
        left_tokens
        + special_tokens_dict['mention_in_sentence_start']
        + mention_tokens
        + special_tokens_dict['mention_in_sentence_end']
        + right_tokens
    )

    return " ".join(final_tokens)


def parse_args():
    """
    Parse input arguments
    """
    cfg = GlobalConfig()
    parser = argparse.ArgumentParser(description='ranker train')

    parser.add_argument('--dictionary_path',  type=str)
    parser.add_argument('--queries_dir',  type=str)
    
    parser.add_argument('--dictionary_max_length',  type=int)
    parser.add_argument('--queries_max_length',  type=int)

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

    if args.queries_max_length:
        cfg.tokenize.queries_max_length = args.queries_max_length

    if args.dictionary_max_length:
        cfg.tokenize.dictionary_max_length = args.dictionary_max_length

    return cfg


def buid_full_query(q, window_words, special_tokens_dict):
    mention, cui, semantic_type, start, end, text = q
    ctx = get_context(text, int(start), int(end), special_tokens_dict=special_tokens_dict,  window=window_words)
    return (
        f" {special_tokens_dict['mention_name_start']} {mention} {special_tokens_dict['mention_name_end']} "
        f" {special_tokens_dict['context_start']} {ctx} {special_tokens_dict['context_end']} "
        f" {special_tokens_dict['type_start']} {semantic_type} {special_tokens_dict['type_end']} "
    )



def tokenize_queries(queries, tokenizer, queries_paths, cfg:GlobalConfig):


    window_words = cfg.tokenize.query_tokens_window_words_in_text
    max_length = cfg.tokenize.queries_max_length
    batch_size = cfg.tokenize.tokenize_batch_size
    special_tokens_dict = cfg.tokenize.special_tokens_dict
    

    print(f"Building full queries")
    full_queries = [buid_full_query(q, window_words, special_tokens_dict) for q in queries]
    print(f"We have: {len(full_queries)} queries..")
    print(f"First 5 queries: {full_queries[:5]}")

    queries_cuis = [q[1] for q in queries]
    queries_cuis = [q.replace("MESH:", "") for q in queries_cuis]
    np.save(queries_paths["ids"], queries_cuis)
    N = len(queries_cuis)


    input_ids_mmap = np.memmap(
        queries_paths['inp'],
        mode="w+",
        dtype=np.int32,
        shape=(N, max_length)
    )
    att_mask_mmap = np.memmap(
        queries_paths['att'],
        mode="w+",
        dtype=np.int32,
        shape=(N, max_length)
    )

    meta = {"shape": (N, max_length)}
    with open(queries_paths['meta'], "w") as f:
        json.dump(meta, f)


    for start in tqdm(range(0, N, batch_size), desc=f"Tokenizing"):
        end = min(start+batch_size, N)

        enc = tokenizer(
            full_queries[start:end],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        input_ids_mmap[start:end] = np.asarray(enc["input_ids"], np.int32)
        att_mask_mmap[start:end] = np.asarray(enc["attention_mask"], np.int32)
        del enc

    input_ids_mmap.flush()
    att_mask_mmap.flush()

    lengths = []

    for q in tqdm(full_queries, desc="Measuring token lengths"):
        encoded = tokenizer(
            q,
            add_special_tokens=True,   # includes [CLS] and [SEP]
            truncation=False           # we want true length, not truncated
        )
        lengths.append(len(encoded["input_ids"]))

    lengths = np.array(lengths)

    print(f"Total queries: {len(lengths)}")
    print(f"Mean length: {np.mean(lengths):.1f}")
    print(f"Median length: {np.median(lengths)}")
    print(f"95th percentile: {np.percentile(lengths, 95)}")
    print(f"99th percentile: {np.percentile(lengths, 99)}")
    print(f"Max length: {np.max(lengths)}")

    return True



def tokenize_dictionary(cuis, names, paths_key, tokenizer, cfg:GlobalConfig, semantics=None ):
    max_length = cfg.tokenize.dictionary_max_length
    batch_size = cfg.tokenize.tokenize_batch_size
    special_tokens_dict=  cfg.tokenize.special_tokens_dict


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
        batch_texts = names[start:end].tolist()
        batch_texts = [f"{special_tokens_dict['mention_name_start']} {n} {special_tokens_dict['mention_name_end']} " for n in batch_texts]
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
        
    # names is your np.array or list of dictionary names
    sample_size = min(200000, len(names))
    sample_indices = random.sample(range(len(names)), sample_size)
    sample_texts = [names[i] for i in sample_indices]

    print(f"Sampling {len(sample_texts)} out of {len(names)} dictionary entries")
    # tokenize in batches for speed
    batch_size = 16000
    lengths = []

    for start in tqdm(range(0, len(sample_texts), batch_size), desc="Measuring lengths"):
        end = min(start + batch_size, len(sample_texts))
        batch = [f"{special_tokens_dict['mention_name_start']} {t} {special_tokens_dict['mention_name_end']}" 
                for t in sample_texts[start:end]]
        enc = tokenizer(batch, add_special_tokens=True, truncation=False)
        lengths.extend([len(x) for x in enc["input_ids"]])

    lengths = np.array(lengths)
    print(f"Mean={np.mean(lengths):.1f}, Median={np.median(lengths)}, 95th={np.percentile(lengths,95)}, Max={np.max(lengths)}")
    return True



if __name__=="__main__":
    cfg = parse_args()
    set_start_method("spawn", force=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, use_fast=True)
    special_tokens = cfg.tokenize.special_tokens
    tokenizer.add_special_tokens(special_tokens)
    
    meta = {"len_tokenizer": len(tokenizer)}
    with open(cfg.paths.tokenizer_meta_path, "w") as f:
        json.dump(meta, f)



    if not cfg.tokenize.skip_tokenize_dictionary:
        print(f"Reading dictionary...")
        train_dictionary=load_dictionary(cfg.paths.dictionary_raw_path)
        dictionary_names = train_dictionary[:, 0]
        dictionary_cuis  = np.char.replace(train_dictionary[:, 1], "MESH:", "")
        tokenize_dictionary(cuis=dictionary_cuis,names= dictionary_names ,  paths_key="dict", tokenizer = tokenizer, cfg=cfg)


    if not cfg.tokenize.skip_tokenize_queries:
        print(f"Reading queries...")
        train_queries = load_queries(cfg.paths.queries_raw_dir)
        tokenize_queries(train_queries, tokenizer, paths["queries"], cfg)


