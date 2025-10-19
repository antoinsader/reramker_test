import numpy as np
from tqdm import tqdm
import config as confs
import os
import json
import logging
import argparse


def load_mmap_shape(fp):
    with open(fp) as f:
        meta = json.load(f)
    return tuple(meta["shape"])



LOGGER = logging.getLogger()



def parse_args():
    parser = argparse.ArgumentParser(description="splitting train file")
    #Required


    #optional
    parser.add_argument(
        '--train_percentage',
        type=float,
        default=0.8,
        help='Fraction of data used for training (0 < value <= 1)'
    )


    args = parser.parse_args()
    return args


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def main(args):
    print(f"args: {args}")
    assert args.train_percentage > 0.0 and args.train_percentage < 1.0, f'Train percentage should be between 0  and 1'
    print("Reading..")
    queries_input_ids_mmap_path = confs.paths["queries"]["inp"]
    queries_attention_mask_mmap_path = confs.paths["queries"]["att"]
    queries_cuis_path = confs.paths["queries"]["ids"]
    queries_meta = confs.paths["queries"]["meta"]


    test_queries_input_ids_mmap_path = confs.paths["test"]["inp"]
    test_queries_attention_mask_mmap_path = confs.paths["test"]["att"]
    test_queries_cuis_path = confs.paths["test"]["ids"]
    test_meta = confs.paths["test"]["meta"]


    query_inputs =   np.memmap(
                    queries_input_ids_mmap_path,
                        mode="r+",
                        dtype=np.int32,
                        shape=load_mmap_shape(queries_meta) 
                )

    query_attention =  np.memmap(
                    queries_attention_mask_mmap_path,
                        mode="r+",
                        dtype=np.int32,
                        shape=load_mmap_shape(queries_meta) 
        )

    query_cuis =  np.load(queries_cuis_path)

    print(f"queries train now: {len(query_cuis)} records")


    n = query_inputs.shape[0]
    split_idx = int(n * args.train_percentage)
    print(f"queries train should be after: {split_idx} records")


    train_inputs = query_inputs[:split_idx]
    train_attention = query_attention[:split_idx]
    train_cuis = query_cuis[:split_idx]

    test_inputs = query_inputs[split_idx:]
    test_attention = query_attention[split_idx:]
    test_cuis = query_cuis[split_idx:]


    test_inputs_mmap = np.memmap(
        test_queries_input_ids_mmap_path,
        mode="w+",
        dtype=np.int32,
        shape=test_inputs.shape
    )
    test_inputs_mmap[:] = test_inputs[:]
    test_inputs_mmap.flush()

    test_attention_mmap = np.memmap(
        test_queries_attention_mask_mmap_path,
        mode="w+",
        dtype=np.int32,
        shape=test_attention.shape
    )
    test_attention_mmap[:] = test_attention[:]
    test_attention_mmap.flush()


    meta = {"shape": test_attention_mmap.shape}
    with open( test_meta, "w") as f:
        json.dump(meta, f)

    np.save(os.path.join(test_queries_cuis_path), test_cuis)

    print(f"created test with len: {len(test_cuis)}")

    train_inputs_mmap = np.memmap(
        queries_input_ids_mmap_path,
        mode="w+",
        dtype=np.int32,
        shape=train_inputs.shape
    )
    train_inputs_mmap[:] = train_inputs[:]
    train_inputs_mmap.flush()

    train_attention_mmap = np.memmap(
        queries_attention_mask_mmap_path,
        mode="w+",
        dtype=np.int32,
        shape=train_attention.shape
    )
    train_attention_mmap[:] = train_attention[:]
    train_attention_mmap.flush()

    meta = {"shape": train_inputs.shape}
    with open(queries_meta, "w") as f:
        json.dump(meta, f)

    np.save(os.path.join(queries_cuis_path), train_cuis)


    with open(queries_meta) as f:
        print("Train shape:", json.load(f)["shape"])

    with open(test_meta) as f:
        print("Test shape:", json.load(f)["shape"])


    print("Train CUIs:", len(train_cuis))
    print("Test CUIs:", len(test_cuis))



if __name__=="__main__":
    init_logging()
    LOGGER.info("Tokenizing..")
    args= parse_args()
    main(args)


