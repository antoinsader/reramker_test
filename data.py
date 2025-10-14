import numpy as np
from tqdm import tqdm
import glob
import os


def load_dictionary(dictionary_path):
    name_cui_map = {}
    data = []
    with open(dictionary_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if line == "": continue
            cui, name = line.split("||")
            data.append((name,cui))
    data = np.array(data)
    return data

def load_queries(data_dir, filter_composite=True, filter_duplicate=True, filter_cuiless=True):
    data = []
    concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
    for concept_file in tqdm(concept_files):
        with open(concept_file, "r", encoding='utf-8') as f:
            concepts = f.readlines()

        for concept in concepts:
            concept = concept.split("||")
            mention = concept[3].strip()
            cui = concept[4].strip()
            is_composite = (cui.replace("+","|").count("|") > 0)

            # filter composite cui
            if filter_composite and is_composite:
                continue
            # filter cuiless
            if filter_cuiless and cui == '-1':
                continue

            data.append((mention,cui))
    if filter_duplicate:
        data = list(dict.fromkeys(data))
    data = np.array(data)
    return data



