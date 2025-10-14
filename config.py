import os


tokens_dir = "./data/tokens"
os.makedirs(tokens_dir, exist_ok=True)

paths = {
    "queries": {
        "inp": os.path.join(tokens_dir, "_q_inp.mmap"),
        "att": os.path.join(tokens_dir, "_q_att.mmap") ,
        "ids":  os.path.join(tokens_dir, "_q_ids.npy") ,
        "meta":  os.path.join(tokens_dir, "_q_meta.json")
    },
    "dict": {
        "inp": os.path.join(tokens_dir, "_d_inp.mmap"),
        "att": os.path.join(tokens_dir, "_d_att.mmap") ,
        "ids":  os.path.join(tokens_dir, "_d_ids.npy") ,
        "meta":  os.path.join(tokens_dir, "_d_meta.json")
        
    }
}

os.makedirs("./data/dict_embs_cache", exist_ok=True)
last_dictionary_embeding_np_file = "./data/dict_embs_cache/d.npy"


max_length = 25
num_epochs = 5


cands_num = 5
train_batch_size = 32
build_faiss_batch_size = 10000
search_faiss_batch_size = 10000

learning_rate = 0.0001
weight_decay=0.01
num_workers = 4

# tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer_name = 'dmis-lab/biobert-base-cased-v1.1'
encoder_model_name = 'dmis-lab/biobert-base-cased-v1.1'



dictionary_path = './data/raw/train_dictionary.txt'
queries_dir = './data/raw/traindev'


os.makedirs('./data/embeds', exist_ok=True)
dict_embs_npy = './data/embeds/dict.npy'


loss_type = 'info_nce_loss'
# loss_type = 'marginal_nll'
