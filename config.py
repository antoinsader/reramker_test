import argparse
import os
num_workers = 4


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
        
    },
    "test": {
        "inp": os.path.join(tokens_dir, "_t_inp.mmap"),
        "att": os.path.join(tokens_dir, "_t_att.mmap") ,
        "ids":  os.path.join(tokens_dir, "_t_ids.npy") ,
        "meta":  os.path.join(tokens_dir, "_t_meta.json")
    }
}

os.makedirs("./data/dict_embs_cache", exist_ok=True)
last_dictionary_embeding_np_file = "./data/dict_embs_cache/d.npy"


max_length = 25
train_batch_size = 16
learning_rate = 0.0001

# tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer_name = 'dmis-lab/biobert-base-cased-v1.1'
encoder_model_name = 'dmis-lab/biobert-base-cased-v1.1'



dictionary_path = './data/raw/train_dictionary.txt'
queries_dir = './data/raw/traindev'
test_queries_dir = './data/raw/test'

os.makedirs('./data/embeds', exist_ok=True)
dict_embs_npy = './data/embeds/dict.npy'


logs_dir = "./logs"
global_log_path = "./logs/logger_all.json"
result_encoders_dir = "./output"


faiss_cluster_samples_num = 300_000
build_faiss_batch_size = 10000
search_faiss_batch_size = 10000


weight_decay=0.01
num_epochs = 10
# loss_type = 'info_nce_loss'
loss_type = 'marginal_nll'

normalize_query_forward = True
normalize_candidates_forward = True

normalize_query_faiss_search = True
normalize_dictionary_faiss_build = True
normalize_faiss_samples= True

loss_score_temperature = 0.2 # if small dict 0.07


topk = 15



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='ranker train')

    # Required
    parser.add_argument('--training_log_name', required=True,
                        help='Training log name')
    parser.add_argument('--faiss_index_name', type=str, required=True,
                        help='Either IndexHNSWFlat or IndexFlatIP')



    # optional
    parser.add_argument('--encoder_model_name',
                        help='Directory for pretrained model', default=encoder_model_name)

    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=train_batch_size, type=int)
    
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=weight_decay, type=float)
    parser.add_argument('--topk',  type=int, 
                        default=topk)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=learning_rate, type=float)
    
    parser.add_argument('--num_workers', default=num_workers, type=int)
    parser.add_argument('--build_faiss_batch_size',
                        help='Batch size for building faiss index',
                        default=build_faiss_batch_size, type=int)
    parser.add_argument('--faiss_cluster_samples_num',
                        help='When faiss will build clusters, how many samples',
                        default=faiss_cluster_samples_num, type=int)



    parser.add_argument('--num_epochs',
                        help='epochs to train',
                        default=num_epochs, type=int)

    parser.add_argument('--search_faiss_batch_size',
                        help='search_faiss_batch_size',
                        default=search_faiss_batch_size, type=int)
    parser.add_argument('--loss_type',
                        help='Either marginal_nll or info_nce_loss', default=loss_type)

    parser.add_argument('--save_debug_pkls',  action="store_true")
    parser.add_argument('--skip_train',  action="store_true")
    parser.add_argument('--skip_eval',  action="store_true")
    parser.add_argument('--encoder_to_eval',
                        help='Dir of the encoder to eval')


    args = parser.parse_args()
    return args
