import argparse
from dataclasses import dataclass, field
import math
import os



tokens_dir = "./data/tokens"
os.makedirs(tokens_dir, exist_ok=True)

paths = {
    "queries": {
        "inp": os.path.join(tokens_dir, "_q_inp.mmap"),
        "att": os.path.join(tokens_dir, "_q_att.mmap") ,
        "ids":  os.path.join(tokens_dir, "_q_ids.npy") ,
        "meta":  os.path.join(tokens_dir, "_q_meta.json"),
        "semantics_pkl":  os.path.join(tokens_dir, "_q_semantics.pkl"),
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





















@dataclass
class PathsConfig:
    tokens_dir : str = './data/tokens'
    logs_dir: str = "./logs"
    output_dir: str = "./output"
    chkpnts_dir: str = "./checkpoints"
    embeds_dir: str = "./data/embeds"
    raw_dir: str = "./data/raw"
    draft_dir: str = "./data/draft"

    global_log_path: str = f"./logs/logger_all.json"
    
    dictionary_raw_path = "./data/raw/train_dictionary.txt"
    queries_raw_dir = "./data/raw/traindev"
    tokenizer_meta_path = "./data/tokenizer.json"

    result_encoder_dir = None
    checkpoint_dir = None
    checkpoint_path = None
    faiss_path = None

    def __post_init__(self):
        assert os.path.isdir(self.raw_dir)
        os.makedirs(self.tokens_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.embeds_dir, exist_ok=True)
        os.makedirs(self.chkpnts_dir, exist_ok=True)
        os.makedirs(self.draft_dir, exist_ok=True)

    def set_result_encoder_dir(self, dir):
        self.result_encoder_dir = dir
        self.checkpoint_dir = os.path.join(dir, "checkpoints")
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "last.pt")
        self.faiss_path  = self.result_encoder_dir + "/faiss_index.faiss"
        os.makedirs(self.result_encoder_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


@dataclass
class TokensConfig:
    dictionary_max_length = 64
    queries_max_length = 80
    tokenize_batch_size : int = 32000
    raw_test_dir:str = None
    test_split_from_train: bool = True
    test_split_percentage: float = 0.8

    skip_tokenize_dictionary: bool = False
    skip_tokenize_queries: bool = False


    query_tokens_window_words_in_text = 5 #5 words before mention start, 5 words after mention start

    special_tokens = {
        'additional_special_tokens': [
            '[MENTION_CONTEXT_START]', '[MENTION_CONTEXT_END]',  # existing
            '[MENTION_NAME_START]', '[MENTION_NAME_END]',
            '[CONTEXT_START]', '[CONTEXT_END]',
            '[TYPE_START]', '[TYPE_END]'
        ]
    }
    special_tokens_dict = {
        "mention_name_start": "[MENTION_NAME_START]",
        "mention_name_end": "[MENTION_NAME_END]",
        "mention_in_sentence_start": "[MENTION_CONTEXT_START]",
        "mention_in_sentence_end": "[MENTION_CONTEXT_END]",
        "context_start": "[CONTEXT_START]",
        "context_end": "[CONTEXT_END]",
        "type_start": "[TYPE_START]",
        "type_end": "[TYPE_END]",
        
    }



@dataclass
class LoggerConfig:
    tag:str="train"
    train_log_name: str = ""



@dataclass
class ModelConfig:
    model_name : str = 'dmis-lab/biobert-base-cased-v1.1'
    pooling : str =  'hybrid' #[mean, cls, hybrid]
    normalize: bool = True
    hidden_size: int = 768


@dataclass
class TrainingConfig:
    num_epochs: int = 13
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.001
    num_workers: int = 8
    topk: int = 20
    loss_type: str = "marginal_nll" # info_nce_loss
    optimizer_name: str = "AdamW" # Adam
    use_amp: bool = True
    loss_temperature: float = 0.2 # if small dict 0.07
    save_batch_output_pkl:bool = False
    save_checkpoints:bool = True
    load_last_checkpoint:bool = True

    inject_hard_negatives:bool= True
    hard_negatives_num:int= 7
    inject_hard_positives:bool= True
    hard_positives_num:int= 2

    freeze_lower_layer_epoch_max:int=2

@dataclass
class FaissConfig:
    cluster_samples: int = 1_000_000
    build_batch_size: int = 4096
    search_batch_size: int = 4096
    index_name: str = "IndexHNSWFlat"
    num_quantizers = 32
    nbits= 8
    hnsw_efConstruction = 200
    hnsw_efSearch = 256
    nrprobe = 32

    def num_clusters(self, dictionary_size):
        return int(math.sqrt(dictionary_size) * 2)




@dataclass
class GlobalConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    tokenize: TokensConfig = field(default_factory=TokensConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    faiss: FaissConfig = field(default_factory=FaissConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)

    skip_eval: bool = False
    skip_train: bool = False
    eval_encoder_dir:str = ""
    eval_faiss_dir:str = ""

    def to_dict(self):
        return {
            "model": vars(self.model),
            "train": vars(self.train),
            "faiss": vars(self.faiss),
            "logger": vars(self.logger),
            "paths": vars(self.paths)
        }
class CheckPointModel:
    def __init__(self, chkpt):
        self.model_state= chkpt['model_state']
        self.optimizer_state = chkpt['optimizer_state']
        self.scheduler_state = chkpt['scheduler_state']
        self.scaler_state = chkpt['scaler_state']
        self.epoch = chkpt['epoch']
        self.faiss_index_path = chkpt['faiss_index_path']

class LogDataModel:
    def __init__(self, row):
        self.training_log_name= row['training_log_name']


def parse_args():
    """
    Parse input arguments
    """
    cfg = GlobalConfig()
    parser = argparse.ArgumentParser(description='ranker train')

    # Required
    parser.add_argument('--training_log_name', required=True,
                        help='Unique name for the training session')

    parser.add_argument('--faiss_index_name', type=str, required=True,
                        help='Either IndexHNSWFlat or IndexFlatIP')



    # optional
    parser.add_argument('--encoder_model_name',
                        help='Directory for pretrained model', required=False)
    
    parser.add_argument('--num_workers', help='Num workers ', type=int, required=False)

    parser.add_argument('--num_epochs', help='train num epochs', type=int, required=False)
    parser.add_argument('--train_batch_size', help='train batch size', type=int, required=False)
    parser.add_argument('--topk', help='train topk candidates', type=int, required=False)


    parser.add_argument('--learning_rate', help='train learning rate', type=float, required=False)
    parser.add_argument('--weight_decay', help='train weight decay', type=float, required=False)
    parser.add_argument('--loss_type', help='Either marginal_nll or info_nce_loss', type=str, required=False)


    parser.add_argument('--build_faiss_batch_size', help='Batch size when building faiss index ', type=int, required=False)
    parser.add_argument('--search_faiss_batch_size', help='Batch size when searching in faiss ', type=int, required=False)
    parser.add_argument('--faiss_clustering_samples_size', help='Num of random samples to create faiss clusters', type=int, required=False)

    parser.add_argument('--encoder_to_eval', help='Dir of the encoder to eval', type=str)
    parser.add_argument('--eval_faiss_path', help='path to eval faiss', type=str)


    parser.add_argument('--save_debug_pkls',  action="store_true")
    parser.add_argument('--skip_train',  action="store_true")
    parser.add_argument('--skip_eval',  action="store_true")

    parser.add_argument('--use_amp',  action="store_true")

    args = parser.parse_args()


    if args.training_log_name:
        cfg.logger.train_log_name = args.training_log_name
    if args.faiss_index_name:
        cfg.faiss.index_name = args.faiss_index_name
    if args.encoder_model_name:
        cfg.model.model_name = args.encoder_model_name
    if args.num_workers:
        cfg.train.num_workers = args.num_workers
    if args.num_epochs:
        cfg.train.num_epochs = args.num_epochs
    if args.train_batch_size:
        cfg.train.batch_size = args.train_batch_size
    if args.topk:
        cfg.train.topk = args.topk

    if args.learning_rate:
        cfg.train.learning_rate = args.learning_rate
    if args.weight_decay:
        cfg.train.weight_decay = args.weight_decay
    if args.loss_type:
        assert args.loss_type in ['marginal_nll','info_nce_loss']
        cfg.train.loss_type = args.loss_type
    if args.build_faiss_batch_size:
        cfg.faiss.build_batch_size = args.build_faiss_batch_size
    if args.search_faiss_batch_size:
        cfg.faiss.search_batch_size = args.search_faiss_batch_size
    if args.faiss_clustering_samples_size:
        cfg.faiss.cluster_samples = args.faiss_clustering_samples_size
    if args.encoder_to_eval:
        assert os.path.isdir(args.encoder_to_eval)
        cfg.eval_encoder_dir = args.encoder_to_eval
    if args.eval_faiss_path:
        assert os.path.exists(args.eval_faiss_path)
        cfg.eval_faiss_dir = args.eval_faiss_path

    if args.save_debug_pkls:
        cfg.train.save_batch_output_pkl = args.save_debug_pkls
    if args.skip_train:
        cfg.skip_train = args.skip_train
    if args.skip_eval:
        cfg.skip_eval = args.skip_eval
    if args.use_amp:
        cfg.train.use_amp = args.use_amp

    return cfg

