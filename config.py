import argparse
from dataclasses import field
import os

from attr import dataclass


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





















@dataclass
class PathsConfig:
    tokens_dir : str = './data/tokens'
    logs_dir: str = "./logs"
    output_dir: str = "./output"
    chkpnts_dir: str = "./checkpoints"
    embeds_dir: str = "./data/embeds"



    def __post_init__(self):
        os.makedirs(self.tokens_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.embeds_dir, exist_ok=True)
        os.makedirs(self.chkpnts_dir, exist_ok=True)

@dataclass
class TokensConfig:
    max_length:int = 25
    raw_dictionary_path:str = f"{PathsConfig.raw_dir}/train_dictionary.txt"
    raw_queries_dir:str = os.path.join(PathsConfig.raw_dir ,  "traindev")
    raw_test_dir:str = None

    test_split_from_train: bool = True
    test_split_percentage: float = 0.8



@dataclass
class LoggerConfig:
    global_log_path:str = f"{PathsConfig.logs_dir}/logger_all.json"
    logs_dir:str= f"{PathsConfig.logs_dir}"
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
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 8
    topk: int = 15
    loss_type: str = "marginal_nll" # info_nce_loss
    optimizer_name: str = "AdamW" # Adam
    use_amp: bool = True
    loss_temperature: float = 0.2 # if small dict 0.07
    save_batch_output_pkl:bool = False
    save_checkpoints:bool = True
    load_last_checkpoint:bool = True

@dataclass
class FaissConfig:
    cluster_samples: int = 500_000
    build_batch_size: int = 4096
    search_batch_size: int = 4096
    index_name: str = "IndexHNSWFlat"


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


def parse_args(cfg:GlobalConfig):
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
        assert os.p
        cfg.eval_encoder_dir = args.encoder_to_eval
    if args.save_debug_pkls:
        cfg.train.save_batch_output_pkl = args.save_debug_pkls
    if args.skip_train:
        cfg.skip_train = args.skip_train
    if args.skip_eval:
        cfg.skip_eval = args.skip_eval
    if args.use_amp:
        cfg.train.use_amp = args.use_amp

    return cfg

