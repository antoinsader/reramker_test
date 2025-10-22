

# use float16 as args 



import datetime
import gc, json, psutil, os, torch, time, faiss, logging
import math
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import faiss.contrib.torch_utils
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.retrieval import RetrievalMRR
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from collections import defaultdict


from transformers import AutoModel
from config import CheckPointModel, FaissConfig, GlobalConfig, LoggerConfig, ModelConfig, TrainingConfig, parse_args, paths
from utils import compute_metrics, get_labels, info_nce_loss, load_mmap_shape, marginal_nll, save_pkl

import config


# ====================
# MY LOGGER
# ====================

class MyLogger:
    def __init__(self, logger, log_path, cfg:GlobalConfig):
        self.log_path = log_path
        self.logger = logger
        self.use_cuda = torch.cuda.is_available()
        self.cfg = cfg
        self.process = psutil.Process(os.getpid())
        self.device = "cuda" if self.use_cuda else "cpu"

        self.cpu_memory_used = 0.0
        self.messages = []
        self.one_time_events_set = set()

        fmt = logging.Formatter('%(message)s')
        self.logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        self.logger.addHandler(console)
        file_handler = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)


    def current_cpu_mem_usage(self):
        rss = self.process.memory_info().rss / (1024 ** 2)
        self.cpu_memory_used = rss
        return rss


    def current_gpu_mem_usage(self):
        if self.use_cuda:
            free = torch.cuda.mem_get_info()[0] / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            return (free, total)
        return (0.0,0.0)


    def current_gpu_stats(self):
        """
            alloc (current allocated memory in MB): Memory currently allocated by tensors.
            alloc_peak (peak allocated memory in MB): Highest memory allocated by tensors since the program start or last reset.
            res (current reserved memory in MB): Memory reserved by the caching allocator (includes allocated plus cached blocks).
            res_peak (peak reserved memory in MB): Highest reserved memory since the program start or last reset.

        """
        if not self.use_cuda:
            return (None, None, None, None)
        torch.cuda.synchronize(self.device)
        alloc = torch.cuda.memory_allocated(self.device) / (1024**2)
        alloc_peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        res = torch.cuda.memory_reserved(self.device) / (1024**2)
        res_peak = torch.cuda.max_memory_reserved(self.device) / (1024**2)
        return (alloc, alloc_peak, res, res_peak)



    def log_event(self, event_tag, message=None, t0=None, log_immediate=True, first_iteration_only=False, log_memory=True, epoch=None):
        if first_iteration_only and event_tag in self.one_time_events_set:
            return True

        big_tag = self.cfg.logger.tag

        self.one_time_events_set.add(event_tag)

        msg = f"[{big_tag}-{event_tag}] "
        if epoch:
            msg += f"-epoch_{epoch}"
        if message: 
            msg += f" | {message}"

        if t0:
            elapsed = time.time() - t0
            msg += f" | elapsed time: {elapsed:.5f}seconds "


        if not log_memory:
            return self.logger.info(f"\n{msg}") if log_immediate else self.messages.append(f"\n{msg}")

        msg += f" \n\n | CPU Memory usage: {self.current_cpu_mem_usage():.1f}MB "
        if self.use_cuda:
            (free, total) = self.current_gpu_mem_usage()
            msg += f" | GPU memory total/free: {total:.1f}/{free:.1f}MB"
            (alloc, alloc_peak, res, res_peak) = self.current_gpu_stats()
            msg += f" | CUDA: allocated/peak: {alloc:.1f}/{alloc_peak:.1f}MB, reserved/peak {res:.1f}/{res_peak:.1f}MB"
        msg += f'\n\n'

        return self.logger.info(f"\n{msg}") if log_immediate else self.messages.append(f"\n{msg}")


# ====================
# MY ENCODER
# ====================
class MyEncoder():
    def __init__(self, use_cuda, cfg:ModelConfig):
        self.cfg = cfg
        self.use_cuda = use_cuda

        encoder = AutoModel.from_pretrained(cfg.model_name, use_safetensors=True)
        self.device = "cuda" if use_cuda else "cpu"
        self.encoder = encoder.to(self.device)
        cfg.hidden_size = self.encoder.config.hidden_size

    def get_emb(self, input_ids_tensor, attention_mask_tensor, use_amp=False, use_no_grad=False):
        context = torch.inference_mode() if use_no_grad else torch.enable_grad()
        with context, torch.amp.autocast(device_type="cuda", enabled=(self.use_cuda and use_amp)):
            # Hidden state, (batch, seq_len, hidden)
            emb = self.encoder(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)[0]

        # mean pooling
        mask = attention_mask_tensor.unsqueeze(-1).float()
        mean_emb = (emb * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        cls_emb = emb[:, 0]

        if self.cfg.pooling == 'mean':
            ret = mean_emb
        if self.cfg.pooling == 'cls':
            ret = cls_emb
        else:
            ret = 0.5 * (mean_emb + cls_emb)

        if self.cfg.normalize:
            ret = F.normalize(ret , p=2, dim=1)

        return ret


    def save_state(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.encoder.save_pretrained(dir)
        return True

    def load_state(self, state):
        self.encoder.load_state_dict(state)

    def get_state_dict(self):
        return self.encoder.state_dict()
# =======================
#       CHECKPOINTING
#========================
class CheckPointing:
    def __init__(self,  cfg:GlobalConfig):
        self.global_log_path = cfg.paths.global_log_path
        self.logs_dir =  cfg.paths.logs_dir
        self.log_path = None
        self.current_global_log_number = None
        self.current_entry, self.all_entries = self.get_last_global_obj()
        self.cfg = cfg


        assert self.current_global_log_number is not None
        assert self.log_path is not None
        assert self.current_entry["id"] == self.all_entries[-1]["id"]

        cfg.paths.set_result_encoder_dir(cfg.paths.output_dir + f"/encoder_{self.current_global_log_number}/")
        self.checkpoint_path = cfg.paths.checkpoint_path

        assert self.checkpoint_path is not None


    def get_last_global_obj(self):
        log_global_data = []
        if not os.path.isfile(self.global_log_path):
            with open(self.global_log_path, "w") as f:
                json.dump(log_global_data,f)
        os.makedirs(self.logs_dir, exist_ok=True)


        with open(self.global_log_path, "r") as f:
            log_global_data = json.load(f)

        log_obj = {}
        unfinished = [x for x in log_global_data if not x.get("finished", False)]
        if unfinished:
            LOGGER.info("Caught unfinihsed")
            current_entry = unfinished[-1]
            self.current_global_log_number = current_entry['id']
            self.log_path = current_entry['log_details_file']
            log_obj = current_entry
        else:
            self.current_global_log_number = log_global_data[-1]["id"] + 1 if len(log_global_data) > 0 else 1
            datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_path = self.logs_dir + f"/log_{self.current_global_log_number}_{datestr}.log"

            log_obj = {
                "id": self.current_global_log_number,
                "start_time": datestr,
                "finished": False,
                "log_details_file": self.log_path
            }
            log_global_data.append(log_obj)
            with open(self.global_log_path, "w") as f:
                json.dump(log_global_data, f, indent=2)
        return log_obj, log_global_data



    def log_finished(self, queries_len, dict_len, training_time_str, last_acc_5, last_mrr, last_faiss_recall, last_loss ):
        self.current_entry["finished"] = True
        self.current_entry["end_date"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.current_entry["training_log_name"] = self.cfg.logger.train_log_name
        self.current_entry["queries_size"] = queries_len
        self.current_entry["dictionary_size"] = dict_len
        self.current_entry["trained_period"] = training_time_str
        self.current_entry["log_details_file"] = self.log_path
        self.current_entry["epochs"] = self.cfg.train.num_epochs
        self.current_entry["last_faiss_recall@15"] = last_faiss_recall
        self.current_entry["last_avg_acc_5"] = last_acc_5
        self.current_entry["last_avg_mrr"] = last_mrr
        self.current_entry["last_avg_loss"] = last_loss
        self.current_entry["result_encoder_dir"] = self.cfg.paths.result_encoder_dir
        



        self.all_entries[-1] = self.current_entry
        with open(self.global_log_path, "w") as f:
            json.dump(self.all_entries, f, indent=2)
        return True

    def save_checkpoint(self, chkpt):
        torch.save(chkpt, self.checkpoint_path)
        return True

    def restore_checkpoint(self):
        chkpt = torch.load(self.checkpoint_path)
        return chkpt

# =======================
#       MY MODEL
#========================

class MyModel(nn.Module):
    def __init__(self, use_cuda, encoder : MyEncoder,  cfg:TrainingConfig ):
        super().__init__()

        self.use_cuda = use_cuda
        self.cfg = cfg
        self.encoder = encoder
        self.criterion = info_nce_loss if cfg.loss_type == 'info_nce_loss' else marginal_nll
        self.device = "cuda" if use_cuda else "cpu"

        assert cfg.optimizer_name == 'AdamW', f'Currently only AdamW available'

        self.optimizer = optim.AdamW(
            self.encoder.encoder.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            fused=self.use_cuda
        )


    def forward(self, query_tokens, candidates_tokens):

        if self.use_cuda:
            candidates_tokens["input_ids"] = candidates_tokens["input_ids"].to("cuda", non_blocking=True)
            candidates_tokens["attention_mask"] = candidates_tokens["attention_mask"].to("cuda", non_blocking=True)
            query_tokens["input_ids"] = query_tokens["input_ids"].to("cuda", non_blocking=True)
            query_tokens["attention_mask"] = query_tokens["attention_mask"].to("cuda", non_blocking=True)

        batch_size, topk, max_length = candidates_tokens["input_ids"].size()


        candidates_tokens["input_ids"] = candidates_tokens["input_ids"].view(batch_size * topk, max_length)
        candidates_tokens["attention_mask"] = candidates_tokens["attention_mask"].view(batch_size * topk, max_length)
        #(batch_size * topk , h)
        candidates_embs = self.encoder.get_emb(candidates_tokens["input_ids"], candidates_tokens["attention_mask"], use_amp=self.cfg.use_amp, use_no_grad=False)
        #(b, h)
        query_embeds = self.encoder.get_emb(query_tokens["input_ids"], query_tokens["attention_mask"], use_amp=self.cfg.use_amp, use_no_grad=False)



        candidates_embs = candidates_embs.view(batch_size, topk, -1)
        query_embeds = query_embeds.unsqueeze(1) # [batch_size, 1, hidden]

        score = torch.bmm(query_embeds, candidates_embs.transpose(1, 2)).squeeze(1) #b, topk
        del candidates_embs, query_embeds
        #score (batch_size, topk)
        return score
    
    def get_loss(self, outputs, targets):
        """
            outputs has shape (batch_size, topk)
            targets if marginal_nll then (batch_size, topk) if other (batch_size)
        """
        if self.use_cuda:
            targets = targets.cuda()
            outputs = outputs.cuda()

        outputs = outputs / self.cfg.loss_temperature
        loss = self.criterion(outputs, targets)
        return loss

# ======================
# MY TRAINER
# ======================
class Trainer:
    def __init__(self,  metric_logger: MyLogger, checkpointing: CheckPointing, cfg:GlobalConfig):
        self.cfg = cfg
        
        self.result_encoder_dir = cfg.paths.result_encoder_dir
        self.faiss_path  = cfg.paths.faiss_path

        self.tokens_paths = TokensPaths("dict", "queries")

        self.logger: MyLogger = metric_logger
        self.checkpointing : CheckPointing = checkpointing

        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda"    if self.use_cuda else "cpu"
        self.scaler = torch.amp.GradScaler(enabled=cfg.train.use_amp)
        self.encoder = MyEncoder(self.use_cuda, cfg.model)
        self.model = MyModel(self.use_cuda, self.encoder, cfg.train)
        self.faiss = MyFaiss(cfg, self.tokens_paths, self.encoder, self.faiss_path, self.use_cuda, self.device)
        self.dataset = MyDataset(self.tokens_paths, cfg)

        num_training_steps = len(self.dataset) // self.cfg.train.batch_size * self.cfg.train.num_epochs
        num_warmup_steps = int(0.05 * num_training_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.model.optimizer,
            num_warmup_steps= num_warmup_steps,
            num_training_steps=num_training_steps
        )

        self.topk = cfg.train.topk
        self.chkpoint_path = cfg.paths.checkpoint_path
        

    def train_one_batch(self, dl_item):
        self.model.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(self.use_cuda and self.cfg.train.use_amp)):
            batch_x, batch_y = dl_item
            batch_query_tokens, batch_candidates_tokens = batch_x
            batch_scores = self.model(batch_query_tokens, batch_candidates_tokens)
            loss = self.model.get_loss(batch_scores, batch_y)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.model.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.encoder.encoder.parameters(), max_norm=1.0)
        
        
        self.scaler.step(self.model.optimizer)
        self.scaler.update()
        self.scheduler.step()



        acc, mrr = compute_metrics(batch_scores.detach().cpu(), batch_y.cpu(), k=5)




        if self.cfg.train.save_batch_output_pkl:
            save_pkl(batch_x, f"{self.cfg.paths.draft_dir}/last_batch_x.pkl")
            save_pkl(batch_y, f"{self.cfg.paths.draft_dir}/last_batch_y.pkl")
            save_pkl(batch_scores, f"{self.cfg.paths.draft_dir}/last_batch_scores.pkl")

        del batch_x, batch_y, batch_scores
        return acc, mrr, loss.item()

    def train_one_epoch(self, epoch):
        torch.cuda.empty_cache()
        gc.collect()

        t0 = time.time()
        self.faiss.build_faiss(self.cfg.faiss.build_batch_size)
        self.logger.log_event(f"Faiss index built finished", t0=t0, epoch = epoch)

        t0 = time.time()
        candidates_idxs = self.faiss.search_faiss(self.cfg.faiss.search_batch_size) # (queries_N, topk)
        candidates_idxs = candidates_idxs.astype(np.int64)
        self.logger.log_event("Search in faiss finished ", t0=t0, epoch=epoch)

        self.dataset.set_candidates(candidates_idxs)
        recall_faiss = self.faiss.compute_faiss_recall_at_k(candidates_idxs, self.dataset.query_cuis, self.dataset.dict_cuis, k=self.topk)
        self.logger.log_event(f"Faiss recall@{self.topk}", message= f"{recall_faiss:.4f}", epoch=epoch, log_memory=False)

        my_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            pin_memory=self.use_cuda,
            num_workers=self.cfg.train.num_workers,
            persistent_workers=False
        )

        t0 = time.time()
        epoch_loss, epoch_acc, epoch_mrr = 0.0, 0.0, 0.0
        n_batches = 0

        for i, dl_item in tqdm(enumerate(my_loader), total=len(my_loader), desc=f"epoch@{epoch} - Training batches"):
            acc, mrr, loss = self.train_one_batch(dl_item)
            epoch_acc += acc
            epoch_mrr += mrr
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_mrr = epoch_mrr / max(1, n_batches)
        avg_acc = epoch_acc / max(1, n_batches)
        self.logger.log_event(f"Epoch train finished", message=f"avg_loss: {avg_loss:.5f}, avg acc@5: {avg_acc}, avg mrr: {avg_mrr} ", t0=t0, epoch=epoch)
        del my_loader
        torch.cuda.empty_cache()
        gc.collect()
        return avg_loss, avg_mrr, avg_acc, recall_faiss


    def restore_chkpoint(self):
        chkpt = self.checkpointing.restore_checkpoint()
        self.model.encoder.load_state(chkpt['model_state'])
        self.model.optimizer.load_state_dict(chkpt['optimizer_state'])
        self.scheduler.load_state_dict(chkpt['scheduler_state'])
        self.scaler.load_state_dict(chkpt['scaler_state'])
        self.faiss.load_faiss_index(chkpt['faiss_index_path'])
        return chkpt["epoch"] + 1

    def save_checkpoint(self,epoch):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.encoder.get_state_dict(),
            "optimizer_state": self.model.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "faiss_index_path": self.faiss.save_index(),
        }
        self.checkpointing.save_checkpoint(ckpt)
        return True

    def train(self):
        t0 = time.time()
        start_epoch = 1
        if self.cfg.train.load_last_checkpoint:
            if os.path.exists(self.chkpoint_path):
                self.logger.log_event("checkpoint resotred", f" restoring checkpoint from: {self.chkpoint_path}", log_memory=False)
                start_epoch = self.restore_chkpoint()

        avg_loss, avg_mrr, avg_acc, last_faiss_recall = 0.0, 0.0, 0.0, 0.0
        assert int(start_epoch) > 0 and int(start_epoch) < self.cfg.train.num_epochs 
        for epoch in range(start_epoch, self.cfg.train.num_epochs + 1):
            avg_loss, avg_mrr, avg_acc, last_faiss_recall = self.train_one_epoch(epoch)
            if self.cfg.train.save_checkpoints:
                self.save_checkpoint(epoch)


        training_time = time.time() - t0
        training_time_str = f"{int(training_time/60/60)}h, {int(training_time/60 % 60)}mins, {int(training_time % 60)}secs"
        self.checkpointing.log_finished(
            queries_len=len(self.dataset.query_cuis),
            dict_len=len(self.dataset.dict_cuis),
            training_time_str=training_time_str,
            last_acc_5 = avg_acc,
            last_mrr = avg_mrr,
            last_faiss_recall= last_faiss_recall,
            last_loss=avg_loss
        )

        self.encoder.save_state(self.result_encoder_dir)
        self.save_checkpoint(epoch='last')

        self.logger.log_event("Train finished", t0=t0, log_memory=False)
        self.logger.log_event("Main info: " , message=self.checkpointing.current_entry, log_memory=False)




class TokensPaths():
    def __init__(self,dict_paths_key, query_paths_key):
        self.dict_inp_path = paths[dict_paths_key]['inp']
        self.dict_att_path = paths[dict_paths_key]['att']
        self.dict_cuis_path = paths[dict_paths_key]['ids']
        self.dict_shape = load_mmap_shape(paths[dict_paths_key]['meta'])


        self.query_inp_path = paths[query_paths_key]['inp']
        self.query_att_path = paths[query_paths_key]['att']
        self.query_cuis_path = paths[query_paths_key]['ids']
        self.query_shape = load_mmap_shape(paths[query_paths_key]['meta'])


class MyFaiss():
    def __init__(self, cfg: GlobalConfig, tokens_paths:TokensPaths, encoder : MyEncoder, save_index_path, use_cuda, device):
        self.cfg = cfg.faiss
        self.use_cuda = use_cuda
        self.device = device
        self.tokens_paths = tokens_paths
        self.encoder = encoder
        self.save_index_path = save_index_path

        self.faiss_index_name =cfg.faiss.index_name
        self.faiss_cluster_samples_num = cfg.faiss.cluster_samples

        self.use_amp = cfg.train.use_amp
        self.topk = cfg.train.topk
        self.hidden_size = cfg.model.hidden_size

        self.faiss_index = None

    def load_faiss_index(self, path):
        assert os.path.exists(path),f'Path faiss {path} not exists'
        gpu_resources = faiss.StandardGpuResources()
        index = faiss.read_index(path)
        if self.use_cuda:
            co = faiss.GpuClonerOptions()
            co.allowCpuCoarseQuantizer = True
            index = faiss.index_cpu_to_gpu(gpu_resources, 0 , index, co)

        self.faiss_index = index

    def save_index(self):
        faiss.write_index(faiss.index_gpu_to_cpu(self.faiss_index), self.save_index_path)
        return self.save_index_path


    def train_sample(self, N, hidden_size):
        assert self.faiss_index is not None
        dictionary_inputs =   np.memmap(self.tokens_paths.dict_inp_path,
            mode="r",
            dtype=np.int32,
            shape=self.tokens_paths.dict_shape
        )
        dictionary_att = np.memmap( self.tokens_paths.dict_att_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.dict_shape
        )
        assert dictionary_att.shape[0] == N, f"Something is wrong! N={N}, dtionary att shape is: {dictionary_att.shape}"


        sample_size= self.faiss_cluster_samples_num
        sample_indices = torch.randperm(N)[:sample_size]
        samples_batch_size = 8_000
        samples_embeds = torch.empty((sample_size, hidden_size), dtype=torch.float32)


        cursor = 0
        for start in tqdm(range(0, len(sample_indices), samples_batch_size),  desc="embed sample and train clusters"):
            end = min(start+samples_batch_size, len(sample_indices))
            batch_idx = sample_indices[start:end]


            inp  = torch.as_tensor(dictionary_inputs[batch_idx], device=self.device)
            att = torch.as_tensor(dictionary_att[batch_idx],device=self.device)

            batch_embeds = self.encoder.get_emb(inp, att, use_amp=self.use_amp, use_no_grad=True)
            batch_embeds = batch_embeds.contiguous()
            samples_embeds[cursor : cursor+(end-start)] = batch_embeds
            cursor += (end -start)
            del batch_embeds, inp, att
        del dictionary_att, dictionary_inputs
        self.faiss_index.train(samples_embeds)
        del samples_embeds
        torch.cuda.empty_cache()
        gc.collect()
    
        

    def init_index(self, hidden_size, N):
        if self.faiss_index_name == 'IndexHNSWFlat':
            LOGGER.info(f"USING IndexHNSWFlat index")
            assert self.use_cuda, f'It is better to use_cuda when index is IndexHNSWFlat'
            assert N > 1_000_000, f"for {N}, it is better to use the flat index"

            gpu_resources = faiss.StandardGpuResources()
            LOGGER.info(f"FAISS INDEX are being built and trained")

            num_clusters = int(math.sqrt(N) * 2)

            num_bytes = 32 # num bytes per vector in PQ
            quantizer = faiss.IndexHNSWFlat(hidden_size, 32)
            quantizer.hnsw.efConstruction = 200
            quantizer.hnsw.efSearch = 256

            index = faiss.GpuIndexIVFPQ(gpu_resources, quantizer, hidden_size, num_clusters, num_bytes, 8)
            index.useFloat16LookupTables = self.use_amp
            self.faiss_index = index
            self.train_sample(N, hidden_size)
            LOGGER.info("Training clusters finsihed ")
            return True
        else:
            assert N <= 1_000_000, f"for {N}, it is better to use the IndexHNSWFlat  index"
            if self.use_cuda:
                gpu_resources = faiss.StandardGpuResources()
                #Index configurations
                index_conf = faiss.GpuIndexFlatConfig()
                index_conf.device = torch.cuda.current_device()
                index_conf.useFloat16 = bool(self.use_cuda)

                #make the index (this index is on gpu)
                self.faiss_index = faiss.GpuIndexFlatIP(gpu_resources, hidden_size, index_conf)
            else:
                #make normal cpu index 
                self.faiss_index = faiss.IndexFlatIP(hidden_size)
            return True


    def build_faiss(self, batch_size):
        N = self.tokens_paths.dict_shape[0]
        hidden_size = self.hidden_size

        if self.faiss_index is None:
            self.init_index(hidden_size, N)
        assert self.faiss_index is not None

        self.faiss_index.reset()

        dictionary_inputs = np.memmap(
                self.tokens_paths.dict_inp_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.dict_shape
            )
        dictionary_att = np.memmap(
                self.tokens_paths.dict_att_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.dict_shape
        )

        if self.faiss_index_name == 'IndexHNSWFlat':
            self.train_sample(N, hidden_size)



        for start in tqdm(range(0, N, batch_size), desc="Building faiss index"):
            end = min(start + batch_size, N)
            inp  = torch.as_tensor(dictionary_inputs[start:end], device=self.device)
            att = torch.as_tensor(dictionary_att[start:end],device=self.device)
            embs = self.encoder.get_emb(inp, att, use_amp=self.use_amp, use_no_grad=True)
            self.faiss_index.add(embs.contiguous())
            del inp, att, embs
        del dictionary_inputs, dictionary_att
        torch.cuda.empty_cache()
        gc.collect()

    def search_faiss(self, batch_size):


        (tokens_size, max_length ) = self.tokens_paths.query_shape
        N = tokens_size
        candidates = np.zeros((N,self.topk))
        faiss_index = self.faiss_index

        query_inputs = np.memmap(
                self.tokens_paths.query_inp_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.query_shape
        )
        query_att = np.memmap(
                  self.tokens_paths.query_att_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.query_shape
        )

        for start in range(0, N,batch_size):
            end = min(start + batch_size, N)
            inp  = torch.as_tensor(query_inputs[start:end], device=self.device)
            att = torch.as_tensor(query_att[start:end],device=self.device)
            embs = self.encoder.get_emb(inp, att, use_amp=self.use_amp, use_no_grad=True)
            if self.use_cuda:
                embs = embs.contiguous()
            else:
                embs = embs.cpu().numpy().astype(np.float32)

            _, chunk_cand_idxs = faiss_index.search(embs, self.topk)

            candidates[start:end] = chunk_cand_idxs.cpu().detach().numpy()
            del inp, att, embs
        del query_inputs, query_att
        gc.collect()
        return candidates


    def compute_faiss_recall_at_k(self,cands_idxs ,query_cuis, dict_cuis, k=10):
        assert cands_idxs is not None
        correct = 0
        num_queries = len(query_cuis)
        dict_cuis = np.array(dict_cuis)

        for i in range(num_queries):
            q_cui = query_cuis[i]
            retreived_cuis = dict_cuis[cands_idxs[i, :k] ]
            if q_cui in retreived_cuis:
                correct += 1
        return correct / max(num_queries, 1)



class MyDataset(Dataset):
    def __init__(self,tokens_paths: TokensPaths, cfg: GlobalConfig):
        self.tokens_paths  = tokens_paths
        self.topk = cfg.train.topk
        self.loss_type = cfg.train.loss_type
        self.all_candidates_idxs = None


        self.query_cuis  = np.load(self.tokens_paths.query_cuis_path)
        self.dict_cuis  = np.load(self.tokens_paths.dict_cuis_path)
        
        self.inject_hard_negatives = cfg.train.inject_hard_negatives
        self.hard_negatives_num = cfg.train.hard_negatives_num
        self.inject_hard_positives = cfg.train.inject_hard_positives
        self.hard_positives_num = cfg.train.hard_positives_num

        self.query_inputs = np.memmap(
                self.tokens_paths.query_inp_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.query_shape
            )

        self.query_att = np.memmap(
                self.tokens_paths.query_att_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.query_shape
            )

        self.dictionary_inputs = np.memmap(
                self.tokens_paths.dict_inp_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.dict_shape
            )
        self.dictionary_att = np.memmap(
                self.tokens_paths.dict_att_path,
                mode="r",
                dtype=np.int32,
                shape=self.tokens_paths.dict_shape
            )

        self.last_epoch_cands = None

        self.dictionary_cui_to_idx = defaultdict(list)
        for idx, cui in enumerate(self.dict_cuis):
            self.dictionary_cui_to_idx[cui].append(idx)

    def __len__(self,):
        return len(self.query_inputs)


    def change_candidates_pool(self):
        """
            all_candidates_idxs are current candidates (N queries, topk)
            last_epoch_cands are candidates from last_epoch (N queries, topk)
            return new_cands (N queries, topk)

        """
        assert self.all_candidates_idxs is not None, "Candidates are not set"

        inj_hard_negatives = (self.last_epoch_cands is not None) and self.inject_hard_negatives

        if self.last_epoch_cands is None:
            return self.all_candidates_idxs

        num_queries, topk = self.all_candidates_idxs.shape
        new_cands = self.all_candidates_idxs.clone()

        for q_idx in range(num_queries):
            q_cui = self.query_cuis[q_idx]
            cand_idxs = new_cands[q_idx].tolist()

            if self.inject_hard_positives:
                positive_indexes = self.dictionary_cui_to_idx.get(q_cui, [])
                if len(positive_indexes) > 0:
                    available_positives = list(set(positive_indexes) - set(cand_idxs))
                    if available_positives:
                        n_pos = min(self.hard_positives_num, len(available_positives))
                        chosen_pos = np.random.choice(available_positives, size=n_pos, replace=False)
                        new_cands[q_idx, -n_pos:] = torch.from_numpy(chosen_pos)

            if inj_hard_negatives:
                last_epoch_cands_idxs = self.last_epoch_cands[q_idx]
                prev_cuis = self.dict_cuis[last_epoch_cands_idxs]
                neg_mask = prev_cuis != q_cui
                hards_negs = last_epoch_cands_idxs[neg_mask]
                if len(hards_negs) > 0:
                    n_neg = min(self.hard_negatives_num, len(hards_negs))
                    chosen_negs = np.random.choice(hards_negs, size=n_neg, replace=False)
                    new_cands[q_idx, :n_neg] = torch.from_numpy(chosen_negs)

        return new_cands

    def set_candidates(self,cands):
        self.all_candidates_idxs = torch.as_tensor(cands, dtype=torch.long)
        new_cands = self.change_candidates_pool()
        self.last_epoch_cands = self.all_candidates_idxs.clone()
        self.all_candidates_idxs = new_cands




    def __getitem__(self, query_idx):
        assert self.all_candidates_idxs is not None
        query_tokens = {
            "input_ids": self.query_inputs[query_idx],
            "attention_mask": self.query_att[query_idx],
        }
        candidate_idxs = self.all_candidates_idxs[query_idx]
        assert len(candidate_idxs) == self.topk

        query_cui = self.query_cuis[query_idx] #1

        query_positive_idxs = self.dictionary_cui_to_idx.get(query_cui, [])
        query_positive_idxs= [q for q in query_positive_idxs if q != query_idx]
        assert len(query_positive_idxs) > 0, f"Query idx: {query_idx} with cui: {query_cui} does not have any positives idxs"



        intersection = list(set(query_positive_idxs) & set(candidate_idxs))
        if len(intersection) < 2:
            needed = 2 - len(intersection)
            available_to_add = list(set(query_positive_idxs) - set(candidate_idxs))
            if len(available_to_add) < needed:
                needed = len(available_to_add)
            if needed > 0:
                new_positives = np.random.choice(available_to_add, size=needed, replace=False)
                # Replace last few elements of candidate_idxs with new positives
                candidate_idxs[-needed:] = torch.from_numpy(new_positives)




        candidate_tokens = {
            "input_ids": self.dictionary_inputs[candidate_idxs],
            "attention_mask": self.dictionary_att[candidate_idxs],
        }

        query_candidates_cuis = np.array(self.dict_cuis)[candidate_idxs] #(batch_size, topk)
        # labels = (candidates_cuis == query_cui).astype(np.float32)
        labels = get_labels(query_candidates_cuis, query_cui, self.loss_type) #if error_type == 'info_nce_loss', will return [batch_size] for each item is the first match, for marginal_nll error_type will return  (batch_size, topk) for each item 0 if false, 1 for true

        return (query_tokens, candidate_tokens), labels



class Evaluater:
    def __init__(self, encoder_dir, faiss_path,    cfg:GlobalConfig):
        self.cfg = cfg
        
        self.encoder_dir = encoder_dir
        cfg.model.model_name = self.encoder_dir
        self.faiss_path = faiss_path

        cfg.paths.faiss_path = self.faiss_path

        self.tokens_paths = TokensPaths("dict", "test")
        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        self.dataset = MyDataset(self.tokens_paths, cfg )
        self.encoder = MyEncoder(self.use_cuda, cfg.model)
        self.faiss = MyFaiss(cfg, self.tokens_paths, self.encoder, self.faiss_path, self.use_cuda, self.device)
        self.model = MyModel(self.use_cuda, self.encoder, cfg.train)
        self.faiss.load_faiss_index(self.faiss_path)

        self.topk = cfg.train.topk



    def eval(self):
        self.model.eval()
        self.faiss.build_faiss(self.cfg.faiss.build_batch_size)
        cands_idxs = self.faiss.search_faiss(self.cfg.faiss.search_batch_size) # (queries_N, topk)
        cands_idxs = cands_idxs.astype(np.int64)
        self.dataset.set_candidates(cands_idxs)
        my_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            pin_memory=self.use_cuda,
            num_workers=self.cfg.train.num_workers,
            persistent_workers=False
        )
        
        total_loss = 0.0
        total_mrr = 0.0
        total_acc = 0.0
        total_samples = 0
        n_eval = 0

        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=self.use_cuda):
            for batch_x, batch_y in tqdm(my_loader, desc="Evaluating"):
                batch_y = batch_y.to(self.device)
                batch_size = batch_y.size(0)


                query_tokens, candidate_tokens = batch_x

                query_tokens = {k: v.to(self.device) for k, v in query_tokens.items()}
                candidate_tokens = {k: v.to(self.device) for k, v in candidate_tokens.items()}
                batch_x = (query_tokens, candidate_tokens)

                # Forward pass
                query_tokens, candidate_tokens = batch_x
                batch_pred = self.model(query_tokens, candidate_tokens)  # [batch_size, hidden_size]
                loss = self.model.get_loss(batch_pred, batch_y)

                acc_k, mrr = compute_metrics(batch_pred.detach().cpu(), batch_y.cpu(), k=self.topk)

                total_loss += loss.item() * batch_size
                total_mrr += mrr
                total_acc += acc_k
                total_samples += batch_size
                n_eval += 1
        avg_loss = total_loss / total_samples
        avg_mrr = total_mrr / n_eval
        avg_acc = total_acc / n_eval

        LOGGER.info(f"[Eval] Loss={avg_loss:.5f}, MRR={avg_mrr:.5f}, ACC@{self.topk}={avg_acc:.5f}")
        return avg_loss, avg_mrr, avg_acc




LOGGER = logging.getLogger()





def train(cfg: GlobalConfig):
    LOGGER.info("ARGS: ")
    LOGGER.info(cfg)
    
    chkpointing = CheckPointing(cfg)
    logger = MyLogger(LOGGER, chkpointing.log_path, cfg)

    trainer = Trainer(logger, chkpointing, cfg)
    trainer.train()
    
    torch.cuda.empty_cache()
    gc.collect()
    return cfg.paths.result_encoder_dir

def eval(cfg:GlobalConfig):
    eval_dir = cfg.eval_encoder_dir
    faiss_dir = cfg.eval_faiss_dir
    e = Evaluater(eval_dir, faiss_dir, cfg)
    e.eval()


if __name__ == "__main__":
    cfg :GlobalConfig = parse_args()
    

    if not cfg.skip_train:
        cfg.logger.tag = "train"
        result_encoder_dir = train(cfg)

    if not cfg.skip_eval:
        cfg.logger.tag = "eval"
        eval(cfg)




# python process.py --training_log_name='small_dictionary_flat_faiss' --faiss_index_name='IndexFlatIP' --num_workers=48 --loss_type='info_nce_loss'



# python process.py --training_log_name='big_dictionary' --faiss_index_name='IndexHNSWFlat' --num_workers=16

# eval:
# python process.py --training_log_name='big_dictionary' --faiss_index_name='IndexHNSWFlat' --num_workers=16 --skip_train --encoder_to_eval='./output/encoder_1'  --eval_faiss_path='./output/encoder_1/faiss_index.faiss'





