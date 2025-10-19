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


from transformers import AutoModel
from config import parse_args, paths, dict_embs_npy
from config import normalize_query_faiss_search, normalize_query_forward, normalize_candidates_forward
from utils import compute_metrics, get_labels, info_nce_loss, load_mmap_shape, marginal_nll

import config



class MyLogger:
    def __init__(self, logger, use_cuda, global_log_path, logs_dir, tag="train"):
        self.use_cuda = use_cuda
        self.global_log_path = global_log_path
        self.logs_dir=  logs_dir

        self.device = "cuda" if self.use_cuda else "cpu"
        self.logger = logger
        self.tag = tag
        self.process = psutil.Process(os.getpid())

        self.cpu_memory_used = 0.0
        self.messages = []
        self.one_time_events_set = set()

        #log_path is where the log of the current training
        # log_global_data is the array holding the all log and will be used to write the last log
        # current_global_log_number is the number used in the current log
        self.log_path,  self.log_global_data, self.current_global_log_number = self._init_logging()


    def _init_logging(self):

        log_global_data = []

        if not os.path.isfile(self.global_log_path):
            with open(self.global_log_path, "w") as f:
                json.dump(log_global_data,f)

        with open(self.global_log_path, "r") as f:
            log_global_data = json.load(f)


        last_log_number  = log_global_data[-1]["id"] if len(log_global_data) > 0  else 0
        current_global_log_number = last_log_number + 1 
        log_global_data.append({"id": current_global_log_number})

        with open(self.global_log_path, "w") as f:
            json.dump(log_global_data, f)


        os.makedirs(self.logs_dir, exist_ok=True)
        datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        log_path = self.logs_dir + f"/log_{current_global_log_number}_{datestr}.log"
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(message)s')

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        self.logger.addHandler(console)
        
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)


        return log_path,  log_global_data,current_global_log_number


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



    def log_event(self, event_tag, t0=None, log_immediate=True, first_iteration_only=False, only_elapsed_time=False, epoch=None):
        if first_iteration_only and event_tag in self.one_time_events_set:
            return True


        self.one_time_events_set.add(event_tag)

        msg = f"[{self.tag}-{event_tag}] "
        if epoch:
            msg += f"-epoch_{epoch}"

        if t0:
            elapsed = time.time() - t0
            msg += f" | elapsed time: {elapsed:.5f}seconds "


        if only_elapsed_time:
            return self.logger.info(f"\n{msg}") if log_immediate else self.messages.append(f"\n{msg}")

        msg += f" | CPU Memory usage: {self.current_cpu_mem_usage():.1f}MB "
        if self.use_cuda:
            (free, total) = self.current_gpu_mem_usage()
            msg += f" | GPU memory total/free: {total:.1f}/{free:.1f}MB"
            (alloc, alloc_peak, res, res_peak) = self.current_gpu_stats()
            msg += f" | CUDA: allocated/peak: {alloc:.1f}/{alloc_peak:.1f}MB, reserved/peak {res:.1f}/{res_peak:.1f}MB"

        return self.logger.info(f"\n{msg}") if log_immediate else self.messages.append(f"\n{msg}")



class MyEncoder():
    def __init__(self,encoder, use_cuda):
        self.use_cuda = use_cuda
        self.encoder = encoder
        if self.use_cuda:
            self.encoder = self.encoder.to("cuda")
            # self.encoder = torch.compile(self.encoder)
    def get_emb(self, input_ids_tensor, atts_tensor, use_amp=False, use_inference=False):

        if use_inference:
            with torch.inference_mode():
                with torch.amp.autocast(device_type="cuda",enabled=(self.use_cuda and use_amp)):
                    emb = self.encoder(input_ids=input_ids_tensor, attention_mask=atts_tensor)[0]  # token embeddings
        else:
            with torch.amp.autocast(device_type="cuda",enabled=(self.use_cuda and use_amp)):
                emb = self.encoder(input_ids=input_ids_tensor, attention_mask=atts_tensor)[0]  # token embeddings

        mask = atts_tensor.unsqueeze(-1).expand(emb.size()).float()
        embs = (emb * mask).sum(1) / mask.sum(1)  # mean pooling

        return embs
    def save_state(self, path):
        self.encoder.save_pretrained(path)
        return True
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
    def __init__(self,tokens_paths, topk, encoder, faiss_index_name, use_cuda, device):
        self.topk = topk
        self.tokens_paths = tokens_paths

        self.use_cuda = use_cuda
        self.device=device
        self.encoder = encoder
        self.faiss_index_name =faiss_index_name
        self.faiss_index = None
        self.last_epoch_candidates_idxs = None

    def set_last_epoch_candidates_idxs(self, cands_idxs):
        self.last_epoch_candidates_idxs = np.array(cands_idxs, dtype=np.int64).flatten()


    def init_index(self, hidden_size, N):
        if self.faiss_index_name == 'IndexHNSWFlat':
            LOGGER.info(f"USING IndexHNSWFlat index")
            assert self.use_cuda, f'It is better to use_cuda when index is IndexHNSWFlat'
            assert N > 1_000_000, f"for {N}, it is better to use the flat index"
            
            num_clusters = int(math.sqrt(N) * 2)

            num_bytes = 32 # num bytes per vector in PQ
            gpu_resources = faiss.StandardGpuResources()
            quantizer = faiss.IndexHNSWFlat(hidden_size, 32)
            quantizer.hnsw.efConstruction = 200
            quantizer.hnsw.efSearch = 64
            
            index = faiss.GpuIndexIVFPQ(gpu_resources, quantizer, hidden_size, num_clusters, num_bytes, 8)
            index.useFloat16LookupTables = True
            #train clusters on 320k random samples
            sample_size= 210_000
            sample_indices = torch.randperm(N)[:sample_size]
            samples_batch_size = 32_000
            samples_embeds = torch.empty((sample_size, hidden_size), dtype=torch.float32)


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
            assert dictionary_att.shape[0] == N, f"Something is wrong! N={N}, dtionary att shape is: {dictionary_att.shape}"

            cursor = 0
            for start in tqdm(range(0, len(sample_indices), samples_batch_size),  desc="embed sample and train clusters"):
                end = min(start+samples_batch_size, len(sample_indices))
                batch_idx = sample_indices[start:end]


                inp  = torch.as_tensor(dictionary_inputs[batch_idx], device=self.device)
                att = torch.as_tensor(dictionary_att[batch_idx],device=self.device)
            
                batch_embeds = self.encoder.get_emb(inp, att, use_amp=False, use_inference=True)
                batch_embeds = batch_embeds.contiguous()
                samples_embeds[cursor : cursor+(end-start)] = batch_embeds
                cursor += (end -start)
                del batch_embeds, batch_input_ids, batch_att_mask
                torch.cuda.empty_cache()
            index.train(samples_embeds)
            LOGGER.info("Training clusters finsihed ")
            del samples_embeds
            torch.cuda.empty_cache()
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

        N = self.tokens_paths.dict_shape[0]
        hidden_size = self.encoder.encoder.config.hidden_size
        embeddings = np.memmap(dict_embs_npy, dtype=np.float32, mode="w+", shape=(N, hidden_size))
        
        assert hidden_size is not None
        if self.faiss_index is None:
            self.init_index(hidden_size, N)

        if self.last_epoch_candidates_idxs is None:
            embed_indices = np.arange(N)
        else:
            embed_indices = self.last_epoch_candidates_idxs

        M = len(embed_indices)

        for start in tqdm(range(0, M, batch_size), desc="Building faiss index"):
            end = min(start + batch_size, M)
            batch_idxs = embed_indices[start:end]

            inp  = torch.as_tensor(dictionary_inputs[batch_idxs], device=self.device)
            att = torch.as_tensor(dictionary_att[batch_idxs],device=self.device)
            embs = self.encoder.get_emb(inp, att, use_amp=False, use_inference=True)
            # embs = F.normalize(embs, p=2, dim=1)
            embeddings[batch_idxs] = embs.cpu().numpy()
            del inp, att, embs


        self.faiss_index.reset()
        self.faiss_index.add(np.array(embeddings))
        embeddings.flush()
        del dictionary_inputs, dictionary_att
        torch.cuda.empty_cache()
        gc.collect()

    def search_faiss(self, batch_size):
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

        (tokens_size, max_length ) = self.tokens_paths.query_shape
        N = tokens_size
        candidates = np.zeros((N,self.topk))
        faiss_index = self.faiss_index
        for start in range(0, N,batch_size):
            end = min(start + batch_size, N)
            inp  = torch.as_tensor(query_inputs[start:end], device=self.device)
            att = torch.as_tensor(query_att[start:end],device=self.device)
            embs = self.encoder.get_emb(inp, att, use_amp=False, use_inference=True)
            if normalize_query_faiss_search:
                embs = F.normalize(embs, p=2, dim=1)
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



class MyDataset(Dataset):
    def __init__(self,tokens_paths, topk):
        self.tokens_paths  = tokens_paths
        self.topk = topk
        self.all_candidates_idxs = None
        self.query_cuis  = np.load(self.tokens_paths.query_cuis_path)



        self.dict_cuis  = np.load(self.tokens_paths.dict_cuis_path)

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

    def __len__(self,):
        return len(self.query_inputs)

    def set_candidates(self,cands):
        self.all_candidates_idxs = torch.as_tensor(cands, dtype=torch.long)
    def __getitem__(self, query_idx):
        assert self.all_candidates_idxs is not None
        query_tokens = {
            "input_ids": self.query_inputs[query_idx],
            "attention_mask": self.query_att[query_idx],
        }
        candidate_idxs = self.all_candidates_idxs[query_idx]

        assert len(candidate_idxs) == self.topk
        candidate_tokens = {
            "input_ids": self.dictionary_inputs[candidate_idxs],
            "attention_mask": self.dictionary_att[candidate_idxs],
        }

        query_cui = self.query_cuis[query_idx] #1
        query_candidates_cuis = np.array(self.dict_cuis)[candidate_idxs] #(batch_size, topk)
        # labels = (candidates_cuis == query_cui).astype(np.float32)
        labels = get_labels(query_candidates_cuis, query_cui) #if error_type == 'info_nce_loss', will return [batch_size] for each item is the first match, for marginal_nll error_type will return  (batch_size, topk) for each item 0 if false, 1 for true

        return (query_tokens, candidate_tokens), labels



class MyModel(nn.Module):
    def __init__(self, encoder, learning_rate,weight_decay, use_cuda):
        super(MyModel, self).__init__()

        self.encoder = encoder
        self.learning_rate = learning_rate
        self.use_cuda = use_cuda
        self.optimizer = optim.AdamW(
            self.encoder.encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
            fused=self.use_cuda
        )

        self.criterion = info_nce_loss if args.loss_type == 'info_nce_loss' else marginal_nll




    def forward(self, x_batch):
        query_tokens, candidates_tokens = x_batch

        if self.use_cuda:
            candidates_tokens["input_ids"] = candidates_tokens["input_ids"].to("cuda", non_blocking=True)
            candidates_tokens["attention_mask"] = candidates_tokens["attention_mask"].to("cuda", non_blocking=True)
            query_tokens["input_ids"] = query_tokens["input_ids"].to("cuda", non_blocking=True)
            query_tokens["attention_mask"] = query_tokens["attention_mask"].to("cuda", non_blocking=True)
            
        batch_size, topk, max_length = candidates_tokens["input_ids"].size()


        #(b, h)
        query_embeds = self.encoder.get_emb(query_tokens["input_ids"], query_tokens["attention_mask"], use_amp=True, use_inference=False)


        candidates_tokens["input_ids"] = candidates_tokens["input_ids"].view(batch_size * topk, max_length)
        candidates_tokens["attention_mask"] = candidates_tokens["attention_mask"].view(batch_size * topk, max_length)
        #(batch_size * topk , h)
        candidates_embs = self.encoder.get_emb(candidates_tokens["input_ids"], candidates_tokens["attention_mask"], use_amp=True, use_inference=False)

        if normalize_candidates_forward:
            candidates_embs = F.normalize(candidates_embs, p=2, dim=1)


        candidates_embs = candidates_embs.view(batch_size, topk, -1)
        if normalize_query_forward:
            query_embeds = F.normalize(query_embeds, p=2, dim=1)
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
        loss = self.criterion(outputs, targets)
        return loss




LOGGER = logging.getLogger()





def train(use_cuda, device, lg, args):

    

    model = AutoModel.from_pretrained(args.encoder_model_name, use_safetensors=True)
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # transformer = model._first_module().auto_model  # the underlying Hugging Face model
    LOGGER.info("ARGS: ")
    LOGGER.info(args)

    t0 = time.time()
    encoder = MyEncoder(model, use_cuda)
    tokens_paths = TokensPaths("dict", "queries")
    my_faiss = MyFaiss(
        tokens_paths=tokens_paths, 
        topk= args.topk, 
        encoder=encoder, 
        faiss_index_name=args.faiss_index_name, 
        use_cuda = use_cuda, 
        device=device
    )
    my_ds = MyDataset(tokens_paths, args.topk)
    my_model = MyModel(encoder, args.learning_rate, args.weight_decay, use_cuda)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_cuda)


    num_training_steps = len(my_ds) // args.train_batch_size * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        my_model.optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    lg.log_event("Classes loaded", t0=t0)

    LOGGER.info("Starting training..")

    train_start_t0 = time.time()


    result_encoder_dir = config.result_encoders_dir + f"/encoder_{lg.current_global_log_number}/" 
    os.makedirs(result_encoder_dir, exist_ok=True)

    epoch_acc, epoch_mrr = 0.0, 0.0

    for epoch in range(args.num_epochs):
        t0 = time.time()
        my_faiss.build_faiss(args.build_faiss_batch_size)
        lg.log_event("FAISS index built finished  ", t0=t0, epoch=epoch)

        t0 = time.time()
        cands_idxs = my_faiss.search_faiss(args.search_faiss_batch_size)
        lg.log_event("Search in faiss ", t0=t0, epoch=epoch)


        t0 = time.time()
        my_ds.set_candidates(cands_idxs) 
        my_faiss.set_last_epoch_candidates_idxs(cands_idxs)
        my_loader = torch.utils.data.DataLoader(
            my_ds, 
            batch_size=args.train_batch_size, 
            shuffle=True, 
            pin_memory=use_cuda, 
            num_workers=args.num_workers,
            persistent_workers=True)
        lg.log_event("Data loader loadeed: ", t0=t0, epoch=epoch)


        t0 = time.time()
        train_loss, train_steps = 0.0, 0
        epoch_acc, epoch_mrr = 0.0, 0.0
        n_eval = 0
        for i, (batch_x, batch_y) in tqdm(enumerate(my_loader), total=len(my_loader), desc="training batches", unit="batch" ):
            my_model.optimizer.zero_grad(set_to_none=True)
            if use_cuda:
                with torch.amp.autocast("cuda"):
                    batch_y_pred = my_model(batch_x)
                    loss = my_model.get_loss(batch_y_pred, batch_y)
                scaler.scale(loss).backward()
                scaler.step(my_model.optimizer)
                scaler.update()
                scheduler.step()
            else:
                batch_y_pred = my_model(batch_x)  
                loss = my_model.get_loss(batch_y_pred, batch_y) 
                loss.backward()
                my_model.optimizer.step()
            train_loss += loss.item()
            train_steps += 1
            
            acc_k, mrr = compute_metrics(batch_y_pred.detach().cpu(), batch_y.cpu(), k=5)
            epoch_acc += acc_k
            epoch_mrr += mrr
            n_eval += 1


            del batch_x, batch_y, batch_y_pred

        train_loss /= (train_steps + 1e-9)
        lg.log_event("Epoch finished training", t0=t0, epoch=epoch)

        LOGGER.info(f"Epoch {epoch}: avg_train_loss={train_loss:.5f}, acc@5={epoch_acc/n_eval:.5f}, mrr={epoch_mrr/n_eval:.5f}")


    encoder.save_state(result_encoder_dir)
    training_time = time.time()-t0
    training_time_str = f"{int(training_time/60/60)}h, {int(training_time/60 % 60)}mins, {int(training_time % 60)}secs"


    with open(config.global_log_path, "w") as f:
        lg.log_global_data[-1]["training_log_name"] = args.training_log_name
        lg.log_global_data[-1]["queries size"]  = len(my_ds.query_cuis)
        lg.log_global_data[-1]["dictionary size"]  = len(my_ds.dict_cuis)
        lg.log_global_data[-1]["finished time"]  = training_time_str
        lg.log_global_data[-1]["log details file"]  = lg.log_path
        lg.log_global_data[-1]["epochs"]  = args.num_epochs
        lg.log_global_data[-1]["acc@5"]  = epoch_acc
        lg.log_global_data[-1]["mrr"]  = epoch_mrr
        lg.log_global_data[-1]["encoder dir"]  = result_encoder_dir
        json.dump(lg.log_global_data,f)


    LOGGER.info(f"LOGS saved in {lg.log_path} and in global with the name: {args.training_log_name}")
    torch.cuda.empty_cache()
    gc.collect()
    lg.log_event("Train finished ", t0=train_start_t0)
    return result_encoder_dir

def eval(use_cuda, device, lg, result_encoder_dir, args):
    
    model = AutoModel.from_pretrained(result_encoder_dir, use_safetensors=True)
    encoder = MyEncoder(model, use_cuda)
    tokens_paths = TokensPaths("dict", "test")
    my_ds = MyDataset(tokens_paths, args.topk)
    my_faiss = MyFaiss(
        tokens_paths=tokens_paths, 
        topk= args.topk, 
        encoder=encoder, 
        faiss_index_name=args.faiss_index_name, 
        use_cuda = use_cuda, 
        device=device
    )

    my_model = MyModel(encoder, args.learning_rate, args.weight_decay, use_cuda)
    my_model.eval()
    my_faiss.build_faiss(args.build_faiss_batch_size)
    cands_idxs = my_faiss.search_faiss(args.search_faiss_batch_size)
    my_ds.set_candidates(cands_idxs) 


    eval_loader = torch.utils.data.DataLoader(
            my_ds, 
            batch_size=args.train_batch_size, 
            shuffle=True, 
            pin_memory=use_cuda, 
            num_workers=args.num_workers,
            persistent_workers=True)
    total_loss = 0.0
    total_mrr = 0.0
    total_acc = 0.0
    total_samples = 0
    n_eval = 0
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_cuda):
        for batch_x, batch_y in tqdm(eval_loader, desc="Evaluating"):
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            batch_y = batch_y.to(device)
            batch_size = batch_y.size(0)
            
            
            query_tokens, candidate_tokens = batch_x
            query_tokens = {k: v.to(device) for k, v in query_tokens.items()}
            candidate_tokens = {k: v.to(device) for k, v in candidate_tokens.items()}
            batch_x = (query_tokens, candidate_tokens)

            # Forward pass
            batch_pred = my_model(batch_x)  # [batch_size, hidden_size]
            loss = my_model.get_loss(batch_pred, batch_y)

            acc_k, mrr = compute_metrics(batch_pred.detach().cpu(), batch_y.cpu(), k=args.topk)


            total_loss += loss.item() * batch_size
            total_mrr += mrr
            total_acc += acc_k
            total_samples += batch_size
            n_eval += 1

    avg_loss = total_loss / total_samples
    avg_mrr = total_mrr / n_eval
    avg_acc = total_acc / n_eval

    LOGGER.info(f"[Eval] Loss={avg_loss:.5f}, MRR={avg_mrr:.5f}, ACC@{args.topk}={avg_acc:.5f}")
    return avg_loss, avg_mrr, avg_acc

if __name__ == "__main__":
    args= parse_args()
    use_cuda = torch.cuda.is_available()
    print(f"use cuda: {use_cuda}" )
    device = torch.device("cuda" if use_cuda else 'cpu')
    lg = MyLogger(LOGGER, use_cuda, global_log_path=config.global_log_path, logs_dir=config.logs_dir, tag="train")

    result_encoder_dir = train(use_cuda, device, lg, args)
    eval(use_cuda, device, lg, result_encoder_dir, args)
    
    


# python process.py --training_log_name='small_dictionary_flat_faiss' --faiss_index_name='IndexFlatIP' --num_workers=48 --loss_type='info_nce_loss'

