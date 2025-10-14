from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import faiss.contrib.torch_utils
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.retrieval import RetrievalMRR

from tqdm import tqdm
import gc

from transformers import AutoModel
from config import paths, cands_num, train_batch_size, learning_rate, encoder_model_name, weight_decay, num_workers,build_faiss_batch_size, num_epochs, search_faiss_batch_size, dict_embs_npy
from utils import info_nce_loss, load_mmap_shape, marginal_nll



class MyEncoder():
    def __init__(self,encoder, use_cuda):
        self.use_cuda = use_cuda
        self.encoder = encoder
        if self.use_cuda:
            self.encoder = self.encoder.to("cuda")
            self.encoder = torch.compile(self.encoder)
    def get_emb(self, input_ids_tensor, atts_tensor):
        with torch.amp.autocast(device_type="cuda",enabled=self.use_cuda):
            emb = self.encoder(input_ids=input_ids_tensor, attention_mask=atts_tensor)[0]  # token embeddings
            mask = atts_tensor.unsqueeze(-1).expand(emb.size()).float()
            embs = (emb * mask).sum(1) / mask.sum(1)  # mean pooling
            
        return embs
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
    def __init__(self,tokens_paths, cands_num, encoder, use_cuda, device):
        self.cands_num = cands_num
        self.tokens_paths = tokens_paths

        self.use_cuda = use_cuda
        self.device=device
        self.encoder = encoder
        self.faiss_index = None
        self.last_epoch_candidates_idxs = None

    def set_last_epoch_candidates_idxs(self, cands_idxs):
        self.last_epoch_candidates_idxs = np.array(cands_idxs, dtype=np.int64).flatten()


    def init_index(self, hidden_size):
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

        self.init_index(hidden_size)

        if self.last_epoch_candidates_idxs is None:
            embed_indices = np.arange(N)
        else:
            embed_indices = self.last_epoch_candidates_idxs

        M = len(embed_indices)

        with torch.inference_mode():
            for start in tqdm(range(0, M, batch_size), desc="Building faiss index"):
                end = min(start + batch_size, M)
                batch_idxs = embed_indices[start:end]

                inp  = torch.as_tensor(dictionary_inputs[batch_idxs], device=self.device)
                att = torch.as_tensor(dictionary_att[batch_idxs],device=self.device)
                embs = self.encoder.get_emb(inp, att)
                embs = F.normalize(embs, p=2, dim=1)
                embeddings[batch_idxs] = embs.cpu().numpy()
                del inp, att, embs

        self.faiss_index.add(np.array(embeddings))
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
        candidates = np.zeros((N,self.cands_num))
        faiss_index = self.faiss_index
        with torch.inference_mode():
            for start in range(0, N,batch_size):
                end = min(start + batch_size, N)
                inp  = torch.as_tensor(query_inputs[start:end], device=self.device)
                att = torch.as_tensor(query_att[start:end],device=self.device)
                embs = self.encoder.get_emb(inp, att)
                embs = F.normalize(embs, p=2, dim=1)
                if self.use_cuda:
                    embs = embs.contiguous()
                else:
                    embs = embs.cpu().numpy().astype(np.float32)

                _, chunk_cand_idxs = faiss_index.search(embs, self.cands_num)
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
                mode="r+",
                dtype=np.int32,
                shape=self.tokens_paths.query_shape
            )
        self.query_att = np.memmap(
                self.tokens_paths.query_att_path,
                mode="r+",
                dtype=np.int32,
                shape=self.tokens_paths.query_shape
            )
    
        self.dictionary_inputs = np.memmap(
                self.tokens_paths.dict_inp_path,
                mode="r+",
                dtype=np.int32,
                shape=self.tokens_paths.dict_shape
            )
        self.dictionary_att = np.memmap(
                self.tokens_paths.dict_att_path,
                mode="r+",
                dtype=np.int32,
                shape=self.tokens_paths.dict_shape
            )
        

        
        
    def __len__(self,):
        return len(self.query_inputs)
    
    def set_candidates(self,cands):
        self.all_candidates_idxs = torch.from_numpy(cands.astype(np.int64))
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
        
        query_cui = self.query_cuis[query_idx]
        candidates_cuis = np.array(self.dict_cuis)[candidate_idxs]
        labels = (candidates_cuis == query_cui).astype(np.float32)
        
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

        self.criterion = info_nce_loss
    
    def forward(self, x_batch):
        query_tokens, candidates_tokens = x_batch
        batch_size, topk, max_length = candidates_tokens["input_ids"].size()
        query_embeds = self.encoder.get_emb(query_tokens["input_ids"], query_tokens["attention_mask"])
        
        candidates_tokens["input_ids"] = candidates_tokens["input_ids"].view(batch_size * topk, max_length)
        candidates_tokens["attention_mask"] = candidates_tokens["attention_mask"].view(batch_size * topk, max_length)
        candidates_embs = self.encoder.get_emb(candidates_tokens["input_ids"], candidates_tokens["attention_mask"])
        candidates_embs = F.normalize(candidates_embs, p=2, dim=1)
        candidates_embs = candidates_embs.view(batch_size, topk, -1)

        query_embeds = F.normalize(query_embeds, p=2, dim=1)
        query_embeds = query_embeds.unsqueeze(1) # [batch_size, 1, hidden]

        score = torch.bmm(query_embeds, candidates_embs.transpose(1, 2)).squeeze(1)
        return score
    
    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
            outputs = outputs.cuda()
        loss = self.criterion(outputs, targets)
        return loss

    
def main():

    model = AutoModel.from_pretrained(encoder_model_name, use_safetensors=True)
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # transformer = model._first_module().auto_model  # the underlying Hugging Face model


    use_cuda = torch.cuda.is_available()
    print(f"use cuda: {use_cuda}" )
    device = torch.device("cuda" if use_cuda else 'cpu')


    encoder = MyEncoder(model, use_cuda)
    tokens_paths = TokensPaths("dict", "queries")
    my_faiss = MyFaiss(tokens_paths, cands_num, encoder, use_cuda, device )
    my_ds = MyDataset(tokens_paths, cands_num)

    my_model = MyModel(encoder, learning_rate, weight_decay, use_cuda)

    scaler = torch.amp.GradScaler(device="cuda", enabled=use_cuda)



    # num_classes = cands_num
    # acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
    # acc5 = acc5.to(device)
    # mrr = RetrievalMRR()


    for epoch in range(num_epochs):
        my_faiss.build_faiss(build_faiss_batch_size)
        cands_idxs = my_faiss.search_faiss(search_faiss_batch_size)
        my_ds.set_candidates(cands_idxs) 
        my_faiss.set_last_epoch_candidates_idxs(cands_idxs)
        my_loader = torch.utils.data.DataLoader(
            my_ds, 
            batch_size=train_batch_size, 
            shuffle=True, 
            pin_memory=use_cuda, 
            num_workers=num_workers,
            persistent_workers=True)
        train_loss, train_steps = 0.0, 0
        for i, (batch_x, batch_y) in tqdm(enumerate(my_loader), total=len(my_loader), desc="training batches", unit="batch" ):
            my_model.optimizer.zero_grad(set_to_none=True)
            
            if use_cuda:
                with torch.cuda.amp.autocast():
                    batch_y_pred = my_model(batch_x)  
                    loss = my_model.get_loss(batch_y_pred, batch_y)
                scaler.scale(loss).backward()
                scaler.step(my_model.optimizer)
                scaler.update()
            else:
                batch_y_pred = my_model(batch_x)  
                loss = my_model.get_loss(batch_y_pred, batch_y) 
                loss.backward()
                my_model.optimizer.step()
        
            train_loss += loss.item()
            train_steps += 1

            # batch_y_pred = batch_y_pred.to(device)
            # batch_y = batch_y.to(device) if isinstance(batch_y, torch.Tensor) else batch_y
            # batch_indexes = torch.arange(start=i*train_batch_size, 
            #                  end=i*train_batch_size + batch_y_pred.size(0),
            #                  device=batch_y_pred.device)
            # batch_indexes = batch_indexes.unsqueeze(1).expand_as(batch_y_pred)
            # mrr.update(batch_y_pred, batch_y.int(), batch_indexes)
            # target = batch_y.argmax(dim=1) 
            # acc5.update(batch_y_pred, target)
            del batch_x, batch_y, batch_y_pred
            if use_cuda:
                torch.cuda.empty_cache()

        # acc5 = acc5.compute()  # Returns a tensor
        # mrr = mrr.compute()
        train_loss /= (train_steps + 1e-9)

        print(f"Epoch {epoch}  avg_train_loss: {train_loss}")
        # acc5.reset()  # Reset for next epoch
        # mrr.reset()
        
        torch.cuda.empty_cache()
        gc.collect()
        



if __name__ == "__main__":
    main()