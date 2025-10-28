import os, torch, json, gc
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn, optim
from transformers import get_linear_schedule_with_warmup

from config import GlobalConfig
from process import MyEncoder, MyDataset, TokensPaths


def fine_tune_embeddings(cfg: GlobalConfig, num_epochs=1, lr=1e-4, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load dataset
    tokens_paths = TokensPaths("dict", "queries")
    dataset = MyDataset(tokens_paths, cfg)

    # --- Build encoder
    encoder = MyEncoder(use_cuda=torch.cuda.is_available(), cfg=cfg)
    model = encoder.encoder
    embedding_layer = model.embeddings.word_embeddings
    print(f"Embedding size: {embedding_layer.weight.shape}")

    # --- Freeze everything except embeddings
    for p in model.parameters():
        p.requires_grad = False
    embedding_layer.weight.requires_grad = True

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    # small cosine embedding loss
    criterion = nn.CosineEmbeddingLoss()

    # randomly sample some training examples
    N = len(dataset.query_cuis)
    sample_indices = np.random.choice(N, size=min(30000, N), replace=False)
    print(f"Training on {len(sample_indices)} query samples")

    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        np.random.shuffle(sample_indices)
        losses = []
        # replace inside training loop
        for start in tqdm(range(0, len(sample_indices), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
            end = min(start + batch_size, len(sample_indices))
            batch_idx = sample_indices[start:end]

            # positives
            q_inp = torch.as_tensor(dataset.query_inputs[batch_idx], device=device)
            q_att = torch.as_tensor(dataset.query_att[batch_idx], device=device)
            d_pos_inp = torch.as_tensor(dataset.dictionary_inputs[batch_idx], device=device)
            d_pos_att = torch.as_tensor(dataset.dictionary_att[batch_idx], device=device)

            # negatives (random dictionary entries)
            neg_idx = np.random.choice(len(dataset.dictionary_inputs), size=len(batch_idx), replace=False)
            d_neg_inp = torch.as_tensor(dataset.dictionary_inputs[neg_idx], device=device)
            d_neg_att = torch.as_tensor(dataset.dictionary_att[neg_idx], device=device)

            # embeddings
            q_emb = encoder.get_emb(q_inp, q_att, use_amp=False, use_no_grad=False)
            d_pos_emb = encoder.get_emb(d_pos_inp, d_pos_att, use_amp=False, use_no_grad=False)
            d_neg_emb = encoder.get_emb(d_neg_inp, d_neg_att, use_amp=False, use_no_grad=False)

            # build batch of positive and negative pairs
            x1 = torch.cat([q_emb, q_emb], dim=0)
            x2 = torch.cat([d_pos_emb, d_neg_emb], dim=0)
            target = torch.cat([
                torch.ones(q_emb.size(0), device=device),
                -torch.ones(q_emb.size(0), device=device)
            ])

            loss = criterion(x1, x2, target)


        print(f"Epoch {epoch+1} mean loss: {np.mean(losses):.6f}")

    # --- Save tuned encoder
    output_dir = "./output/fine_tuned_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    encoder.save_state(output_dir)
    print(f"Saved fine-tuned encoder to {output_dir}")

    return output_dir


if __name__ == "__main__":
    cfg = GlobalConfig()
    fine_tune_embeddings(cfg, num_epochs=4, lr=3e-4, batch_size=128)
