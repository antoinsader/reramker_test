import os
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn, optim

from config import GlobalConfig
from process import MyEncoder, MyDataset, TokensPaths


def fine_tune_embeddings(cfg: GlobalConfig, num_epochs=3, lr=3e-5, batch_size=128):
    """
    Fine-tunes the token embeddings (including special tokens) using cosine contrastive loss.
    Only the embedding layer is updated — all encoder layers are frozen.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -----------------------
    # Load dataset
    # -----------------------
    tokens_paths = TokensPaths("small_dict", "queries")
    dataset = MyDataset(tokens_paths, cfg)
    dict_cuis = dataset.dict_cuis
    query_cuis = dataset.query_cuis
    cui_to_dict_idxs = dataset.dictionary_cui_to_idx

    # -----------------------
    # Build encoder
    # -----------------------
    encoder = MyEncoder(use_cuda=torch.cuda.is_available(), cfg=cfg)
    model = encoder.encoder
    embedding_layer = model.embeddings.word_embeddings
    print(f"Embedding size: {embedding_layer.weight.shape}")

    # Freeze all except embeddings
    for p in model.parameters():
        p.requires_grad = False
    embedding_layer.weight.requires_grad = True

    model.to(device)
    model.train()

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    criterion = nn.CosineEmbeddingLoss(margin=0.3)

    # -----------------------
    # Training sample setup
    # -----------------------
    N = len(query_cuis)
    sample_indices = np.random.choice(N, size=min(50000, N), replace=False)
    print(f"Training on {len(sample_indices)} query samples")

    for epoch in range(num_epochs):
        np.random.shuffle(sample_indices)
        losses = []
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for start in tqdm(range(0, len(sample_indices), batch_size), desc="Batches"):
            end = min(start + batch_size, len(sample_indices))
            batch_idx = sample_indices[start:end]

            # -----------------------
            # Build query and dictionary pairs
            # -----------------------
            pos_idx = []
            neg_idx = []
            for i in batch_idx:
                q_cui = query_cuis[i]

                # Positive: same CUI if exists, else random
                positives = cui_to_dict_idxs.get(q_cui, [])
                if positives:
                    pos_idx.append(random.choice(positives))

                # Negative: random different CUI
                while True:
                    j = np.random.randint(len(dict_cuis))
                    if dict_cuis[j] != q_cui:
                        neg_idx.append(j)
                        break


            q_inp = torch.as_tensor(dataset.query_inputs[batch_idx], device=device)
            q_att = torch.as_tensor(dataset.query_att[batch_idx], device=device)
            d_pos_inp = torch.as_tensor(dataset.dictionary_inputs[pos_idx], device=device)
            d_pos_att = torch.as_tensor(dataset.dictionary_att[pos_idx], device=device)
            d_neg_inp = torch.as_tensor(dataset.dictionary_inputs[neg_idx], device=device)
            d_neg_att = torch.as_tensor(dataset.dictionary_att[neg_idx], device=device)

            # -----------------------
            # Get embeddings
            # -----------------------
            q_emb = encoder.get_emb(q_inp, q_att, use_amp=False, use_no_grad=False)
            d_pos_emb = encoder.get_emb(d_pos_inp, d_pos_att, use_amp=False, use_no_grad=False)
            d_neg_emb = encoder.get_emb(d_neg_inp, d_neg_att, use_amp=False, use_no_grad=False)

            # -----------------------
            # Compute contrastive loss
            # -----------------------
            x1 = torch.cat([q_emb, q_emb], dim=0)
            x2 = torch.cat([d_pos_emb, d_neg_emb], dim=0)
            target = torch.cat([
                torch.ones(q_emb.size(0), device=device),
                -torch.ones(q_emb.size(0), device=device)
            ])
            loss = criterion(x1, x2, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            del q_inp, q_att, d_pos_inp, d_pos_att, d_neg_inp, d_neg_att
            del q_emb, d_pos_emb, d_neg_emb, loss
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}: mean loss = {np.mean(losses):.6f}")

    # -----------------------
    # Save fine-tuned encoder
    # -----------------------
    output_dir = "./output/fine_tuned_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    encoder.save_state(output_dir)
    print(f"Saved fine-tuned encoder to: {output_dir}")

    return output_dir


if __name__ == "__main__":
    cfg = GlobalConfig()
    fine_tune_embeddings(cfg, num_epochs=3, lr=2e-5, batch_size=256)
