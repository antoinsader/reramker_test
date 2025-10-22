# Concept retreival 

Deep learning retrieval system for learning the matching between mentions (queries) to dictionary entries using encoders and FAISS vector search.

I am using this model to train on ncbi-dataset 
I have concept files in traindev, each one contains multiple medical concepts (mention) with their corresponding CUI (id)
and the model will learn how to relate those mentions with the corresponding terms in train dictionary containing the same CUIs but maybe with different names 
The model has been trained with 4m dictionary records successfully and gave result of:
accuracy@5: 
mrr: 
average_loss: 

The model trying to retreive candidates for each query from the dictionary, then fine-tune the encoder based on marginal_nll criterion to retreive the correct candidates 
The num of candidates is specified with --topk argument during training.

The FAISS search can be Flat brute search by specifying the --faiss_index_name as IndexFlatIP or IndexHNSWFlat (look into FAISS documentation for more info about the index types)
but in general if you have big dictionary (bigger than 1m) it is advisable to use IndexHNSWFlat with cuda, otherwise for small dictionaries you can use IndexFlatIP which will be more accurate




Each time you run the process.py which having the training, a new object will be created in logs/logger_all.json file, which will have highlights of the training process, and a link to the .log file containing all the details of the training.


---

## Installation:

If you are on **Linux with a CUDA GPU**, the easiest way to set up everything is by running: 

```bash
    bash install_ds.sh
```

This scripts: 
- Extract dictionary and traindev zip files from /raw/ into ./data/raw/
- Create python virtual environment
- Install required libraries

If you're on another OS or prefer manual setup, create your own python environment, and install dependencies from 'requirements.txt'

But be careful:
- faiss-gpu-cu12 is the faiss using cuda 12, if you don't have cuda, you can use normal faiss
- Make sure that inside the folder data/raw, you have traindev/ folder containing .concept files and train_dictionary.txt file

---

## Tokenize:

The `1_tokenize.py` script reads the dictionary (`train_dictionary.txt`) and the `traindev/` folder of query concept files, then tokenizes them using a Hugging Face tokenizer (by default, BioBERT).

The tokenized results are saved in:

```
data/tokens/
 ├── _d_inp.mmap      ← dictionary input_ids
 ├── _d_att.mmap      ← dictionary attention masks
 ├── _q_inp.mmap      ← query input_ids
 ├── _q_att.mmap      ← query attention masks
 ├── _d_ids.npy       ← dictionary CUIs
 ├── _q_ids.npy       ← query CUIs
 └── ...
```

These tokenized files are what the later stages (`split` and `train`) will use.  
Run:
```bash
python 1_tokenize.py
```



---
## Split:

You can create a **test set** in one of two ways:

1. **Manually** create a folder `data/raw/test/` containing `.concept` files for your test queries.

2. **Automatically** split your existing tokenized queries using:
   ```bash
   python split.py --train_percentage 0.8
   ```
   This will keep 80% of your tokenized queries for training and 20% for testing.

After this step, `data/tokens` will contain both train and test query token files.
---

## Train

Training and evaluation are both handled by `process.py`.

### Arguments

| Argument | Required | Description |
|-----------|-----------|-------------|
| `--training_log_name` | ✅ | Unique name for this training session |
| `--faiss_index_name` | ✅ | Either `IndexHNSWFlat` or `IndexFlatIP` |
| `--encoder_model_name` | ❌ | Pretrained encoder model name or directory |
| `--num_workers` | ❌ | Number of workers for DataLoader |
| `--num_epochs` | ❌ | Total number of epochs |
| `--train_batch_size` | ❌ | Training batch size |
| `--topk` | ❌ | Number of retrieved candidates per query |
| `--learning_rate` | ❌ | Learning rate |
| `--weight_decay` | ❌ | Weight decay for optimizer |
| `--loss_type` | ❌ | Either `marginal_nll` or `info_nce_loss` |
| `--build_faiss_batch_size` | ❌ | Batch size when building FAISS index |
| `--search_faiss_batch_size` | ❌ | Batch size when searching FAISS |
| `--faiss_clustering_samples_size` | ❌ | Number of samples to build FAISS clusters |
| `--encoder_to_eval` | ❌ | Directory of encoder to evaluate |
| `--eval_faiss_path` | ❌ | Path to FAISS index for evaluation |
| `--save_debug_pkls` | ❌ | Save debug pickles |
| `--skip_train` | ❌ | Skip training (only eval) |
| `--skip_eval` | ❌ | Skip evaluation |
| `--use_amp` | ❌ | Enable automatic mixed precision |

### Basic training example

```bash
python process.py --training_log_name='big_dictionary' --faiss_index_name='IndexHNSWFlat' --num_workers=16
```

### What happens during training

1. The script builds a **FAISS index** for the dictionary embeddings.
2. For each epoch:
   - Queries are embedded.
   - Top-K candidate matches are retrieved from FAISS.
   - The encoder is trained using contrastive or NLL loss.
   - Metrics such as loss, MRR, accuracy@K, and FAISS recall@K are logged.
3. The encoder and FAISS index are saved in:
   ```
   output/encoder_{N}/
     ├── config.json
     ├── model.safetensors
     ├── faiss_index.faiss
     └── checkpoints/
   ```
4. Evaluation runs automatically at the end of training (unless `--skip_eval` is set).

---

## Evaluation

By default, evaluation is performed right after training using the trained encoder and FAISS index.

To **only run evaluation**, skip training and specify both the encoder directory and FAISS index path:

```bash
python process.py \
  --training_log_name='big_dictionary_eval' \
  --faiss_index_name='IndexHNSWFlat' \
  --num_workers=16 \
  --skip_train \
  --encoder_to_eval='./output/encoder_1' \
  --eval_faiss_path='./output/encoder_1/faiss_index.faiss'
```

During evaluation:
- The dictionary FAISS index is rebuilt.
- Queries from the test set are embedded.
- The system reports **Loss**, **Mean Reciprocal Rank (MRR)**, and **Accuracy@K** over the test queries.

---

## Output structure

```
output/
 ├── encoder_1/
 │    ├── model.safetensors
 │    ├── config.json
 │    ├── faiss_index.faiss
 │    └── checkpoints/
 ├── encoder_2/
 │    └── ...
logs/
 ├── logger_all.json
 ├── log_1_2025-10-22_12-00-00.log
 └── ...
```

---

## Notes

- All major directories (`data/tokens`, `data/raw`, `logs`, `output`) are auto-created if missing.
- Mixed precision (AMP) can significantly speed up training on GPUs.
- FAISS index choice:
  - `IndexFlatIP`: fast, simple, suitable for small dictionaries.
  - `IndexHNSWFlat`: hierarchical search for large dictionaries (>1M entries).

---

**Author:** Starky