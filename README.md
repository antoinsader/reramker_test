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
- Make sure that inside 