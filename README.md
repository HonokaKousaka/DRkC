# Coresets for Deletion-Robust k-Center Clustering

This is the official repository of CIKM 2024 paper *Coresets for Deletion-Robust k-Center Clustering*.

## Setup

This implementation is based on Python 3. To run the code, you need the following dependencies.

- numpy==1.26.4

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- dataset # adjacency matrices and edges of 5 datasets from graphic perspective
    |-- adult_complete.npy # the adjacency matrix of Adult dataset
    |-- adult_edges.npy # the edges of Adult dataset in non-descending order
    |-- ...
|-- original_dataset # 5 datasets, each with 1,000 elements
    |-- adult.npy # Adult dataset with 1,000 dataset
    |-- ...
|-- functions.py # all algorithm functions
|-- table_complete_deletion.py # Complete-knowledge Deletion for the table in the paper
|-- table_half_deletion.py # Half-knowledge Deletion for the table in the paper
|-- table_lossMax_deletion.py #  Loss-maximizing Deletion for the table in the paper
|-- fix_k_complete_deletion.py # Complete-knowledge Deletion with k fixed
|-- fix_k_half_deletion.py # Half-knowledge Deletion with k fixed
|-- fix_k_lossMax_deletion.py # Loss-maximizing Deletion with k fixed
```

## Run our codes
You can use the codes in ```functions.py``` if you want to try our algorithm in your own way.

If you want to reproduce the results in the paper, you can run our codes in the following steps.

1. Make sure there are 2 directories named ```results_1``` and ```results_2``` in the same directory that our codes are in.

2. You can run our codes like the script in the below: 
```python
python table_complete_deletion.py
python table_half_deletion.py
python table_lossMax_deletion.py
python fix_k_complete_deletion.py
python fix_k_half_deletion.py
python fix_k_lossMax_deletion.py
```

## Citation

If you find this code working for you, please cite:



