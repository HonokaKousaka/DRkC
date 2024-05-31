# Coresets for Deletion-Robust k-Center Clustering

This is the official repository of the CIKM 2024 paper submission titled *Coresets for Deletion-Robust k-Center Clustering*.

## Setup

This implementation is based on Python 3. To run the code, you need the following dependencies.

- numpy==1.26.4

To install the dependencies, you may use pip as follows.

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
|-- fix_k_GBGMM.py # GBGMM Deletion with k fixed
|-- fix_k_WBGreedy.py # WBGreedy Deletion with k fixed
|-- fix_k_WBNN.py # WBNN Deletion with k fixed
|-- table_GBGMM.py # GBGMM Deletion for the table in the paper
|-- table_WBGreedy.py #  WBGreedy Deletion for the table in the paper
|-- table_WBNN.py # WBNN Deletion for the table in the paper
```

## Run our code
You can use the code in ```functions.py``` if you want to try our algorithm in your own way.

If you want to reproduce the results in the paper, you can run our code in the following steps.

1. Make sure there are 2 directories named ```results_1``` and ```results_2``` in the same directory that our code are in.

2. You can run our code as in the script in the below: 
```python
python fix_k_GBGMM.py
python fix_k_WBGreedy.py
python fix_k_WBNN.py
python table_GBGMM.py
python table_WBGreedy.py
python table_WBNN.py
```



