# SparseIM

(SIGIR 2022)

## Runing the experiments

### dataset and preprocessing

#### Data downloading

* [Reddit](http://snap.stanford.edu/jodie/reddit.csv)
* [Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)

#### Data preprocessing

We use the dense `npy` format to save the features in binary format. If edge features or nodes features are absent, it will be replaced by a vector of zeros. The dataset will be processed automatically when runing the `Train.py`.

### Runing the model
#### Pretrain mode
##### Pretrain the GNN model
```{bash}
python3 Train.py  --data wikipedia/reddit/mooc --mode Pretrain 
```
##### Then Training the Sparsification Strategy 
```{bash}
python3 Train.py  --data wikipedia/reddit/mooc --mode SparseIM
```
#### End to End mode
Jointly train the GNN and SparseIM.
```{bash}
python3 Train.py  --data wikipedia/reddit/mooc --mode End2End 
```



### Requirements

* python>=3.7

* Dependency

  ```python
  dgl==0.7.0
  numpy==1.18.1
  scikit-learn==0.23.2
  torch==1.9.0
  tqdm==4.48.2
  ```



#### General flag

Optional arguments are described in args.py.



