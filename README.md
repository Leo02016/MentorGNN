### Dependencies
* PyTorch
* torch_geometric
* pandas
* sklearn

### 
# For citeseer data set and pubmed dataset, please unzip the files in [citeseer file](https://github.com/Leo02016/MentorGNN/blob/main/data/citeseer/preprocessed_data.zip) and [Pubmed file](https://github.com/Leo02016/MentorGNN/blob/main/data/pubmed/preprocessed_data.zip) first. 

### Command
Single-graph knowledge transfer: cora-> reddit1, where GCN is used as backbone.
```
python main.py --gpu 0 --graph_src cora --graph_dst reddit1 --dropout 0.5 --lr 0.003 --patience 1000 --base_model gcn
```

Single-graph knowledge transfer: cora-> reddit1, where GAT is used as backbone. 
Notice that the required GPU memory of GAT model will be larger than 16GB.
```
python main.py --gpu 0 --graph_src cora --graph_dst reddit1 --dropout 0.5 --lr 0.003 --patience 1000 --base_model gat
```

Multi-graph knowledge transfer: cora+pubmed+citeseer-> reddit, where GCN is used as backbone.
```
python main.py --gpu 0 --graph_src cora+pubmed+citeseer  --graph_dst reddit1 --dropout 0.3 --lr 0.002 --patience 1000 --base_model gcn --beta 2
```
