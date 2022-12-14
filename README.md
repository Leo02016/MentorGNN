# Required packages
* PyTorch
* torch_geometric
* pandas
* sklearn

# Unzip data sets 
For citeseer data set and pubmed dataset, please unzip the files [citeseer file](https://github.com/Leo02016/MentorGNN/blob/main/data/citeseer/preprocessed_data.zip) and [Pubmed file](https://github.com/Leo02016/MentorGNN/blob/main/data/pubmed/preprocessed_data.zip) first. 


# Environment and Installation:
1. conda env create -f environment.yml
2. conda activate MentorGNN

# Command
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
python main.py --gpu 0 --graph_src cora+pubmed+citeseer  --graph_dst reddit1 --dropout 0.5 --lr 0.003 --patience 1000 --base_model gcn --beta 2
```

Multi-graph knowledge transfer: cora+pubmed+citeseer-> reddit, where GAT is used as backbone.
```
python main.py --gpu 0 --graph_src cora+pubmed+citeseer  --graph_dst reddit1 --dropout 0.5 --lr 0.003 --patience 1000 --base_model gat --beta 2
```

# Reference
@inproceedings{DBLP:conf/cikm/ZhouZF0H22,
  author    = {Dawei Zhou and
               Lecheng Zheng and
               Dongqi Fu and
               Jiawei Han and
               Jingrui He},
  editor    = {Mohammad Al Hasan and
               Li Xiong},
  title     = {MentorGNN: Deriving Curriculum for Pre-Training GNNs},
  booktitle = {Proceedings of the 31st {ACM} International Conference on Information
               {\&} Knowledge Management, Atlanta, GA, USA, October 17-21, 2022},
  pages     = {2721--2731},
  publisher = {{ACM}},
  year      = {2022}
}
