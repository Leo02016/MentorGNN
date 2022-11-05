import numpy as np
import torch
import os
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix
import pickle as pkl


class Graph(object):
    # graph statistics
    adj = None
    feats = None
    train_idx = []
    test_idx = []
    edge_src = []
    edge_dst = []
    num_nodes = 0
    num_feats = 0
    index = None
    attr_id = 1


class Opt(object):
    in_dim = 100
    h_dim = 50
    out_dim = 10
    num_bases = None
    num_hidden_layers = 2
    link_list = []
    link_list_label = []
    name = 'target'


def label_encoding(label, num_class):
    unique_label_map = {}
    unique_label = np.unique(label)
    for i in range(num_class):
        unique_label_map[unique_label[i]] = i
    arr = np.zeros((len(label), num_class))
    for i in range(len(label)):
        arr[i, unique_label_map[label[i]]] = 1.0
    return arr


# sampling negative edge
def edge_negative_sampling(edge_list, graph):
    index = np.random.permutation(np.arange(edge_list[0, :].shape[0], dtype=int))
    num = int(edge_list[0, :].shape[0] * 0.1)
    graph.link_list = np.array(edge_list[:, index[:num]]).T
    adj = csr_matrix((np.ones(edge_list.shape[1]), (edge_list[0, :], edge_list[1, :])),
                     shape=(graph.num_nodes, graph.num_nodes))
    count = 0
    i, j = 0, 0
    negative_list = []
    while count < num:
        if i == j or adj[i, j] == 1.0:
            i = np.random.randint(0, adj.shape[0])
            j = np.random.randint(0, adj.shape[0])
        else:
            negative_list.append([i, j])
            count += 1
    graph.link_list = np.concatenate([graph.link_list, np.array(negative_list)])
    graph.link_list_label = np.concatenate([np.ones(num), np.zeros(num)])
    index = np.random.permutation(np.arange(graph.link_list.shape[0], dtype=int))
    graph.link_list = graph.link_list[index, :]
    graph.link_list_label = graph.link_list_label[index]


def graph_K_fold_split(args, graph, index, graph_data, fold=10, stage='source', use_cuda=True):
    if os.path.exists('./data/{}/preprocessed_data.pkl'.format(graph_data)):
        data = pkl.load(open('./data/{}/preprocessed_data.pkl'.format(graph_data), "rb"))
        graph.feats = torch.FloatTensor(data['feats'])
        class_num = np.unique(data['labels']).shape[0]
        graph.labels = torch.LongTensor(label_encoding(data['labels'].reshape(-1, ), class_num)).max(dim=1)[1]
        graph.num_nodes = graph.feats.shape[0]
        graph.edge_src = data['edge_list'][:, 0]
        graph.edge_dst = data['edge_list'][:, 1]
        graph.num_feats = data['feats'].shape[1]
        if stage == 'source':
            graph.train_idx = data['train_idx'][index]
            graph.test_idx = data['test_idx'][index]
            graph.val_idx = data['val_idx'][index]
        else:
            graph.train_idx = data['train_idx'][index].reshape(-1, )
            graph.test_idx = data['test_idx'][index].reshape(-1, )
            graph.val_idx = data['val_idx'][index].reshape(-1, )
            edge_list = data['edge_list'].T
            edge_negative_sampling(edge_list, graph)
    else:
        graph.feats = pkl.load(open('./data/{}/'.format(graph_data) + graph_data + ".x.pkl", 'rb'))
        graph.labels = pkl.load(open('./data/{}/'.format(graph_data) + graph_data + ".y.pkl", 'rb'))
        with open('./data/{}/'.format(graph_data) + graph_data + '.edgelist', 'r') as f:
            graph.num_nodes = int(next(f).strip())
            edge_list = np.array(list(map(lambda x: x.strip().split(' '), f.readlines())), dtype=np.int)
            graph.edge_src = edge_list[:, 0]
            graph.edge_dst = edge_list[:, 1]
        kf = StratifiedKFold(n_splits=fold, shuffle=True)
        train, val, test = {}, {}, {}
        count = 0
        for train_index, test_index in kf.split(graph.feats, graph.labels):
            test[count] = train_index
            kf2 = StratifiedKFold(n_splits=4, shuffle=True)
            for train_index, val_index in kf2.split(graph.feats[test_index], graph.labels[test_index]):
                train[count] = test_index[val_index]
                val[count] = test_index[train_index]
                break
            count += 1
            if count >= fold:
                break
        data = {'feats': graph.feats, 'labels': graph.labels, 'edge_list': edge_list,
                'train_idx': train, 'val_idx': val, 'test_idx': test}
        pkl.dump(data, open('./data/{}/preprocessed_data.pkl'.format(graph_data), 'wb'))
        graph.num_feats = graph.feats.shape[1]
        graph.train_idx = data['train_idx'][index]
        graph.test_idx = data['test_idx'][index]
        graph.val_idx = data['val_idx'][index]
        graph.num_feats = data['feats'].shape[1]
        graph.feats = torch.FloatTensor(graph.feats)
        class_num = np.unique(data['labels']).shape[0]
        graph.labels = torch.LongTensor(label_encoding(graph.labels, class_num)).max(dim=1)[1]
    graph.adj = np.zeros((graph.num_nodes, graph.num_nodes))
    graph.adj = graph.adj + np.identity(graph.num_nodes)
    for i, j in data['edge_list']:
        graph.adj[i, j] = 1
        graph.adj[j, i] = 1
    graph.adj = torch.FloatTensor(graph.adj)
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        graph.feats = graph.feats.cuda()
        graph.labels = graph.labels.cuda()
        graph.adj = graph.adj.cuda()
    if stage == 'target':
        graph.link_labels = torch.tensor(graph.link_list_label).reshape(-1, 1)
        graph.attr_id = graph.attr_id
        graph.link_index = graph.link_list
        graph.link_index = graph.link_list
        if use_cuda:
            graph.link_labels = graph.link_labels.cuda()
    return graph


def load_graph(args, idx):
    graph_src = args.graph_src.split("+")
    source_graph = []
    source_opt = []
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    i = 0
    for item in graph_src:
        graph_1 = Graph()
        graph_1 = graph_K_fold_split(args, graph_1, idx, item, fold=5, use_cuda=use_cuda)
        source_graph.append(graph_1)
        opt = Opt()
        opt.in_dim = graph_1.num_feats
        opt.num_nodes = graph_1.num_nodes
        opt.num_graph = len(graph_src)
        opt.class_num = torch.unique(graph_1.labels).shape[0]
        opt.name = 'source'
        source_opt.append(opt)
        source_graph[i].edge_list = torch.tensor(np.stack([source_graph[i].edge_src, source_graph[i].edge_dst])).long()
        if use_cuda:
            source_graph[i].edge_list = source_graph[i].edge_list.cuda()
        i = i + 1
    # graph in the target domain
    target_graph = Graph()
    target_graph = graph_K_fold_split(args, target_graph, idx, args.graph_dst, 5, stage='target', use_cuda=use_cuda)
    opt = Opt()
    opt.in_dim = target_graph.num_feats
    opt.num_nodes = target_graph.num_nodes
    opt.num_graph = len(graph_src)
    opt.class_num = torch.unique(target_graph.labels).shape[0]
    opt.name = 'target'
    target_graph.edge_list = torch.tensor(np.stack([target_graph.edge_src, target_graph.edge_dst])).long()
    if use_cuda:
        target_graph.edge_list = target_graph.edge_list.cuda()
    return source_graph, target_graph, source_opt, opt
