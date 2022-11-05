import torch.nn as nn
import torch
from copy import deepcopy as copy
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False, dropout=0.0):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=1)
        return y


class GAT_block(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT_block, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        emb = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        return emb


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class simple_GCN_model(nn.Module):
    def __init__(self, source_opt, dropout=0.5, use_cuda=False):
        super(simple_GCN_model, self).__init__()
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.gc1 = GCNConv(source_opt.in_dim, source_opt.h_dim)
        hidden_layers = GCNConv(source_opt.h_dim, source_opt.h_dim)
        self.gc2 = self.clones(hidden_layers, source_opt.num_hidden_layers)
        self.gc3 = nn.Linear(source_opt.h_dim, source_opt.class_num)
        self.dropout = dropout

    def clones(self, module, N):
        return nn.ModuleList([copy(module) for _ in range(N)])

    def forward(self, graph, edge_list, train_idx):
        h = self.gc1(graph.feats, edge_list)
        h = F.dropout(h, self.dropout, training=self.training)
        for layer in self.gc2:
          h = F.relu(layer(h, edge_list))
        h = F.normalize(h, p=2, dim=1)
        prediction = F.log_softmax(self.gc3(h), dim=1)
        criterion = F.nll_loss
        clf_loss = criterion(prediction[train_idx], graph.labels[train_idx])
        return clf_loss, prediction, h


class DiffPoolingBlock(torch.nn.Module):
    def __init__(self, in_dim, n_clusters):
        super().__init__()
        self.gcn = GraphConv(in_dim, n_clusters, normalize_embedding=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feats, adj):
        return self.softmax(self.gcn(feats, adj))


class MentorGNN(nn.Module):
    def __init__(self, source_opt, target_opt, dropout=0.5, super_node=[500, 100], use_cuda=False, coef=1,
                 base_model='gcn', alpha=0.1, beta=1, gamma=0.1, n_head=1, n_hid=50, gpu=0):
        super(MentorGNN, self).__init__()
        self.device = torch.device('cuda:{}'.format(gpu) if use_cuda else 'cpu')
        self.dropout = dropout
        self.use_cuda = use_cuda
        out_dim = target_opt.h_dim
        class_num = target_opt.class_num
        num_graph = source_opt[0].num_graph
        self.num_graph = num_graph
        self.num_class = class_num
        self.diff_pool_layer_s = nn.ModuleList(self.build_diff_pool_layer(source_opt[i], super_node)
                                               for i in range(num_graph))
        self.diff_pool_layer_t = self.build_diff_pool_layer(target_opt, super_node)
        self.transation_fun = nn.ModuleList(self.s2t_mapping_block(source_opt[i].h_dim, super_node)
                                         for i in range(num_graph))
        self.link_mlp = nn.Sequential(nn.Linear(2*(1+num_graph)*out_dim, out_dim), nn.ReLU(inplace=True),
                                      nn.BatchNorm1d(out_dim), nn.Linear(out_dim, 1), nn.Sigmoid())
        self.label_mlp = GraphAttentionLayer(target_opt.h_dim * (num_graph+1), class_num, dropout=dropout,
                                             alpha=0.2, concat=False)
        # The weight for different source graph. default weight is 1 for all source graphs.
        self.weight = [1 for _ in range(num_graph)]
        assert target_opt.h_dim == n_hid * n_head
        if base_model == 'gcn':
            self.gcn = simple_GCN_model(target_opt, dropout)
            self.gcn.name = 'gcn'
        else:
            self.gcn = GAT_block(target_opt.in_dim, n_hid, dropout, 0.2, n_head)
            self.gcn.name = 'gat'
        # coefficient for knowledge transfer loss, node classification loss, link prediction loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_curriculum = False
        self.lambda_1 = coef
        self.lambda_2 = 10

    def s2t_mapping_block(self, out_dim, super_node):
        return nn.ModuleList(nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(inplace=True))
                             for _ in range(len(super_node)))

    def build_diff_pool_layer(self, obj, super_node):
        P = []
        for i in range(len(super_node)):
            P.append(DiffPoolingBlock(obj.h_dim, super_node[i]))
        # for i, corsen_matrix in enumerate(P):
        #     self.add_module('{}_{}'.format(obj.name, i), corsen_matrix)
        P = nn.ModuleList(P)
        return P

    def cross_entropy(self, in_, target, eps=1e-15):
        return (-(target * torch.log(in_+eps)) - ((1 - target)*torch.log(1-in_+eps))).mean(dim=1)

    def curriculum(self, loss, lambda_1=1, lambda_2=1):
        if lambda_2 == 0:
            return loss <= lambda_1
        else:
            find_max = torch.max(torch.stack([torch.zeros_like(loss), 1 - (loss - lambda_1)/lambda_2]), dim=0).values
            find_min = torch.min(torch.stack([torch.ones_like(loss), find_max]), dim=0).values
            return find_min

    def curriculum_regularizer(self, weight, lambda_1=1, lambda_2=1):
        return 1/2*lambda_2 *weight**2 - (lambda_1 + lambda_2)*weight

    def forward(self, g_s, g_t, emb_source, prediction=True, train_index=None, coef3=0.95, coef2=0.95):
        if self.gcn.name == 'gcn':
            _, _, emb_target = self.gcn(g_t, g_t.edge_list, g_t.train_idx)
        else:
           emb_target = self.gcn(g_t.feats, g_t.adj)
        emb_t = [[] for _ in range(len(self.diff_pool_layer_t) + 1)]
        emb_t[0] = emb_target
        # emb_s is a 4d tensor: num_graph by num_layer by num_super_node by feature_dim
        emb_s = [[[] for _ in range(len(self.diff_pool_layer_s[i]) + 1)] for i in range(self.num_graph)]
        P_s = [[[] for _ in range(len(self.diff_pool_layer_s[i]))] for i in range(self.num_graph)]
        for i in range(self.num_graph):
            adj_s = [None for _ in range(len(self.diff_pool_layer_s[i]))]
            adj_s[0] = g_s[i].adj
            emb_s[i][0] = emb_source[i]
            for j in range(len(self.diff_pool_layer_s[i])):
                # get the differentiable pooling matrix
                P_s[i][j] = self.diff_pool_layer_s[i][j](emb_s[i][j], adj_s[j])
                # update the feature matrix of the coarse graph
                emb_s[i][j + 1] = torch.mm(P_s[i][j].T, emb_s[i][j])
                if j < len(self.diff_pool_layer_s[i]) - 1:
                    # update the adjacency matrix for the coarse graph
                    adj_s[j + 1] = torch.mm(torch.mm(P_s[i][j].T, adj_s[j]), P_s[i][j])
        adj_t = [None for _ in range(len(self.diff_pool_layer_t))]
        P_t = [[] for _ in range(len(self.diff_pool_layer_t))]
        adj_t[0] = g_t.adj
        for i in range(len(self.diff_pool_layer_t)):
            P_t[i] = self.diff_pool_layer_t[i](emb_t[i], adj_t[i])
            # update the feature matrix of the coarse graph
            emb_t[i + 1] = torch.mm(P_t[i].T, emb_t[i])
            if i < len(self.diff_pool_layer_t) - 1:
                # update the adjacency matrix for the coarse graph
                adj_t[i + 1] = torch.mm(torch.mm(P_t[i].T, adj_t[i]), P_t[i])
        transfered_knowledge = [[[] for _ in range(len(emb_s[i]) - 1)] for i in range(self.num_graph)]
        for i in range(self.num_graph):
            for j in range(len(emb_s[i]) - 1):
                transfered_knowledge[i][j] = self.transation_fun[i][j](emb_s[i][j + 1])
        if not prediction:
            # knowledge transfer loss
            l1 = 0
            for i in range(self.num_graph):
                # transferred knowledge loss
                l1 += self.weight[i] * (transfered_knowledge[i][-1] - emb_t[-1]).norm(2) / emb_t[-1].shape[0]
            l1 = l1/self.num_graph
            # node classification loss
            l2 = 0
            # final_emb = torch.cat([emb_target, torch.stack(transfered_knowledge, dim=1)])
            # hid = F.dropout(emb_target, self.dropout, training=self.training)
            # label_prediction = F.softmax(F.elu(self.label_mlp(final_emb, g_t.adj)), dim=1)
            # l2 = self.cross_entropy(label_prediction[train_index], F.one_hot(g_t.labels[train_index], self.num_class))
            if self.use_curriculum:
                w2 = self.curriculum(l2, lambda_1=coef2, lambda_2=self.lambda_1)
            else:
                w2 = torch.tensor(1)
            # link prediction loss
            l3 = 0
            for i in range(len(self.diff_pool_layer_t)):
                reconstruct_knowledge = torch.cat([transfered_knowledge[j][i] for j in range(self.num_graph)], dim=1)
                for j in range(i+1, 0, -1):
                    reconstruct_knowledge = torch.matmul(P_t[j-1], reconstruct_knowledge)
                knowledge = torch.cat([reconstruct_knowledge, emb_t[0]], dim=1)
                # node classification loss
                label_prediction = F.softmax(F.elu(self.label_mlp(knowledge, g_t.adj)), dim=1)
                l2 += self.cross_entropy(label_prediction[train_index],
                                        F.one_hot(g_t.labels[train_index], self.num_class))
                # link prediction
                edge_pair = torch.cat((knowledge[g_t.link_index[:, 0], :], knowledge[g_t.link_index[:, 1], :]), dim=1)
                prediction = F.softmax(self.link_mlp(edge_pair), dim=1)
                l3 += self.cross_entropy(prediction, g_t.link_labels)
                if self.use_curriculum:
                    w3 = self.curriculum(l3, lambda_1=coef3, lambda_2=self.lambda_2)
                else:
                    w3 = torch.tensor(1)
            final_loss = self.alpha * l1 + self.beta * (w2*l2).mean() + self.gamma * (w3*l3).mean()
            if self.use_curriculum:
                final_loss += self.beta * self.curriculum_regularizer(w2, coef2, self.lambda_1).mean() + \
                              self.gamma * self.curriculum_regularizer(w3, coef3, self.lambda_2).mean()
            return final_loss, label_prediction
        else:
            for i in range(len(self.diff_pool_layer_t)):
                reconstruct_knowledge = torch.cat([transfered_knowledge[j][i] for j in range(self.num_graph)], dim=1)
                for j in range(i+1, 0, -1):
                    reconstruct_knowledge = torch.matmul(P_t[j-1], reconstruct_knowledge)
                knowledge = torch.cat([reconstruct_knowledge, emb_t[0]], dim=1)
                label_prediction = F.softmax(F.elu(self.label_mlp(knowledge, g_t.adj)), dim=1)
                break
            return 0, label_prediction
