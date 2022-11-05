import argparse
import time
from model import simple_GCN_model, MentorGNN
from sklearn.metrics import accuracy_score
import random
from data_loader import *


def main(args, source_graph, target_graph, source_opt, target_opt, coefficient=0.4):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if len(source_graph) != source_opt[0].num_graph:
        raise ValueError('Length of source_graph does not match the number of graph in source domain.')

    # --------- train gcn to get the embedding for the graph in source domain and store in 'final_emb_s' list --------#
    final_emb_s = []
    for i in range(source_opt[0].num_graph):
        model = simple_GCN_model(source_opt[i], args.dropout, use_cuda=use_cuda).float()
        if use_cuda:
            model.cuda()
            torch.cuda.manual_seed(args.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("start pre-training graph in source domain.....")
        model.train()
        edge_list = torch.tensor(np.stack([source_graph[i].edge_src, source_graph[i].edge_dst])).long()
        if use_cuda:
            edge_list = edge_list.cuda()
        for epoch in range(1, 1501):
            optimizer.zero_grad()
            loss, pred_s, _ = model(source_graph[i], edge_list,  source_graph[i].train_idx)
            loss.backward()
            optimizer.step()
        _, pred_s, emb_s = model(source_graph[i], edge_list, source_graph[i].test_idx)
        final_emb_s.append(emb_s.detach())

    # ---------- based on the fixed embedding, update the mapping function ------------- #
    model_kt = MentorGNN(source_opt, target_opt, args.dropout, use_cuda=use_cuda, base_model=args.base_model,
                         gpu=args.gpu, alpha=args.alpha, beta=args.beta, gamma=args.gamma).float()
    if use_cuda:
        model_kt.cuda()
    optimizer = torch.optim.Adam(model_kt.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_idx = target_graph.train_idx
    test_idx = target_graph.test_idx
    val_idx = target_graph.val_idx
    emb_s = final_emb_s
    coef = 1000000
    coef2 = 1000000
    meter = {'val_acc': 0, 'patience': 0}
    t1 = time.time()
    t_total = time.time()
    for epoch in range(1, args.epochs+1):
        optimizer.zero_grad()
        loss, prediction = model_kt(source_graph, target_graph, emb_s, train_index=train_idx, prediction=False,
                                    coef2=coef2, coef3=coef)
        logits = torch.max(prediction, dim=1)[1].cpu().numpy()
        loss.backward()
        optimizer.step()
        val_acc = accuracy_score(target_graph.labels[val_idx].cpu().numpy(), logits[val_idx])
        train_acc = accuracy_score(target_graph.labels[train_idx].cpu().numpy(), logits[train_idx])
        if meter['val_acc'] < val_acc:
            meter['val_acc'] = val_acc
            torch.save(model_kt.state_dict(), 'best_model_{}_{}.pkl'.format(args.graph_src, args.graph_dst))
            meter['patience'] = 0
        if epoch == 5:
            coef = 0.99
            coef2 = coefficient
        if epoch % 50 == 0:
            print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Val Acc: {:.4f} | time: {:.4f}".format(
                epoch, train_acc, loss.item(), val_acc, time.time()-t1))
            t1 = time.time()
            coef -= 0.01
            coef2 += 0.01
        meter['patience'] += 1
        if meter['patience'] >= args.patience:
            break
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('Loading best model...')
    model_kt.load_state_dict(torch.load('best_model_{}_{}.pkl'.format(args.graph_src, args.graph_dst)))
    model_kt.eval()
    _, prediction = model_kt(source_graph, target_graph, emb_s, train_index=test_idx, prediction=True)
    logits = torch.max(prediction, dim=1)[1].cpu().numpy()
    test_acc = accuracy_score(target_graph.labels[test_idx].cpu().numpy(), logits[test_idx])
    print("Test Accuracy: {:.4f}".format(test_acc))
    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MentorGNN')
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--graph_src", type=str, default='cora', help="graph in the source domain")
    parser.add_argument("--graph_dst", type=str, default='graph2', help="graph in the target domain")
    parser.add_argument("--base_model", type=str, default='gcn', help="base model (gcn or gat)")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=3000, help="number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay rate")
    parser.add_argument("--use-self-loop", default=False, action='store_true', help="include self feature as a special relation")
    parser.add_argument('--seed', type=int, default=26, help='Random seed.')
    parser.add_argument('--patience', type=int, default=1000, help='Patience')
    parser.add_argument('--alpha', type=int, default=0.1, help='coefficient for knowledge transfer loss')
    parser.add_argument('--beta', type=int, default=2, help='coefficient for node classification loss')
    parser.add_argument('--gamma', type=int, default=0.1, help='coefficient for link prediction loss')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args)
    avg_acc = []
    for i in range(5):
        print('\n\n------- begin {}-th round ---------'.format(i+1))
        source_graph, target_graph, source_opt, target_opt = load_graph(args, i)
        test_acc = main(args, source_graph, target_graph, source_opt, target_opt)
        avg_acc.append(test_acc)
    print('Average Accuracy: {:.4f} +/- {:.4f}'.format(np.mean(avg_acc), np.std(avg_acc)))
