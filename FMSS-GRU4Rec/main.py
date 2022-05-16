import argparse
from FMSS import Server, Clients


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=-1,
                        help='embedding vector dimensionality, -1 for one-hot embeddings.')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden layer dimensionality.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate for the output of GRU')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--early_stop', type=int, default=50, help='Patience for early stop')
    parser.add_argument('--path', type=str, default='./ML100K/', help='Data path')
    parser.add_argument('--train_data', type=str, default='train_5.csv', help='train dataset')
    parser.add_argument('--test_data', type=str, default='test_5.csv', help='test dataset')
    parser.add_argument('--rho', type=int, default=0, help='rho')
    parser.add_argument('--c', type=int, default=0, help='c')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('\n'.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]))
    # construct clients
    clients = Clients(args)
    # construct the server
    server = Server(args, clients)
    server.train()
