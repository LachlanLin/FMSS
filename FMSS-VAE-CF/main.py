import argparse
from FMSS import Server, Clients


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hiddenDim', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--early_stop', type=int, default=50, help='Patience for early stop')
    parser.add_argument('--path', type=str, default='./ML100K/', help='Data path')
    parser.add_argument('--train_data', type=str, default='ML100K-copy1-train', help='train dataset')
    parser.add_argument('--test_data', type=str, default='ML100K-copy1-test', help='test dataset')
    parser.add_argument('--n', type=int, default=943, help='Users num')
    parser.add_argument('--m', type=int, default=1682, help='Items num')
    parser.add_argument('--c', type=int, default=0, help='c')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # construct clients
    clients = Clients(args)
    # construct the server
    server = Server(args, clients)
    server.train()
