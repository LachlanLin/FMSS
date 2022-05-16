import numpy as np
import torch
from torch.utils.data import DataLoader
from model.multi_vae import MultiVAE
from dataset import ClientsSampler, ClientsDataset, TestDataset
from metric import Recall_Precision_F1_OneCall_at_k_batch, NDCG_binary_at_k_batch, AUC_at_k_batch


class Clients:
    def __init__(self, args):
        self.n = args.n
        self.m = args.m
        self.c = args.c
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MultiVAE([args.hiddenDim, args.m])
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.clients_data = ClientsDataset(args.path + args.train_data, args.n, args.m)

    def FMSS_perturb(self, clients_grads, x):
        for uid in range(x.shape[0]):
            clients_to_send = np.random.choice(x.shape[0], self.c, replace=False)
            for suid in clients_to_send:
                for name in clients_grads[uid]:
                    random_nums = np.random.random_sample(clients_grads[uid][name].shape)
                    random_nums = random_nums.astype(np.float32)
                    random_nums = torch.from_numpy(random_nums)
                    random_nums = random_nums.to(self.device)
                    clients_grads[uid][name] -= random_nums
                    clients_grads[suid][name] += random_nums

        return clients_grads

    def train(self, uids, model_param_state_dict, anneal):
        # receive model parameters from the server
        self.model.load_state_dict(model_param_state_dict)
        x = []
        for uid in uids:
            x.append(self.clients_data[uid].view(1, -1))
        x = torch.cat(x, 0)
        x = x.to(self.device)
        # each client computes gradients using its private data
        clients_grads = {}
        for uid in range(x.shape[0]):
            x_u = x[uid].view(1, -1)
            recon_batch, mu, logvar = self.model(x_u)
            loss = self.model.loss_function(recon_batch, x_u, mu, logvar, anneal)
            self.optimizer.zero_grad()
            loss.backward()
            grad_u = {}
            for name, param in self.model.named_parameters():
                grad_u[name] = param.grad.detach().clone()
            clients_grads[uid] = grad_u
        # perturb the original gradients
        perturb_grads = self.FMSS_perturb(clients_grads, x.cpu().numpy())
        # send the gradients of each client to the server
        return perturb_grads


class Server:
    def __init__(self, args, clients):
        self.clients = clients
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.total_anneal_steps = 200000
        self.anneal_cap = 0.2
        self.log_interval = 100
        self.batch_size = args.batch_size
        self.update_count = 0.0
        self.test_data = DataLoader(
            TestDataset(args.path + args.train_data, args.path + args.test_data, args.n, args.m),
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False)
        self.model = MultiVAE([args.hiddenDim, args.m])
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def aggregate_gradients(self, clients_grads):
        clients_num = len(clients_grads)
        aggregated_gradients = {}
        for uid, grads_dict in clients_grads.items():
            for name, grad in grads_dict.items():
                if name in aggregated_gradients:
                    aggregated_gradients[name] = aggregated_gradients[name] + grad / clients_num
                else:
                    aggregated_gradients[name] = grad / clients_num

        for name, param in self.model.named_parameters():
            if param.grad is None:
                param.grad = aggregated_gradients[name].detach().clone()
            else:
                param.grad += aggregated_gradients[name]

    def train(self):
        best_ndcg = -np.inf
        best_epoch = 0
        patience = self.early_stop
        for epoch in range(self.epochs):
            # train phase
            self.model.train()
            uid_seq = DataLoader(ClientsSampler(self.clients.n), batch_size=self.batch_size, shuffle=True)
            for uids in uid_seq:
                # sample clients to train the model
                if self.total_anneal_steps > 0:
                    anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
                else:
                    anneal = self.anneal_cap
                self.update_count += 1
                self.optimizer.zero_grad()
                # send the model to the clients and let them start training
                clients_grads = self.clients.train(uids, self.model.state_dict(), anneal)
                # aggregate the received gradients
                self.aggregate_gradients(clients_grads)
                # update the model
                self.optimizer.step()

            # evaluate phase
            ndcg5_list = []
            recall5_list = []
            precision5_list = []
            f1_list = []
            oneCall_list = []
            auc_list = []

            self.model.eval()
            with torch.no_grad():
                for x, test_x in self.test_data:
                    x = x.to(self.device)
                    recon_batch, mu, logvar = self.model(x)
                    recon_batch = recon_batch.cpu().numpy()
                    recon_batch[x.cpu().numpy().nonzero()] = -np.inf
                    test_x = test_x.detach().numpy()
                    n_5 = NDCG_binary_at_k_batch(recon_batch, test_x, 5)
                    r_5, p_5, f_5, o_5 = Recall_Precision_F1_OneCall_at_k_batch(recon_batch, test_x, 5)
                    auc_b = AUC_at_k_batch(x.cpu().numpy(), recon_batch, test_x)
                    ndcg5_list.append(n_5)
                    recall5_list.append(r_5)
                    precision5_list.append(p_5)
                    f1_list.append(f_5)
                    oneCall_list.append(o_5)
                    auc_list.append(auc_b)

            ndcg5_list = np.concatenate(ndcg5_list)
            recall5_list = np.concatenate(recall5_list)
            precision5_list = np.concatenate(precision5_list)
            f1_list = np.concatenate(f1_list)
            oneCall_list = np.concatenate(oneCall_list)
            auc_list = np.concatenate(auc_list)

            ndcg5_list[np.isnan(ndcg5_list)] = 0
            ndcg5 = np.mean(ndcg5_list)
            recall5_list[np.isnan(recall5_list)] = 0
            recall5 = np.mean(recall5_list)
            precision5_list[np.isnan(precision5_list)] = 0
            precision5 = np.mean(precision5_list)
            f1_list[np.isnan(f1_list)] = 0
            f1 = np.mean(f1_list)
            oneCall_list[np.isnan(oneCall_list)] = 0
            oneCAll = np.mean(oneCall_list)
            auc_list[np.isnan(auc_list)] = 0
            auc = np.mean(auc_list)

            print(
                "Epoch: {:3d} | Pre@5: {:5.4f} | Rec@5: {:5.4f} | F1@5: {:5.4f} | NDCG@5: {:5.4f} | 1-call@5: {:5.4f} | AUC: {:5.4f}".format(
                    epoch + 1, precision5, recall5, f1, ndcg5, oneCAll, auc), flush=True)

            if ndcg5 > best_ndcg:
                best_ndcg = ndcg5
                best_epoch = epoch + 1
                patience = self.early_stop
            else:
                patience -= 1
                if patience == 0:
                    break
        print('epoch of best ndcg@5({:5.4f})'.format(best_ndcg), best_epoch, flush=True)
