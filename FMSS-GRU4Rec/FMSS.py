import numpy as np
import torch
from torch.utils.data import DataLoader
from model import GRU4Rec
from dataset import ClientsSampler, ClientsDataset, TestDataset
from metric import Recall_Precision_F1_OneCall_at_k_batch, NDCG_binary_at_k_batch, AUC_at_k_batch


class Clients:
    def __init__(self, args):
        self.neg_num = args.batch_size - 1  # the same as that of session-parallel mini-batches
        self.clients_data = ClientsDataset(args.path + args.train_data, self.neg_num)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GRU4Rec(args, self.clients_data.get_maxid())
        self.model.to(self.device)
        self.lr = args.lr
        self.rho = args.rho
        self.c = args.c
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr)
        self.first_iter = [True] * args.n
        self.fake_items = {}

    def FMSS_perturb(self, clients_grads):
        rho = max(self.rho - self.neg_num, 0)
        uids = list(clients_grads.keys())
        for uid in uids:
            clients_to_send = np.random.choice(uids, self.c, replace=False).tolist()
            seq = self.clients_data.seq[uid]
            fake_candidate = np.setdiff1d(self.clients_data.item_set, seq)
            fake_num = int(min(rho * seq.shape[0], fake_candidate.shape[0]))
            if self.first_iter[uid]:
                fake_items = np.random.choice(fake_candidate, fake_num, replace=False)
                self.fake_items[uid] = fake_items
                self.first_iter[uid] = False
            else:
                fake_items = self.fake_items[uid]
            for suid in clients_to_send:
                for name in clients_grads[uid]:
                    # only apply fake marks to the ID-sensitive parameters
                    if name == 'gru_layer.weight_ih_l0':
                        sent_items = np.zeros([clients_grads[uid][name].shape[1]])
                        sent_items[fake_items] = 1.0
                        sent_items[seq] = 1.0
                        random_nums = np.random.random_sample(clients_grads[uid][name].shape) * sent_items.reshape(1,
                                                                                                                   -1)
                    else:
                        random_nums = np.random.random_sample(clients_grads[uid][name].shape)
                    random_nums = random_nums.astype(np.float32)
                    random_nums = torch.from_numpy(random_nums)
                    random_nums = random_nums.to(self.device)
                    # for the convenience of programming, we send the entire tensor, i.e., random_nums,
                    # but actually only the non-zero value of it needs to be sent
                    clients_grads[uid][name] -= random_nums
                    clients_grads[suid][name] += random_nums

        return clients_grads

    def train(self, uids, model_param_state_dict):
        # receive model parameters from the server
        self.model.load_state_dict(model_param_state_dict)
        # each client computes gradients using the private data
        clients_grads = {}
        for uid in uids:
            uid = uid.item()
            input_seq, target_seq, input_len, neg_seq = self.clients_data[uid]
            input_seq = torch.from_numpy(input_seq).unsqueeze(0).to(self.device)
            target_seq = torch.from_numpy(target_seq).unsqueeze(0).to(self.device)
            neg_seq = torch.from_numpy(neg_seq).unsqueeze(0).to(self.device)
            input_len = torch.tensor(input_len).unsqueeze(0)
            seq_out = self.model(input_seq, input_len)
            padding_mask = (torch.not_equal(input_seq, 0)).float().unsqueeze(-1).to(self.device)
            loss = self.model.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
            self.optimizer.zero_grad()
            loss.backward()
            grad_u = {}
            for name, param in self.model.named_parameters():
                grad_u[name] = param.grad.detach().clone()
            clients_grads[uid] = grad_u
        # perturb the original gradients
        perturb_grads = self.FMSS_perturb(clients_grads)
        # send the gradients of each client to the server
        return perturb_grads

    def cal_popularity(self, user_list):
        n = len(user_list)
        item_counts = np.zeros([n, self.clients_data.get_maxid() + 1], dtype=np.int32)
        for i in range(n):
            uid = user_list[i]
            input_seq, target_seq, input_len, _ = self.clients_data[uid]
            item_counts[i, input_seq[:input_len]] = 1
            item_counts[i, target_seq[input_len - 1]] = 1
        perturbed_item_counts = self.popularity_perturb(item_counts)
        return perturbed_item_counts

    def popularity_perturb(self, item_counts):
        perturbed_item_counts = item_counts.copy()
        for uid in range(item_counts.shape[0]):
            clients_to_send = np.random.choice(item_counts.shape[0], self.c, replace=False)
            for suid in clients_to_send:
                x_u = item_counts[uid]
                fake_candidate = np.where(x_u == 0)[0]
                fake_candidate = fake_candidate[1:]  # remove 0 item ID
                fake_num = int(min(self.rho * x_u.sum(), fake_candidate.shape[0]))
                fake_items = np.random.choice(fake_candidate, fake_num, replace=False)
                fake_x = np.zeros_like(x_u)
                fake_x[fake_items] = 1
                sent_items = fake_x + x_u
                random_nums = np.random.randint(0, 3, x_u.shape) * sent_items
                perturbed_item_counts[uid] -= random_nums
                perturbed_item_counts[suid] += random_nums
        return perturbed_item_counts


class Server:
    def __init__(self, args, clients):
        self.clients = clients
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_data = DataLoader(
            TestDataset(args.path + args.train_data, args.path + args.test_data),
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False)
        self.model = GRU4Rec(args, self.clients.clients_data.get_maxid())
        self.model.to(self.device)
        for name, param in self.model.named_parameters():
            print(name)
            print(param.shape, param.device)
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=args.lr)

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
        uid_seq = DataLoader(
            ClientsSampler(self.clients.clients_data.get_user_set()),
            batch_size=self.batch_size,
            shuffle=True)
        # calculate the popularity of the items
        perturbed_item_counts_list = list()
        for uids in uid_seq:
            perturbed_item_counts = self.clients.cal_popularity(uids.tolist())
            perturbed_item_counts_list.append(perturbed_item_counts)
        item_popularity = np.concatenate(perturbed_item_counts_list).sum(axis=0)
        # send item popularity to all the clients
        self.clients.clients_data.set_popularity(item_popularity)

        # epochs
        best_ndcg = -np.inf
        best_epoch = 0
        patience = self.early_stop
        for epoch in range(self.epochs):
            # train phase
            self.model.train()
            for uids in uid_seq:
                # sample clients to train the model
                self.optimizer.zero_grad()
                # send the model to the clients and let them start training
                clients_grads = self.clients.train(uids, self.model.state_dict())
                # aggregate the received gradients
                self.aggregate_gradients(clients_grads)
                # update the model
                self.optimizer.step()

            # evaluate phase
            self.model.eval()
            ndcg5_list = []
            recall5_list = []
            precision5_list = []
            f1_list = []
            oneCall_list = []
            auc_list = []
            with torch.no_grad():
                for input_seq, input_len, train_vec, target_vec in self.test_data:
                    input_seq = input_seq.to(self.device)
                    pro = self.model(input_seq, input_len)
                    recon_batch = []
                    for i in range(input_seq.shape[0]):
                        recon_batch.append(pro[i, input_len[i] - 1, :].view(1, -1))
                    recon_batch = torch.cat(recon_batch)
                    recon_batch = recon_batch.detach().cpu().numpy()
                    train_vec = train_vec.numpy()
                    target_vec = target_vec.numpy()
                    recon_batch[train_vec.nonzero()] = -np.inf
                    n_5 = NDCG_binary_at_k_batch(recon_batch[:, 1:], target_vec[:, 1:], 5)
                    r_5, p_5, f_5, o_5 = Recall_Precision_F1_OneCall_at_k_batch(recon_batch[:, 1:], target_vec[:, 1:],
                                                                                5)
                    auc_b = AUC_at_k_batch(train_vec[:, 1:], recon_batch[:, 1:], target_vec[:, 1:])
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
