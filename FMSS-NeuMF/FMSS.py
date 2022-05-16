import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from model.SharedNeuMF import SharedNeuMF
from dataset import ClientsDataset, ClientsSampler, TestDataset


class Clients:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rho = args.rho
        self.c = args.c
        self.n = args.n
        self.m = args.m
        self.model = SharedNeuMF(args.n, args.m, args.embed_dim)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.clients_data = ClientsDataset(args.path + args.train_data)
        self.gmf_user_embedding = torch.zeros([args.n, args.embed_dim])
        nn.init.kaiming_normal_(self.gmf_user_embedding, mode='fan_out')
        self.gmf_user_embedding.requires_grad = True
        self.ncf_user_embedding = torch.zeros([args.n, args.embed_dim])
        nn.init.kaiming_normal_(self.ncf_user_embedding, mode='fan_out')
        self.ncf_user_embedding.requires_grad = True
        self.gmf_user_embedding.to(self.device)
        self.ncf_user_embedding.to(self.device)
        self.embed_optimizer = torch.optim.Adam([self.gmf_user_embedding, self.ncf_user_embedding], lr=args.lr)
        self.first_iter = [True] * args.n
        self.fake_items = {}

    def FMSS_perturb(self, clients_grads):
        users = np.array(list(clients_grads.keys()))
        for uid in users:
            clients_to_send = np.random.choice(users, self.c, replace=False)
            i_u, _ = self.clients_data[uid]
            i_u = i_u.numpy()
            fake_candidate = np.setdiff1d(np.arange(self.m), i_u)
            fake_num = int(min(self.rho * i_u.shape[0], fake_candidate.shape[0]))
            if self.first_iter[uid]:
                fake_items = np.random.choice(fake_candidate, fake_num, replace=False)
                self.fake_items[uid] = fake_items
                self.first_iter[uid] = False
            else:
                fake_items = self.fake_items[uid]
            for suid in clients_to_send:
                for name in clients_grads[uid]:
                    # only apply fake marks to the ID-sensitive parameters
                    if name == 'gmf_item_embedding.weight' or name == 'ncf_item_embedding.weight':
                        sent_items = np.zeros([self.m])
                        sent_items[fake_items] = 1.0
                        sent_items[i_u] = 1.0
                        random_nums = np.random.random_sample(clients_grads[uid][name].shape) * sent_items.reshape(-1,
                                                                                                                   1)
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
        self.embed_optimizer.zero_grad()
        for uid in uids:
            uid = uid.item()
            iids, ratings = self.clients_data[uid]
            iids = iids.to(self.device)
            ratings = ratings.to(self.device)
            ncf_u_embed = self.ncf_user_embedding[uid].view(1, -1).expand([iids.shape[0], -1]).to(self.device)
            gmf_u_embed = self.gmf_user_embedding[uid].view(1, -1).expand([iids.shape[0], -1]).to(self.device)
            preds = self.model(ncf_u_embed, gmf_u_embed, iids)
            loss = self.model.loss_function(preds, ratings)
            self.optimizer.zero_grad()
            loss.backward()
            grad_u = {}
            for name, param in self.model.named_parameters():
                grad_u[name] = param.grad.detach().clone()
            clients_grads[uid] = grad_u
        self.embed_optimizer.step()
        # perturb the original gradients
        perturb_grads = self.FMSS_perturb(clients_grads)
        # send the gradients of each client to the server
        return perturb_grads


class Server:
    def __init__(self, args, clients):
        self.clients = clients
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_data = DataLoader(TestDataset(args.path + args.test_data),
                                    batch_size=args.batch_size,
                                    num_workers=2,
                                    pin_memory=True,
                                    shuffle=False)
        self.model = SharedNeuMF(args.n, args.m, args.embed_dim)
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
        best_rmse = np.inf
        best_epoch = 0
        patience = self.early_stop
        for epoch in range(self.epochs):
            # train phase
            self.model.train()
            uid_seq = DataLoader(ClientsSampler(self.clients.n), batch_size=self.batch_size, shuffle=False)
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
            ae_list = []
            se_list = []
            with torch.no_grad():
                for uids, iids, ratings in self.test_data:
                    uids = uids.to(self.device)
                    iids = iids.to(self.device)
                    ncf_u_embed = self.clients.ncf_user_embedding[uids].to(self.device)
                    gmf_u_embed = self.clients.gmf_user_embedding[uids].to(self.device)
                    preds = self.model(ncf_u_embed, gmf_u_embed, iids)
                    preds = preds.detach().cpu().numpy()
                    preds = np.where(preds > 5, 5, preds)
                    preds = np.where(preds < 1, 1, preds)
                    ratings = ratings.detach().cpu().numpy()
                    ae_batch = np.abs(preds - ratings)
                    se_batch = np.square(preds - ratings)
                    ae_list.append(ae_batch)
                    se_list.append(se_batch)

            ae_list = np.concatenate(ae_list)
            se_list = np.concatenate(se_list)
            mae = np.mean(ae_list)
            rmse = np.sqrt(np.mean(se_list))

            print("Epoch: {:3d} | MAE: {:5.4f} | RMSE: {:5.4f}".format(epoch + 1, mae, rmse))
            if rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch + 1
                patience = self.early_stop
            else:
                patience -= 1
                if patience == 0:
                    break
        print('epoch of best RMSE({:5.4f})'.format(best_rmse), best_epoch, flush=True)
