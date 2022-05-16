import torch
from torch import nn
from torch.nn import functional as F


class SharedNeuMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, dropout=0, init=None):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout)
        self.gmf_item_embedding = nn.Embedding(self.num_items, self.embed_dim)
        self.ncf_item_embedding = nn.Embedding(self.num_items, self.embed_dim)
        self.hidden_layer1 = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.hidden_layer2 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.output_layer = nn.Linear(self.embed_dim + self.embed_dim // 2, 1, bias=False)
        if init is not None:
            self.init_embedding(init)
        else:
            self.init_embedding(0)

    def init_embedding(self, init):
        nn.init.kaiming_normal_(self.ncf_item_embedding.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.gmf_item_embedding.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.hidden_layer1.weight, mode='fan_out', a=init)
        nn.init.constant_(self.hidden_layer1.bias, 0.0)
        nn.init.kaiming_normal_(self.hidden_layer2.weight, mode='fan_out', a=init)
        nn.init.constant_(self.hidden_layer2.bias, 0.0)
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_out', a=init)

    def forward(self, ncf_u_latent, gmf_u_latent, items):
        ncf_i_latent = self.dropout(self.ncf_item_embedding(items))
        ncf_ui_latent = torch.cat([ncf_u_latent, ncf_i_latent], 1)
        ncf_h = F.relu(self.hidden_layer2(F.relu(self.hidden_layer1(ncf_ui_latent))))
        gmf_i_latent = self.dropout(self.gmf_item_embedding(items))
        gmf_h = gmf_u_latent * gmf_i_latent
        h = torch.cat([gmf_h, ncf_h], 1)
        preds = self.output_layer(h)
        # do not apply sigmoid since we use NeuMF to predict rating
        # preds = torch.sigmoid(preds)
        preds = preds.squeeze(dim=-1)
        return preds

    def loss_function(self, preds, label_list):
        criterion = nn.MSELoss(reduction='sum')
        loss = criterion(preds, label_list)
        return loss
