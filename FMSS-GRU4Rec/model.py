import torch
from torch import nn


class GRU4Rec(nn.Module):
    def __init__(self, args, item_maxid):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.embed_dim != -1:
            self.item_embedding = nn.Embedding(item_maxid + 1, args.embed_dim, padding_idx=0)
            self.gru_layer = nn.GRU(
                input_size=args.embed_dim,
                hidden_size=args.hidden_size,
                batch_first=True)
        else:
            self.item_one_hot_embedding = torch.eye(item_maxid + 1, dtype=torch.float32).to(self.device)
            self.gru_layer = nn.GRU(
                input_size=item_maxid + 1,
                hidden_size=args.hidden_size,
                batch_first=True)
        self.dropout = nn.Dropout(p=args.dropout)
        self.output_layer = nn.Linear(args.hidden_size, item_maxid + 1, bias=False)
        self.init_param()

    def init_param(self):
        if self.embed_dim != -1:
            nn.init.kaiming_normal_(self.item_embedding.weight, mode='fan_out')
        nn.init.orthogonal_(self.gru_layer.weight_ih_l0)
        nn.init.orthogonal_(self.gru_layer.weight_hh_l0)
        nn.init.constant_(self.gru_layer.bias_ih_l0, 0.0)
        nn.init.constant_(self.gru_layer.bias_hh_l0, 0.0)
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_out')

    def forward(self, x, x_lens):
        max_len = x.shape[1]
        if self.embed_dim != -1:
            seq_embedding = self.item_embedding(x)
        else:
            seq_embedding = self.item_one_hot_embedding[x.long().flatten()]
            seq_embedding = seq_embedding.view(x.shape[0], x.shape[1], -1)
        seq_embedding = torch.nn.utils.rnn.pack_padded_sequence(seq_embedding, x_lens, batch_first=True,
                                                                enforce_sorted=False)
        gru_out, _ = self.gru_layer(seq_embedding)
        gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=max_len)
        gru_out = self.dropout(gru_out)
        seq_out = torch.tanh(self.output_layer(gru_out))
        return seq_out

    def loss_function(self, seq_out, padding_mask, target, neg, seq_len):
        neg_output = torch.gather(seq_out, 2, neg.long())
        target_output = torch.gather(seq_out, 2, target.unsqueeze(-1).long())
        # top1
        err = torch.sigmoid(neg_output - target_output.expand(-1, -1, neg_output.shape[2])) + torch.sigmoid(
            torch.square(neg_output))
        err = err * padding_mask.expand(-1, -1, err.shape[2])
        loss = torch.mean(err)
        # bpr
        # err = torch.log(torch.sigmoid(target_output - neg_output))
        # loss = -torch.mean(err)
        return loss
