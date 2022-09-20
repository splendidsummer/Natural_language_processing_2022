import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiLSTM_SoftAtten(nn.Module):  # 100, 64,
    def __init__(self, num_classes, vocab_size,
                 pos_size, drugtype_size, num_lstm_layers,
                 embedding_dim1, embedding_dim2, embedding_dim3,
                 hidden_dim, dropout, bi=True):
        super(BiLSTM_SoftAtten, self).__init__()

        self.embedding1 = nn.Embedding(vocab_size, embedding_dim1, padding_idx=0).to(device)
        self.embed_dropout1 = nn.Dropout(dropout)
        self.embedding2 = nn.Embedding(pos_size, embedding_dim2, padding_idx=0).to(device)
        self.embed_dropout2 = nn.Dropout(dropout)
        self.embedding3 = nn.Embedding(drugtype_size, embedding_dim3, padding_idx=0).to(device)
        self.embed_dropout3 = nn.Dropout(dropout)
        embedding_dim = embedding_dim1 + embedding_dim2 + embedding_dim3
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                            num_layers=num_lstm_layers, dropout=dropout,
                            bidirectional=True)

        hidden_dim = 2 * hidden_dim
        self.query_dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def soft_attention_net(self, x, query):
        d_k = torch.tensor(query.size(-1), dtype=torch.float32)
        scores = torch.matmul(query, x.transpose(1, 2)) / torch.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, x).sum(1)

        return context

    def forward(self, xw, xp, xdt, lens):  # we can add one more hidden input,  init hidden state somewhere
        """
		The forward method takes in the input and the previous hidden state
		"""
        embW = self.embed_dropout1(self.embedding1(xw))
        embP = self.embed_dropout2(self.embedding2(xp))
        embDT = self.embed_dropout3(self.embedding3(xdt))

        embs = torch.cat([embW, embP, embDT], dim=-1)
        lens = lens.cpu()

        pack = nn.utils.rnn.pack_padded_sequence(embs, lens, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(pack)  # out is the hidden state, _ is memory state
        out, lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # Get the last output to do the predict
        query = self.query_dropout(out)
        out = self.soft_attention_net(out, query)
        # out = out[:, -1]
        out = self.classifier(out)
        return out


class BiLSTM_SelfAtten(nn.Module):  # 100, 64,
    def __init__(self, num_classes, vocab_size,
                 pos_size, drugtype_size, num_lstm_layers,
                 embedding_dim1, embedding_dim2, embedding_dim3,
                 hidden_dim, dropout, bi=True):
        super(BiLSTM_SelfAtten, self).__init__()

        self.embedding1 = nn.Embedding(vocab_size, embedding_dim1, padding_idx=0).to(device)
        self.embed_dropout1 = nn.Dropout(dropout)
        self.embedding2 = nn.Embedding(pos_size, embedding_dim2, padding_idx=0).to(device)
        self.embed_dropout2 = nn.Dropout(dropout)
        self.embedding3 = nn.Embedding(drugtype_size, embedding_dim3, padding_idx=0).to(device)
        self.embed_dropout3 = nn.Dropout(dropout)
        embedding_dim = embedding_dim1 + embedding_dim2 + embedding_dim3
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                            num_layers=num_lstm_layers, dropout=dropout,
                            bidirectional=True)

        hidden_dim = 2 * hidden_dim
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.classifier = nn.Linear(hidden_dim, num_classes)

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, x):  # x:[batch, seq_len, hidden_dim*2]

        u = torch.tanh(torch.matmul(x, self.w_omega))  # [batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score  # [batch, seq_len, hidden_dim*2]

        context = torch.sum(scored_x, dim=1)  # [batch, hidden_dim*2]
        return context

    def forward(self, xw, xp, xdt, lens):  # we can add one more hidden input,  init hidden state somewhere
        """
		The forward method takes in the input and the previous hidden state
		"""
        embW = self.embed_dropout1(self.embedding1(xw))
        embP = self.embed_dropout2(self.embedding2(xp))
        embDT = self.embed_dropout3(self.embedding3(xdt))

        embs = torch.cat([embW, embP, embDT], dim=-1)
        lens = lens.cpu()

        pack = nn.utils.rnn.pack_padded_sequence(embs, lens, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(pack)  # out is the hidden state, _ is memory state
        out, lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out.permute(1, 0, 2)  # [batch, seq_len, hidden_dim*2]
        out = self.attention_net(out)

        out = self.classifier(out)
        return out
