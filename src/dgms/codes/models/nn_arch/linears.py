import torch.nn as nn

class LinearEncoder(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim, 
        h_dim, 
        add_clf=False, 
        with_sigmoid=True
    ):
        super(LinearEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, output_dim)
        )
        self.add_clf = add_clf
        if add_clf:
            self.clf = nn.Linear(output_dim, 1)
            self.sigmoid = nn.Sigmoid() if with_sigmoid else None

    def forward(self, x):
        """_summary_

         Args:
            x (Tensor): Tensor of shape (bs, n_lead=1, seqlen)

        Returns:
            _type_: _description_
        """
        x = self.fc(x) # -> bs, n_lead, h_dim
        if self.add_clf:
            x = self.clf(x)
            if self.sigmoid is not None:
                x = self.sigmoid(x.squeeze(-1))
            else:
                x = x.squeeze(-1)
        return x

class LinearChunkEncoder(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim,
        chunk_len, 
        h_dim, 
        add_clf=False,
        with_sigmoid=True
    ):
        super(LinearChunkEncoder, self).__init__()
        n_chunks = input_dim // chunk_len
        self.chunk_len = chunk_len
        self.linear_emb = nn.Linear(chunk_len, h_dim)

        self.fc = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )
        self.fc2 = nn.Linear(h_dim*n_chunks, output_dim)

        self.add_clf = add_clf
        if add_clf:
            self.clf = nn.Linear(output_dim, 1)
            self.sigmoid = nn.Sigmoid() if with_sigmoid else None

    def forward(self, x):
        """_summary_

         Args:
            x (Tensor): Tensor of shape (bs, n_lead=1, seqlen)

        Returns:
            _type_: _description_
        """
        x = x.reshape(x.size(0), -1, self.chunk_len) # bs, n_chunks, chunk_len
        x = self.linear_emb(x) # bs, n_chunks, h_dim
        x = self.fc(x) # -> bs, n_lead, h_dim
        x = x.reshape(x.size(0), -1)
        x = self.fc2(x)
        if self.add_clf:
            x = self.clf(x)
            if self.sigmoid is not None:
                x = self.sigmoid(x.squeeze(-1))
            else:
                x = x.squeeze(-1)
        return x
    

class LinearDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, h_dim):
        super(LinearDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, output_dim)
        )

    def forward(self, x):

        x = self.fc(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x
