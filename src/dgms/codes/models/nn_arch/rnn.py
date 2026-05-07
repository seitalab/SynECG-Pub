import math
import torch
import torch.nn as nn

class RNNEncoder(nn.Module):

    def __init__(
        self, 
        seqlen: int, 
        enc_out_dim: int, 
        h_dim: int, 
        gru_dim: int, 
        chunk_len: int, 
        device: str,
        add_clf: bool = False
    ):
        super(RNNEncoder, self).__init__()

        self.step_len = chunk_len
        self.n_step = seqlen // self.step_len
        assert seqlen % self.step_len == 0

        self.gru_dim = gru_dim
        self.h_dim = h_dim
        self.device = device
        
        self.emb = nn.Linear(
            self.step_len, self.h_dim)
        self.rnn = nn.GRU(
            self.h_dim, self.gru_dim, bidirectional=True)
        self.act = nn.ELU()
        self.h_0 = nn.Parameter(
            torch.zeros(1, self.gru_dim, device=self.device))
        self.nn_s = nn.Linear(self.n_step, enc_out_dim)
        self.nn_d = nn.Linear(self.gru_dim*2, enc_out_dim)

        self.add_clf = add_clf
        if add_clf:
            self.clf = nn.Linear(enc_out_dim, 1)
            self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x (Tensor): (bs, num_lead=1, seqlen)
        """
        bs = x.size(0)
        x = x.view(bs, self.n_step, self.step_len)
        x = self.emb(x) # -> bs, n_step, emb_dim
        self.init_hidden()

        x = x.permute(1, 0, 2) # (n_step, bs, H_in=emb_dim)
        hidden = self.h_0.expand(2, bs, self.gru_dim).contiguous()
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x, hidden)

        # n_step, bs, D * H_out -> bs, D * H_out, n_step
        out = out.permute(1, 2, 0)
        out_s = self.nn_s(self.act(out)) # -> bs, D * H_out, enc_out_dim
        out_s = out_s.swapaxes(1, 2).mean(dim=-1)
    
        out_d = self.nn_d(self.act(out.swapaxes(1, 2))) # -> bs, n_step, enc_out_dim
        out_d = out_d.swapaxes(1, 2).mean(dim=-1)

        out = self.act(out_d + out_s)
        if self.add_clf:
            out = self.sigmoid(self.clf(out))
        return out
    
    def init_hidden(self):
        self.h_0 = nn.Parameter(
            torch.zeros(1, self.gru_dim, device=self.device))

class HierarchicalRNNDecoder(nn.Module):

    def __init__(self, z_dim, seqlen, chunk_len, h_dim):
        super(HierarchicalRNNDecoder, self).__init__()

        self.z2h  = nn.Linear(z_dim, h_dim)
        self.z2h0 = nn.Linear(z_dim, h_dim)

        self.rnn_1 = nn.GRU(h_dim, h_dim)
        self.out_1 = nn.Linear(h_dim, h_dim)
        self.act_1 = nn.ELU()
        
        self.rnn_2 = nn.GRU(h_dim, h_dim)
        self.out_2 = nn.Linear(h_dim, 1)
        self.act_2 = nn.ELU()
        
        self.seqlen  = seqlen
        self.steplen = chunk_len
        self.top_rep = math.ceil(seqlen/chunk_len)

    def forward(self, z):
        """
        Args:
            z (torch.Tensor): tensor of size (bs, z_dim).
        Returns:
            output (torch.Tensor): 
        """
        h0 = self.z2h0(z)
        h0 = h0.unsqueeze(0) # bs, h_dim -> 1, bs, h_dim
        input_h = self.z2h(z)
        input_h = input_h.repeat(self.top_rep, 1, 1)
        input_h = input_h.view(self.top_rep, input_h.size(1), -1)
        outputs_top, _ = self.rnn_1(input_h, h0)
        outputs_top = self.act_1(outputs_top)

        h0_sec = self.out_1(outputs_top[:1])
        outputs_sec = []
        for output in outputs_top:
            input_sec = self.out_1(output)
            input_sec = input_sec.repeat(self.steplen, 1, 1)
            input_sec = input_sec.view(
                self.steplen, input_sec.size(1), -1)
            out_sec, h0_sec = self.rnn_2(input_sec, h0_sec)
            outputs_sec.append(out_sec)
        
        outputs_sec = outputs_sec[:self.seqlen]
        outputs_sec = torch.stack(outputs_sec)
        output = self.out_2(self.act_2(outputs_sec))
        output = output.squeeze(-1).permute(2, 0, 1) # seqlen, step, bs, 1 -> bs, seqlen, step
        output = output.view(z.size(0), -1)
        output = output.unsqueeze(1) # bs, seqlen -> bs, 1, seqlen
        return output

class RNNDecoder(nn.Module):

    def __init__(self, z_dim, seqlen, chunk_len, h_dim):
        super(RNNDecoder, self).__init__()
        
        self.chunk_len = chunk_len

        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=h_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.conv1 = nn.Conv1d(h_dim*2, h_dim*2, kernel_size=16, stride=1, padding='same')
        self.leaky_relu1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv1d(h_dim*2, h_dim, kernel_size=16, stride=1, padding='same')
        self.leaky_relu2 = nn.LeakyReLU()
        
        self.upsample1 = nn.Upsample(scale_factor=2)
        
        self.conv3 = nn.Conv1d(h_dim, h_dim//2, kernel_size=16, stride=1, padding='same')
        self.leaky_relu3 = nn.LeakyReLU()
        
        self.conv4 = nn.Conv1d(h_dim//2, h_dim//4, kernel_size=16, stride=1, padding='same')
        self.leaky_relu4 = nn.LeakyReLU()
        
        self.upsample2 = nn.Upsample(scale_factor=2)
        
        self.conv5 = nn.Conv1d(h_dim//4, 1, kernel_size=16, stride=1, padding='same')
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: (batch_size, z_dim)
        x = x.unsqueeze(-1) # -> bs, z_dim, 1
        x, _ = self.lstm(x)
        # x shape: (batch_size, z_dim, 128)
        
        x = x.transpose(1, 2)
        # x shape: (batch_size, 128, z_dim)
        
        x = self.leaky_relu1(self.conv1(x))
        print(x.size())
        x = self.leaky_relu2(self.conv2(x))
        print(x.size())
        
        x = self.upsample1(x)
        print(x.size())
        
        x = self.leaky_relu3(self.conv3(x))
        print(x.size())
        x = self.leaky_relu4(self.conv4(x))
        print(x.size())
        
        x = self.upsample2(x)
        print(x.size())
        
        x = self.tanh(self.conv5(x))
        print(x.size())
        
        # x shape: (batch_size, 1, 400)
        return x.transpose(1, 2)  # (batc