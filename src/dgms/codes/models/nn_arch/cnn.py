import torch.nn as nn

class CNN01(nn.Module):

    def __init__(self, seqlen, output_dim, h_dim, add_clf=False, with_sigmoid=True):
        super(CNN01, self).__init__()

        self.cnn = nn.Sequential(
            # Input: N x 1 x seqlen
            nn.Conv1d(
                1, h_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(h_dim, h_dim * 2, 4, 2, 1),
            self._block(h_dim * 2, h_dim * 4, 4, 2, 1),
            nn.Conv1d(h_dim * 4, 1, kernel_size=4, stride=1, padding=1),
        )
        dim = seqlen // 2 // 2 // 2 - 1 # 5000 -> 624
        self.fc = nn.Linear(dim, output_dim)

        self.add_clf = add_clf
        if add_clf:
            self.clf = nn.Linear(output_dim, 1)
            self.sigmoid = nn.Sigmoid() if with_sigmoid else None

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.cnn(x).view(x.size(0), -1)
        x = self.fc(x)
        if self.add_clf:
            x = self.clf(x)
            if self.sigmoid is not None:
                x = self.sigmoid(x.squeeze(-1))
            else:
                x = x.squeeze(-1)        
        return x

class CNN02(nn.Module):

    def __init__(self, seqlen, output_dim, h_dim, add_clf=False, with_sigmoid=True):
        super(CNN02, self).__init__()

        self.conv1 = nn.Conv1d(1, h_dim, kernel_size=16, stride=1, padding='same')
        self.leaky_relu1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv1d(h_dim, h_dim*2, kernel_size=16, stride=1, padding='same')
        self.leaky_relu2 = nn.LeakyReLU()
        
        self.max_pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(h_dim*2, h_dim*4, kernel_size=16, stride=1, padding='same')
        self.leaky_relu3 = nn.LeakyReLU()
        
        self.conv4 = nn.Conv1d(h_dim*4, h_dim*8, kernel_size=16, stride=1, padding='same')
        self.leaky_relu4 = nn.LeakyReLU()
        
        self.max_pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(seqlen//4*h_dim*8, output_dim)

        self.add_clf = add_clf
        if add_clf:
            self.clf = nn.Linear(output_dim, 1)
            self.sigmoid = nn.Sigmoid() if with_sigmoid else None

    def forward(self, x):
        # x shape: (batch_size, seqlen, 1)

        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        
        x = self.max_pool1(x)
        
        x = self.leaky_relu3(self.conv3(x))
        x = self.leaky_relu4(self.conv4(x))
        
        x = self.max_pool2(x)
        
        x = self.flatten(x)
        x = self.dense(x)

        if self.add_clf:
            x = self.clf(x)
            if self.sigmoid is not None:
                x = self.sigmoid(x.squeeze(-1))
            else:
                x = x.squeeze(-1)        
        return x

