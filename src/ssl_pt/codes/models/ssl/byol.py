import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.models.ssl.simclr import TokenSelector

def byol_loss(pred1, pred2, target1, target2):

    pred1 = F.normalize(pred1, dim=1)
    pred2 = F.normalize(pred2, dim=1)
    target1 = F.normalize(target1, dim=1)
    target2 = F.normalize(target2, dim=1)

    loss1 = 2 - 2 * (pred1 * target2).sum(dim=-1).mean()
    loss2 = 2 - 2 * (pred2 * target1).sum(dim=-1).mean()
    return loss1 + loss2

class BYOL(TokenSelector):

    def __init__(
        self, 
        encoder, 
        encoder_out_dim,
        projection_dim,
        hidden_dim,
        token_selection: str="cls"
    ):
        super(BYOL, self).__init__()

        self.online_encoder = encoder
        self.target_encoder = encoder
        self.token_selection = token_selection

        # Projection MLP
        self.online_projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        self.target_projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Disable gradients for target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target(self, tau=0.996):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data
        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data

    def forward(self, sequences_tuple, return_loss_only=True):
        sequences1, sequences2 = sequences_tuple

        # Online network
        online_proj1 = self.online_projector(
            self._select_token(self.online_encoder(sequences1)))
        online_proj2 = self.online_projector(
            self._select_token(self.online_encoder(sequences2)))
        
        # Target network
        with torch.no_grad():
            target_proj1 = self.target_projector(
                self._select_token(self.target_encoder(sequences1)))
            target_proj2 = self.target_projector(
                self._select_token(self.target_encoder(sequences2)))
        
        # Predictions
        pred1 = self.predictor(online_proj1)
        pred2 = self.predictor(online_proj2)
        
        loss = byol_loss(pred1, pred2, target_proj1.detach(), target_proj2.detach())
        assert return_loss_only, "Not implemented"
        return loss

    def forward_encoder(self, x):
        return self.online_encoder(x)
