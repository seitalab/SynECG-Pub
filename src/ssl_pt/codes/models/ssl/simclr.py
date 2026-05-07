import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenSelector(nn.Module):

    def _select_token(self, z):

        if self.token_selection is None:
            return z

        if self.token_selection == "cls":
            return z[:, 0]
        else:
            raise NotImplementedError(f"Unknown token selection: {self.token_selection}")


class SimCLR(TokenSelector):

    def __init__(
        self, 
        encoder, 
        encoder_out_dim,
        projection_dim,
        temperature,
        token_selection: str="cls"
    ):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.temperature = temperature
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim),
            nn.ReLU(),
            nn.Linear(encoder_out_dim, projection_dim)
        )
        
        self.token_selection = token_selection

    def forward_loss(self, z1, z2):
        """
        Args:
            z1 (torch.Tensor): (N, D)
            z2 (torch.Tensor): (N, D)
        Returns:
            loss (torch.Tensor): ()
        """
        batch_size = z1.shape[0]

        features = torch.cat([z1, z2], dim=0)
        features = nn.functional.normalize(features, dim=1)

        # Calculate similarity matrix
        similarity_matrix = torch.matmul(features, features.T)

        # Mask for positive samples (diagonal blocks)
        mask = torch.eye(batch_size * 2, dtype=torch.bool, device=z1.device)
        positives = similarity_matrix[mask].view(batch_size * 2, 1)

        # Mask for negative samples (off-diagonal elements)
        negatives = similarity_matrix[~mask].view(batch_size * 2, -1)
    
        # Compute logits
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature

        # Labels are all zeros because for each row, the first element is the positive sample
        labels = torch.zeros(batch_size * 2, dtype=torch.long, device=z1.device)
            
        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
        
    def forward_encoder(self, sequences):
        h = self.encoder(sequences)
        z = self.projector(h)
        return z

    def forward(self, sequences_tuple, return_loss_only=True):
        sequences1, sequences2 = sequences_tuple
        z1 = self.forward_encoder(sequences1) # bs, dim, seqlen -> bs, dim, seqlen
        z2 = self.forward_encoder(sequences2) # bs, dim, seqlen -> bs, dim, seqlen

        z1 = self._select_token(z1)
        z2 = self._select_token(z2)

        loss = self.forward_loss(z1, z2)
        assert return_loss_only, "Not implemented"
        return loss