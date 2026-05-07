import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.models.ssl.dino import EMASSL

class iBOT(EMASSL):

    # Required for iBOT_CNN.
    emb_dim_ratio = 1

    def __init__(
        self, 
        encoder, 
        encoder_out_dim,
        projection_dim,
        hidden_dim,
        ibot_mask_ratio,
        temperature_student,
        temperature_teacher,
        center_momentum,
        center_patch_momentum
    ):
        super(iBOT, self).__init__()

        self.embed_dim = int(encoder_out_dim * self.emb_dim_ratio)
        self.projection_dim = projection_dim

        self.student = encoder
        self.teacher = encoder

        # Share the parameters of projection heads for [CLS] token and patch tokens 
        # (from iBOT paper)
        self.proj_student = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.proj_teacher = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.proj_teacher.parameters():
            param.requires_grad = False

        self.ibot_mask_ratio = ibot_mask_ratio

        self.temperature_student = temperature_student
        self.temperature_teacher = temperature_teacher

        self.register_buffer("center", torch.zeros(1, projection_dim))
        self.center_momentum = center_momentum
        self.register_buffer("center_patch", torch.zeros(1, projection_dim))
        self.center_patch_momentum = center_patch_momentum

    @torch.no_grad()
    def update_center_patch(self, teacher_output_patch):
        batch_center = torch.sum(teacher_output_patch, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output_patch)

        self.center_patch = self.center_patch * self.center_patch_momentum + \
            batch_center * (1 - self.center_patch_momentum)

    def forward_encoder(self, x, use_student=True):

        if use_student:
            x, mask = self.mask_input(x)
            features = self.student(x)
            cls_token = self.proj_student(features[:, 0, :])
            feats = features[:, 1:, :].reshape(-1, features.size(2))
            patch = self.proj_student(feats)
            patch = patch.reshape(features.size(0), features.size(1) - 1, -1)
        else:
            with torch.no_grad():
                features = self.teacher(x)
                cls_token = self.proj_teacher(features[:, 0, :])
                feats = features[:, 1:, :].reshape(-1, features.size(2))
                patch = self.proj_teacher(feats)
                patch = patch.reshape(features.size(0), features.size(1) - 1, -1)
            mask = None
        return cls_token, patch, mask

    def mask_input(self, x):
        batch_size = x.size(0)
        n_chunks = x.size(1)
        mask = torch.rand(batch_size, n_chunks) < self.ibot_mask_ratio
        mask = mask.to(x.device)
        return x * mask.unsqueeze(-1), mask

    def calc_loss_mim(self, patch_s, patch_t, mask):

        patch_s = patch_s / self.temperature_student
        patch_t_ = F.softmax((patch_t - self.center_patch) / self.temperature_teacher, dim=-1)

        loss = torch.sum(-patch_t_ * F.log_softmax(patch_s, dim=-1), dim=-1)
        loss = loss * mask

        self.update_center_patch(patch_t)
        return loss.mean()

    def forward(self, sequences_tuple, return_loss_only=True):
        """
        Args:
            sequence (torch.Tensor): Tensor of shape (batch_size, n_lead, sequence_length).
        """
        sequences1, sequences2 = sequences_tuple

        cls_token_s1, patch_s1, mask_s1 =\
            self.forward_encoder(sequences1)
        cls_token_t1, patch_t1, _ =\
            self.forward_encoder(sequences1, use_student=False)
        cls_token_s2, patch_s2, mask_s2 =\
            self.forward_encoder(sequences2)
        cls_token_t2, patch_t2, _ =\
            self.forward_encoder(sequences2, use_student=False)

        # loss
        loss_mim1 = self.calc_loss_mim(patch_s1, patch_t1.detach(), mask_s1)
        loss_mim2 = self.calc_loss_mim(patch_s2, patch_t2.detach(), mask_s2)

        # cls loss: same with DINO.
        loss_cls1 = self.calc_loss(cls_token_t2.detach(), cls_token_s1)
        loss_cls2 = self.calc_loss(cls_token_t1.detach(), cls_token_s2)
        loss = loss_mim1 + loss_mim2 + loss_cls1 + loss_cls2

        assert return_loss_only
        return loss
        
class iBOT_CNN(iBOT):
    
    emb_dim_ratio = 0.5

    def forward_encoder(self, x, use_student):

        if use_student:
            x, mask = self.mask_input(x)
            features = self.student(x)
            # Feature dim must be divisible by 2.
            assert features.size(1) % 2 == 0
            cls_token = self.proj_student(features[:, :self.embed_dim])
            patch = self.proj_student(features[:, self.embed_dim:])
        else:
            with torch.no_grad():
                features = self.teacher(x)
                # Feature dim must be divisible by 2.
                assert features.size(1) % 2 == 0
                cls_token = self.proj_teacher(features[:, :self.embed_dim])
                patch = self.proj_teacher(features[:, self.embed_dim:])
            mask = None
        return cls_token, patch, mask
    
    def forward(self, sequences_tuple, return_loss_only=True):
        """
        Args:
            sequence (torch.Tensor): Tensor of shape (batch_size, n_lead, sequence_length).
        """
        sequences1, sequences2 = sequences_tuple

        cls_token_s1, patch_s1, mask_s1 =\
            self.forward_encoder(sequences1, use_student=True)
        cls_token_t1, patch_t1, _ =\
            self.forward_encoder(sequences1, use_student=False)
        cls_token_s2, patch_s2, mask_s2 =\
            self.forward_encoder(sequences2, use_student=True)
        cls_token_t2, patch_t2, _ =\
            self.forward_encoder(sequences2, use_student=False)

        # loss
        loss_mim1 = self.calc_loss_mim(patch_s1, patch_t1.detach(), mask_s1)
        loss_mim2 = self.calc_loss_mim(patch_s2, patch_t2.detach(), mask_s2)

        # cls loss: same with DINO.
        loss_cls1 = self.calc_loss(cls_token_t2.detach(), cls_token_s1)
        loss_cls2 = self.calc_loss(cls_token_t1.detach(), cls_token_s2)
        loss = loss_mim1 + loss_mim2 + loss_cls1 + loss_cls2

        assert return_loss_only
        return loss
 