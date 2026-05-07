import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.models.ssl.simclr import TokenSelector

class EMASSL(TokenSelector):

    def calc_loss(self, teacher_output, student_output):
        teacher_out = F.softmax(
            (teacher_output - self.center) / self.temperature_teacher, dim=-1)
        student_out = F.log_softmax(
            student_output / self.temperature_student, dim=-1)
        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()

        # Update center
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)

        self.center = self.center * self.center_momentum + \
            batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def update_target(self, tau=0.996):
        # Update teacher network by EMA.
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data = tau * teacher_param.data + (1 - tau) * student_param.data
        for student_param, teacher_param in zip(self.proj_student.parameters(), self.proj_teacher.parameters()):
            teacher_param.data = tau * teacher_param.data + (1 - tau) * student_param.data


class DINO(EMASSL):

    def __init__(
        self, 
        encoder, 
        encoder_out_dim,
        projection_dim,
        hidden_dim,
        temperature_student,
        temperature_teacher,
        center_momentum,
        token_selection: str="cls"
    ):
        super(DINO, self).__init__()

        self.embed_dim = encoder_out_dim

        self.student = encoder
        self.teacher = encoder

        self.proj_student = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.proj_teacher = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.proj_teacher.parameters():
            param.requires_grad = False

        self.temperature_student = temperature_student
        self.temperature_teacher = temperature_teacher
        self.token_selection = token_selection

        self.register_buffer("center", torch.zeros(1, projection_dim))
        self.center_momentum = center_momentum

    def forward_encoder(self, x, use_student=True):

        if use_student:
            out = self._select_token(self.student(x))
            return self.proj_student(out)

        # Teacher
        with torch.no_grad():
            out = self._select_token(self.teacher(x))
            out = self.proj_teacher(out)
        return out

    
    def forward(self, sequences_tuple, return_loss_only=True):
        """
        Args:
            sequence (torch.Tensor): Tensor of shape (batch_size, n_lead, sequence_length).
        """
        sequences1, sequences2 = sequences_tuple

        seq1_student = self.forward_encoder(sequences1)#, use_student=True)
        seq2_student = self.forward_encoder(sequences2)#, use_student=True)

        seq1_teacher = self.forward_encoder(sequences1, use_student=False)
        seq2_teacher = self.forward_encoder(sequences2, use_student=False)

        loss1 = self.calc_loss(seq1_teacher.detach(), seq2_student)
        loss2 = self.calc_loss(seq2_teacher.detach(), seq1_student)
        loss = (loss1 + loss2) / 2
        assert return_loss_only
        return loss