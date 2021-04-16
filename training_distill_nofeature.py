import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from utils import plot


class DistillationTrainer:
    def __init__(self,
                 directory,
                 student_model,
                 teacher_model,
                 trainloader,
                 testloader,
                 lr,
                 ):
        # models
        # dataset ou dataloader ?
        self.logdir = directory
        # Optimizer
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=lr)
        self.writer = SummaryWriter(log_dir=directory)

    def do_epoch(self, device, train=True):
        dico = dict(total_loss=0.0,
                    loss_classif=0.0,
                    loss_logits=0.0,
                    accuracy=0.0,
                    )
        total = 0
        correct = 0

        loader = self.trainloader if train else self.testloader
        self.student_model.train(train)

        for i, (images, labels) in tqdm(enumerate(loader)):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            with torch.set_grad_enabled(train):

                logits_teacher, f2, f4 = self.teacher_model(images)
                logits_student, fA, fB = self.student_model(images)

                loss_classif = self.classif_loss(logits_student, labels)
                loss_logits = self.logit_distillation_loss(logits_teacher, logits_student)

                total_loss=loss_classif+loss_logits

                dico['loss_classif'] += loss_classif.item()
                dico['loss_logits'] += loss_logits.item()
                dico['total_loss'] += total_loss.item()
                # dico['total_loss'] += total_loss.item()

                _, predicted_indices = torch.max(logits_student.data, 1)
                total += labels.size(0)
                correct += (predicted_indices == labels).sum().item()

            # Backprpagation and optimization
            if train:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                # fin de l'itération

        # fin de l'epoch
        dico = {
            key: (value / (i + 1))
            for key, value in dico.items()
        }

        dico['accuracy'] = correct / total
        return dico  # average error for an iteration

    def classif_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def logit_distillation_loss(self, logits_teacher, logits_student, T=4):
        softmax_op = nn.Softmax(dim=1)
        mseloss_fn = nn.MSELoss()
        return mseloss_fn(softmax_op(logits_student / T), softmax_op(logits_teacher) / T)
    #
    # def feature_distillation_loss(self, fA, fB, f2, f4):
    #     mseloss_fn = nn.MSELoss()
    #     return mseloss_fn(fA, f2) + mseloss_fn(fB, f4)



    def train_model(self, num_epochs, device):

        for epoch in range(num_epochs):
            train_dico= self.do_epoch(device, train=True)
            print(
                f'TRAIN: Epoch[{epoch + 1}/{num_epochs}], Loss_Classif:{train_dico["loss_classif"]:.4f}, Loss_Logits:{train_dico["loss_logits"]:.4f},  Accuracy:{train_dico["accuracy"]:.4f}')

            test_dico = self.do_epoch(device, train=False)
            print(f'TRAIN: Epoch[{epoch + 1}/{num_epochs}], Loss_Classif:{test_dico["loss_classif"]:.4f}, Loss_Logits:{test_dico["loss_logits"]:.4f}, Accuracy:{test_dico["accuracy"]:.4f}')

            plot(self.writer, epoch, train_dico, test_dico)
