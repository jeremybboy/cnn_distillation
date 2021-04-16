import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import plot


class ClassifTrainer:
    def __init__(self,
                 directory,
                 model,
                 trainloader,
                 testloader,
                 lr):
        # models
        # dataset ou dataloader ?
        self.logdir = directory
        # Optimizer
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.writer = SummaryWriter(log_dir=directory)


    def do_epoch(self, device, train=True):
        dico = dict(loss_classif=0.0,
                    accuracy=0.0)

        total = 0
        correct = 0

        loader = self.trainloader if train else self.testloader
        self.model.train(train)

        for i, (images, true_labels) in tqdm(enumerate(loader)):
            # Move tensors to the configured device
            images = images.to(device)
            true_labels = true_labels.to(device)

            # Forward pass
            with torch.set_grad_enabled(train):
                logits, f2, f4 = self.model(images)  # model's forward method
                loss = self.loss(logits, true_labels)
                dico['loss_classif'] += loss.item()

                _, predicted_indices = torch.max(logits.data, 1)
                total += true_labels.size(0)
                correct += (predicted_indices == true_labels).sum().item()

            # Backprpagation and optimization
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            #fin de l'itération

        #fin de l'epoch
        dico = {
            key: (value / (i+1))
            for key, value in dico.items()
        }

        dico['accuracy'] = correct/total
        return dico #average error for an iteration


    def loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels)


    def train_model(self, num_epochs, device):

        for epoch in range(num_epochs):
            train_dico = self.do_epoch(device, train=True)
            print(f'TRAIN : Epoch [{epoch + 1}/{num_epochs}], Loss: {train_dico["loss_classif"]:.4f}, Acc:{train_dico["accuracy"]:.4f}')

            test_dico= self.do_epoch(device, train=False)
            print(f'TEST : Epoch [{epoch + 1}/{num_epochs}], Loss: {test_dico["loss_classif"]:.4f}, Acc:{test_dico["accuracy"]:.4f}')

            plot(self.writer, epoch, train_dico, test_dico)



    # def save_losses(self, train, test, name="losses.pdf"):
    #     fig = plt.Figure()
    #     plt.plot(train, color="black")
    #     plt.plot(test, color="black", linestyle='dashed')
    #
    #     # plt.title('Classification Loss')
    #     # plt.ylabel('LOSS')
    #     # plt.xlabel('EPOCHS')
    #
    #     plt.savefig(os.path.join(self.logdir, name))
    #     plt.close('all')

