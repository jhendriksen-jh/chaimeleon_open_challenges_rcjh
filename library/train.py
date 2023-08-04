import os
import tqdm
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

PROSTATE_LOSS = torch.nn.BCEWithLogitsLoss(reduction='mean')

def create_optimizer(model, lr=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


def create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=os.cpu_count() - 1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_device(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    return device


class Trainer():
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device, scheduler=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.train_time = []
        self.val_time = []
        self.best_val_acc = 0
        self.best_epoch = 0
        self.best_val_loss = 1e99

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch()
            self.val_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.val_acc[-1] > self.best_val_acc:
                self.best_val_acc = self.val_acc[-1]
                self.best_val_loss = self.val_loss[-1]
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), 'best_model.pt')
                print(f'New best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch} - train acc: {self.train_acc[-1]:.4f}')
            else:
                print(f'Validation accuracy did not improve from {self.best_val_acc:.4f} to {self.val_acc[-1]:.4f} at epoch {self.best_epoch} - train acc: {self.train_acc[-1]:.4f}')
    
    def train_epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        start_time = time.time()
        self.model.train()
        for batch_idx, (images, metadata, targets) in enumerate(self.train_loader):
            if self.model.model_data_type == 'images':
                data = images.to(self.device)
            elif self.model.model_data_type == 'metadata':
                data = metadata.squeeze().to(self.device)
            else:
                data = images.to(self.device), metadata.squeeze().to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, targets)
            loss.backward()
            self.optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            epoch_loss += loss.item()
            epoch_acc += pred.eq(targets.argmax(dim=1, keepdim=True).view_as(pred)).sum().item()
        self.train_acc.append(epoch_acc / len(self.train_loader.dataset))
        self.train_loss.append(epoch_loss / len(self.train_loader.dataset))
        self.train_time.append(time.time() - start_time)
        
        return self.train_loss, self.train_acc, self.train_time

    def val_epoch(self):
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for batch_idx, (images, metadata, targets) in enumerate(self.val_loader):
                if self.model.model_data_type == 'images':
                    data = images.to(self.device)
                elif self.model.model_data_type == 'metadata':
                    data = metadata.squeeze().to(self.device)
                else:
                    data = images.to(self.device), metadata.squeeze().to(self.device)
                targets = targets.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, targets)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(targets.argmax(dim=1, keepdim=True).view_as(pred)).sum().item()
            val_loss /= len(self.val_loader.dataset)
            val_acc /= len(self.val_loader.dataset)
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)
            self.val_time.append(time.time() - start_time)
        
        return self.val_loss, self.val_acc, self.val_time

    def plot_loss(self):
        plt.plot(self.train_loss, label='train_loss')
        plt.plot(self.val_loss, label='val_loss')
        plt.legend()
        plt.show()

    def plot_acc(self):
        plt.plot(self.train_acc, label='train_acc')
        plt.plot(self.val_acc, label='val_acc')
        plt.legend()
        plt.show()