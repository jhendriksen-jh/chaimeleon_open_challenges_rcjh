import os
import tqdm
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# PROSTATE_LOSS = torch.nn.BCEWithLogitsLoss(reduction="mean")
PROSTATE_LOSS = torch.nn.BCEWithLogitsLoss(
    reduction="mean", pos_weight=torch.tensor([4], device="cuda")
)
# LUNG_LOSS = torch.nn.MSELoss(reduction="mean")
LUNG_LOSS = torch.nn.MultiLabelSoftMarginLoss(reduction="sum")


def create_optimizer(model, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


def create_dataloader(
    dataset, batch_size=16, shuffle=True, num_workers=os.cpu_count() - 1
):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def get_device(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    return device


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        evaluation_function=None,
        scheduler=None,
        training_dir="./",
        event_filter=False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.eval_function = evaluation_function
        self.training_dir = training_dir
        self.event_filter = event_filter
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.train_time = []
        self.val_time = []
        self.val_score = [0]
        self.val_score_dicts = []
        self.best_score = 0
        self.best_val_acc = 0

        self.best_epoch = 0
        self.best_val_loss = 1e99

    def train(self, epochs, training_timestamp):
        for epoch in range(epochs):
            self.train_epoch()
            self.val_epoch()
            improved_val_acc = self.val_acc[-1] > self.best_val_acc
            if improved_val_acc:
                self.best_val_acc = self.val_acc[-1]
            if self.eval_function is not None:
                improved_score = self.val_score[-1] > self.best_score
                if improved_score:
                    self.best_score = self.val_score[-1]
            else:
                improved_score = False
            if improved_val_acc:
                self.best_val_loss = self.val_loss[-1]
                self.best_epoch = epoch + 1
                torch.save(
                    self.model.state_dict(),
                    f"{self.training_dir}best_val_acc_{self.model.__class__.__name__}.pt",
                )
                print(
                    f"{epoch+1}/{epochs} - New best validation acc: {self.best_val_acc:.4f} acc at epoch {self.best_epoch} - score: {self.val_score[-1]:.4f} - train acc: {self.train_acc[-1]:.4f}"
                )
            if improved_score:
                self.best_val_loss = self.val_loss[-1]
                self.best_epoch = epoch + 1
                torch.save(
                    self.model.state_dict(),
                    f"{self.training_dir}best_val_score_{self.model.__class__.__name__}.pt",
                )
                print(
                    f"{epoch+1}/{epochs} - New best validation score: {self.val_acc[-1]:.4f} acc at epoch {self.best_epoch} - score: {self.best_score:.4f} - train acc: {self.train_acc[-1]:.4f}"
                )
            else:
                print(
                    f"{epoch+1}/{epochs} - Validation perf did not improve with acc at {self.val_acc[-1]:.4f} vs {self.best_val_acc:.4f} and score: {self.val_score[-1]:.4f} vs {self.best_score:.4f} - train acc: {self.train_acc[-1]:.4f}"
                )
            if self.scheduler is not None:
                self.scheduler.step(self.best_score)

        print(
            f"Best validation perf: {self.best_val_acc:.4f} acc at epoch {self.best_epoch} - Best score: {self.best_score:.4f}"
        )
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "train_time": self.train_time,
            "val_loss": self.val_acc,
            "val_score": self.val_score,
            "best_val_acc": self.best_val_acc,
            "best_val_score": self.best_score,
            "best_epoch": self.best_epoch,
            "val_score_dicts": self.val_score_dicts,
        }

    def train_epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        start_time = time.time()
        self.model.train()
        for batch_idx, (images, metadata, targets) in enumerate(self.train_loader):
            if self.model.model_data_type == "images":
                data = images.to(self.device)
            elif self.model.model_data_type == "metadata":
                data = metadata.squeeze().to(self.device)
            else:
                data = images.to(self.device), metadata.squeeze().to(self.device)
            if isinstance(targets, list):
                targets = targets[0].to(self.device)
            else:
                targets = targets.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            if not self.event_filter:
                loss = self.loss_fn(output, targets)
            else:
                batch_inds = []
                for batch_ind, (cur_output, cur_targets, cur_pfs) in enumerate(zip(output, targets, pfs)):
                    if cur_pfs[0]:
                        batch_inds.append(batch_ind)
                loss = self.loss_fn(output[batch_inds], targets[batch_inds])
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            epoch_acc += (
                pred.eq(targets.argmax(dim=1, keepdim=True).view_as(pred)).sum().item()
            )
        self.train_acc.append(epoch_acc / len(self.train_loader.dataset))
        self.train_loss.append(epoch_loss * 10 / len(self.train_loader.dataset))
        self.train_time.append(time.time() - start_time)

        return self.train_loss, self.train_acc, self.train_time

    def val_epoch(self):
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            epoch_targets = []
            epoch_preds = []
            epoch_outputs = []
            for batch_idx, (images, metadata, targets) in enumerate(self.val_loader):
                if self.model.model_data_type == "images":
                    data = images.to(self.device)
                elif self.model.model_data_type == "metadata":
                    data = metadata.squeeze().to(self.device)
                else:
                    data = images.to(self.device), metadata.squeeze().to(self.device)
                if isinstance(targets, list):
                    pfs = targets[1]
                    targets = targets[0].to(self.device)
                else:
                    targets = targets.to(self.device)
                output = self.model(data)
                if not self.event_filter:
                    loss = self.loss_fn(output, targets)
                else:
                    batch_inds = []
                    for batch_ind, (cur_output, cur_targets, cur_pfs) in enumerate(zip(output, targets, pfs)):
                        if cur_pfs[0]:
                            batch_inds.append(batch_ind)
                    loss = self.loss_fn(output[batch_inds], targets[batch_inds])

                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += (
                    pred.eq(targets.argmax(dim=1, keepdim=True).view_as(pred))
                    .sum()
                    .item()
                )
                if self.loss_fn is PROSTATE_LOSS:
                    targets, output, pred = (
                        targets.argmax(dim=1, keepdim=True).to("cpu"),
                        torch.nn.functional.sigmoid(output).to("cpu"),
                        pred.to("cpu"),
                    )

                    epoch_targets.extend([i.item() for i in targets])
                    epoch_preds.extend([i.item() for i in pred])
                    epoch_outputs.extend(
                        [output[k][1].item() for k, i in enumerate(targets)]
                    )
                elif self.loss_fn is LUNG_LOSS:
                    estimates = torch.nn.functional.softmax(output, dim=1).to("cpu")
                    epoch_preds.extend([i.item() for i in pred])
                    epoch_outputs.extend([i for i in estimates])
                    epoch_targets.extend(pfs)

            val_loss /= len(self.val_loader.dataset)
            val_acc /= len(self.val_loader.dataset)
            self.val_loss.append(val_loss * 10)
            self.val_acc.append(val_acc)
            if self.eval_function is not None and self.loss_fn is PROSTATE_LOSS:
                val_score, val_score_dict = self.eval_function(
                    epoch_targets, epoch_outputs, epoch_preds
                )
                self.val_score.append(val_score)
                self.val_score_dicts.append(val_score_dict)
            elif self.eval_function is not None and self.loss_fn is LUNG_LOSS:
                self.val_score.append(self.eval_function(epoch_targets, epoch_preds))
            self.val_time.append(time.time() - start_time)

        return self.val_loss, self.val_acc, self.val_time

    def plot_loss(self):
        plt.plot(self.train_loss[1:], label="train_loss")
        plt.plot(self.val_loss[1:], label="val_loss")
        plt.title(f"Loss - {self.model.__class__.__name__}")
        plt.legend()
        plt.show()

    def plot_acc(self):
        plt.plot(self.train_acc[1:], label="train_acc")
        plt.plot(self.val_acc[1:], label="val_acc")
        plt.plot(self.val_score[1:], label="val_score")
        plt.title(f"Accuracy - {self.model.__class__.__name__}")
        plt.legend()
        plt.show()


class FineTuner:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        evaluation_function=None,
        scheduler=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.eval_function = evaluation_function
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.train_time = []
        self.val_time = []
        self.val_score = [0]
        self.best_score = 0
        self.best_val_acc = 0

        self.best_epoch = 0
        self.best_val_loss = 1e99

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch()
            self.val_epoch()
            improved_val_acc = self.val_acc[-1] > self.best_val_acc
            if improved_val_acc:
                self.best_val_acc = self.val_acc[-1]
            if self.eval_function is not None:
                improved_score = self.val_score[-1] > self.best_score
                if improved_score:
                    self.best_score = self.val_score[-1]
            else:
                improved_score = False
            if improved_val_acc:
                self.best_val_loss = self.val_loss[-1]
                self.best_epoch = epoch + 1
                torch.save(
                    self.model.state_dict(),
                    f"best_val_acc_{self.model.__class__.__name__}.pt",
                )
                print(
                    f"{epoch+1}/{epochs} - New best validation perf: {self.best_val_acc:.4f} acc at epoch {self.best_epoch} - score: {self.best_score:.4f} - train acc: {self.train_acc[-1]:.4f}"
                )
            if improved_score:
                self.best_val_loss = self.val_loss[-1]
                self.best_epoch = epoch + 1
                torch.save(
                    self.model.state_dict(),
                    f"best_val_score_{self.model.__class__.__name__}.pt",
                )
                print(
                    f"{epoch+1}/{epochs} - New best validation perf: {self.best_val_acc:.4f} acc at epoch {self.best_epoch} - score: {self.best_score:.4f} - train acc: {self.train_acc[-1]:.4f}"
                )
            else:
                print(
                    f"{epoch+1}/{epochs} - Validation perf did not improve with acc at {self.val_acc[-1]:.4f} vs {self.best_val_acc:.4f} and score: {self.val_score[-1]:.4f} vs {self.best_score:.4f} - train acc: {self.train_acc[-1]:.4f}"
                )
            if self.scheduler is not None:
                self.scheduler.step(self.best_val_acc)

        print(
            f"Best validation perf: {self.best_val_acc:.4f} acc at epoch {self.best_epoch} - Best score: {self.best_score:.4f}"
        )

    def train_epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        start_time = time.time()
        self.model.train()
        for batch_idx, (images, metadata, targets) in enumerate(self.train_loader):
            data = images.to(self.device)
            if isinstance(targets, list):
                targets = targets[0].to(self.device)
            else:
                targets = targets.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, targets)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            epoch_acc += (
                pred.eq(targets.argmax(dim=1, keepdim=True).view_as(pred)).sum().item()
            )
        self.train_acc.append(epoch_acc / len(self.train_loader.dataset))
        self.train_loss.append(epoch_loss * 10 / len(self.train_loader.dataset))
        self.train_time.append(time.time() - start_time)

        return self.train_loss, self.train_acc, self.train_time

    def val_epoch(self):
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            epoch_targets = []
            epoch_preds = []
            epoch_outputs = []
            for batch_idx, (images, metadata, targets) in enumerate(self.val_loader):
                data = images.to(self.device)
                if isinstance(targets, list):
                    pfs = targets[1]
                    targets = targets[0].to(self.device)
                else:
                    targets = targets.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, targets)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += (
                    pred.eq(targets.argmax(dim=1, keepdim=True).view_as(pred))
                    .sum()
                    .item()
                )
                if self.loss_fn is PROSTATE_LOSS:
                    targets, output, pred = (
                        targets.argmax(dim=1, keepdim=True).to("cpu"),
                        torch.nn.functional.sigmoid(output).to("cpu"),
                        pred.to("cpu"),
                    )

                    epoch_targets.extend([i.item() for i in targets])
                    epoch_preds.extend([i.item() for i in pred])
                    epoch_outputs.extend(
                        [output[k][1].item() for k, i in enumerate(targets)]
                    )
                elif self.loss_fn is LUNG_LOSS:
                    estimates = torch.nn.functional.softmax(output, dim=1).to("cpu")

                    epoch_outputs.extend([i for i in estimates])
                    epoch_targets.extend(pfs)

            val_loss /= len(self.val_loader.dataset)
            val_acc /= len(self.val_loader.dataset)
            self.val_loss.append(val_loss * 10)
            self.val_acc.append(val_acc)
            if self.eval_function is not None and self.loss_fn is PROSTATE_LOSS:
                self.val_score.append(
                    self.eval_function(epoch_targets, epoch_outputs, epoch_preds)
                )
            elif self.eval_function is not None and self.loss_fn is LUNG_LOSS:
                self.val_score.append(self.eval_function(epoch_targets, epoch_outputs))
            self.val_time.append(time.time() - start_time)

        return self.val_loss, self.val_acc, self.val_time

    def plot_loss(self):
        plt.plot(self.train_loss[5:], label="train_loss")
        plt.plot(self.val_loss[5:], label="val_loss")
        plt.title(f"Loss - {self.model.__class__.__name__}")
        plt.legend()
        plt.show()

    def plot_acc(self):
        plt.plot(self.train_acc[5:], label="train_acc")
        plt.plot(self.val_acc[5:], label="val_acc")
        plt.plot(self.val_score[5:], label="val_score")
        plt.title(f"Accuracy - {self.model.__class__.__name__}")
        plt.legend()
        plt.show()
