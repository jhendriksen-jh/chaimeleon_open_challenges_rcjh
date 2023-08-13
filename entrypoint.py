import torch
import numpy as np
from functools import partial
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
from library.datasets import ProstateCancerDataset, LungCancerDataset
from library.models import (
    ProstateImageModel,
    ProstateMetadataModel,
    ProstateCombinedModel,
    get_number_of_parameters,
    LungMetadataModel,
    LungImageModel,
    LungCombinedModel,
)
from library.train import (
    Trainer,
    get_device,
    create_dataloader,
    create_optimizer,
    PROSTATE_LOSS,
    LUNG_LOSS,
)


def prostate_scoring_function(targets, outputs, preds):
    """
    Creates the scoring for the Prostate dataset as defined in challenge
    """
    auc = roc_auc_score(targets, outputs)
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_accuracy = balanced_accuracy_score(targets, preds)
    score = (
        (0.4 * auc)
        + (0.2 * sensitivity)
        + (0.2 * specificity)
        + (0.2 * balanced_accuracy)
    )
    # print(f"\t### AUC: {auc:.3f}, balanced_accuracy: {balanced_accuracy:.3f}, sensitivity: {sensitivity:.3f}, specificity: {specificity:.3f}")
    return score


def lung_scoring_function(trains, days_per_bucket, targets, outputs):
    """
    creates scoring function for the Lung dataset as defined in challenge
    """
    targets = np.array(targets)
    min_time = np.min(targets.T[1])
    max_time = np.max(targets.T[1])
    min_index = int(min_time / days_per_bucket)
    max_index = int(max_time / days_per_bucket)

    truncated_outputs = [i[min_index:max_index] for i in outputs]
    times = np.arange(min_time, max_time - days_per_bucket, days_per_bucket)
    targets = np.array(
        [(i[0], i[1]) for i in targets], dtype=[("event", "bool"), ("time", "float")]
    )
    trains = np.array(
        [(i[0], i[1]) for i in trains], dtype=[("event", "bool"), ("time", "float")]
    )

    cum_auc = cumulative_dynamic_auc(trains, targets, truncated_outputs, times)
    time_auc = cum_auc[1]

    estimates_at_midpoint = [i[: int(max_index / 2)].sum() for i in outputs]
    min_estimate_at_midpoint = min(estimates_at_midpoint)
    max_estimate_at_midpoint = max(estimates_at_midpoint)

    concordance = concordance_index_ipcw(trains, targets, estimates_at_midpoint)
    c_index = concordance[0]

    score = (0.5 * time_auc) + (0.5 * c_index)

    # print(f"\t### Time AUC: {time_auc:.3f}, c_index: {c_index:.3f}, min_estimate_at_midpoint: {min_estimate_at_midpoint:.3f}, max_estimate_at_midpoint: {max_estimate_at_midpoint:.3f}")

    return score


def main(data_directory: str, train: bool = False, cancer: str = None):
    if train and cancer == "prostate":
        train_dataset = ProstateCancerDataset(data_directory)
        val_dataset = ProstateCancerDataset(data_directory, split_type="val")

        image_model = ProstateImageModel()
        metadata_model = ProstateMetadataModel()

        device = get_device()
        train_loader = create_dataloader(train_dataset, batch_size=144)
        val_loader = create_dataloader(val_dataset, batch_size=144)
        optimizer = create_optimizer(image_model)

        num_epochs = 10

        print(
            f"\n######## Training Image Model - {get_number_of_parameters(image_model)} params ########\n"
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, verbose=True
        )
        ProstateImageTrainer = Trainer(
            image_model,
            train_loader,
            val_loader,
            PROSTATE_LOSS,
            optimizer,
            device,
            evaluation_function=prostate_scoring_function,
            scheduler=scheduler,
        )
        ProstateImageTrainer.train(num_epochs)
        print(
            f"Training averaged {sum(ProstateImageTrainer.train_time)/num_epochs:.2f}s per epoch"
        )
        # ProstateImageTrainer.plot_acc()
        # ProstateImageTrainer.plot_loss()
        del ProstateImageTrainer
        del image_model

        print(
            f"\n######## Training Metadata Model - {get_number_of_parameters(metadata_model)} ########\n"
        )

        metadata_optimizer = create_optimizer(metadata_model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            metadata_optimizer, mode="max", factor=0.5, patience=10, verbose=True
        )
        ProstateMetadataTrainer = Trainer(
            metadata_model,
            train_loader,
            val_loader,
            PROSTATE_LOSS,
            metadata_optimizer,
            device,
            evaluation_function=prostate_scoring_function,
            scheduler=scheduler,
        )
        ProstateMetadataTrainer.train(num_epochs)
        print(
            f"Training averaged {sum(ProstateMetadataTrainer.train_time)/num_epochs:.2f}s per epoch"
        )
        # ProstateMetadataTrainer.plot_acc()
        # ProstateMetadataTrainer.plot_loss()
        del ProstateMetadataTrainer
        del metadata_model
        num_epochs = 15
        combo_model = ProstateCombinedModel()
        print(
            f"\n######## Training Combined Model - {get_number_of_parameters(combo_model)} ########\n"
        )

        combo_optimizer = create_optimizer(combo_model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            combo_optimizer, mode="max", factor=0.5, patience=10, verbose=True
        )
        ProstateCombinedTrainer = Trainer(
            combo_model,
            train_loader,
            val_loader,
            PROSTATE_LOSS,
            combo_optimizer,
            device,
            evaluation_function=prostate_scoring_function,
            scheduler=scheduler,
        )
        ProstateCombinedTrainer.train(num_epochs)
        print(
            f"Training averaged {sum(ProstateCombinedTrainer.train_time)/num_epochs:.2f}s per epoch"
        )
        # ProstateCombinedTrainer.plot_acc()
        # ProstateCombinedTrainer.plot_loss()

    elif train and cancer == "lung":
        train_dataset = LungCancerDataset(data_directory)
        val_dataset = LungCancerDataset(data_directory, split_type="val")

        lung_model = LungCombinedModel()

        device = get_device()
        train_loader = create_dataloader(train_dataset, batch_size=144)
        val_loader = create_dataloader(val_dataset, batch_size=144)

        num_epochs = 75

        print(
            f"\n######## Training Lung Model - {get_number_of_parameters(lung_model)} ########\n"
        )

        metadata_optimizer = create_optimizer(lung_model, lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            metadata_optimizer, mode="max", factor=0.25, patience=10, verbose=True
        )

        lung_score_generator = partial(
            lung_scoring_function,
            train_dataset.get_all_ground_truth(),
            train_dataset.days_per_gt_bucket,
        )

        LungModelTrainer = Trainer(
            lung_model,
            train_loader,
            val_loader,
            LUNG_LOSS,
            metadata_optimizer,
            device,
            evaluation_function=lung_score_generator,
            scheduler=scheduler,
        )

        LungModelTrainer.train(num_epochs)
        print(
            f"Training averaged {sum(LungModelTrainer.train_time)/num_epochs:.2f}s per epoch"
        )
        LungModelTrainer.plot_loss()
        LungModelTrainer.plot_acc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="data directory")
    parser.add_argument(
        "--train",
        action="store_true",
        help="runs through training with given parameters",
    )
    parser.add_argument(
        "--cancer",
        type=str,
        default=None,
        choices=["prostate", "lung"],
    )
    input_args = parser.parse_args()
    main(
        data_directory=input_args.data_dir,
        train=input_args.train,
        cancer=input_args.cancer,
    )
