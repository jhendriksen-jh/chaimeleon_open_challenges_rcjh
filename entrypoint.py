import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
from library.datasets import ProstateCancerDataset, LungCancerDataset
from library.models import (
    ProstateImageModel,
    ProstateMetadataModel,
    ProstateCombinedModel,
    get_number_of_parameters,
    LungMetadataModel,
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
    score = (0.4*auc) + (0.2*sensitivity) + (0.2*specificity) + (0.2*balanced_accuracy)
    # print(f"\t### AUC: {auc:.3f}, balanced_accuracy: {balanced_accuracy:.3f}, sensitivity: {sensitivity:.3f}, specificity: {specificity:.3f}")
    return score


def lung_scoring_function(targets, outputs, preds):
    """
    creates scoring function for the Lung dataset as defined in challenge
    """



def main(data_directory: str, train: bool = False, cancer: str = None):
    if train and cancer == 'prostate':
        # import pudb; pudb.set_trace()
        train_dataset = ProstateCancerDataset(data_directory)
        val_dataset = ProstateCancerDataset(data_directory, split_type="val")

        image_model = ProstateImageModel()
        metadata_model = ProstateMetadataModel()

        device = get_device()
        train_loader = create_dataloader(train_dataset, batch_size=144)
        val_loader = create_dataloader(val_dataset, batch_size=144)
        optimizer = create_optimizer(image_model)

        num_epochs = 30

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
        num_epochs = 75
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
    
    elif train and cancer == 'lung':
        # import pudb; pudb.set_trace()
        train_dataset = LungCancerDataset(data_directory)
        val_dataset = LungCancerDataset(data_directory, split_type="val")

        metadata_model = LungMetadataModel()

        device = get_device()
        train_loader = create_dataloader(train_dataset, batch_size=144)
        val_loader = create_dataloader(val_dataset, batch_size=144)

        num_epochs = 50

        print(
            f"\n######## Training Metadata Model - {get_number_of_parameters(metadata_model)} ########\n"
        )

        metadata_optimizer = create_optimizer(metadata_model, lr=0.005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            metadata_optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        LungMetadataModelTrainer = Trainer(
            metadata_model,
            train_loader,
            val_loader,
            LUNG_LOSS,
            metadata_optimizer,
            device,
            scheduler=scheduler,
        )

        LungMetadataModelTrainer.train(num_epochs)
        print(
            f"Training averaged {sum(LungMetadataModelTrainer.train_time)/num_epochs:.2f}s per epoch"
        )
        LungMetadataModelTrainer.plot_loss()
        LungMetadataModelTrainer.plot_acc()
        








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
    main(data_directory=input_args.data_dir, train=input_args.train, cancer=input_args.cancer)
