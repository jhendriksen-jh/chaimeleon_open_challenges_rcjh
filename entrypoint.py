import os
import json
import torch
import torchvision
import numpy as np
import datetime
from functools import partial
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, recall_score, roc_curve, auc, f1_score
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
from library.datasets import ProstateCancerDataset, LungCancerDataset
from library.models import (
    ProstateImageModel,
    ProstateMetadataModel,
    ProstateMetadataModelV2Small,
    ProstateMetadataModelV3,
    ProstateCombinedModel,
    ProstateCombinedModelV1Tiny,
    ProstateCombinedModelV1_1Tiny,
    ProstateCombinedResnet18PretrainedModel,
    get_number_of_parameters,
    LungMetadataModel,
    LungImageModel,
    LungCombinedModel,
)
from library.train import (
    Trainer,
    FineTuner,
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
    roc_auc = roc_auc_score(targets, outputs)
    fpr, tpr, _ = roc_curve(targets, preds)
    auc_chai = auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    sensitivity = tp / (tp + fn)
    sensitivity_recall = recall_score(targets, preds)
    specificity = tn / (tn + fp)
    balanced_accuracy = balanced_accuracy_score(targets, preds)
    score = (
        (0.2 * roc_auc)
        + (0.2 * auc_chai)
        + (0.2 * sensitivity)
        + (0.2 * specificity)
        + (0.2 * balanced_accuracy)
    )
    score_dict = {
        "score": score,
        "roc_auc": roc_auc,
        "auc_chai": auc_chai,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "sensitivity_recall": sensitivity_recall,
        "specificity": specificity,
    }
    # print((f"\t### AUC: {roc_auc:.3f} - auc_chai: {auc_chai:.3f},\n\t balanced_accuracy: {balanced_accuracy:.3f},\n\t"
    #        f"sensitivity: {sensitivity:.3f} - sensitivity_recall: {sensitivity_recall:.3f},\n\t specificity: {specificity:.3f}\n\t"))
    return score, score_dict


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


def get_prostate_gt_split(dataset):
    num_low = len([i for i in dataset.ground_truth_list if i == "Low"])
    num_high = len([i for i in dataset.ground_truth_list if i == "High"])

    assert num_low + num_high == len(
        dataset
    ), "ground truth parsing doesn't match length of dataset"

    high_ratio = num_low / (num_low + num_high)

    return num_low, num_high, high_ratio


def main(data_directory: str, train: bool = False, cancer: str = None):
    if train and cancer == "prostate":
        # image_model = ProstateImageModel()
        # image_model = torchvision.models.resnet18(pretrained=True)
        # num_ftrs = image_model.fc.in_features
        # image_model.fc = torch.nn.Sequential(
        #         torch.nn.Linear(num_ftrs, 2),
        #         torch.nn.Dropout(p=0.025),
        #         torch.nn.Linear(1024, 2),
        #     )

        # optimizer = create_optimizer(image_model)
        for random_seed in [24042]:
            num_epochs = 800
            device = get_device()
            training_batch_size = 48

            # print(
            #     f"\n######## Training {image_model.__class__.__name__} - {get_number_of_parameters(image_model)} params ########\n"
            # )

            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer, mode="max", factor=0.25, patience=75, verbose=True
            # )

            # ProstateImageTrainer = FineTuner(
            #     image_model,
            #     train_loader,
            #     val_loader,
            #     PROSTATE_LOSS,
            #     optimizer,
            #     device,
            #     evaluation_function=prostate_scoring_function,
            #     scheduler=scheduler,
            # )
            # ProstateImageTrainer.train(num_epochs)
            # print(
            #     f"Training averaged {sum(ProstateImageTrainer.train_time)/num_epochs:.2f}s per epoch"
            # )
            # del image_model
            for metadata_model_class in [ProstateMetadataModel, ProstateMetadataModelV2Small, ProstateMetadataModelV3]:
                metadata_model = metadata_model_class()

                train_dataset = ProstateCancerDataset(data_directory, random_seed=random_seed)
                train_gt_details = get_prostate_gt_split(train_dataset)
                print(
                    f"Training Prostate Cancer Dataset - {train_gt_details[0]} low risk cases, {train_gt_details[1]} high risk cases, {train_gt_details[2]} high risk ratio"
                )
                val_dataset = ProstateCancerDataset(
                    data_directory, split_type="val", random_seed=random_seed
                )
                val_gt_details = get_prostate_gt_split(val_dataset)
                print(
                    f"Validation Prostate Cancer Dataset - {val_gt_details[0]} low risk cases, {val_gt_details[1]} high risk cases, {val_gt_details[2]} high risk ratio"
                )
                train_loader = create_dataloader(train_dataset, batch_size=training_batch_size)
                val_loader = create_dataloader(val_dataset, batch_size=16)
                print(
                    f"\n######## Training Metadata Model - {get_number_of_parameters(metadata_model):_} ########\n"
                )
                factor = 0.75
                patience = 60
                training_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                training_dir = f"./training_details/metadata_model_raw_values/{metadata_model.__class__.__name__}/{training_timestamp}/"
                os.makedirs(training_dir, exist_ok=True)
                metadata_optimizer = create_optimizer(metadata_model)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    metadata_optimizer,
                    mode="max",
                    factor=factor,
                    patience=patience,
                    verbose=True,
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
                    training_dir=training_dir,
                )
                training_details = ProstateMetadataTrainer.train(num_epochs, training_timestamp)
                training_details.update(
                    {
                        "total_epochs": num_epochs,
                        "factor": factor,
                        "patience": patience,
                        "model_name": metadata_model.__class__.__name__,
                        "training_batch_size": training_batch_size,
                        "timestamp": training_timestamp,
                        "random_seed": random_seed,
                    }
                )
                with open(f"{training_dir}/training_details.json", "w") as f:
                    json.dump(training_details, f, indent=4)
                print(
                    f"Training averaged {sum(ProstateMetadataTrainer.train_time)/num_epochs:.2f}s per epoch"
                )
                del metadata_model

            # ProstateMetadataTrainer.plot_acc()
            # ProstateMetadataTrainer.plot_loss()
            # import pudb; pudb.set_trace()
            training_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            training_batch_size = 48
            factor = 0.75
            patience = 60
            starting_lr = 0.005
            # slices = [1, 3, 6, 9, 18]
            input_slices = 3
            for starting_lr in [0.01, 0.003, 0.001, 0.0003]:
                for frozen_layers in [[], ["layer1", "layer2", "layer3"], ["layer1", "layer2", "layer3", "layer4"]]:
                    train_dataset = ProstateCancerDataset(
                        data_directory, input_slice_count=input_slices, random_seed=random_seed
                    )
                    val_dataset = ProstateCancerDataset(
                        data_directory,
                        split_type="val",
                        input_slice_count=input_slices,
                        random_seed=random_seed,
                    )
                    train_gt_details = get_prostate_gt_split(train_dataset)
                    print(
                        f"\n\nTraining Prostate Cancer Dataset - {train_gt_details[0]} low risk cases, {train_gt_details[1]} high risk cases, {train_gt_details[2]} high risk ratio"
                    )
                    val_gt_details = get_prostate_gt_split(val_dataset)
                    print(
                        f"Validation Prostate Cancer Dataset - {val_gt_details[0]} low risk cases, {val_gt_details[1]} high risk cases, {val_gt_details[2]} high risk ratio"
                    )

                    train_loader = create_dataloader(
                        train_dataset, batch_size=training_batch_size
                    )
                    val_loader = create_dataloader(val_dataset, batch_size=16)

                    # combo_model = ProstateCombinedModelV1_1Tiny(input_slice_count=input_slices)
                    combo_model = ProstateCombinedResnet18PretrainedModel(frozen_layers=frozen_layers)
                    print(
                        f"\n######## Training Combined Model {input_slices} slices - {get_number_of_parameters(combo_model):_} ########\n"
                    )
                    training_dir = f"./training_details/combined_model_raw_metadata/{combo_model.__class__.__name__}/{training_timestamp}/{frozen_layers}_frozen/"
                    os.makedirs(training_dir, exist_ok=True)
                    combo_optimizer = create_optimizer(combo_model, lr=starting_lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        combo_optimizer,
                        mode="max",
                        factor=factor,
                        patience=patience,
                        verbose=True,
                    )
                    ProstateCombinedPretrainedTrainer = Trainer(
                        combo_model,
                        train_loader,
                        val_loader,
                        PROSTATE_LOSS,
                        combo_optimizer,
                        device,
                        evaluation_function=prostate_scoring_function,
                        scheduler=scheduler,
                        training_dir=training_dir,
                    )
                    training_details = ProstateCombinedPretrainedTrainer.train(
                        num_epochs, training_timestamp=training_timestamp
                    )
                    training_details.update(
                        {
                            "total_epochs": num_epochs,
                            "factor": factor,
                            "patience": patience,
                            "model_name": f"{combo_model.__class__.__name__}_{input_slices}_slice",
                            "training_batch_size": training_batch_size,
                            "timestamp": training_timestamp,
                            "starting_lr": starting_lr,
                            "input_slices": input_slices,
                            "random_seed": random_seed,
                            "frozen_layers": frozen_layers, 
                        }
                    )
                    with open(f"{training_dir}/training_details.json", "w") as f:
                        json.dump(training_details, f, indent=4)
                    print(
                        f"Training averaged {sum(ProstateCombinedPretrainedTrainer.train_time)/num_epochs:.2f}s per epoch"
                    )
                    del combo_model
            
            training_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            training_batch_size = 48
            factor = 0.75
            patience = 60
            starting_lr = 0.005
            slices = [1, 3, 6, 9, 18]
            for starting_lr in [0.01, 0.003, 0.001, 0.0003]:
                for input_slices in slices:
                    for model in [ProstateCombinedModelV1_1Tiny, ProstateCombinedModelV1Tiny]:
                        train_dataset = ProstateCancerDataset(
                            data_directory, input_slice_count=input_slices, random_seed=random_seed
                        )
                        val_dataset = ProstateCancerDataset(
                            data_directory,
                            split_type="val",
                            input_slice_count=input_slices,
                            random_seed=random_seed,
                        )
                        train_gt_details = get_prostate_gt_split(train_dataset)
                        print(
                            f"\n\nTraining Prostate Cancer Dataset - {train_gt_details[0]} low risk cases, {train_gt_details[1]} high risk cases, {train_gt_details[2]} high risk ratio"
                        )
                        val_gt_details = get_prostate_gt_split(val_dataset)
                        print(
                            f"Validation Prostate Cancer Dataset - {val_gt_details[0]} low risk cases, {val_gt_details[1]} high risk cases, {val_gt_details[2]} high risk ratio"
                        )

                        train_loader = create_dataloader(
                            train_dataset, batch_size=training_batch_size
                        )
                        val_loader = create_dataloader(val_dataset, batch_size=16)

                        combo_model = model(input_slice_count=input_slices)
                        # combo_model = ProstateCombinedResnet18PretrainedModel(frozen_layers=frozen_layers)
                        print(
                            f"\n######## Training Combined Model {input_slices} slices - {get_number_of_parameters(combo_model):_} ########\n"
                        )
                        training_dir = f"./training_details/combined_model_raw_metadata/{combo_model.__class__.__name__}/{training_timestamp}/{input_slices}_input_slices/"
                        os.makedirs(training_dir, exist_ok=True)
                        combo_optimizer = create_optimizer(combo_model, lr=starting_lr)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            combo_optimizer,
                            mode="max",
                            factor=factor,
                            patience=patience,
                            verbose=True,
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
                            training_dir=training_dir,
                        )
                        training_details = ProstateCombinedTrainer.train(
                            num_epochs, training_timestamp=training_timestamp
                        )
                        training_details.update(
                            {
                                "total_epochs": num_epochs,
                                "factor": factor,
                                "patience": patience,
                                "model_name": f"{combo_model.__class__.__name__}_{input_slices}_slice",
                                "training_batch_size": training_batch_size,
                                "timestamp": training_timestamp,
                                "starting_lr": starting_lr,
                                "input_slices": input_slices,
                                "random_seed": random_seed,
                            }
                        )
                        with open(f"{training_dir}/training_details.json", "w") as f:
                            json.dump(training_details, f, indent=4)
                        print(
                            f"Training averaged {sum(ProstateCombinedTrainer.train_time)/num_epochs:.2f}s per epoch"
                        )
                        del combo_model
            

                # ProstateCombinedTrainer.plot_acc()
                # ProstateCombinedTrainer.plot_loss()

        # ProstateImageTrainer.plot_acc()
        # ProstateImageTrainer.plot_loss()
        ProstateMetadataTrainer.plot_acc()
        ProstateMetadataTrainer.plot_loss()
        ProstateCombinedPretrainedTrainer.plot_acc()
        ProstateCombinedPretrainedTrainer.plot_loss()
        ProstateCombinedTrainer.plot_acc()
        ProstateCombinedTrainer.plot_loss()

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
            metadata_optimizer, mode="max", factor=0.5, patience=15, verbose=True
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
