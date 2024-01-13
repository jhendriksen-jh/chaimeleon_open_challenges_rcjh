import os
import time
import json
import torch
import torchvision
import numpy as np
import datetime
from functools import partial
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    recall_score,
    roc_curve,
    auc,
    f1_score,
)
from lifelines.utils import concordance_index
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
    ProstateImageResnet18PretrainedModel,
    ProstateCombinedResnet18PretrainedModel_V2_Grid,
    ProstateCombinedResnet18PretrainedModel_V2_1_Grid,
    get_number_of_parameters,
    LungMetadataModel,
    LungImageModel,
    LungCombinedModel,
    LungCombinedResnet18PretrainedModel,
    LungCombinedResnet18PretrainedModelV2AllMeta,
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
        (0.4 * auc_chai)
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


def lung_scoring_function(months_per_gt_bucket, targets, outputs):
    """
    creates scoring function for the Lung dataset as defined in challenge
    """
    targets = np.array(targets).tolist()

    # as they do in their repo
    predictions = []
    ground_truths = []
    ground_truth_events = []
    for (event, survival_time_in_months), raw_prediction in zip(targets, outputs):
        os_months = raw_prediction * months_per_gt_bucket
        prediction = float(os_months)
        predictions.append(prediction)
        ground_truth = float(survival_time_in_months)
        ground_truths.append(ground_truth)
        ground_truth_event = int(event)
        ground_truth_events.append(ground_truth_event)

    c_index = concordance_index(ground_truths, predictions, ground_truth_events)

    return c_index


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
        best_model_performances = []
        start_overall_time = time.time()

        for random_seed in [24042]:
            num_epochs = 600
            device = get_device()
            training_batch_size = 48


            train_dataset = ProstateCancerDataset(
                data_directory, random_seed=random_seed
            )
            train_gt_details = get_prostate_gt_split(train_dataset)
            train_details = f"Training Prostate Cancer Dataset - {train_gt_details[0]} low risk cases, {train_gt_details[1]} high risk cases, {train_gt_details[2]} high risk ratio\n"
            print(train_details)
            val_dataset = ProstateCancerDataset(
                data_directory, split_type="val", random_seed=random_seed
            )
            val_gt_details = get_prostate_gt_split(val_dataset)
            val_details = f"Validation Prostate Cancer Dataset - {val_gt_details[0]} low risk cases, {val_gt_details[1]} high risk cases, {val_gt_details[2]} high risk ratio\n\n"
            print(val_details)

            training_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            training_batch_size = 48
            factor = 0.75
            patience = 60
            starting_lr = 0.005
            # slices = [1, 3, 6, 9, 18]
            input_slices = 3
            for patience in [60, 30]:
                for starting_lr, factor in [
                    (0.05, 0.25),
                    (0.01, 0.5),
                    (0.01, 0.1),
                    (0.007, 0.5),
                    (0.003, 0.5),
                    (0.0005, 0.5),
                ]:
                    for frozen_layers in [
                        [],
                        ["layer2", "layer3"],
                    ]:
                        for pre_model in [
                            ProstateCombinedResnet18PretrainedModel_V2_1_Grid,
                            ProstateCombinedResnet18PretrainedModel_V2_Grid,
                            ProstateCombinedResnet18PretrainedModel,
                        ]:
                            train_dataset = ProstateCancerDataset(
                                data_directory,
                                input_slice_count=input_slices,
                                random_seed=random_seed,
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

                            combo_model = pre_model(frozen_layers=frozen_layers)
                            print(
                                f"\n######## Training {combo_model.__class__.__name__} {frozen_layers} frozen {starting_lr} lr {factor} factor - {get_number_of_parameters(combo_model):_} ########\n"
                            )
                            training_dir = f"./mega_tuning_exp_training_details/pretrained_model_raw_metadata/{combo_model.__class__.__name__}/{training_timestamp}/avg_fill_augmentation_reduced_loss_bias_16_4/{frozen_layers}_frozen/{starting_lr}_starting_learning_rate_{factor}_factor_{patience}_patience/"
                            os.makedirs(training_dir, exist_ok=True)
                            combo_optimizer = create_optimizer(
                                combo_model, lr=starting_lr
                            )
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
                            best_model_performances.append(
                                f"{combo_model.__class__.__name__}_{frozen_layers}_frozen_{starting_lr}_learning_rate_{patience} - best_epoch: {training_details['best_epoch']} - best_acc: {training_details['best_val_acc']} - best_score: {training_details['best_val_score']} - {get_number_of_parameters(combo_model):_} params"
                            )
                            with open(
                                f"{training_dir}/training_details.json", "w"
                            ) as f:
                                json.dump(training_details, f, indent=4)
                            print(
                                f"Training {combo_model.__class__.__name__} averaged {sum(ProstateCombinedPretrainedTrainer.train_time)/num_epochs:.2f}s per epoch"
                            )
                            del combo_model
                            with open(
                                f"./{training_timestamp}_overall_exp_agg_results.txt",
                                "w",
                            ) as f:
                                print(
                                    f"Training for all models complete - {time.time() - start_overall_time:.2f}s - {num_epochs} epochs\n"
                                )
                                f.write(
                                    f"Training for all models complete - {time.time() - start_overall_time:.2f}s - {num_epochs} epochs\n"
                                )
                                print(train_details)
                                f.write(train_details)
                                print(val_details)
                                f.write(val_details)
                                for best_model_performance in best_model_performances:
                                    print(best_model_performance)
                                    f.write(f"\n{best_model_performance}")

        with open(f"./{training_timestamp}_overall_exp_agg_results.txt", "w") as f:
            print(
                f"Training for all models complete - {time.time() - start_overall_time:.2f}s - {num_epochs} epochs\n"
            )
            f.write(
                f"Training for all models complete - {time.time() - start_overall_time:.2f}s - {num_epochs} epochs\n"
            )
            print(train_details)
            f.write(train_details)
            print(val_details)
            f.write(val_details)
            for best_model_performance in best_model_performances:
                print(best_model_performance)
                f.write(f"\n{best_model_performance}")

    elif train and cancer == "lung":
        best_model_performances = []
        random_seed = 42024
        training_batch_size = 48
        starting_lr = 0.001
        patience = 50
        factor = 0.5
        number_of_buckets = 400
        months_per_gt_bucket = 0.5
        device = get_device()

        num_epochs = 500
        training_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for event_filter in [
            # True, 
            False,
            ]:
            for frozen_layers in [
                # [],
                ["layer2", "layer3"],
            ]:
                for number_of_buckets, months_per_gt_bucket in [
                    (400, 0.5),
                    (200, 1),
                    (100, 1.5),
                    (50, 3),
                ]:
                    for lung_model, categorical_metadata in [
                        (
                            LungCombinedResnet18PretrainedModel(
                                frozen_layers=frozen_layers,
                                number_of_buckets=number_of_buckets,
                            ),
                            ["gender", "smoking_status"],
                        ),
                        (
                            LungCombinedResnet18PretrainedModelV2AllMeta(
                                frozen_layers=frozen_layers,
                                number_of_buckets=number_of_buckets,
                            ),
                            [
                                "gender",
                                "smoking_status",
                                "clinical_category",
                                "regional_nodes_category",
                                "metastasis_category",
                            ],
                        ),
                    ]:
                        train_dataset = LungCancerDataset(
                            data_directory,
                            random_seed=random_seed,
                            number_of_buckets=number_of_buckets,
                            months_per_gt_bucket=months_per_gt_bucket,
                            categorical_metadata=categorical_metadata,
                        )
                        val_dataset = LungCancerDataset(
                            data_directory,
                            split_type="val",
                            random_seed=random_seed,
                            number_of_buckets=number_of_buckets,
                            months_per_gt_bucket=months_per_gt_bucket,
                            categorical_metadata=categorical_metadata,
                        )
                        train_loader = create_dataloader(
                            train_dataset, batch_size=training_batch_size
                        )
                        val_loader = create_dataloader(
                            val_dataset, batch_size=training_batch_size
                        )

                        print(
                            f"\n######## Training {lung_model.__class__.__name__} - {get_number_of_parameters(lung_model)} - {frozen_layers} frozen - {number_of_buckets} - buckets {months_per_gt_bucket} months - {event_filter} event filter ########\n"
                        )
                        for starting_lr in [0.05, 0.01, 0.003, 0.1]:
                            print(f"Starting LR: {starting_lr}")

                            training_dir = f"./mega_lung_training/avg_fil_augmentation_event_loss_filter_{event_filter}/{lung_model.__class__.__name__}/{training_timestamp}/{number_of_buckets}_buckets_{months_per_gt_bucket}_months/{frozen_layers}/{starting_lr}_starting_lr/"
                            os.makedirs(training_dir, exist_ok=True)

                            metadata_optimizer = create_optimizer(
                                lung_model, lr=starting_lr
                            )
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                metadata_optimizer,
                                mode="max",
                                factor=factor,
                                patience=patience,
                                verbose=True,
                            )

                            lung_score_generator = partial(
                                lung_scoring_function,
                                train_dataset.months_per_gt_bucket,
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
                                training_dir=training_dir,
                                event_filter=event_filter,
                            )
                            training_details = LungModelTrainer.train(
                                num_epochs, training_timestamp=training_timestamp
                            )
                            print(
                                f"Training averaged {sum(LungModelTrainer.train_time)/num_epochs:.2f}s per epoch"
                            )
                            training_details.update(
                                {
                                    "total_epochs": num_epochs,
                                    "factor": factor,
                                    "patience": patience,
                                    "model_name": f"{lung_model.__class__.__name__}",
                                    "training_batch_size": training_batch_size,
                                    "timestamp": training_timestamp,
                                    "starting_lr": starting_lr,
                                    "random_seed": random_seed,
                                    "number_of_buckets": number_of_buckets,
                                    "months_per_gt_bucket": months_per_gt_bucket,
                                }
                            )
                            best_model_performances.append(
                                f"{lung_model.__class__.__name__}_{starting_lr}_learning_rate_{frozen_layers}_frozen_{number_of_buckets}_buckets_{months_per_gt_bucket}_months_{event_filter}_event_filter - best_epoch: {training_details['best_epoch']} - best_acc: {training_details['best_val_acc']} - best_score: {training_details['best_val_score']} - {get_number_of_parameters(lung_model):_} params"
                            )
                            with open(f"{training_dir}/training_details.json", "w") as f:
                                json.dump(training_details, f, indent=4)
                            with open(
                                f"./{training_timestamp}_overall_exp_agg_results.txt", "w"
                            ) as f:
                                for best_model_performance in best_model_performances:
                                    print(best_model_performance)
                                    f.write(f"\n{best_model_performance}")
        with open(f"./{training_timestamp}_overall_exp_agg_results.txt", "w") as f:
            for best_model_performance in best_model_performances:
                print(best_model_performance)
                f.write(f"\n{best_model_performance}")

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
