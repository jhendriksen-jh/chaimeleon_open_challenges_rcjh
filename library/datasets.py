"""
handling of data provided by chaimeleon 
"""

import os
import json
import torch
import random
import numpy as np
import pandas as pd
import nibabel as nib
import torchvision as tv

from collections import defaultdict
from PIL import Image


class ChaimeleonData:
    def __init__(self, data_directory, input_slice_count: int = 1, random_seed=2038):
        self.random_seed = random_seed
        self.data_directory = data_directory
        self.input_slice_count = input_slice_count
        self.image_arrays = []
        self.raw_cases = {}
        self.prepared_cases = []

        self.process_data_directory()

    def process_data_directory(self, data_directory=None):
        if data_directory is None:
            data_directory = self.data_directory
        self.add_raw_case_data(data_directory)
        self.get_dataset_splits()

    def add_raw_case_data(self, case_directory):
        case_folders = os.listdir(case_directory)
        for case_folder in case_folders:
            case_path = os.path.join(case_directory, case_folder)
            case_files = os.listdir(case_path)
            for case_file in case_files:
                if case_file.endswith(".nii.gz"):
                    case_image = self.load_image_file(
                        os.path.join(case_path, case_file)
                    )
                elif case_file.endswith("ground_truth.json"):
                    case_ground_truth = self.add_raw_ground_truth(
                        os.path.join(case_path, case_file)
                    )
                elif case_file.endswith(".json"):
                    case_metadata = self.add_raw_metadata(
                        os.path.join(case_path, case_file)
                    )
            raw_case_data = {
                "image": case_image,
                "metadata": case_metadata,
                "ground_truth": case_ground_truth,
            }
            self.raw_cases[case_folder] = raw_case_data
        return raw_case_data

    def add_raw_metadata(self, metadata_file):
        with open(metadata_file) as f:
            raw_metadata = json.load(f)
        return raw_metadata

    def add_raw_ground_truth(self, ground_truth_file):
        with open(ground_truth_file) as f:
            raw_ground_truth = json.load(f)
        return raw_ground_truth

    def load_image_file(self, image_file):
        nifti_image = nib.load(image_file)
        nii_data = nifti_image.get_fdata()
        # nii_affine = nifti_image.get_affine()
        # nii_header = nifti_image.get_header()
        nii_chunks = []
        for k in range(self.input_slice_count):
            u = int(np.ceil((k + 1) * (nii_data.shape[-1] / self.input_slice_count)))
            l = int(np.floor((k) * (nii_data.shape[-1] / self.input_slice_count)))
            nii_chunk = np.mean(nii_data[:, :, l:u], axis=2)
            nii_chunk = (
                (nii_chunk - nii_chunk.min()) / (nii_chunk.max() - nii_chunk.min())
            ) * 255
            nii_chunks.append(nii_chunk)
        nii_chunked_image = np.array(nii_chunks).astype(np.uint8)
        nii_chunked_image = np.transpose(nii_chunked_image, (1, 2, 0))
        self.image_arrays.append(nii_chunked_image)
        return nii_chunked_image

    def get_dataset_splits(
        self, train_percentage=0.85, val_percentage=0.15, test_percentage=0.0
    ):
        assert (
            train_percentage + val_percentage + test_percentage == 1
        ), "split percentages must add up to 1"
        dataset_keys = list(self.raw_cases.keys())
        random.seed(self.random_seed)
        random.shuffle(dataset_keys)
        self.train_keys = dataset_keys[: int(len(dataset_keys) * train_percentage)]
        self.val_keys = dataset_keys[
            int(len(dataset_keys) * train_percentage) : int(
                len(dataset_keys) * (train_percentage + val_percentage)
            )
        ]
        self.test_keys = dataset_keys[
            int(len(dataset_keys) * (train_percentage + val_percentage)) :
        ]
        assert len(self.train_keys) + len(self.val_keys) + len(self.test_keys) == len(
            dataset_keys
        )
        self.keys_by_split = {
            "train": self.train_keys,
            "val": self.val_keys,
            "test": self.test_keys,
        }

    def __len__(self):
        return len(self.prepared_cases)


class ProstateCancerDataset(ChaimeleonData):
    def __init__(
        self,
        data_directory,
        split_type="train",
        random_seed=20380119,
        input_slice_count: int = 1,
    ):
        super().__init__(
            data_directory, input_slice_count=input_slice_count, random_seed=random_seed
        )
        self.split_type = split_type
        self.split_keys = self.keys_by_split[split_type]
        # self.categorical_metadata = ['histology_type', 'pirads', 'neural_invasion', 'vascular_invasion', 'lymphatic_invasion']
        # self.categorical_metadata = [
        #     "histology_type",
        #     "neural_invasion",
        #     "vascular_invasion",
        #     "lymphatic_invasion",
        # ]
        self.categorical_metadata = []
        self.numerical_metadata = ["age", "psa"]
        self.image_size = self.image_arrays[0].shape[:-1]
        self.ground_truth_list = []
        self.get_metadata_details()
        self.define_image_transformations(split_type)
        self.prepare_dataset()

    def prepare_dataset(self):
        prepared_cases = []
        for key in self.split_keys:
            raw_case = self.raw_cases[key]
            raw_image = raw_case["image"]
            prepped_metadata = self.normalize_metadata(raw_case["metadata"])
            normalized_ground_truth = self.normalize_ground_truth(
                raw_case["ground_truth"]
            )
            self.ground_truth_list.append(raw_case["ground_truth"])
            prepared_cases.append(
                {
                    key: {
                        "image": raw_image,
                        "metadata": prepped_metadata,
                        "ground_truth": normalized_ground_truth,
                    }
                }
            )
        self.prepared_cases = prepared_cases

    def define_image_transformations(self, split_type):
        if split_type == "train":
            image_transformations = tv.transforms.Compose(
                [
                    tv.transforms.ToTensor(),
                    tv.transforms.RandomAffine(
                        degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
                    ),
                    # tv.transforms.Resize(self.image_size, antialias=True),
                ]
            )
        else:
            image_transformations = tv.transforms.Compose(
                [
                    tv.transforms.ToTensor(),
                    # tv.transforms.Resize(self.image_size, antialias=True),
                ]
            )
        self.image_transformations = image_transformations
        return image_transformations

    def get_metadata_details(self):
        metadata_details = defaultdict(lambda: {"values": []})
        for case, case_data in self.raw_cases.items():
            for key, value in case_data["metadata"].items():
                metadata_details[key]["values"].append(value)

        for details in metadata_details.values():
            details["max"] = max(details["values"])
            details["min"] = min(details["values"])
            details["length"] = len(details["values"])
            values = details["values"]
            unique_values = set(values)
            sorted_values = list(unique_values)
            sorted_values.sort()
            details["sorted_values"] = sorted_values

        self.metadata_details = metadata_details
        return metadata_details

    def normalize_metadata(self, raw_metadata):
        all_encoded_metadata = np.zeros((0, 1))
        for key in self.categorical_metadata:
            current_value = raw_metadata[key]
            num_possible_values = len(self.metadata_details[key]["sorted_values"])
            encoded_location = self.metadata_details[key]["sorted_values"].index(
                current_value
            )
            encoded_metadata = np.zeros((num_possible_values, 1))
            encoded_metadata[encoded_location] = 1
            all_encoded_metadata = np.concatenate(
                (all_encoded_metadata, encoded_metadata), axis=0
            )
        for key in self.numerical_metadata:
            current_value = raw_metadata[key]
            max_value = self.metadata_details[key]["max"]
            min_value = self.metadata_details[key]["min"]
            normalized_value = (current_value - min_value) / (max_value - min_value)
            normalized_value = np.array([[normalized_value]])
            all_encoded_metadata = np.concatenate(
                (all_encoded_metadata, normalized_value), axis=0
            )
        return all_encoded_metadata

    def normalize_ground_truth(self, raw_ground_truth):
        normalized_ground_truth = np.zeros((2, 1))
        if raw_ground_truth == "Low":
            normalized_ground_truth[0] = 1
        elif raw_ground_truth == "High":
            normalized_ground_truth[1] = 1
        else:
            raise ValueError('Ground truth must be either "Low" or "High"')
        return normalized_ground_truth

    def __getitem__(self, idx):
        current_case = list(self.prepared_cases[idx].values())[0]
        current_case_image = current_case["image"]
        current_metadata = current_case["metadata"]
        current_ground_truth = np.squeeze(current_case["ground_truth"])
        return (
            self.image_transformations(current_case_image),
            torch.FloatTensor(current_metadata),
            torch.FloatTensor(current_ground_truth),
        )


class LungCancerDataset(ChaimeleonData):
    def __init__(self, data_directory, split_type="train", random_seed=20380119):
        super().__init__(data_directory)
        self.split_type = split_type
        self.split_keys = self.keys_by_split[split_type]
        self.days_per_gt_bucket = 0.5
        self.categorical_metadata = [
            "gender",
            "smoking_status",
        ]
        self.numerical_metadata = ["age"]
        self.image_size = (224, 224)
        self.get_metadata_details()
        self.define_image_transformations(split_type)
        self.prepare_dataset()

    def prepare_dataset(self):
        prepared_cases = []
        for key in self.split_keys:
            raw_case = self.raw_cases[key]
            raw_image = raw_case["image"]
            prepped_metadata = self.normalize_metadata(raw_case["metadata"])
            normalized_ground_truth = self.normalize_ground_truth(
                raw_case["ground_truth"]
            )
            prepared_cases.append(
                {
                    key: {
                        "image": raw_image,
                        "metadata": prepped_metadata,
                        "ground_truth": normalized_ground_truth,
                        "survival": np.array(
                            [
                                raw_case["ground_truth"]["progression"],
                                raw_case["ground_truth"]["pfs"],
                            ]
                        ),
                    }
                }
            )
        self.prepared_cases = prepared_cases

    def define_image_transformations(self, split_type):
        if split_type == "train":
            image_transformations = tv.transforms.Compose(
                [
                    tv.transforms.RandomAffine(
                        degrees=180, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
                    ),
                    tv.transforms.ToTensor(),
                    tv.transforms.Resize(self.image_size, antialias=True),
                ]
            )
        else:
            image_transformations = tv.transforms.Compose(
                [
                    tv.transforms.ToTensor(),
                    tv.transforms.Resize(self.image_size, antialias=True),
                ]
            )
        self.image_transformations = image_transformations
        return image_transformations

    def get_metadata_details(self):
        metadata_details = defaultdict(lambda: {"values": []})
        for case, case_data in self.raw_cases.items():
            for key, value in case_data["metadata"].items():
                metadata_details[key]["values"].append(value)

        for details in metadata_details.values():
            details["max"] = max(details["values"])
            details["min"] = min(details["values"])
            details["length"] = len(details["values"])
            values = details["values"]
            unique_values = set(values)
            sorted_values = list(unique_values)
            sorted_values.sort()
            details["sorted_values"] = sorted_values

        self.metadata_details = metadata_details
        return metadata_details

    def normalize_metadata(self, raw_metadata):
        all_encoded_metadata = np.zeros((0, 1))
        for key in self.categorical_metadata:
            current_value = raw_metadata[key]
            num_possible_values = len(self.metadata_details[key]["sorted_values"])
            encoded_location = self.metadata_details[key]["sorted_values"].index(
                current_value
            )
            encoded_metadata = np.zeros((num_possible_values, 1))
            encoded_metadata[encoded_location] = 1
            all_encoded_metadata = np.concatenate(
                (all_encoded_metadata, encoded_metadata), axis=0
            )
        for key in self.numerical_metadata:
            current_value = raw_metadata[key]
            max_value = self.metadata_details[key]["max"]
            min_value = self.metadata_details[key]["min"]
            normalized_value = (current_value - min_value) / (max_value - min_value)
            normalized_value = np.array([[normalized_value]])
            all_encoded_metadata = np.concatenate(
                (all_encoded_metadata, normalized_value), axis=0
            )
        return all_encoded_metadata

    def normalize_ground_truth(self, raw_ground_truth):
        normalized_ground_truth = np.zeros((200, 1))
        normalized_gt_bucket = round(raw_ground_truth["pfs"] / self.days_per_gt_bucket)
        if raw_ground_truth["progression"]:
            normalized_ground_truth[normalized_gt_bucket][0] = 1
        else:
            normalized_ground_truth[normalized_gt_bucket][0] = 0
        return normalized_ground_truth

    def get_all_ground_truth(self):
        all_ground_truth = []
        for case in self.split_keys:
            all_ground_truth.append(
                [
                    self.raw_cases[case]["ground_truth"]["progression"],
                    self.raw_cases[case]["ground_truth"]["pfs"],
                ]
            )
        return all_ground_truth

    def __getitem__(self, idx):
        current_case = list(self.prepared_cases[idx].values())[0]
        current_case_image = Image.fromarray(np.asarray(current_case["image"]))
        current_metadata = current_case["metadata"]
        current_ground_truth = np.squeeze(current_case["ground_truth"])
        current_survival = current_case["survival"]
        return (
            self.image_transformations(current_case_image),
            torch.FloatTensor(current_metadata),
            (torch.FloatTensor(current_ground_truth), current_survival),
        )
