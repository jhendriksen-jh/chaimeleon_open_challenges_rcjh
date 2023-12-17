"""
handling of data provided by chaimeleon 
"""

import os
import json
import time
import torch
import random
import numpy as np
import pandas as pd
import nibabel as nib
import torchvision as tv
import SimpleITK as sitk

from collections import defaultdict
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def create_dataloader(
    dataset, batch_size=16, shuffle=True, num_workers=os.cpu_count() - 1
):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


class ChaimeleonData:
    def __init__(
        self,
        data_directory,
        image_path,
        metadata_path,
        input_slice_count: int = 1,
        random_seed=2038,
        lung=False,
    ):
        self.random_seed = random_seed
        self.data_directory = data_directory
        self.input_slice_count = input_slice_count
        self.image_arrays = []
        self.raw_cases = {}
        self.prepared_cases = []
        self.lung = lung
        if self.lung:
            npz_path = "./datasets/train_lung_npz/processed_case_images.npz"
        else:
            npz_path = f"./datasets/train_prostate_npz/processed_case_images_{input_slice_count}.npz"
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        if os.path.exists(npz_path):
            self.raw_case_images = np.load(npz_path, allow_pickle=True)
            self.process_data_directory(data_directory, image_path, metadata_path, case_image_archive = self.raw_case_images)
        elif os.path.exists(f".{npz_path}"):
            npz_path = f".{npz_path}"
            self.raw_case_images = np.load(npz_path, allow_pickle=True)
            self.process_data_directory(data_directory, image_path, metadata_path, case_image_archive = self.raw_case_images)
        else:
            self.process_data_directory(data_directory, image_path, metadata_path)
            self.save_raw_cases_npz_compressed(npz_path)

    def save_raw_cases_npz_compressed(self, npz_path):
        raw_image_archive = {case: details['image'] for case, details in self.raw_cases.items()}
        np.savez_compressed(npz_path, **raw_image_archive)

    def process_data_directory(
        self, data_directory=None, image_path=None, metadata_path=None, case_image_archive=None
    ):
        if data_directory is None:
            data_directory = self.data_directory
        self.add_raw_case_data(data_directory, image_path, metadata_path, case_image_archive=case_image_archive)
        self.get_dataset_splits()

    def add_raw_case_data(self, case_directory, image_path=None, metadata_path=None, case_image_archive=None):
        case_ground_truth = None
        if case_directory is not None:
            case_folders = os.listdir(case_directory)
            if os.path.isdir(os.path.join(case_directory, case_folders[0])):
                for case_folder in tqdm(case_folders, desc="Adding raw case data to ChaimeleonDataset"):
                    case_path = os.path.join(case_directory, case_folder)
                    case_files = os.listdir(case_path)
                    for case_file in case_files:
                        if case_image_archive is None:
                            if case_file.endswith(".nii.gz"):
                                case_image = self.load_image_file(
                                    os.path.join(case_path, case_file), lung=self.lung
                                )
                            elif case_file.endswith(".mha"):
                                case_image = self.load_image_file(
                                    os.path.join(case_path, case_file), lung=self.lung
                                )
                        if case_file.endswith("ground_truth.json"):
                            case_ground_truth = self.add_raw_ground_truth(
                                os.path.join(case_path, case_file)
                            )
                        elif case_file.endswith(".json"):
                            case_metadata = self.add_raw_metadata(
                                os.path.join(case_path, case_file)
                            )
                    if case_image_archive:
                        case_image = case_image_archive[case_folder]
                    raw_case_data = {
                        "image": case_image,
                        "metadata": case_metadata,
                        "ground_truth": case_ground_truth,
                    }
                    self.raw_cases[case_folder] = raw_case_data
            else:
                case_path = case_directory
                for case_file in case_folders:
                    if case_file.endswith(".nii.gz"):
                        case_image = self.load_image_file(
                            os.path.join(case_path, case_file), lung=self.lung
                        )
                    elif case_file.endswith(".mha"):
                        case_image = self.load_image_file(
                            os.path.join(case_path, case_file), lung=self.lung
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
                    "ground_truth": case_ground_truth or None,
                }
                self.raw_cases[case_directory] = raw_case_data
        elif image_path is not None and metadata_path is not None:
            case_image = self.load_image_file(image_path)
            case_metadata = self.add_raw_metadata(metadata_path)
            raw_case_data = {
                "image": case_image,
                "metadata": case_metadata,
                "ground_truth": None,
            }
            self.raw_cases[image_path] = raw_case_data

        return raw_case_data

    def add_raw_metadata(self, metadata_file):
        with open(metadata_file) as f:
            raw_metadata = json.load(f)
        return raw_metadata

    def add_raw_ground_truth(self, ground_truth_file):
        with open(ground_truth_file) as f:
            raw_ground_truth = json.load(f)
        return raw_ground_truth

    def load_image_file(self, image_file, lung=False):
        if image_file.endswith(".nii.gz"):
            nifti_image = nib.load(image_file)
            nii_data = nifti_image.get_fdata()
        else:
            image = sitk.ReadImage(image_file)
            nifti_image = sitk.GetArrayFromImage(image)
            nifti_image = np.transpose(nifti_image, (2, 1, 0))
            nii_data = nifti_image

        if not lung:
            nii_chunks = []
            for k in range(self.input_slice_count):
                u = int(
                    np.ceil((k + 1) * (nii_data.shape[-1] / self.input_slice_count))
                )
                l = int(np.floor((k) * (nii_data.shape[-1] / self.input_slice_count)))
                nii_chunk = np.mean(nii_data[:, :, l:u], axis=2)
                nii_chunk = (
                    (nii_chunk - nii_chunk.min()) / (nii_chunk.max() - nii_chunk.min())
                ) * 255
                nii_chunks.append(nii_chunk)
            nii_chunked_image = np.array(nii_chunks).astype(np.uint8)
            nii_chunked_image = np.transpose(nii_chunked_image, (1, 2, 0))
        else:
            cropping_pixels = 32
            nii_data = nii_data[cropping_pixels:-cropping_pixels , cropping_pixels:-cropping_pixels, cropping_pixels: -cropping_pixels]
            axis_chunks = []
            for a in range(3):
                nii_chunks = []
                for k in range(3):
                    u = int(np.ceil((k + 1) * (nii_data.shape[a] / 3)))
                    l = int(np.floor((k) * (nii_data.shape[a] / 3)))
                    if a == 0:
                        nii_chunk = np.mean(nii_data[l:u, :, :], axis=a)
                    elif a == 1:
                        nii_chunk = np.mean(nii_data[:, l:u, :], axis=a)
                    elif a == 2:
                        nii_chunk = np.mean(nii_data[:, :, l:u], axis=a)
                    nii_chunk = (
                        (nii_chunk - nii_chunk.min())
                        / (nii_chunk.max() - nii_chunk.min())
                    ) * 255
                    nii_chunks.append(nii_chunk)
                nii_chunked_image = np.array(nii_chunks)
                nii_chunked_image = np.transpose(nii_chunked_image, (1, 2, 0))
                axis_chunks.append(nii_chunked_image)
            nii_chunked_image = np.concatenate(axis_chunks, axis=2)
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
            "test": self.val_keys,
            "all": dataset_keys,
        }

    def __len__(self):
        return len(self.prepared_cases)


class ProstateCancerDataset(ChaimeleonData):
    def __init__(
        self,
        data_directory=None,
        image_path=None,
        metadata_path=None,
        split_type="train",
        random_seed=20380119,
        input_slice_count: int = 1,
    ):
        super().__init__(
            data_directory,
            image_path,
            metadata_path,
            input_slice_count=input_slice_count,
            random_seed=random_seed,
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
        self.image_size = (256,256)
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
        if len(prepared_cases) == 1:
            prepared_cases = prepared_cases * 2
        self.prepared_cases = prepared_cases

    def define_image_transformations(self, split_type):
        if split_type == "train":
            image_transformations = tv.transforms.Compose(
                [
                    tv.transforms.ToTensor(),
                    tv.transforms.RandomAffine(
                        degrees=37, translate=(0.12, 0.12), scale=(0.9, 1.1), shear=5
                    ),
                    # tv.transforms.Resize(self.image_size, antialias=True),
                ]
            )
        elif split_type == "test":
            image_transformations = tv.transforms.Compose(
                [
                    tv.transforms.ToTensor(),
                    tv.transforms.Resize(
                        round(self.image_size[0] * random.uniform(0.9, 1.1)), antialias=True
                    ),
                    tv.transforms.RandomAffine(
                        degrees=(-60, -15), shear=(5, 10), translate=(0.2, 0.2)
                    )
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
            if key == "age" and raw_metadata.get(key) is not None:
                current_value = raw_metadata[key]
            elif key == "age" and raw_metadata.get(key) is None:
                key = "patient_age"
                current_value = raw_metadata[key]
            else:
                current_value = raw_metadata[key]
            max_value = self.metadata_details[key]["max"]
            min_value = self.metadata_details[key]["min"]

            if len(self.metadata_details[key]["sorted_values"]) > 1:
                normalized_value = (current_value - min_value) / (max_value - min_value)
                normalized_value = np.array([[normalized_value]])
            # all_encoded_metadata = np.concatenate(
            #     (all_encoded_metadata, normalized_value), axis=0
            # )
            all_encoded_metadata = np.concatenate(
                (all_encoded_metadata, np.array([[current_value]])),
                axis=0,  # use raw numerical values to better generalize with test data
            )
        return all_encoded_metadata

    def normalize_ground_truth(self, raw_ground_truth):
        if raw_ground_truth is not None:
            normalized_ground_truth = np.zeros((2, 1))
            if raw_ground_truth == "Low":
                normalized_ground_truth[0] = 1
            elif raw_ground_truth == "High":
                normalized_ground_truth[1] = 1
            else:
                raise ValueError('Ground truth must be either "Low" or "High"')
        else:
            normalized_ground_truth = None
        return normalized_ground_truth
    
    def metatadata_transformations(self, metadata):
        # import pudb; pudb.set_trace()
        random.seed(time.time())
        if self.split_type == "val" or self.split_type == "all":
            return torch.FloatTensor(metadata)
        else:
            for k in range(len(metadata)):
                numerical_scale = random.uniform(0.93, 1.07)
                metadata[k] = metadata[k] * numerical_scale
            
            return torch.FloatTensor(metadata)

    def __getitem__(self, idx):
        current_case = list(self.prepared_cases[idx].values())[0]
        current_case_image = current_case["image"]
        current_metadata = current_case["metadata"]
        if self.split_type == 'test':
            random.seed(self.random_seed+idx)
            torch.manual_seed(self.random_seed+idx)
        transformed_image = self.image_transformations(current_case_image)
        if current_case["ground_truth"] is not None:
            current_ground_truth = np.squeeze(current_case["ground_truth"])
            return (
                transformed_image,
                self.metatadata_transformations(current_metadata),
                torch.FloatTensor(current_ground_truth),
                # list(self.prepared_cases[idx].items())[0],
                # current_case,
                # current_metadata,
            )
        else:
            return (
                transformed_image,
                self.metatadata_transformations(current_metadata),
            )


class LungCancerDataset(ChaimeleonData):
    def __init__(
        self,
        data_directory=None,
        image_path=None,
        metadata_path=None,
        split_type="train",
        months_per_gt_bucket=1,
        number_of_buckets = 360,
        random_seed=20380119,
    ):
        super().__init__(
            data_directory,
            lung=True,
            image_path=image_path,
            metadata_path=metadata_path,
            random_seed=random_seed,
        )

        self.split_type = split_type
        self.split_keys = self.keys_by_split[split_type]
        self.months_per_gt_bucket = months_per_gt_bucket
        self.number_of_buckets = number_of_buckets
        self.categorical_metadata = [
            "gender",
            "smoking_status",
        ]
        self.categorical_value_possiblities = {
            "gender": ["MALE", "FEMALE"],
            "smoking_status": ["Ex-smoker", "Smoker", "Non-smoker", 'Unknown'],
        }
        self.numerical_metadata = ["age"]
        self.image_size = (256, 256)
        self.get_metadata_details()
        self.define_image_transformations(split_type)
        self.prepare_dataset()

    def prepare_dataset(self):
        prepared_cases = []
        for key in tqdm(self.split_keys, desc=f"Preparing {self.split_type} dataset"):
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
                                raw_case["ground_truth"]["event"],
                                raw_case["ground_truth"]["survival_time_months"],
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
                    tv.transforms.ToTensor(),
                    tv.transforms.RandomAffine(
                        degrees=45, translate=(0.2, 0.2), scale=(0.9, 1.1), shear=5
                    ),
                    tv.transforms.Resize(self.image_size, antialias=True),
                ]
            )
        elif split_type == "test":
            image_transformations = tv.transforms.Compose(
                [
                    tv.transforms.ToTensor(),
                    tv.transforms.Resize(
                        round(self.image_size[0] * random.uniform(0.9, 1.1)), antialias=True
                    ),
                    tv.transforms.RandomAffine(
                        degrees=(-60, -15), shear=(5, 10), translate=(0.2, 0.2)
                    ),
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
    
    def metatadata_transformations(self, metadata):
        # import pudb; pudb.set_trace()
        random.seed(time.time())
        if self.split_type == "val" or self.split_type == "all":
            return torch.FloatTensor(metadata)
        else:
            numerical_scale = random.uniform(0.85, 1.15)
            metadata[-1] = metadata[-1] * numerical_scale

            categorical_flip_odds = random.uniform(0.05, 0.15)
            if random.choices([True, False], weights=[categorical_flip_odds, 1 - categorical_flip_odds], k=1)[0]:
                metadata[:2] = metadata[:2][::-1]
            if random.choices([True, False], weights=[categorical_flip_odds, 1 - categorical_flip_odds], k=1)[0]:
                shuffled_metadata = metadata[2:-1]
                random.shuffle(shuffled_metadata)
                metadata[2:-1] = shuffled_metadata
            
            return torch.FloatTensor(metadata)

    def get_metadata_details(self):
        metadata_details = defaultdict(lambda: {"values": []})
        for case, case_data in self.raw_cases.items():
            for key, value in case_data["metadata"].items():
                if key not in {"clinical_category", "regional_nodes_category", "metastasis_category"}:
                    metadata_details[key]["values"].append(value)
                    if self.categorical_value_possiblities.get(key) is not None:
                        metadata_details[key]["values"].extend(self.categorical_value_possiblities[key])

        for details in metadata_details.values():
            details['values'] = set(details["values"])
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
            # normalized_value = np.array([[normalized_value]])
            normalized_value = np.array([[current_value]])
            all_encoded_metadata = np.concatenate(
                (all_encoded_metadata, normalized_value), axis=0
            )
        return all_encoded_metadata

    def normalize_ground_truth(self, raw_ground_truth):
        normalized_ground_truth = np.zeros((self.number_of_buckets, 1))
        normalized_gt_bucket = round(raw_ground_truth["survival_time_months"] / self.months_per_gt_bucket)
        normalized_ground_truth[normalized_gt_bucket][0] = raw_ground_truth["event"]
        return normalized_ground_truth

    def get_all_ground_truth(self):
        all_ground_truth = []
        for case in self.split_keys:
            all_ground_truth.append(
                [
                    self.raw_cases[case]["ground_truth"]["event"],
                    self.raw_cases[case]["ground_truth"]["survival_time_months"],
                ]
            )
        return all_ground_truth

    def __getitem__(self, idx):
        current_case = list(self.prepared_cases[idx].values())[0]
        current_case_image = current_case["image"].astype(np.float32)
        current_metadata = current_case["metadata"]
        current_ground_truth = np.squeeze(current_case["ground_truth"])
        current_survival = current_case["survival"]
        if current_case["ground_truth"] is not None:
            return (
                self.image_transformations(current_case_image),
                self.metatadata_transformations(current_metadata),
                (torch.FloatTensor(current_ground_truth), current_survival),
            )
        if current_case["ground_truth"] is not None:
            return (
                self.image_transformations(current_case_image),
                self.metatadata_transformations(current_metadata),
            )