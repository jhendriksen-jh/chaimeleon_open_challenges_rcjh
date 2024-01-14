# chaimeleon_open_challenges_rcjh

## Team rcjh:
- Jordan Hendriksen

## Summary

This repository will act as a base for participation in the [chaimeleon open challenges](https://chaimeleon.grand-challenge.org/overview/) - the classification phase at least. Methods used for the classification summarized below.

## Methods - Classification Phase

### Data Preparation

#### Common methods:

Both cancer types make use of a common parent class `ChaimeleonData`. This class encapsulates several methods used for reading and preparing data for use in training and evaluation.

- `process_data_directory`: Responsible for loading the raw data into the dataset. It uses the add_raw_case_data method to add the raw data to the dataset. This method looks for case folders in the specified data directory and adds the raw data for each case folder.
    - `add_raw_case_data`: Loads the image and metadata files for a given case and adds them to the dataset class. If the case_image_archive parameter is specified, the code loads the image data from the archive. If the release flag is set to True, it loads the data from the directory regardless of archive existence.
    - `add_raw_metadata` and `add_raw_ground_truth` methods simply open their respective json files and store the data within the file as dictionaries.
    - `load_image_file`: loads the image files (`.nii.gz` or `.mha`) into numpy arrays and applies a set of preprocessing (to be expanded upon below in the 'Prostate Risk Specific' and 'Lung Overall Survival Specific' sections below) to the image before adding them to the dataset class.
- `get_dataset_splits`: randomly splits the keys for the different cases within the dataset into train, validation, and test sets. A random seed is set - and can be defined at initialization of the dataset - while the percentage of cases targeted for each split is defined when the method is called.

After a dataset is initialized the first time it's images are stored as a compressed numpy array archive in 'npzc' format. During subsequent initializations if the release flag is not set to True the code loads the data from the archive if the archive still exists. This drastically cuts down on IO and computation if no parameters affecting the images loaded are being experimented with.

#### Prostate Risk Specific:

- `load_image_file` preprocessing for prostate data involves two main preprocessing steps:
    - aggregating the 3rd dimension slices into _N_ 'average' slices (defined at initiation of the dataset). For example, with 24 starting slices and a target of 3 'average' slices the first 8 slices will be averaged, the middle 8 will be average, and the last 8 will be averaged, and these slices will be concatenated into one array for use in training/evaluation
    - scaling the values of the 'average' slices between 0 and 1
- Dataset preparation is completed in the `ProstateCancerDataset` class by doing the following for cases in the current metadata split:
    - normalizing metadata (one hot encoding categorical metadata) and loading both encoded categorical and numerical metadata into a vector. For this cancer there are no categorical metadata values provided and so the final vector is structured as [age, PSA]
    - normalizing ground truth values into a two class classification format (as opposed to binary classification that this could be viewed as)
    - Image and metadata augmentation strategies are defined
        - only during train and 'test' - which for this phase was an augmented version of the validation set

#### Lung Overall Survival Specific:

- `load_image_file` preprocessing for lung cases performs three main preprocessing steps:
    - aggregating the scans into a set of 3 'average' slices for each of the anatomical planes - resulting in 9 total slices that are concatenated into one array for use in training/evaluation
    - cropping 32 pixels off each edge of the initial tensor
    - scaling the values of the 'average' slices between 0 and 1
- Dataset preparation is completed in the `LungCancerDataset` class by doing the following for cases in the current metadata split:
    - normalizing metadata (one hot encoding categorical metadata) and loading both encoded categorical and numerical metadata into a vector. This was structured as [encoded categorical metadata (which values and order decided at runtime), numerical metadata]
    - normalizing ground truth values into an _N_ value classification format. At initialization _N_ is specified as the number of buckets the ground truth values will fall into with the number of months per bucket as specified. True ground truth values are divided by the number of months per bucket and the result is rounded - with that becoming the index of the buckets where the ground truth is then labeled at.
    - Image and metadata augmentation strategies are defined
        - only during train and 'test' - which for this phase was an augmented version of the validation set

### Models and Training

#### Prostate Risk

- Training:
    - BCE loss with bias balance loss towards 'high risk' cases
    - Tracked a variety of statistics and saved models based on both accuracy (number of validation samples with the correct classification) and score (weighted average of a few metrics surrounding classification and confidence) 
    - Tracked variables required in order to recreate or _very nearly_ recreate a model by using the same training parameters

- Models:
    - Various from scratch image models
    - Various from scratch metadata models
    - Various from scratch combined image metadata models
        - experimented with a variety of average image slices from a singular slice to 24 slices
    - Image based model from Resnet18 base
    - Combined image and metadata models from Resnet18 base (all with 3 average slices)
        - `ProstateCombinedResnet18PretrainedModel` (no resnet layers frozen) - two versions submitted to validation set with one version submitted to classification test set
        - `ProstateCombinedResnet18PretrainedModel_V2_1_Grid` (resnet layers 2 and 3 frozen) - one versions submitted to validation set 

#### Lung Overall Survival

- Training:
    - MultiLabel soft margin loss (used with training for submitted models)
    - MSE loss (used during experiments early on)
    - Tracked a variety of statistics and saved models based on both accuracy (number of validation samples with the correct classification) and score (concordance index)
    - Tracked variables required in order to recreate or _very nearly_ recreate a model by using the same training parameters

- Models:
    - From scratch image model
    - From scratch metadata model
    - From scratch combined image metadata model
    - Combined image and metadata models from Resnet18 base (trained with various resnet layers frozen)
        - `LungCombinedResnet18PretrainedModel` - used for all three validation submissions and the test submission
            - 400 buckets with 0.5 months per bucket (v1 and v3) - submitted to test set - no resnet layers frozen
            - 50 buckets with 3 months per bucket (v2) - resnet layers 2 and 3 frozen
            - 400 buckets with 0.5 months per bucket (v3) - resnet layers 2 and 3 frozen

### Evaluation

The actual evaluations were run inside a set of jupyter notebooks - `/notebooks/evaluate_prostate_models_V2.ipynb` and `/notebooks/evaluate_lung_models.ipynb`. Evaluations were performed by generating scores for each model on three datasets - the validation set used during training and two test sets. 

In this phase due to a small number of cases in each cancers dataset the test sets were generated by applying various strengths of translation, rotation, skew, and scaling augmentations to the validation set to create images that the model 'had not' directly seen before. There are flaws in this system but it's the one that was chosen for the classification phase.

After generating scores for each model on each of the three datasets the scores were ranked and an average rank was calculated. Models were ordered by average rank and their individual scores against the different sets were compared to one another to try to determine which model might perform the best on the Grand Challenge datasets. I.e. a model with the best average rank because it did best on the validation set but faired more poorly on the test sets compared to the second or third average ranked model might not be the best choice as it seems to generalize more poorly.