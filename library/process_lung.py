import SimpleITK as sitk
from pathlib import Path
import json
import numpy as np
import torch
import random

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from library.models import LungCombinedResnet18PretrainedModel
from library.datasets import LungCancerDataset, create_dataloader

class Lungcancerosprediction(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # path to image file
        self.image_input_dir = "/input/images/chest-ct/"
        self.image_input_path = list(Path(self.image_input_dir).glob("*.mha"))
        if self.image_input_path:
            self.image_input_path = str(self.image_input_path[0])

        # load clinical information
        # dictionary with patient_age and psa information
        self.clinical_info_path = "/input/clinical-information-lung-ct.json"
        # with open("/input/clinical-information-lung-ct.json") as fp:
        #     self.clinical_info = json.load(fp)

        # path to output files
        self.os_output_file = Path("/output/overall-survival-months.json")

    def predict(self, image_path=None, clinical_info_path=None):
        """
        Your algorithm goes here
        """        
        if image_path is None:
            image_path = self.image_input_path
        if clinical_info_path is None:
            clinical_info_path = self.clinical_info_path

        number_of_buckets = 400
        months_per_gt_bucket = 0.5

        dataset = LungCancerDataset(
            image_path=image_path,
            metadata_path=clinical_info_path,
            split_type="all",
            number_of_buckets=number_of_buckets,
            months_per_gt_bucket=months_per_gt_bucket,
            release=True,
        )

        eval_loader = create_dataloader(dataset, batch_size=2, shuffle=False)
        for data in eval_loader:
            (eval_images, self.clinical_info) = data


        # read image
        # image = sitk.ReadImage(str(self.image_input_path))
        clinical_info = self.clinical_info
        print('Clinical info: ')
        print(clinical_info)

        # TODO: Add your inference code here
        model_path = "./library/release_models/lung/v1/20231212_400buck_05m_003lr_unfrozen_best_val_score_LungCombinedResnet18PretrainedModel.pt"
        model = LungCombinedResnet18PretrainedModel(number_of_buckets=number_of_buckets)
        print(f"model_built - {model_path}")
        if not torch.cuda.is_available():
            model_state_dict = torch.load(
                f"{model_path}", map_location=torch.device("cpu")
            )
            print(f"model_loaded - cpu")
        else:
            model_state_dict = torch.load(f"{model_path}")
            print(f"model_loaded - gpu")
        model.load_state_dict(model_state_dict)
        model.eval()

        outputs = model((eval_images, self.clinical_info.squeeze()))
        predictions = torch.argmax(outputs, dim=1)
        predicted_bucket = predictions[0].item()
        print('Predicted bucket: ', predicted_bucket)
        overall_survival = predicted_bucket*months_per_gt_bucket
    
        print('OS (months): ', overall_survival)

        # save case-level class
        with open(str(self.os_output_file), 'w') as f:
            json.dump(overall_survival, f)


if __name__ == "__main__":
    # Lungcancerosprediction().predict()
    Lungcancerosprediction().predict(image_path="./datasets/eval_lung/case_0093.mha", clinical_info_path="./datasets/eval_lung/case_0093.json")