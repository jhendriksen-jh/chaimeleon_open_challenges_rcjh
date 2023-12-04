import SimpleITK as sitk
from pathlib import Path
import json
import torch
import numpy as np

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from library.models import ProstateCombinedResnet18PretrainedModel

class Prostatecancerriskprediction(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        self.input_slice_count = 3

        # # path to image file
        # self.image_input_dir = "/input/images/axial-t2-prostate-mri/"
        # self.image_input_path = list(Path(self.image_input_dir).glob("*.mha"))[0]

        # # load clinical information
        # # dictionary with patient_age and psa information
        # with open("/input/psa-and-age.json") as fp:
        #     self.clinical_info = json.load(fp)

        # path to output files
        self.risk_score_output_file = Path("/output/prostate-cancer-risk-score.json")
        self.risk_score_likelihood_output_file = Path("/output/prostate-cancer-risk-score-likelihood.json")
    
    def predict(self, image_path: str = None, clinical_info_path: str = None):
        # import pudb; pudb.set_trace()
        """
        Your algorithm goes here
        """        
        if image_path:
            self.image_input_path = Path(image_path)
        if clinical_info_path:
            with open(clinical_info_path, 'r') as f:
                self.clinical_info = json.load(f)
    
        # read image
        image = sitk.ReadImage(str(self.image_input_path))
        clinical_info = self.clinical_info
        print('Clinical info: ')
        print(clinical_info)

        # TODO: Add your inference code here
        risk_scores = ['Low', 'High']

        np_image = sitk.GetArrayFromImage(image)
        nii_chunks = []
        for k in range(self.input_slice_count):
            u = int(np.ceil((k + 1) * (np_image.shape[0] / self.input_slice_count)))
            l = int(np.floor((k) * (np_image.shape[0] / self.input_slice_count)))
            nii_chunk = np.mean(np_image[l:u, :, :], axis=0)
            nii_chunk = (
                (nii_chunk - nii_chunk.min()) / (nii_chunk.max() - nii_chunk.min())
            ) * 255
            nii_chunks.append(nii_chunk)
        nii_chunked_image = np.array(nii_chunks).astype(np.uint8)
        # nii_chunked_image = np.transpose(nii_chunked_image, (1, 2, 0))
        image_tensor = torch.from_numpy(nii_chunked_image)

        if clinical_info.get("patient_age"):
            age = int(clinical_info["patient_age"])
        else:
            age = int(clinical_info["age"])
        psa = float(clinical_info["psa"])
        clinical_tensor = torch.FloatTensor([[[age, psa]]])

        model_path = "./library/release_models/20231127_best_val_score_unfrozen_01lr_raw_meta_ProstateCombinedResnet18PretrainedModel.pt"
        model = ProstateCombinedResnet18PretrainedModel(eval_mode=True)
        model_state_dict = torch.load(f"{model_path}")
        model.load_state_dict(model_state_dict)
        model.eval()

        prediction_scores = model((image_tensor, clinical_tensor))
        print(prediction_scores)
        prediction = torch.argmax(prediction_scores, dim=1)
        risk_score = risk_scores[prediction.item()]
        risk_score_likelihood = torch.softmax(prediction_scores,dim=1)[0][prediction.item()].item()

        print('Risk score: ', risk_score)
        print('Risk score likelihood: ', risk_score_likelihood)

        # save case-level class
        # with open(str(self.risk_score_output_file), 'w') as f:
        #     json.dump(risk_score, f)

        # # save case-level likelihood
        # with open(str(self.risk_score_likelihood_output_file), 'w') as f:
        #     json.dump(float(risk_score_likelihood), f)


if __name__ == "__main__":
    # Prostatecancerriskprediction().predict()
    Prostatecancerriskprediction().predict(image_path="./datasets/eval_prostate/case_0157.mha", clinical_info_path="./datasets/eval_prostate/case_0157.json")
    # Prostatecancerriskprediction().predict(image_path="./datasets/eval_prostate/case_0279/case_0279.mha", clinical_info_path="./datasets/eval_prostate/case_0279/case_0279.json")