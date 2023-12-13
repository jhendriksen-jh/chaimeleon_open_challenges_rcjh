"""
Model class definitions
"""
import torch
import torch.nn as nn
from torchvision import models
import pickle as pkl


def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


######## PROSTATE CANCER MODELS ########


class ProstateImageModel(nn.Module):
    def __init__(self):
        super(ProstateImageModel, self).__init__()
        self.model_data_type = "images"

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3
        )  # 240
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding="same"
        )  # 120
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same"
        )  # 120
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # 60
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )  # 30
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same"
        )  # 30
        self.maxpool3 = nn.MaxPool2d(kernel_size=3)  # 10
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding="same"
        )  # 10
        self.conv7 = nn.Conv2d(
            in_channels=384, out_channels=512, kernel_size=3, stride=2, padding=1
        )  # 5
        self.downpool1 = nn.Conv2d(
            in_channels=16, out_channels=384, kernel_size=12, stride=12, padding=1
        )  # 120 -> 15
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=5 * 5 * 512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=2)
        self.drop = nn.Dropout(p=0.2)
        self.drop2d = nn.Dropout2d(p=0.1)

    def forward(self, x):
        out = self.conv1(x)
        residuals_1 = out
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.maxpool2(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.maxpool3(out)
        out = self.relu(out)

        out = self.conv6(out)
        out += self.downpool1(residuals_1)
        out = self.drop2d(out)
        out = self.relu(out)

        out = self.conv7(out)
        out = self.relu(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class ProstateMetadataModelV2Small(nn.Module):
    def __init__(self):
        super(ProstateMetadataModelV2Small, self).__init__()
        self.model_data_type = "metadata"

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=96)
        self.fc5 = nn.Linear(in_features=96, out_features=192)
        self.fc6 = nn.Linear(in_features=192, out_features=64)
        self.fc7 = nn.Linear(in_features=64, out_features=2)
        self.drop = nn.Dropout(p=0.05)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        res_1d = out
        out = self.relu(out)

        out = self.fc4(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.fc5(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.fc6(out)
        out += res_1d
        out = self.drop(out)
        out = self.relu(out)

        out = self.fc7(out)

        return out


class ProstateMetadataModelV3(nn.Module):
    """average poolng of tensor instead of fc layers"""

    def __init__(self):
        super(ProstateMetadataModelV3, self).__init__()
        self.model_data_type = "metadata"

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=96)
        self.fc5 = nn.Linear(in_features=96, out_features=192)
        self.fc6 = nn.Linear(in_features=192, out_features=512)
        self.cv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2)
        self.cv2 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3)
        # self.fc7 = nn.Linear(in_features=64, out_features=2)
        self.drop = nn.Dropout(p=0.05)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        res_1d = out
        out = self.relu(out)

        out = self.fc4(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.fc5(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.fc6(out)
        out = self.drop(out)
        out = self.relu(out)

        out = out.view((out.shape[0], 8, 8, 8))

        out = self.cv1(out)
        out += res_1d.view((out.shape[0], 1, 8, 8))
        out = self.relu(out)

        out = self.cv2(out)
        out = self.relu(out)

        out = torch.mean(out, dim=(2, 3))

        return out


class ProstateMetadataModel(nn.Module):
    def __init__(self):
        super(ProstateMetadataModel, self).__init__()
        self.model_data_type = "metadata"

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=256)
        self.fc5 = nn.Linear(in_features=256, out_features=512)
        self.fc6 = nn.Linear(in_features=512, out_features=64)
        self.fc7 = nn.Linear(in_features=64, out_features=2)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.fc5(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.fc6(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.fc7(out)

        return out


class ProstateCombinedModelV1Tiny(nn.Module):
    def __init__(self, input_slice_count: int):
        super(ProstateCombinedModelV1Tiny, self).__init__()
        self.model_data_type = "both"

        self.relu = nn.ReLU(inplace=True)

        # metadata layers
        self.meta_fc1 = nn.Linear(in_features=2, out_features=16)
        self.meta_fc2 = nn.Linear(in_features=16, out_features=32)
        self.meta_fc3 = nn.Linear(in_features=32, out_features=64)
        self.meta_fc4 = nn.Linear(in_features=64, out_features=128)
        # self.meta_fc5 = nn.Linear(in_features=64, out_features=128)
        self.meta_drop = nn.Dropout(p=0.05)

        # image layers
        self.image_conv1 = nn.Conv2d(
            in_channels=input_slice_count,
            out_channels=32,
            kernel_size=7,
            stride=2,
            padding=3,
        )  # 256 -> 128
        self.image_conv2 = nn.Conv2d(
            in_channels=32, out_channels=48, kernel_size=5, stride=1, padding="same"
        )  # 128
        self.image_conv3 = nn.Conv2d(
            in_channels=48, out_channels=48, kernel_size=3, stride=1, padding="same"
        )  # 128
        self.image_maxpool2 = nn.MaxPool2d(kernel_size=2)  # 64
        self.image_conv4 = nn.Conv2d(
            in_channels=48, out_channels=64, kernel_size=3, stride=2, padding=1
        )  # 32
        self.image_conv5 = nn.Conv2d(
            in_channels=64, out_channels=96, kernel_size=3, stride=1, padding="same"
        )  # 32
        self.image_maxpool3 = nn.MaxPool2d(kernel_size=3)  # 10
        self.image_conv6 = nn.Conv2d(
            in_channels=96, out_channels=128, kernel_size=5, stride=1, padding="same"
        )  # 10
        self.image_conv7 = nn.Conv2d(
            in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1
        )  # 5
        self.image_conv8 = nn.Conv2d(
            in_channels=196, out_channels=256, kernel_size=3, stride=1, padding="same"
        )  # 5
        self.image_downpool1 = nn.Conv2d(
            in_channels=32, out_channels=128, kernel_size=12, stride=12, padding=1
        )  # 120 -> 15
        self.image_flatten = nn.Flatten()
        self.image_fc1 = nn.Linear(in_features=5 * 5 * 256, out_features=64)
        self.image_fc2 = nn.Linear(in_features=64, out_features=2)
        self.image_drop = nn.Dropout(p=0.05)
        self.image_drop2d = nn.Dropout2d(p=0.01)

        # combo layers
        self.combo_fc1 = nn.Linear(in_features=32, out_features=48 * 64 * 64)
        self.combo_fc2 = nn.Linear(in_features=128, out_features=10 * 10 * 128)

    def forward(self, data):
        image, meta = data
        # meta layers
        meta_out = self.meta_fc1(meta)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc2(meta_out)
        combo_in1 = meta_out
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc3(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc4(meta_out)
        meta_out = self.meta_drop(meta_out)
        meta_out = self.relu(meta_out)

        # meta_out = self.meta_fc5(meta_out)
        # meta_out = self.meta_drop(meta_out)
        combo_in2 = meta_out

        # combo layers - consider FC that can be reshaped to match dimensions of conv outputs and added (112x112 for fc2)
        combo_out1 = self.combo_fc1(combo_in1)
        combo_out2 = self.combo_fc2(combo_in2)

        # image_layers
        image_out = self.image_conv1(image)
        residuals_1 = image_out
        image_out = self.relu(image_out)
        image_out = self.image_conv2(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv3(image_out)
        image_out = self.image_maxpool2(image_out)

        combo_image_enhancer = combo_out1.view(image_out.shape)
        image_out += combo_image_enhancer

        image_out = self.relu(image_out)

        image_out = self.image_conv4(image_out)
        image_out = self.relu(image_out)
        image_out = self.image_conv5(image_out)
        image_out = self.image_maxpool3(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv6(image_out)
        image_out += self.image_downpool1(residuals_1)
        image_out += combo_out2.view(image_out.shape)

        image_out = self.image_drop2d(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv7(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv8(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_flatten(image_out)
        image_out = self.image_fc1(image_out)
        image_out = self.image_drop(image_out)
        image_out = self.relu(image_out)
        image_out = self.image_fc2(image_out)

        return image_out


class ProstateCombinedModelV1_1Tiny(nn.Module):
    """additional residual layer and only second combination layer"""

    def __init__(self, input_slice_count: int = 1):
        super(ProstateCombinedModelV1_1Tiny, self).__init__()
        self.model_data_type = "both"

        self.relu = nn.ReLU(inplace=True)

        # metadata layers
        self.meta_fc1 = nn.Linear(in_features=2, out_features=16)
        self.meta_fc2 = nn.Linear(in_features=16, out_features=32)
        self.meta_fc3 = nn.Linear(in_features=32, out_features=64)
        self.meta_fc4 = nn.Linear(in_features=64, out_features=128)
        self.meta_fc5 = nn.Linear(in_features=128, out_features=32)
        self.meta_drop = nn.Dropout(p=0.05)

        # image layers
        self.image_conv1 = nn.Conv2d(
            in_channels=input_slice_count,
            out_channels=32,
            kernel_size=7,
            stride=2,
            padding=3,
        )  # 256 -> 128
        self.image_conv2 = nn.Conv2d(
            in_channels=32, out_channels=48, kernel_size=5, stride=1, padding="same"
        )  # 128
        self.image_conv3 = nn.Conv2d(
            in_channels=48, out_channels=48, kernel_size=3, stride=1, padding="same"
        )  # 128
        self.image_maxpool2 = nn.MaxPool2d(kernel_size=2)  # 64
        self.image_conv4 = nn.Conv2d(
            in_channels=48, out_channels=64, kernel_size=3, stride=2, padding=1
        )  # 32
        self.image_conv5 = nn.Conv2d(
            in_channels=64, out_channels=96, kernel_size=3, stride=1, padding="same"
        )  # 32
        self.image_maxpool3 = nn.MaxPool2d(kernel_size=3)  # 10
        self.image_conv6 = nn.Conv2d(
            in_channels=96, out_channels=128, kernel_size=5, stride=1, padding="same"
        )  # 10
        self.image_conv7 = nn.Conv2d(
            in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1
        )  # 5
        self.image_conv8 = nn.Conv2d(
            in_channels=196, out_channels=256, kernel_size=3, stride=1, padding="same"
        )  # 5
        self.image_downpool1 = nn.Conv2d(
            in_channels=32, out_channels=128, kernel_size=12, stride=12, padding=1
        )  # 128 -> 10
        self.image_downpool2 = nn.Conv2d(
            in_channels=48, out_channels=196, kernel_size=12, stride=12, padding=1
        )
        self.image_flatten = nn.Flatten()
        self.image_fc1 = nn.Linear(in_features=5 * 5 * 256, out_features=64)
        self.image_fc2 = nn.Linear(in_features=64, out_features=2)
        self.image_drop = nn.Dropout(p=0.05)
        self.image_drop2d = nn.Dropout2d(p=0.01)

        # combo layers
        # self.combo_fc1 = nn.Linear(
        #     in_features=32, out_features=48 * 64 * 64
        # )
        self.combo_fc2 = nn.Linear(in_features=32, out_features=10 * 10 * 128)

    def forward(self, data):
        image, meta = data
        # meta layers
        meta_out = self.meta_fc1(meta)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc2(meta_out)
        meta_res1 = meta_out
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc3(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc4(meta_out)
        meta_out = self.meta_drop(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc5(meta_out)
        meta_out += meta_res1
        # meta_out = self.meta_drop(meta_out)
        combo_in2 = meta_out

        # combo layers - consider FC that can be reshaped to match dimensions of conv outputs and added (112x112 for fc2)
        # combo_out1 = self.combo_fc1(combo_in1)
        combo_out2 = self.combo_fc2(combo_in2)

        # image_layers
        image_out = self.image_conv1(image)
        residuals_1 = image_out
        image_out = self.relu(image_out)
        image_out = self.image_conv2(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv3(image_out)
        image_out = self.image_maxpool2(image_out)
        residuals_2 = image_out

        # combo_image_enhancer = combo_out1.view(image_out.shape)
        # image_out += combo_image_enhancer

        image_out = self.relu(image_out)

        image_out = self.image_conv4(image_out)
        image_out = self.relu(image_out)
        image_out = self.image_conv5(image_out)
        image_out = self.image_maxpool3(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv6(image_out)
        image_out += self.image_downpool1(residuals_1)
        image_out += combo_out2.view(image_out.shape)

        image_out = self.image_drop2d(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv7(image_out)
        image_out += self.image_downpool2(residuals_2)
        image_out = self.relu(image_out)

        image_out = self.image_conv8(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_flatten(image_out)
        image_out = self.image_fc1(image_out)
        image_out = self.image_drop(image_out)
        image_out = self.relu(image_out)
        image_out = self.image_fc2(image_out)

        return image_out


class ProstateCombinedModel(nn.Module):
    def __init__(self, input_slice_count: int):
        super(ProstateCombinedModel, self).__init__()
        self.model_data_type = "both"

        self.relu = nn.ReLU(inplace=True)

        # metadata layers
        self.meta_fc1 = nn.Linear(in_features=2, out_features=32)
        self.meta_fc2 = nn.Linear(in_features=32, out_features=64)
        self.meta_fc3 = nn.Linear(in_features=64, out_features=128)
        self.meta_fc4 = nn.Linear(in_features=128, out_features=256)
        self.meta_fc5 = nn.Linear(in_features=256, out_features=512)
        # self.meta_fc6 = nn.Linear(in_features = 512, out_features=256)
        # self.meta_fc7 = nn.Linear(in_features = 64, out_features=2)
        self.meta_drop = nn.Dropout(p=0.05)

        # image layers
        self.image_conv1 = nn.Conv2d(
            in_channels=input_slice_count,
            out_channels=32,
            kernel_size=7,
            stride=2,
            padding=3,
        )  # 256 -> 128
        self.image_conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding="same"
        )  # 128
        self.image_conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"
        )  # 128
        self.image_maxpool2 = nn.MaxPool2d(kernel_size=2)  # 64
        self.image_conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )  # 32
        self.image_conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same"
        )  # 32
        self.image_maxpool3 = nn.MaxPool2d(kernel_size=3)  # 10
        self.image_conv6 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=5, stride=1, padding="same"
        )  # 10
        self.image_conv7 = nn.Conv2d(
            in_channels=512, out_channels=756, kernel_size=3, stride=2, padding=1
        )  # 5
        self.image_conv8 = nn.Conv2d(
            in_channels=756, out_channels=1024, kernel_size=3, stride=1, padding="same"
        )  # 5
        self.image_downpool1 = nn.Conv2d(
            in_channels=32, out_channels=512, kernel_size=12, stride=12, padding=1
        )  # 120 -> 15
        self.image_flatten = nn.Flatten()
        self.image_fc1 = nn.Linear(in_features=5 * 5 * 1024, out_features=384)
        self.image_fc2 = nn.Linear(in_features=384, out_features=2)
        self.image_drop = nn.Dropout(p=0.05)
        self.image_drop2d = nn.Dropout2d(p=0.01)

        # combo layers
        self.combo_fc1 = nn.Linear(
            in_features=2, out_features=input_slice_count * 256 * 256
        )
        self.combo_fc2 = nn.Linear(in_features=512, out_features=10 * 10 * 512)
        # self.combo_output = nn.Linear(in_features = 512, out_features=2)

    def forward(self, data):
        image, meta = data
        # meta layers
        combo_in1 = meta
        meta_out = self.meta_fc1(meta)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc2(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc3(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc4(meta_out)
        meta_out = self.meta_drop(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc5(meta_out)
        meta_out = self.meta_drop(meta_out)
        combo_in2 = meta_out
        # meta_out = self.relu(meta_out)

        # meta_out = self.meta_fc6(meta_out)
        # meta_out = self.meta_drop(meta_out)
        # meta_out = self.relu(meta_out)

        # meta_out = self.meta_fc7(meta_out)

        # combo layers - consider FC that can be reshaped to match dimensions of conv outputs and added (112x112 for fc2)
        combo_out1 = self.combo_fc1(combo_in1)
        combo_out2 = self.combo_fc2(combo_in2)

        # image_layers
        combo_image_enhancer = combo_out1.view(image.shape)
        image += combo_image_enhancer
        image_out = self.image_conv1(image)
        residuals_1 = image_out
        image_out = self.relu(image_out)
        image_out = self.image_conv2(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv3(image_out)
        image_out = self.image_maxpool2(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv4(image_out)
        image_out = self.relu(image_out)
        image_out = self.image_conv5(image_out)
        image_out = self.image_maxpool3(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv6(image_out)
        image_out += self.image_downpool1(residuals_1)
        image_out += combo_out2.view(image_out.shape)

        image_out = self.image_drop2d(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv7(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv8(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_flatten(image_out)
        image_out = self.image_fc1(image_out)
        image_out = self.image_drop(image_out)
        image_out = self.relu(image_out)
        image_out = self.image_fc2(image_out)

        # combined_out = torch.cat((image_out, meta_out), dim=1)
        # combined_out = self.combo_output(combined_out)

        return image_out


class ProstateCombinedResnet18PretrainedModel(nn.Module):
    def __init__(self, frozen_layers=[], eval_mode=False):
        super(ProstateCombinedResnet18PretrainedModel, self).__init__()
        # import pudb; pudb.set_trace()
        self.eval_mode = eval_mode
        self.model_data_type = "both"
        self.relu = nn.ReLU(inplace=True)

        # self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT, progress=True)
        with open("./library/resnet18.pkl", "rb") as f:
            self.resnet18 = pkl.load(f)
        resnet_layer_list = list(self.resnet18.children())
        self.resnet_layers = {
            "conv1": 0,
            "bn1": 1,
            "relu": 2,
            "maxpool": 3,
            "layer1": 4,
            "layer2": 5,
            "layer3": 6,
            "layer4": 7,
            "avgpool": 8,
            "fc": 9,
        }
        self.resnet_layers = {
            k: resnet_layer_list[v] for k, v in self.resnet_layers.items()
        }

        for layer in frozen_layers:
            for param in self.resnet_layers[layer].parameters():
                param.requires_grad = False
        # layer3 -> 256x16x16

        # metadata layers
        self.meta_fc1 = nn.Linear(in_features=2, out_features=16)
        self.meta_fc2 = nn.Linear(in_features=16, out_features=32)
        self.meta_fc3 = nn.Linear(in_features=32, out_features=64)
        self.meta_fc4 = nn.Linear(in_features=64, out_features=128)
        self.meta_fc5 = nn.Linear(in_features=128, out_features=32)
        self.meta_drop = nn.Dropout(p=0.05)

        self.image_drop = nn.Dropout(p=0.05)
        self.image_drop2d = nn.Dropout2d(p=0.01)

        self.combo_fc = nn.Linear(in_features=32, out_features=3)

        self.final_fc1 = nn.Linear(in_features=512, out_features=256)
        self.final_fc2 = nn.Linear(in_features=256, out_features=2)

        self.resnet_layers["fc"] = self.final_fc1

    def forward(self, data):
        # import pudb; pudb.set_trace()
        image, meta = data
        # meta layers
        meta_out = self.meta_fc1(meta)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc2(meta_out)
        meta_res1 = meta_out
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc3(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc4(meta_out)
        meta_out = self.meta_drop(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc5(meta_out)
        meta_out += meta_res1
        # meta_out = self.meta_drop(meta_out)
        combo_in2 = meta_out

        # combo layers
        combo_out = self.combo_fc(combo_in2)

        if self.eval_mode:
            image_in1 = combo_out[:, :, 0] * image[0, :, :]
            image_in2 = combo_out[:, :, 1] * image[1, :, :]
            image_in3 = combo_out[:, :, 2] * image[2, :, :]
            image = torch.cat(
                (
                    image_in1.unsqueeze(0),
                    image_in2.unsqueeze(0),
                    image_in3.unsqueeze(0),
                ),
                dim=0,
            ).unsqueeze(0)
        else:
            image_in1 = combo_out[:, 0][:, None, None, None] * image[:, 0][:, None]
            image_in2 = combo_out[:, 1][:, None, None, None] * image[:, 1][:, None]
            image_in3 = combo_out[:, 2][:, None, None, None] * image[:, 2][:, None]
            image = torch.cat((image_in1, image_in2, image_in3), dim=1)

        # image layers
        for layer_name, resnet_layer in self.resnet_layers.items():
            image = resnet_layer(image)
            if layer_name == "avgpool":
                image = image.reshape(image.size(0), -1)
            if layer_name == "layer4":
                image = self.image_drop2d(image)

        image = self.relu(image)

        output = self.meta_drop(image)
        output = self.final_fc2(output)

        return output


class ProstateImageResnet18PretrainedModel(nn.Module):
    def __init__(self, frozen_layers=[]):
        super(ProstateImageResnet18PretrainedModel, self).__init__()
        # import pudb; pudb.set_trace()
        self.model_data_type = "both"
        self.relu = nn.ReLU(inplace=True)

        self.resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT, progress=True
        )
        resnet_layer_list = list(self.resnet18.children())
        self.resnet_layers = {
            "conv1": 0,
            "bn1": 1,
            "relu": 2,
            "maxpool": 3,
            "layer1": 4,
            "layer2": 5,
            "layer3": 6,
            "layer4": 7,
            "avgpool": 8,
            "fc": 9,
        }
        self.resnet_layers = {
            k: resnet_layer_list[v] for k, v in self.resnet_layers.items()
        }

        for layer in frozen_layers:
            for param in self.resnet_layers[layer].parameters():
                param.requires_grad = False
        # layer3 -> 256x16x16

        # metadata layers
        # self.meta_fc1 = nn.Linear(in_features=2, out_features=16)
        # self.meta_fc2 = nn.Linear(in_features=16, out_features=32)
        # self.meta_fc3 = nn.Linear(in_features=32, out_features=64)
        # self.meta_fc4 = nn.Linear(in_features=64, out_features=128)
        # self.meta_fc5 = nn.Linear(in_features=128, out_features=32)
        self.meta_drop = nn.Dropout(p=0.05)

        self.image_drop = nn.Dropout(p=0.05)
        self.image_drop2d = nn.Dropout2d(p=0.01)

        # self.combo_fc = nn.Linear(in_features=32, out_features=3)

        self.final_fc1 = nn.Linear(in_features=512, out_features=256)
        self.final_fc2 = nn.Linear(in_features=256, out_features=2)

        self.resnet_layers["fc"] = self.final_fc1

    def forward(self, data):
        # import pudb; pudb.set_trace()
        image, meta = data
        # meta layers
        # meta_out = self.meta_fc1(meta)
        # meta_out = self.relu(meta_out)

        # meta_out = self.meta_fc2(meta_out)
        # meta_res1 = meta_out
        # meta_out = self.relu(meta_out)

        # meta_out = self.meta_fc3(meta_out)
        # meta_out = self.relu(meta_out)

        # meta_out = self.meta_fc4(meta_out)
        # meta_out = self.meta_drop(meta_out)
        # meta_out = self.relu(meta_out)

        # meta_out = self.meta_fc5(meta_out)
        # meta_out += meta_res1
        # # meta_out = self.meta_drop(meta_out)
        # combo_in2 = meta_out

        # # combo layers
        # combo_out = self.combo_fc(combo_in2)

        # image_in1 = combo_out[:, 0][:, None, None, None] * image[:, 0][:,None]
        # image_in2 = combo_out[:, 1][:, None, None, None] * image[:, 1][:,None]
        # image_in3 = combo_out[:, 2][:, None, None, None] * image[:, 2][:,None]
        # image = torch.cat((image_in1, image_in2, image_in3), dim=1)

        # image layers
        for layer_name, resnet_layer in self.resnet_layers.items():
            image = resnet_layer(image)
            if layer_name == "avgpool":
                image = image.reshape(image.size(0), -1)
            if layer_name == "layer4":
                image = self.image_drop2d(image)

        image = self.relu(image)

        output = self.meta_drop(image)
        output = self.final_fc2(output)

        return output


######## LUNG CANCER MODELS ########


class LungMetadataModel(nn.Module):
    def __init__(self):
        super(LungMetadataModel, self).__init__()
        self.model_data_type = "metadata"

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=6, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=1024)
        self.fc6 = nn.Linear(in_features=1024, out_features=512)
        self.fc7 = nn.Linear(in_features=512, out_features=200)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.fc5(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.fc6(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.fc7(out)

        return out


class LungImageModel(nn.Module):
    def __init__(self):
        super(LungImageModel, self).__init__()
        self.model_data_type = "images"

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3
        )  # 240
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding="same"
        )  # 120
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same"
        )  # 120
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # 60
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )  # 30
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same"
        )  # 30
        self.maxpool3 = nn.MaxPool2d(kernel_size=3)  # 10
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding="same"
        )  # 10
        self.conv7 = nn.Conv2d(
            in_channels=384, out_channels=512, kernel_size=3, stride=2, padding=1
        )  # 5
        self.downpool1 = nn.Conv2d(
            in_channels=16, out_channels=384, kernel_size=12, stride=12, padding=1
        )  # 120 -> 15
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=5 * 5 * 512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=200)
        self.drop = nn.Dropout(p=0.2)
        self.drop2d = nn.Dropout2d(p=0.1)

    def forward(self, x):
        out = self.conv1(x)
        residuals_1 = out
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.maxpool2(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.maxpool3(out)
        out = self.relu(out)

        out = self.conv6(out)
        out += self.downpool1(residuals_1)
        out = self.drop2d(out)
        out = self.relu(out)

        out = self.conv7(out)
        out = self.relu(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class LungCombinedModel(nn.Module):
    def __init__(self, number_of_buckets = 360):
        super(LungCombinedModel, self).__init__()
        self.model_data_type = "both"

        self.relu = nn.ReLU(inplace=True)

        # metadata layers
        self.meta_fc1 = nn.Linear(in_features=7, out_features=16)
        self.meta_fc2 = nn.Linear(in_features=16, out_features=32)
        self.meta_fc3 = nn.Linear(in_features=32, out_features=128)
        self.meta_fc3_down = nn.Linear(in_features=128, out_features=32)
        self.meta_fc4 = nn.Linear(in_features=128, out_features=256)
        self.meta_fc5 = nn.Linear(in_features=256, out_features=512)
        # self.meta_fc6 = nn.Linear(in_features = 512, out_features=256)
        # self.meta_fc7 = nn.Linear(in_features = 64, out_features=2)
        self.meta_drop = nn.Dropout(p=0.1)

        # image layers
        self.image_conv1 = nn.Conv2d(
            in_channels=9, out_channels=16, kernel_size=7, stride=2, padding=3
        )  # 224 -> 112
        self.image_conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding="same"
        )  # 112
        self.image_conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same"
        )  # 112
        self.image_maxpool2 = nn.MaxPool2d(kernel_size=2)  # 56
        self.image_conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )  # 28
        self.image_conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same"
        )  # 28
        self.image_maxpool3 = nn.MaxPool2d(kernel_size=3)  # 9
        self.image_conv6 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding="same"
        )  # 9
        self.image_conv7 = nn.Conv2d(
            in_channels=384, out_channels=512, kernel_size=3, stride=2, padding=1
        )  # 5
        self.image_downpool1 = nn.Conv2d(
            in_channels=16, out_channels=384, kernel_size=12, stride=12, padding=1
        )  # 120 -> 15
        self.image_flatten = nn.Flatten()
        self.image_fc1 = nn.Linear(in_features=5 * 5 * 512, out_features=512)
        self.image_fc2 = nn.Linear(in_features=512, out_features=number_of_buckets)
        self.image_drop = nn.Dropout(p=0.2)
        self.image_drop2d = nn.Dropout2d(p=0.1)

        # combo layers
        self.combo_fc1 = nn.Linear(in_features=32, out_features=9)
        # self.combo_output = nn.Linear(in_features = 512, out_features=2)

    def forward(self, data):
        # import pudb; pudb.set_trace()
        image, meta = data
        # meta layers
        meta_out = self.meta_fc1(meta)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc2(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc3(meta_out)
        meta_out = self.relu(meta_out)
        combo_in1 = meta_out

        meta_out = self.meta_fc4(meta_out)
        meta_out = self.meta_drop(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc5(meta_out)
        meta_out = self.meta_drop(meta_out)
        # meta_out = self.relu(meta_out)

        # meta_out = self.meta_fc6(meta_out)
        # meta_out = self.meta_drop(meta_out)
        # meta_out = self.relu(meta_out)

        # meta_out = self.meta_fc7(meta_out)

        # combo layers - consider FC that can be reshaped to match dimensions of conv outputs and added (112x112 for fc2)
        combo_in1 = self.meta_fc3_down(combo_in1)
        combo_out1 = self.combo_fc1(combo_in1)

        # image_layers
        enhanced_images = []
        for i in range(image.shape[1]):
            enhanced_image = combo_out1[:,i][:, None, None, None] * image[:, i][:, None]
            enhanced_images.append(enhanced_image)
        image_in = torch.cat(enhanced_images, dim=1)

        image_out = self.image_conv1(image_in)
        residuals_1 = image_out
        image_out = self.relu(image_out)
        image_out = self.image_conv2(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv3(image_out)
        image_out = self.image_maxpool2(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv4(image_out)
        image_out = self.relu(image_out)
        image_out = self.image_conv5(image_out)
        image_out = self.image_maxpool3(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv6(image_out)
        image_out += self.image_downpool1(residuals_1)

        image_out = self.image_drop2d(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_conv7(image_out)
        image_out = self.relu(image_out)

        image_out = self.image_flatten(image_out)
        image_out = self.image_fc1(image_out)
        image_out = self.image_drop(image_out)
        image_out = self.relu(image_out)
        image_out = self.image_fc2(image_out)

        return image_out


class LungCombinedResnet18PretrainedModel(nn.Module):
    def __init__(self, frozen_layers=[], number_of_buckets = 360):
        super(LungCombinedResnet18PretrainedModel, self).__init__()
        # import pudb; pudb.set_trace()
        self.model_data_type = "both"
        self.relu = nn.ReLU(inplace=True)

        # self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT, progress=True)
        with open("./library/resnet18.pkl", "rb") as f:
            self.resnet18 = pkl.load(f)
        resnet_layer_list = list(self.resnet18.children())
        self.resnet_layers = {
            "conv1": 0,
            "bn1": 1,
            "relu": 2,
            "maxpool": 3,
            "layer1": 4,
            "layer2": 5,
            "layer3": 6,
            "layer4": 7,
            "avgpool": 8,
            "fc": 9,
        }
        self.resnet_layers = {
            k: resnet_layer_list[v] for k, v in self.resnet_layers.items()
        }

        for layer in frozen_layers:
            for param in self.resnet_layers[layer].parameters():
                param.requires_grad = False
        # layer3 -> 256x16x16

        # metadata layers
        self.meta_fc1 = nn.Linear(in_features=7, out_features=16)
        self.meta_fc2 = nn.Linear(in_features=16, out_features=32)
        self.meta_fc3 = nn.Linear(in_features=32, out_features=64)
        self.meta_fc4 = nn.Linear(in_features=64, out_features=128)
        self.meta_fc5 = nn.Linear(in_features=128, out_features=32)
        self.meta_drop = nn.Dropout(p=0.05)

        self.image_drop = nn.Dropout(p=0.05)
        self.image_drop2d = nn.Dropout2d(p=0.01)

        self.combo_fc = nn.Linear(in_features=32, out_features=9)

        self.final_fc1 = nn.Linear(in_features=512, out_features=256)
        self.final_fc2 = nn.Linear(in_features=256, out_features=number_of_buckets)

        self.resnet_layers["fc"] = self.final_fc1

    def forward(self, data):
        # import pudb; pudb.set_trace()
        image, meta = data
        # meta layers
        meta_out = self.meta_fc1(meta)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc2(meta_out)
        meta_res1 = meta_out
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc3(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc4(meta_out)
        meta_out = self.meta_drop(meta_out)
        meta_out = self.relu(meta_out)

        meta_out = self.meta_fc5(meta_out)
        meta_out += meta_res1
        # meta_out = self.meta_drop(meta_out)
        combo_in2 = meta_out

        # combo layers
        combo_out = self.combo_fc(combo_in2)

  
        enhanced_images = []
        enhanced_planes = []
        for i in range(image.shape[1]):
            enhanced_image = combo_out[:,i][:, None, None, None] * image[:, i][:, None]
            enhanced_images.append(enhanced_image)
            if len(enhanced_images) == 3:
                enhanced_slice = torch.cat(enhanced_images, dim=1)
                enhanced_slice = torch.mean(enhanced_slice, dim=1, keepdim=True)
                enhanced_images = []
                enhanced_planes.append(enhanced_slice)
        image = torch.cat(enhanced_planes, dim=1)

        # image layers
        for layer_name, resnet_layer in self.resnet_layers.items():
            image = resnet_layer(image)
            if layer_name == "avgpool":
                image = image.reshape(image.size(0), -1)
            if layer_name == "layer4":
                image = self.image_drop2d(image)

        image = self.relu(image)

        output = self.meta_drop(image)
        output = self.final_fc2(output)

        return output