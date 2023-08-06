"""
Model class definitions
"""
import torch
import torch.nn as nn



def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ProstateImageModel(nn.Module):
    def __init__(self):
        super(ProstateImageModel, self).__init__()
        self.model_data_type = 'images'

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3) # 240
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding='same') # 120
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same') # 120
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # 60
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # 30
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same') # 30
        self.maxpool3 = nn.MaxPool2d(kernel_size=3) # 10
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels=384, kernel_size=3, stride=1, padding='same') # 10
        self.conv7 = nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, stride=2, padding=1) # 5
        self.downpool1 = nn.Conv2d(in_channels = 16, out_channels=384, kernel_size=12, stride=12, padding=1) # 120 -> 15
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=5*5*512, out_features=256)
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


class ProstateMetadataModel(nn.Module):
    def __init__(self):
        super(ProstateMetadataModel, self).__init__()
        self.model_data_type ='metadata'

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features = 12, out_features=16)
        self.fc2 = nn.Linear(in_features = 16, out_features=32)
        self.fc3 = nn.Linear(in_features = 32, out_features=128)
        self.fc4 = nn.Linear(in_features = 128, out_features=256)
        self.fc5 = nn.Linear(in_features = 256, out_features=512)
        self.fc6 = nn.Linear(in_features = 512, out_features=64)
        self.fc7 = nn.Linear(in_features = 64, out_features=2)
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
    

class ProstateCombinedModel(nn.Module):
    def __init__(self):
        super(ProstateCombinedModel, self).__init__()
        self.model_data_type ='both'

        self.relu = nn.ReLU(inplace=True)

        # metadata layers
        self.meta_fc1 = nn.Linear(in_features = 12, out_features=16)
        self.meta_fc2 = nn.Linear(in_features = 16, out_features=32)
        self.meta_fc3 = nn.Linear(in_features = 32, out_features=128)
        self.meta_fc4 = nn.Linear(in_features = 128, out_features=256)
        self.meta_fc5 = nn.Linear(in_features = 256, out_features=512)
        # self.meta_fc6 = nn.Linear(in_features = 512, out_features=256)
        # self.meta_fc7 = nn.Linear(in_features = 64, out_features=2)
        self.meta_drop = nn.Dropout(p=0.1)

        # image layers
        self.image_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3) # 224 -> 112
        self.image_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding='same') # 112
        self.image_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same') # 112
        self.image_maxpool2 = nn.MaxPool2d(kernel_size=2) # 56
        self.image_conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # 28
        self.image_conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same') # 28
        self.image_maxpool3 = nn.MaxPool2d(kernel_size=3) # 9
        self.image_conv6 = nn.Conv2d(in_channels = 256, out_channels=384, kernel_size=3, stride=1, padding='same') # 9
        self.image_conv7 = nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, stride=2, padding=1) # 5
        self.image_downpool1 = nn.Conv2d(in_channels = 16, out_channels=384, kernel_size=12, stride=12, padding=1) # 120 -> 15
        self.image_flatten = nn.Flatten()
        self.image_fc1 = nn.Linear(in_features=5*5*512, out_features=256)
        self.image_fc2 = nn.Linear(in_features=256, out_features=2)
        self.image_drop = nn.Dropout(p=0.2)
        self.image_drop2d = nn.Dropout2d(p=0.1)

        # combo layers
        self.combo_fc1 = nn.Linear(in_features = 12, out_features=224*224)
        self.combo_fc2 = nn.Linear(in_features = 512, out_features=9*9*384)
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
        # import pudb; pudb.set_trace()
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

        image_out = self.image_flatten(image_out)
        image_out = self.image_fc1(image_out)
        image_out = self.image_drop(image_out)
        image_out = self.relu(image_out)
        image_out = self.image_fc2(image_out)

        # combined_out = torch.cat((image_out, meta_out), dim=1)
        # combined_out = self.combo_output(combined_out)

        return image_out