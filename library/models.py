"""
Model class definitions
"""
import torch.nn as nn



def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ProstateImageModel(nn.Module):
    def __init__(self):
        super(ProstateImageModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3) # 240
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding='same') # 120
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same') # 120
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # 60
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # 30
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same') # 30
        self.maxpool3 = nn.MaxPool2d(kernel_size=3) # 10
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels=384, kernel_size=3, stride=1, padding='same') # 10
        self.conv7 = nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, stride=2, padding=1) # 5
        self.downpool1 = nn.Conv2d(in_channels = 16, out_channels=384, kernel_size=5, stride=8, padding=2) # 120 -> 15
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=5*5*512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=2)
        self.drop = nn.Dropout(p=0.2)
        self.drop2d = nn.Dropout2d(p=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        residuals_1 = x
        x = self.relu(x)

        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.maxpool3(x)
        x = self.relu(x)

        x = self.conv6(x)
        x += self.downpool1(residuals_1)
        x = self.drop2d(x)
        x = self.relu(x)

        x = self.conv7(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class ProstateMetadataModel(nn.Module, ChaimeleonChallengeModel):
    def __init__(self):
        super(ProstateMetadataModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features = 16, out_features=16)
        self.fc2 = nn.Linear(in_features = 16, out_features=32)
        self.fc3 = nn.Linear(in_features = 32, out_features=128)
        self.fc4 = nn.Linear(in_features = 128, out_features=256)
        self.fc5 = nn.Linear(in_features = 256, out_features=512)
        self.fc6 = nn.Linear(in_features = 512, out_features=64)
        self.fc7 = nn.Linear(in_features = 64, out_features=2)
        self.drop = nn.Dropout(p=0.1)

        self.model = self

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.drop(x)
        x = self.relu(x)

        x = self.fc5(x)
        x = self.drop(x)
        x = self.relu(x)

        x = self.fc6(x)

        return x