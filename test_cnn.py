#this is the paret where we gona load out trained and saved model and we couldd test it
import torch
import torch.nn as nn


class cnn(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.activation = nn.GELU()
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(128 * 2 * 2, 10)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.linear(x)

        return x
model=cnn()
#now lets losd previous saved model
model.load_state_dict(torch.load("best.cnn.pth",weights_only=True))
model.eval()
from torchvision import datasets,transforms
transform=transforms.ToTensor()
test_dataset=datasets.CIFAR10(
    root="bibi",
    train=False,
    download=True,
    transform=transform
)
#our dataset hase only this type of images and also it coluld predict this thype of images because we only trained on these
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
image,label=test_dataset[10]#now we taking an image on dataset where on index 10
image=image.unsqueeze(0)#it adds a 1 at starting index of the particualr image of rgb is 1 in the begining on that image of rgb
#it adds the dim which acts as batch_size so that cnn could process it
with torch.no_grad():
    
    
    output=model(image)
    prediction=output.argmax(dim=1).item()
    print("Predicted:", classes[prediction])
print("Actual:", classes[label])#final prediction as we choose in index according to image so that it predicts
