import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# -----------------------------
# 1. Residual Block
# -----------------------------

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=1
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=1
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.activation = nn.GELU()


    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.activation(out)

        return out


# -----------------------------
# 2. CNN Model
# -----------------------------

class cnn(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 32, 3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.activation = nn.GELU()
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(
            32, 64, 3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(
            64, 128, 3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.25)

        self.residual = ResidualBlock(128)

        self.conv4 = nn.Conv2d(
            128, 256, 3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(
            256, 512, 3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2)
        self.dropout5 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(
            512 * 1 * 1,
            10
        )


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

        x = self.residual(x)

        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.pool5(x)
        x = self.dropout5(x)

        x = self.flatten(x)

        x = self.linear(x)

        return x


# -----------------------------
# 3. Load trained model
# -----------------------------

model = cnn()

model.load_state_dict(
    torch.load(
        "best2.cnn.pth",
        weights_only=True,
        map_location="cpu"
    )
)

model.eval()


# -----------------------------
# 4. CIFAR-10 classes
# -----------------------------

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


# -----------------------------
# 5. Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


# -----------------------------
# 6. Streamlit UI
# -----------------------------

st.title("CIFAR-10 CNN Classifier")

st.write(
    "Upload an image and let my CNN predict its CIFAR-10 class."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# 7. Prediction
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image"
    )

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = output.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0, prediction
        ].item()

    st.subheader(
        f"Prediction: {classes[prediction]}"
    )

    st.write(
        f"Confidence: {confidence * 100:.2f}%"
    )
