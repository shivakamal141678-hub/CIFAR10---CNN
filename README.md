# CIFAR-10 CNN

## About
- CNN built from scratch using PyTorch
- CIFAR-10 image classification
- 10-class classification
- Best test accuracy: **88.58%**

## Dataset
- 50,000 training images
- 10,000 test images
- RGB images
- Image size: 32×32
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Model Architecture
- Conv2D
- Batch Normalization
- GELU activation
- Max Pooling
- Dropout
- Residual Block
- Fully Connected Layer

## Training
- Optimizer: Adam
- Loss: Cross Entropy
- Batch size: 64
- Initial learning rate: 0.001
- Weight decay: 1e-4
- Data augmentation: Random Crop + Random Horizontal Flip

## Results
| Model | Test Accuracy |
|---|---:|
| Initial CNN | 86.56% |
| CNN + Residual Block | 88.37% |
| Fine-tuned Model | **88.58%** |

## Error Analysis
- Evaluated class-wise accuracy
- Generated a confusion matrix
- Identified **cat vs dog** as the major confusion
- Found 212 cat/dog cross-confusion cases

## Experiments
- Normalization
- AdamW
- Additional residual block
- Label smoothing
- Checkpoint-based fine-tuning

Some experiments reduced performance, so they were rejected.

## What I Learned
- CNN tensor shapes and data flow
- Convolution and pooling
- Batch normalization and dropout
- Residual connections
- Model checkpointing
- Fine-tuning
- Confusion matrix analysis
- Class-wise evaluation
- Controlled ML experiments

## Project Structure
```text
cifar10-cnn/
├── cifar.py
├── test_cnn.py
├── cat_dog_errors.png
├── one_cat_dog_error.png
└── .gitignore
How to Run
.can upload images which i have trained on yes and see result and how confident my model is
the images i want you to upload and predict only which are below
Airplane
Automobile (cars, not trucks)
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck (pickup trucks, large trucks)
and link to check
https://shivakamal141678-hub-cifar10---cnn-app-ugxt66.streamlit.app/
.and code is on github
git clone https://github.com/shivakamal141678-hub/CIFAR10---CNN.git
cd CIFAR10---CNN
python cifar.py
Future Work
Improve the model through targeted error analysis
Build a real-world computer vision application
Deploy the model for practical use
Author

Shiva Kamal

B.Tech CSE — Artificial Intelligence & Data Science


Available next action: :contentReference[oaicite:0]{index=0}
