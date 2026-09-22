#here we are importing pytorch its an farme work used for deep learning
import torch
#and importing nuralnetwork
import torch.nn as nn
#and here we are importing dataset of cifar10 from torchvision
from torchvision import datasets, transforms
#and here we are impoeting dataloader
from torch.utils.data import DataLoader



#now we creating our resedualarcethure
class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
#this is our first convlayer
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        #and here is batchnorm
        #it acts as an stablizer it takes our feature maps and normalize numer inside
        self.bn1 = nn.BatchNorm2d(channels)

#this is our 2nd conv layer
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
#and here is our activation where its gelu widely used
#its adds up non_liniarity to our model so that the model could learn complex patterns
        self.activation = nn.GELU()
    #now we are giving it the forward pass    

    def forward(self, out):

        identity = out
#now here we passing out through every layer which we have took
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.activation(out)

        return out #now it retuns the features which we have created as an arecthure


#now here is thecrucial part where we create our cnn layers so that the datacoud pass through multiple layers which we have created as flow
class cnn(nn.Module):

    def __init__(self):

        super().__init__()
#its our 1st conv layer
        self.conv1 = nn.Conv2d(3, 32, 3,padding=1)
        #and its batchnorm which acts as an stablizer
        self.bn1 = nn.BatchNorm2d(32)
        #now here is our activation function which adds non_liniarity so that our model could learn complicated patterns
        self.activation = nn.GELU()
        #this is our max pooling where it reduces the size of feature maps
        #reduces computation by reducing our feature map 
        #it also prevents from overfitting
        self.pool1 = nn.MaxPool2d(2)
        # dropout deactivate some of it nurons so that the learning could speard acros other nurons 
        #and its about 0.25 and its less
        self.dropout1 = nn.Dropout(0.25)
#this is our layer2
        self.conv2 = nn.Conv2d(32, 64, 3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        #layer3

        self.conv3 = nn.Conv2d(64, 128, 3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.25)
         #now here we gonna add residual layer which we created before
         #where it jumps and skips a pawt of layer and gets to aplus point
         #and we added it before 4th layer it could be placed according to the models perfomance 1 or 2 or 3th layer
        self.residual = ResidualBlock(128)

        #layer4
        self.conv4 = nn.Conv2d(128, 256, 3,padding=1)
        self.bn4= nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(0.25)
        #5th layer
        self.conv5 = nn.Conv2d(256, 512, 3,padding=1)
        self.bn5= nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2)
        self.dropout5 = nn.Dropout(0.25)

           #now this layer adds up that grid which we got from before
        self.flatten = nn.Flatten()

#now here we adding of nn where the values passes through every layer
#which has 512 and that 10 are the categories where we got from our cifar10 dataset
        self.linear = nn.Linear(512*1*1,10)
        #now its time to forward pass


    def forward(self, x):
#layer1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)
#layer2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)
#layer3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
#adding resedual as we added above as same as before
        x = self.residual(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        #layer4

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        #5th layer
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.pool5(x)
        x = self.dropout5(x)
      
           
        x = self.flatten(x)

        x = self.linear(x)

        return x


#now we are tellin it that model is cnn

model = cnn()
#now when we print this model we get the moth con and resedual arecthure which we have created
print(model)


#this is our data aurgmentation
#this is where we make the pictures upside down and crom so that model could learn more complecated visuals

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),#it flips image
    transforms.RandomCrop(32, padding=4), #it crops and which is 32features and padding is where it adds up a boundary of number to entire grid so that it could also make shure learning edges
    transforms.ToTensor()#it changes the image which we can see in to numbers because meachine cant see as we see so we transfom them in to numbers tensors
])
#it transforms our pixeld imaes to tensors where model could understand it

test_transform = transforms.ToTensor()

#the dataset we taking is cifar10 saving downloading training
train_dataset = datasets.CIFAR10(
    root="bibi",
    download=True,
    train=True,
    transform=transform
)

#its test dataset downloadis true  train is false because we are testing
test_dataset = datasets.CIFAR10(
    root="bibi",
    download=True,
    train=False,
    transform=test_transform
)
#lets add loader so tat we could put batch_size and shuffle and size is 64 per batch 1 epoch mean it goes through whole data set respective to batch_size
#we can imagin the math behind

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
#now lets test the loader


test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


#now lets add loss fn and optimizer
loss_fn = nn.CrossEntropyLoss()
#here we go optimizer with lr=0.001 and weight_deacy 
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,weight_decay=1e-4
)
#2nd learning sheduler it shedules the lr after 10 epochs so model could take small steps
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)


#now its time to train the model
best_accuracy=0#where we start it as best acc as 0
#totally 50 rounds where each round goes through whole datasst 1 so it repated for 50 epochs so that model could decrease in loss

for epoch in range(50):  
    #training model

    model.train()
#runnin_loss is0
    running_loss = 0

    correct = 0
    total = 0
#lets add for loop so that it could adjust weights and parameters

    for inputs, targets in train_loader:
        #this is the step we are telling it to clear previous gradents

        optimizer.zero_grad()
        #now lets give forward pass

        output = model(inputs)
#now lets add loss fn to the output and targets
        loss = loss_fn(output, targets)
#lets calcuate the gradents by using backprop 
        loss.backward()
#lets add those calcuated gradents tithe model
        optimizer.step()
       

#let it predict rrunning loss first as we gave before
        running_loss += loss.item()
#lets check predicted output and armax dim 1 is were it gives the highes probability num or logits to us
        predicted = output.argmax(dim=1)

        total += targets.size(0)
#lets check weather predicted is equal to targets
        correct += (predicted == targets).sum().item()
#lets give name as train_acc and out of 100 let it see how many correct our model gets

    train_accuracy = 100 * correct / total

#now its time for testing this is where we could understand that our model could perfom on unseen images
    model.eval()#this tells the model that iam gonna test the model no weights adjusted no gradents

    correct = 0#correct as 0
    total = 0 #total as 0
#this is step were everything is stop and no gradents flow nothing to adjust

    with torch.no_grad():
        #leyts fun for loop for it and check

        for inputs, targets in test_loader:

            output = model(inputs)

            predicted = output.argmax(dim=1)

            total += targets.size(0)

            correct += (predicted == targets).sum().item()


    test_accuracy = 100 * correct / total

#lets print what we got
    print(
        "Epoch:", epoch + 1,
        "Loss:", running_loss / len(train_loader),
        "Train Accuracy:", train_accuracy,
        "Test Accuracy:", test_accuracy
    )
    #also adding sheduler where after 10 it goes as per our shedule
    scheduler.step()
#lets add condition in it
if test_accuracy>best_accuracy:#we are telling model if best is test to save the model 
    #for example while training at any particular round model could get better we dont know so that at best acc model saves it and we can save those weights and biases  and a;so can load for fine tuning
    
    best_accuracy=test_accuracy
    torch.save(model.state_dict(),"best.cnn.pth")#this is used to save the model which we have trained and also we could load it so our model no need to train it longer to use
    
    print("model saved")#finally print the model wfter all before executed so that the model is saved on a best acc and testa ass



  
