import pandas as pd
import numpy as np
import torch
import glob
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pdb
import os
import torchvision
import time

#How to load and prepare photos of dogs and cats for modeling.
#How to develop a convolutional neural network for photo classification from scratch and improve model performance.
#How to develop a model for photo classification using transfer learning.



X = []
Y = []
torch.cuda.empty_cache()

training_path='./Dataset/train/'
training_filename = glob.glob(training_path + '*.csv')
test_path='./Dataset/test/'
test_filename = glob.glob(test_path + '*.csv')        
torch.manual_seed(42)

class Dataset_Interpreter(Dataset):
    def __init__(self,file_names, transforms=None):
        self.file_names = file_names
        self.transforms = transforms

    def __len__(self):
        return (len(self.file_names))

    def __getitem__(self,idx):
        data = pd.read_csv(self.file_names[idx])
        ii = data.iloc[250:1100,:138]
        qq = data.iloc[250:1100,138:276]
        complexqq = qq.astype(float)*1j
        iq = ii.add(complexqq, fill_value=0)
        X =abs(iq)
        Y= 3
        if(os.path.basename(self.file_names[idx]).startswith('A')):
            Y = 0
        elif(os.path.basename(self.file_names[idx]).startswith('B')):
            Y = 1
        elif(os.path.basename(self.file_names[idx]).startswith('')):
            Y = 2
    
        X_array = np.array(X)
        
        dataset_X = torch.FloatTensor(X_array)
        dataset_X = dataset_X.unsqueeze(0) #so that the format is accepted for eg. [50, 600, 276] --> [50, 1, 600, 276]
        dataset_Y = torch.tensor(Y,dtype=torch.long)
        
        #pdb.set_trace() 
        if self.transforms is not None:
            dataset_X = self.transforms(dataset_X)   
        return dataset_X,dataset_Y

mytransform = torchvision.transforms.RandomAffine(degrees= 10, translate=(0.25, 0.5), 
scale=(1.2, 2.0), shear=0.1)
train_data = Dataset_Interpreter(file_names=training_filename ,transforms=None)
#test_data = Dataset_Interpreter(file_names=test_filename ,transforms=None)
train_ds, test_data = torch.utils.data.random_split(train_data, (320, 80))
BATCH_SIZE = 10
train_iterator = DataLoader(train_ds, shuffle=True, batch_size= BATCH_SIZE)
test_iterator = DataLoader(test_data, shuffle=False, batch_size= 10)
print(len(train_data))
print(len(train_ds.indices), len(test_data.indices))



class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)


def ResNet18(img_channels=3, num_classes=1000):
    return ResNet(18, Block, img_channels, num_classes)


def ResNet34(img_channels=3, num_classes=1000):
    return ResNet(34, Block, img_channels, num_classes)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(50, Block, img_channels, num_classes)


def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(101, Block, img_channels, num_classes)


def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(152, Block, img_channels, num_classes)



model = ResNet50(img_channels=1, num_classes=4)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    print("cuda")
    model = model.cuda()
    criterion = criterion.cuda()

#train_losses = []
#train_counter = []


def train(epoch):
    total=0
    correct=0
    model.train()
    for batch_idx, (data, target) in enumerate(train_iterator):

        if torch.cuda.is_available():
            data=data.cuda()
            target=target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        #accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        loss.backward()
        optimizer.step()
    
    print('Train Epoch: {}  Loss: {:.3f}| Acc:{:.3f}'.format(epoch, loss.item(),correct / total))
    #train_losses.append(loss.item())
    #train_counter.append((batch_idx*64) + ((epoch-1)*len(train_iterator.dataset)))

def test():
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_iterator:
            if torch.cuda.is_available():
                data=data.cuda()
                target=target.cuda()

            output = model(data)
            loss = criterion(output, target)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Test : Loss: {:.3f}, Acc:{:.3f}\n'.format(
    loss, correct / len(test_iterator.dataset)))

#test() #ramdonly allocate test
n_epochs = 20
# training the model

for epoch in range(1, n_epochs + 1):
    #t0 = time.time()
    train(epoch)
    test()
    #print('{:.4f} seconds'.format((time.time()-t0)))
#
