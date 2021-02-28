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

#How to load and prepare photos of dogs and cats for modeling.
#How to develop a convolutional neural network for photo classification from scratch and improve model performance.
#How to develop a model for photo classification using transfer learning.

A_path = './Dataset/GestureA/0 degree/'
A_filenames = glob.glob(A_path + '*.csv')
B_path = './Dataset/GestureC/0 degree/'
B_filenames = glob.glob(B_path + '*.csv')
X = []
Y = []
if(B_filenames[0].startswith('B')):
    print('hello')
training_path='./Dataset/train/'
training_filename = glob.glob(training_path + '*.csv')

# for files in training_filename:
    
#     #print(os.path.basename(files))
#     data = pd.read_csv(files)
    
#     ii = data.iloc[:550,:138]
#     qq = data.iloc[:550,138:276]
#     complexqq = qq.astype(float)*1j
#     iq = ii.add(complexqq, fill_value=0)
#     X.append(abs(iq))
#     if(os.path.basename(files).startswith('A')):
#         Y.append(0.0)
#     else:
#         Y.append(1.0)
    
# X_array = np.array(X)
# dataset_X = torch.FloatTensor(X_array)
# dataset_X = dataset_X.unsqueeze(1) #so that the format is accepted for eg. [50, 600, 276] --> [50, 1, 600, 276]
# dataset_Y = torch.tensor(Y,dtype=torch.long)
        


class Dataset_Interpreter(Dataset):
    def __init__(self,file_names, transforms=None):
        self.file_names = file_names
        self.transforms = transforms

    def __len__(self):
        return (len(self.file_names))

    def __getitem__(self,idx):
        data = pd.read_csv(self.file_names[idx])
        ii = data.iloc[:550,:138]
        qq = data.iloc[:550,138:276]
        complexqq = qq.astype(float)*1j
        iq = ii.add(complexqq, fill_value=0)
        X =abs(iq)
        Y= 1
        if(os.path.basename(self.file_names[idx]).startswith('A')):
            Y = 0
    
        X_array = np.array(X)
        
        dataset_X = torch.FloatTensor(X_array)
        dataset_X = dataset_X.unsqueeze(0) #so that the format is accepted for eg. [50, 600, 276] --> [50, 1, 600, 276]
        dataset_Y = torch.tensor(Y,dtype=torch.long)
        
        #pdb.set_trace()    
        return dataset_X,dataset_Y

train_data = Dataset_Interpreter(file_names=training_filename ,transforms=None)
BATCH_SIZE = 10
train_iterator = DataLoader(train_data, shuffle=True, batch_size= BATCH_SIZE)


class Net(nn.Module):
    def __init__(self):
        

        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5), #stride=1, padding=0 is a default
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(141504, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        
        return x


model = Net()
print(model)

# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
# defining the loss function
criterion = nn.BCEWithLogitsLoss()


train_losses = []
train_counter = []

def train(epoch,criterion):
  model.train()
  for batch_idx, (data, target) in enumerate(train_iterator):
    optimizer.zero_grad()
    output = model(data)
    target = target.unsqueeze(1)
    target = target.type_as(output)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_iterator.dataset),100. * batch_idx / len(train_iterator), loss.item()))
    train_losses.append(loss.item())
    train_counter.append((batch_idx*64) + ((epoch-1)*len(train_iterator.dataset)))


# defining the number of epochs
n_epochs = 3


# training the model
for epoch in range(1, n_epochs + 1):
  train(epoch,criterion)
