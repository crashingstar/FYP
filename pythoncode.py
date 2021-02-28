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



A_path = './Dataset/GestureA/0 degree/'
A_filenames = glob.glob(A_path + '*.csv')
B_path = './Dataset/GestureB/0 degree/'
B_filenames = glob.glob(B_path + '*.csv')
X = []
Y = []

class Dataset_Interpreter(Dataset):
    def __init__(self,file_names, transforms=None):
        self.file_names = file_names
        #self.label = label
        self.transforms = transforms

    def __len__(self):
        return (len(self.file_names))

    def __getitem__(self,idx):
        data = pd.read_csv(self.file_names[idx])
        ii = data.iloc[:600,:138]
        qq = data.iloc[:600,138:276]
        complexqq = qq*1j
        iq = ii.add(complexqq, fill_value=0)
        X =abs(iq)
        Y = 0
        
        X_array = np.array(X)
       
        
        dataset_X = torch.FloatTensor(X_array)
        dataset_X = dataset_X.unsqueeze(0) #so that the format is accepted for eg. [50, 600, 276] --> [50, 1, 600, 276]
        dataset_Y = torch.tensor(Y,dtype=torch.long)
        
        #pdb.set_trace()    
        return dataset_X,dataset_Y

train_data = Dataset_Interpreter(file_names=B_filenames ,transforms=None)
train_data.class_to_idx
BATCH_SIZE = 5
train_iterator = DataLoader(train_data, shuffle=True, batch_size= BATCH_SIZE)


class Net(nn.Module):
    def __init__(self):
        

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(310464, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 310464)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
print(model)

# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
# defining the loss function
criterion = nn.CrossEntropyLoss()


train_losses = []
train_counter = []

def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_iterator):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_iterator.dataset),100. * batch_idx / len(train_iterator), loss.item()))
    train_losses.append(loss.item())
    train_counter.append((batch_idx*64) + ((epoch-1)*len(train_iterator.dataset)))


# defining the number of epochs
n_epochs = 10


# training the model
for epoch in range(1, n_epochs + 1):
  train(epoch)
