import pandas as pd
import numpy as np
import torch
import glob
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_path = './Dataset/GestureA/0 degree/'
x_filenames = glob.glob(x_path + '*.csv')
X = []
Y = []

for filename in x_filenames:
    data = pd.read_csv(filename)
    ii = data.iloc[:600,:138]
    qq = data.iloc[:600,138:276]
    complexqq = qq*1j
    iq = ii.add(complexqq, fill_value=0)
    X.append(abs(iq))
    Y.append(0)
    
X_array = np.array(X)
Y_array = np.array(Y)
dataset_X = torch.FloatTensor(X_array)
dataset_X = dataset_X.unsqueeze(1) #so that the format is accepted for eg. [50, 600, 276] --> [50, 1, 600, 276]
dataset_Y = torch.tensor(Y_array,dtype=torch.long)
X_train, X_test, Y_train, Y_test = train_test_split( dataset_X,dataset_Y, test_size=0.1, random_state=42)
print(Y_train.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(54*144*63, 1024)  
        #self.conv1 = nn.Conv2d(1, 32, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.cnn_layers(x)
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

model = Net()
print(model)

# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.07)
# defining the loss function
criterion = nn.CrossEntropyLoss()

def train(epoch, x_train, y_train, x_val, y_val):
    model.train()
    tr_loss = 0

    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

# defining the number of epochs
n_epochs = 25
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
for epoch in range(n_epochs):
    train(epoch,X_train,Y_train,X_test,Y_test)