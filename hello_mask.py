#!/usr/bin/env python
# coding: utf-8

# # Mukh O Mukhosh, the face mask detector

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from PIL import Image
import os


# ## dataset read/transform/prepocessing

# In[2]:


root_path = "./data"


# In[3]:


train_data_path = os.path.join(root_path, "train")


# In[4]:


print(train_data_path)


# In[5]:


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(250),
    transforms.ToTensor()
])


# In[6]:


train_data = datasets.ImageFolder(train_data_path, train_transform)


# In[7]:


print(train_data)


# In[8]:


print(train_data.classes)


# In[9]:


print(train_data.class_to_idx)


# In[10]:


print(train_data.imgs)


# ## train image dimension analysis using Pandas

# In[11]:


df = pd.DataFrame(train_data.imgs)


# In[12]:


print(df.head())


# In[13]:


print(df[0].describe())


# In[14]:


def image_size(img):
    with Image.open(img) as im:
        return im.size

img_sizes = [image_size(img[0]) for img in train_data.imgs]


# In[15]:


print(img_sizes)


# In[16]:


df_image_sizes = pd.DataFrame(img_sizes)


# In[17]:


print(df_image_sizes)


# In[18]:


print(df_image_sizes[0].describe())


# ## data loader

# In[19]:


train_loader = DataLoader(train_data, batch_size=10, shuffle=True)


# ## convolutional model

# In[20]:


class FaceMaskDetectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(61*61*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 61*61*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


# In[21]:


torch.manual_seed(42)
model = FaceMaskDetectorModel()
print(model)

# ## loss function & optimizer

# In[22]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ## training

# In[23]:


'''for b, (x_train, y_train) in enumerate(train_loader):
    print("b", b)
    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    break'''


# In[ ]:

import time
start_time = time.time()

epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1
        print("b",b)
        # Apply the model
        y_pred = model(X_train)  # we don't flatten X-train here
        print("y_pred",y_pred)
        loss = criterion(y_pred, y_train)
 
        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print interim results
        if b%600 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{10*b:6}/60000]  loss: {loss.item():10.8f}  accuracy: {trn_corr.item()*100/(10*b):7.3f}%')
        
    train_losses.append(loss)
    train_correct.append(trn_corr)
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

