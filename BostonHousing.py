#!/usr/bin/env python
# coding: utf-8
# based on https://discuss.pytorch.org/t/pytorch-fails-to-over-fit-boston-housing-dataset/40365

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy  as np
import sklearn
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pandas as pd

boston = load_boston()
X,y   = (boston.data, boston.target)
dim = X.shape[1]


X.shape

y

# Skip the next four lines if BostonHousing.csv is not available.
house = pd.read_csv('BostonHousing.csv')
print(house.head(10))
house.hist(column='medv', bins=50)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)
num_train = X_train.shape[0]
X_train


torch.set_default_dtype(torch.float64)
net = nn.Sequential(
    nn.Linear(dim, 50, bias = True), nn.ELU(),
    nn.Linear(50,   50, bias = True), nn.ELU(),
    nn.Linear(50,   50, bias = True), nn.Sigmoid(),
    nn.Linear(50,   1)
)
criterion = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr = .0005)


num_epochs = 8000
#from torch.utils.data import TensorDataset, DataLoader
y_train_t =torch.from_numpy(y_train).clone().reshape(-1, 1)
x_train_t =torch.from_numpy(X_train).clone()
#dataset = TensorDataset(torch.from_numpy(X_train).detach().clone(), torch.from_numpy(y_train).reshape(-1,1).detach().clone())
#loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
losssave = []
stepsave = []

for i in range(num_epochs):
    y_hat = net(x_train_t)
    loss = criterion(y_train_t,net(x_train_t))
    losssave.append(loss.item())
    stepsave.append(i)
    loss.backward()
    opt.step()
    opt.zero_grad()
    y_hat_class = (y_hat.detach().numpy())
    accuracy = np.sum(y_train.reshape(-1,1)== y_hat_class )/len(y_train)
    if i > 0 and i % 100 == 0:
        print('Epoch %d, loss = %g acc = %g ' % (i, loss,  accuracy))

ss=np.array(stepsave)
ss.shape
sl =np.array(losssave)
sl.shape
#print (y_hat_class)
#print(y_train.reshape(-1,1))
#ss.reshape(8000)
#sl.reshape(8000)

py = net(torch.DoubleTensor(X_train))
plt.plot(sl, '+')
plt.xlabel('Actual value of training set')
plt.ylabel('Prediction')
plt.show()

ypred = net(torch.from_numpy(X_test).detach())
err = ypred.detach().numpy() - y_test
mse = np.mean(err*err)
print(np.sqrt(mse))
plt.plot(ypred.detach().numpy(),y_test, '+')
plt.show()


model = MLPRegressor(
    hidden_layer_sizes=(50,50,50),
    alpha = 0,
    activation='relu',
    batch_size=128,
    learning_rate_init = 1e-3,
    solver = 'adam',
    learning_rate = 'constant',
    verbose = False,
    n_iter_no_change = 1000,
    validation_fraction = 0.0,
    max_iter=1000)
model.fit(X_train, y_train)

py = model.predict(X_test)
err = y_test - py
mse = np.mean(err**2)
rmse = np.sqrt(mse)
print('rmse for test %g' % rmse)
plt.subplot(121)
plt.plot(y_test, py, '+')
plt.show()
err = y_train - model.predict(X_train)
mse = np.mean(err**2)

plt.plot(py)
py.mean()

