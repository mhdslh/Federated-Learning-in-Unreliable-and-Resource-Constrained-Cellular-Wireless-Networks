import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
import os
import pickle
import scipy.io

cpath = os.path.dirname(__file__)
IMAGE_DATA = False
image = 1 if IMAGE_DATA else 0
filename = '{}/data/train/all_data_{}_random_niid.pkl'.format(cpath, image)
with open(filename, 'rb') as f:
    train_data = pickle.load(f)
    
# Simulation parameters
N = 100 # this can be obtained from train_data['users'] 
n = np.array(train_data['num_samples']) # n.shape = (N,)
T = 400 # total number of iterations
input_dim = 784
output_dim = 10
ETA_0 = 1 # Learning rate at iteration 0. In the original paper, it has been choosen from the set {1, 0.1, 0.001}
LAMBDA = 0.0001 # Regularization parameter (Weight decay)

def mlp(sizes, hidden_activation=nn.ReLU, output_activation=nn.Identity): 
    # nn.CrossEntropyLoss does the softmax operation. 
    # Thus, we can use nn.Identity as the activation function of the output layer. 
    layers = []
    for l in range(len(sizes)-1):
        act = hidden_activation if l<len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[l], sizes[l+1]), act()]
    return nn.Sequential(*layers)

def init_weights(m):
    for param in m.parameters():
        nn.init.uniform_(param, -0.1,0.1)

sizes = [input_dim, 300, 300, output_dim]
Network = mlp(sizes) 
criterion = nn.CrossEntropyLoss()

init_weights(Network)

X_cat = torch.tensor([])
y_cat = torch.tensor([])
n_data = np.sum(n)
for i in range(N):
    X = train_data['user_data'][i]['x']
    y = train_data['user_data'][i]['y']
    X_cat = torch.cat((X_cat,X), dim=0)
    y_cat = torch.cat((y_cat,y))
    
LOSS = np.zeros(T+1)
ACCURACY = np.zeros(T+1)

for ROUND in range(T+1):
    print(ROUND, T+1)
    
    ETA = ETA_0/(1+ROUND)
    Optimizer = optim.SGD(Network.parameters(), lr=ETA, weight_decay=LAMBDA)
   
    Optimizer.zero_grad()
    predicted = Network(X_cat) # predicted.size() = [n_samples, output_dim]
    loss = criterion(predicted, y_cat.long())
    loss.backward()
    Optimizer.step()
    
    LOSS[ROUND] = loss
    y_hat = torch.argmax(predicted, dim=1)
    ACCURACY[ROUND] = torch.sum( y_hat==y_cat ).item()/n_data
    
plt.plot(LOSS)
plt.plot(ACCURACY)

filename = '{}/data/test/all_data_{}_random_niid.pkl'.format(cpath, image)
with open(filename, 'rb') as f:
    test_data = pickle.load(f)
n = np.array(test_data['num_samples'])
X_cat = torch.tensor([])
y_cat = torch.tensor([])
n_data = np.sum(n)
for i in range(N):
    X = test_data['user_data'][i]['x']
    y = test_data['user_data'][i]['y']
    X_cat = torch.cat((X_cat,X), dim=0)
    y_cat = torch.cat((y_cat,y))   
with torch.no_grad():
    predicted = Network(X_cat)
    y_hat = torch.argmax(predicted, dim=1)
    TEST_ACCURACY = torch.sum( y_hat==y_cat ).item()/n_data
    print(TEST_ACCURACY)

DICT = {'LOSS_MNIST': LOSS, 'ACCURACY_MNIST': ACCURACY, 'TEST_ACCURACY_MNIST': TEST_ACCURACY}
scipy.io.savemat('Step2_Output_Benchmark_MNIST.mat', DICT)