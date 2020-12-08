import torch
import torch.nn as nn
import torch.optim as optim

import copy

import numpy as np
import matplotlib.pyplot as plt

import os
import pickle
import scipy.io

# Load Data
print('>>> Load Data')
cpath = os.path.dirname(__file__)
dataset = 'MNIST'

if dataset == 'MNIST':
    image = 0
    filename = '{}/data/train/all_data_{}_random_niid.pkl'.format(cpath, image)
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
    filename = '{}/data/test/all_data_{}_random_niid.pkl'.format(cpath, image)
    with open(filename, 'rb') as f:
        test_data = pickle.load(f)
    input_dim = 784
    output_dim = 10
    batch_size = 64
    ETA_0 = 1 # Learning rate at iteration 0. In the original paper, it has been choosen from the set {1, 0.1, 0.001}


        
if dataset == 'SYNTHETIC':
    ALPHA = 1
    BETA = 1
    filename = '{}/data/train/all_data_ALPHA_{}_BETA_{}.pkl'.format(cpath, ALPHA, BETA)
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
    filename = '{}/data/test/all_data_ALPHA_{}_BETA_{}.pkl'.format(cpath, ALPHA, BETA)
    with open(filename, 'rb') as f:
        test_data = pickle.load(f)
    input_dim = 60
    output_dim = 10
    batch_size = 25
    ETA_0 = 0.1 # Learning rate at iteration 0. In the original paper, it has been choosen from the set {1, 0.1, 0.001}



# Set Simulation Parameters
print('>>> Set Simulation Parameters')
N = 100 # Number of devices. Can be obtained by train_data['users']
M = 20 # Number of subchannels
n = np.array(train_data['num_samples']) # Number of training data samples at each device 
                                        # n.shape = (N,)
p = n/np.sum(n)                         # p.shape = (N,)
LAMBDA = 0.0001 # Regularization parameter (Weight decay)
E = 1 # number of SGD steps between each two consecutive communication steps
K = 200 # number of rounds
Matlab_content = scipy.io.loadmat('Step1_Output_U_ell_1_2_and_3.mat')
ell = 2
if ell==1:
    U = Matlab_content['U_ell_1'] # U.shape = (1,N)
if ell==2:
    U = Matlab_content['U_ell_2'] # U.shape = (1,N)
if ell==3:
    U = Matlab_content['U_ell_3'] # U.shape = (1,N)
    
n_iterations = 10 # Average the Global Loss over the number of simulations

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
Network = {}
Optimizer = {}
Network['Server'] = mlp(sizes) 
criterion = nn.CrossEntropyLoss()
 
def Broadcast(ROUND): # t = time
    for i in range(N):
        Network['Device_'+str(i)] = copy.deepcopy(Network['Server']) 
        ETA = ETA_0/(1+ROUND)
        Optimizer['Device_'+str(i)] = optim.SGD(Network['Device_'+str(i)].parameters(), lr=ETA, weight_decay=LAMBDA)
    return Network, Optimizer

def Participants(alg, pr_scheduling=np.zeros(N)):
    if alg == 'Sahu':
        S = np.random.choice(N, size=(1,M), replace=True, p=p) # Sampling (Scheduling)
                                                               # S.shape = (1,M)
    if alg == 'Proposed-U':
        S = np.random.choice(N, size=(1,M), replace=False) # Sampling (Scheduling)
                                                           # S.shape = (1,M)
    if alg == 'Proposed-W':
        S = np.random.choice(N, size=(1,M), replace=True, p=pr_scheduling) # Sampling (Scheduling)
                                                                           # S.shape = (1,M)  
    temp = np.random.rand(1,M) # temp.shape = (1,M)
    return [s for k, s in enumerate(S[0]) if temp[0,k]<U[0,s]] # returns a list of the participants' indices   
        
def local_SGD(active_devices, Network, Optimizer, E):  
    devices = list(dict.fromkeys(active_devices)) # remove the repeated elements
    for i in devices:
        idx = np.random.randint(0, n[i], E*batch_size) # idx.shape = (E*batch_size,)
        X = train_data['user_data'][i]['x']
        y = train_data['user_data'][i]['y']
        samples_x = X[idx,:] # samples_x.size() = [E * batch size, input_dim]
        samples_y = y[idx].long() # samples_y.size = [E * batch size]
        for _ in range(E):
            input_batch = samples_x[_*batch_size:(_+1)*batch_size,:] # input_batch.size() = [batch_size, input_dim]
            target_batch = samples_y[_*batch_size:(_+1)*batch_size] # target_batch.size() = [batch_size]
            Optimizer['Device_'+str(i)].zero_grad()
            predicted = Network['Device_'+str(i)](input_batch) # predicted.size() = [output_dim]
            loss = criterion(predicted, target_batch)
            loss.backward()
            Optimizer['Device_'+str(i)].step()
    return Network
        
def Aggregation(active_devices, Network, alg, q=np.ones(N)):
    # q depends on the scheduling policy and pr_scheduling
    # for Proposed-U algorithm: q = M/N*np.ones(N)
    # for Proposed-W algorithm: q = M*pr_scheduling
    with torch.no_grad():
        Server_Params = list(Network['Server'].parameters())
        # Network['Server'] weights change when we change Server_Params values
         
        if alg == 'Sahu':
            N_active = len(active_devices)
            if N_active>0:
                for l in range(len(Server_Params)): 
                    Server_Params[l] -= Server_Params[l]
                for i in active_devices:
                    Device_Params = list(Network['Device_'+str(i)].parameters())
                    for l in range(len(Server_Params)):
                        Server_Params[l] += 1/M * Device_Params[l] 
                        # Server_Params[l] += 1/N_active * Device_Params[l]
        
        if alg == 'Proposed-U' or alg == 'Proposed-W':
            for i in active_devices:
                Device_Params = list(Network['Device_'+str(i)].parameters())
                for l in range(len(Server_Params)):
                    Server_Params[l] += p[i]/(q[i]*U[0,i]) * (Device_Params[l]-Server_Params[l])
        
        # l = 0
        # for param in Network['Server'].parameters():
        #     param.copy_(Server_Params[l])                      
        #     l += 1
    return Network
            
def Performance(Network, Dataset):
    Global_Loss = 0
    Accuracy = 0
    n_data = np.sum( np.array(Dataset['num_samples']) )
    for i in range(N):
        X = Dataset['user_data'][i]['x']
        y = Dataset['user_data'][i]['y']
        with torch.no_grad():
            predicted = Network['Server'](X) # predicted.size() = [n_samples, output_dim]
            loss = criterion(predicted, y.long())
            y_hat = torch.argmax(predicted, dim=1) # y_hat.size() = [n_samples]
            Accuracy += torch.sum( y_hat==y ).item()/n_data
            Global_Loss += p[i]*loss
    return Global_Loss, Accuracy    

# if alg = 'Proposed-U':
#    set q = M/N*np.ones(N)
# if alg = 'Proposed-W':
#    define pr_scheduling with size (N,)
#    set q = M*pr_scheduling

# STEP 1
#    Server Broadcasts the global model parameters to all devices at the 
#    beginning of the round
# STEP 2
#    Determine the set of active devices (devices that successfully 
#    transmit their local model parameters to the server at the end of this
#    round)
#    Remark: When alg = 'Proposed-W', we also need to pass pr_scheduling to 
#    the function as
# STEP 3
#    Perform local update only on the devices that have successful 
#    transmission (doing so for other devices is not helpful)
# STEP 4
#    Server aggregates the local models to update the global model parameters
#    Remark1: When alg = 'Proposed-U', q=M/N*np.ones(N)
#    Remark2: When alg = 'Proposed-W', q=M*pr_scheduling
# Step 5
#    We calculate the global objective function 
#    This is performed at the beginning of the next round

alg = 'Proposed-W'

if alg=='Proposed-U':
    pr = 1/N*np.ones(N) # has no effect
    q = M/N*np.ones(N)
if alg=='Proposed-W':
    pr = 1/N*np.ones(N)
    q = M*pr

LOSS = np.zeros((n_iterations, K+1))
ACCURACY = np.zeros((n_iterations, K+1))
TEST_ACCURACY = np.zeros(n_iterations)

for itr in range(n_iterations):
    init_weights(Network['Server'])
    LOSS[itr,0], ACCURACY[itr,0] = Performance(Network, train_data)
    
    for ROUND in range(K):
        print('iteration {}/{} | round {}/{}' .format(itr+1, n_iterations, ROUND+1, K))
        Network, Optimizer = Broadcast(ROUND)
        active_devices = Participants(alg, pr_scheduling=pr)
        Network = local_SGD(active_devices, Network, Optimizer, E)
        Network = Aggregation(active_devices, Network, alg, q=q)
        LOSS[itr,ROUND+1], ACCURACY[itr,ROUND+1] = Performance(Network, train_data)
    
    no_need, TEST_ACCURACY[itr] = Performance(Network, test_data)

plt.figure(1) 
plt.plot(np.mean(LOSS,0))
plt.plot(np.mean(ACCURACY,0))

# plt.legend([''])
plt.ylabel('Global Loss', fontsize=12)
plt.xlabel('Round (K)', fontsize=12)
plt.grid(True)
plt.axis([0, K, 0, 3])
plt.tight_layout()
plt.show()

DICT = {}
DICT['LOSS'] = LOSS
DICT['ACCURACY'] = ACCURACY
DICT['TEST_ACCURACY'] = TEST_ACCURACY
DICT['K'] = K
DICT['ell'] = ell
DICT['E'] = E
DICT['dataset'] = dataset
DICT['alg'] = alg

scipy.io.savemat('Step3_Output_SchemeII_MNIST_200_2_1.mat', DICT)