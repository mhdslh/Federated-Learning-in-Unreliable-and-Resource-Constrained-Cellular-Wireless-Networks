import torch
import numpy as np
import pickle
import os
cpath = os.path.dirname(__file__)

SAVE = True
DATASET_FILE = os.path.join(cpath, 'data')
np.random.seed(6)

N = 100

ALPHA = 1 # controls how much local models differ from each other
BETA = 1 # controls how much the local data at each device differs from that of other devices

input_dim = 60
output_dim = 10
n = np.random.lognormal(4, 2, (1,N)).astype(int) + 50 # n[0,i] is the total number of data at device i
                                                      # n.shape = (1,N)
n_train = np.round(0.9*n)
n_test = n - n_train

Train_Dataset = {} # Dataset['Device_i']=(x_i, y_i)
Test_Dataset = {} 
mu = np.random.normal(0, ALPHA, (1,N)) # mu.shape = (1,N)
B = np.random.normal(0, BETA, (1,N)) # B.shape = (1,N)
# To have a same model at all devices run Lines 27-29
# mu = np.random.normal(0, ALPHA, 1) # mu.shape = (1,N)
# W_i = np.random.normal(mu, 1, (output_dim, input_dim))
# b_i = np.random.normal(mu, 1, (output_dim, 1))
for i in range(N):
    W_i = np.random.normal(mu[0,i], 1, (output_dim, input_dim)) # W_i.shape = (output_dim, input_dim)
    b_i = np.random.normal(mu[0,i], 1, (output_dim, 1)) # b_i.shape = (output_dim, 1)
    
    v = np.random.normal(B[0,i], 1, input_dim)
    cov = np.diag( np.power(np.arange(1,input_dim+1),-1.2) )
    X_i = np.random.multivariate_normal(v, cov, n[0,i]) # X_i.shape = (n_samples, input_dim)
    temp = np.matmul(W_i,X_i.T) + b_i # temp.shape = (output_dim, n_samples)
    # argmax(softmax(x)) = argmax(x)
    y_i = np.argmax(temp, axis=0) # y_i.shape = (n_samples,)
    idx = int(n_train[0,i])
    Train_Dataset['Device_'+str(i)] = (X_i[0:idx],y_i[0:idx])
    Test_Dataset['Device_'+str(i)] = (X_i[idx::],y_i[idx::])


# Setup directory for train/test data
print('>>> Set data path for Synthetic.')
train_path = '{}/data/train/all_data_ALPHA_{}_BETA_{}.pkl'.format(cpath, ALPHA, BETA)
test_path = '{}/data/test/all_data_ALPHA_{}_BETA_{}.pkl'.format(cpath, ALPHA, BETA)


dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}


# Setup 1000 users
for i in range(N):
    uname = i

    train_data['users'].append(uname)
    train_X, train_y = Train_Dataset['Device_'+str(i)]
    x = torch.tensor(train_X, dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.float32)
    train_data['user_data'][uname] = {'x': x, 'y': y}
    train_data['num_samples'].append(n_train[0,i])
    
    test_data['users'].append(uname)
    test_X, test_y = Test_Dataset['Device_'+str(i)]
    x = torch.tensor(test_X, dtype=torch.float32)
    y = torch.tensor(test_y, dtype=torch.float32)
    test_data['user_data'][uname] = {'x': x, 'y': y}
    test_data['num_samples'].append(n_test[0,i]) 

print('>>> User data distribution: {}'.format(train_data['num_samples']))
print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))


# Save user data
if SAVE:
    with open(train_path, 'wb') as outfile:
        pickle.dump(train_data, outfile)
        
    with open(test_path, 'wb') as outfile:
        pickle.dump(test_data, outfile)

print('>>> Save data.')