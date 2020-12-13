from scipy.stats import norm
import numpy as np 
import torch
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from models import LogisticRegression as LR_torch

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter

np.random.seed(0)

# class Dataset(torch.utils.data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, list_IDs, labels, X):
#         'Initialization'
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.X = X
# 
#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)
# 
#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]
# 
#         # Load data and get label
#         X = self.X[ID]
#         y = self.labels[ID]
# 
#         return X, y


####################################### CODE SARTS HERE ############################## 
E = 100 # number of environments
ntrain=50 
sigma = 2.5 # amount of noise 
signal_noise=3
K = 100 # number of predictors # Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

max_epochs = 5
l1_coef = 5e-3
reg=False

rand_strong_coefs = 3.
inv_weak_coefs = 0.3

# for every environment, pick at random a strongest predictor (does not have to be unique) + random small coefficients for all others
# EXCEPT for a subset of predictors which are concistent across all environments

# random normal matrix size (N,K)
#X = np.ones((E,K)) # data A = np.random.normal(0, 1, (3, 3))
X = np.random.normal(0, sigma, (E, K))
A = np.ones((E,K)) # coefficients 

y = np.random.randint(2,size=E)



# true invariant predictor indices 
true_pred_indices = [2,4,10,20,32]

X_train = X[:ntrain]
X_test = X[ntrain+1:]
y_train = y[:ntrain]
y_test = y[ntrain+1:]

print(f"""Parameters: 
Number of environments: {E}
Number of training exampels: {ntrain} 
Noise: {sigma} 
Number of predictors: {K} 
Random strong coefficient value: {rand_strong_coefs} 
True invariant coefficient value: {inv_weak_coefs}""")


for e in range(E):
  
  # pick a subset of best predictors at random 
  rand_best_indices = np.random.randint(K, size=10)
  
  # check to make sure these randoms aren't out true pred
  condition = True
  while condition:
    for i,rbp in enumerate(rand_best_indices):
      if rbp in true_pred_indices:
        rand_best_indices[i] = np.random.randint(K, size=1)
        condition = True
      else:
        condition = False
  
  # set the coefficients at the right row 
  # that separates each class
  if y[e] == 1:
    X[e,true_pred_indices] = inv_weak_coefs +  np.random.normal(0,signal_noise)
    X[e,rand_best_indices] = rand_strong_coefs +  np.random.normal(0,signal_noise)
    
  elif y[e] == 0:
    X[e,true_pred_indices] = 0 +  np.random.normal(0,signal_noise) #-inv_weak_coefs 
    X[e,rand_best_indices] = -rand_strong_coefs +  np.random.normal(0,signal_noise)
      
    
print(f"X shape: {X.shape}, y shape {y.shape}")

np.save('data/X_synthetic.npy',X)
np.save('data/y_synthetic.npy',y)

# 
# # train 
# clf = LogisticRegression(random_state=0).fit(X_train, y_train)
# y_pred = clf.predict(X_train)
# print(confusion_matrix(y_train,y_pred))
# print(f"Score on training set: {clf.score(X_train,y_train)}")
# 
# # test 
# y_pred = clf.predict(X_test)
# print(confusion_matrix(y_test,y_pred))
# print(f"Score on training set: {clf.score(X_test,y_test)}")
# 
# fc_dim = X.shape[1]
# model = LR_torch(fc_dim)
# 
# # CUDA for PyTorch
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True
# 
# 
# 
# # Datasets
# partition = {'train': [i for i in range(ntrain)], 'validation': [i for i in range((ntrain+1),E)]} # IDs
# #labels = {k,v for k,v in zip([i for i in range(ntrain)],y_train)} # Labels
# labels = dict(zip([i for i in range(ntrain)], y_train))
# 
# # Generators
# #training_set = Dataset(partition['train'], labels,X_train)
# #training_generator = torch.utils.data.DataLoader(training_set, **params)
# 
# #validation_set = Dataset(partition['validation'], labels,X_test)
# #validation_generator = torch.utils.data.DataLoader(validation_set, **params)
# 
# """ Instantiate model on device """
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# model.to(device)
# criterion.to(device)
# 
# start_ind = 0
# 
# X_train = torch.from_numpy(X_train)
# X_test = torch.from_numpy(X_test)
# y_train = torch.from_numpy(y_train)
# y_test = torch.from_numpy(y_test)
# 
# 
# # Loop over epochs
# for epoch in range(max_epochs):
#     # Training
#     #for local_batch, local_labels in training_generator:
#     #for end_ind in range(0,ntrain,params['batch_size']):
#     local_batch = X_train[:]
#     local_labels = y_train[:]
#     # Transfer to GPU
#     local_batch, local_labels = local_batch.to(device), local_labels.to(device)
# 
#     # Model computations
#     optimizer.zero_grad()
#     
#     outputs = model(local_batch.float())
#     loss = criterion(outputs, local_labels)
#     
#     if reg:
#         l1_crit = nn.L1Loss(reduction='sum')
#         reg_loss = 0
#         for param in model.parameters():
#             reg_loss += l1_crit(param.to(device), torch.from_numpy(np.zeros(param.shape)).to(device))
# 
#         loss += l1_coef * reg_loss
#       
#     
#     
#     loss.backward()
#     optimizer.step()
#     print(f'training loss: {loss}')
#     #start_ind = end_ind
# 
#         # if batch_number % 10 == 0:    # print every 10 samples
#         #   print(f"training loss: {loss}")
# 
#     # Validation
#     start_ind = 0
#     with torch.set_grad_enabled(False):
#         
#         #for local_batch, local_labels in validation_generator:
#         #for end_ind in range(ntrain,E,params['batch_size']):
#         # Transfer to GPU
#         local_batch = X_test
#         local_labels = y_test
#         local_batch, local_labels = local_batch.to(device), local_labels.to(device)
# 
#         # Model computations
#         outputs = model(local_batch.float())
#         val_loss = criterion(outputs, local_labels)
#         
#     print(f"validation loss: {val_loss}")
