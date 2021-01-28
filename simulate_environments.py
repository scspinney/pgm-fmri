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

from training import evaluate

np.random.seed(0)


E = 100 # number of environments
ntrain=30 
sigma = 1 # amount of noise 
signal_noise=1.5
K = 100 # number of predictors # Parameters

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


for e in range(E):
  # new seed
  np.random.seed(e)
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

print(f"""Parameters: 
Number of environments: {E}
Number of training exampels: {ntrain} 
Noise: {sigma} 
Number of predictors: {K} 
Random strong coefficient value: {rand_strong_coefs} 
True invariant coefficient value: {inv_weak_coefs}
""")
