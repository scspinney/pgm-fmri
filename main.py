import json

params_file = 'main_parameters.json'

def train(params_file):
  
  with open(params_file,'r') as f:
    params = 

# nifti_file = 'sub-0' + str(subject) + '_task-stopsignal_run-01_bold.nii.gz'

number_of_subjects = 3
subsampler = Subsample(2,2,2)
flat = Flatten()

composed_transform = transforms.Compose([subsampler,flat])

data_format = 'npy' # 'npy' or 'nifti'

number_of_subjects = 2


subjects_individual_train_datasets = []
subjects_individual_test_datasets = []

files_paths = []

# partition = {'train':list(range(0,91)), 'test':list(range(91,182))}
partition = {'train':list(range(0,123)), 'test':list(range(123,246))}


# For having flexibility in defining various train and test partitions when using
# the mixed dataset of all subjects, we may use the list of partitions and labels
# as below

partitions_mixed = dict.fromkeys(['train','test'])
partitions_mixed['train'],partitions_mixed['test'] = [],[]

labels_mixed = dict.fromkeys(['train','test'])
labels_mixed['train'],labels_mixed['test'] = [],[]

for subject in range(number_of_subjects):
    # one might change the partitions for each subject.
    partitions_mixed['train'].append(list(range(0,91)))
    partitions_mixed['test'].append(list(range(91,182)))
    labels_mixed['train'].append((np.zeros(91)).astype(int))
    labels_mixed['test'].append((np.zeros(91)).astype(int))

if data_format is 'nifti':

    labels = {'train':(np.zeros(91)).astype(int), 'test':(np.zeros(91)).astype(int)}

    for subject in range(1,number_of_subjects+1):
        path = 'sub-0' + str(subject) + '_task-stopsignal_run-01_bold.nii.gz'
        files_paths.append(path)

    subjects_individual_train_datasets.append(fmriDatasetSubject(root_dir, path, partition['train'], labels['train'], data_format, composed_transform))
    subjects_individual_test_datasets.append(fmriDatasetSubject(root_dir, path, partition['test'], labels['test'], data_format, composed_transform))

    mixed_dataset_train = fmriDatasetAllSubjects(root_dir, files_paths, partitions_mixed['train'], labels_mixed['train'], data_format, composed_transform)
    mixed_dataset_test = fmriDatasetAllSubjects(root_dir, files_paths,  partitions_mixed['test'], labels_mixed['test'], data_format, composed_transform)

else:

    subjects_event_paths = '/content/drive/MyDrive/Mila Fall 2020/Probabilistic Graphical Models/Project/Data/data/y.npy'
    events = np.load(subjects_event_paths, encoding='bytes')
    events[np.where(events == -1)] = 1
    
    labels = dict.fromkeys(['train','test'])
    labels['train'],labels['test'] = events[partition['train']],events[partition['test']]

    files_paths = '/content/drive/MyDrive/Mila Fall 2020/Probabilistic Graphical Models/Project/Data/data/X.npy'

    mixed_dataset_train = fmriDatasetAllSubjects(root_dir, files_paths, partition['train'], labels['train'], data_format, composed_transform)
    mixed_dataset_test = fmriDatasetAllSubjects(root_dir, files_paths,  partition['test'], labels['test'], data_format, composed_transform)

# print(labels)
print(getattr(mixed_dataset_train, 'subject_frames').shape)
print(mixed_dataset_train[57][0].shape)

# print(getattr(b,'subject_frames').shape)
print(mixed_dataset_train,mixed_dataset_test)
print(getattr(mixed_dataset_train,'subject_frames').shape)
print(getattr(mixed_dataset_test,'subject_frames').shape)
print(mixed_dataset_train[0][0].shape[0])

# How to access these custom datasets
# print(subjects_individual_datasets[0][2])
# print(mixed_dataset[4])

def storeData(object, file_name, root_dir):
    with open(root_dir+file_name, 'wb') as f:
        pickle.dump(object, f)					 
        f.close() 

def loadData(file_name, root_dir): 
    with open(root_dir+file_name, 'rb') as f:
        db = pickle.load(f) 
        f.close()
        return db


# storeData(subjects_individual_train_datasets, 'subjects_individual_train_datasets', root_dir)
# storeData(subjects_individual_test_datasets, 'subjects_individual_test_datasets', root_dir)

# storeData(mixed_dataset_train, 'mixed_dataset_train', root_dir)
# storeData(mixed_dataset_test, 'mixed_dataset_test', root_dir)


mixed_dataset_train = loadData('mixed_dataset_train', root_dir)
mixed_dataset_test = loadData('mixed_dataset_test', root_dir)
# print((getattr(mixed_dataset_train,'subject_frames').shape[3]))
# print(getattr(mixed_dataset_test,'subject_frames').shape)


# We weight the training loss by the weight of labels in the training set.
Y = getattr(mixed_dataset_train,'labels')
weights = [len(np.where(Y == 0)[0]), len(np.where(Y == 1)[0])]/np.max([len(np.where(Y == 0)[0]), len(np.where(Y == 1)[0])])
print(weights)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


fc_dim = mixed_dataset_train[0][0].shape[0]
# now pass fcdim to the Net class from models.py
model = Net(fc_dim)

# Parameters
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 6}

max_epochs = 3

# Since the labels are probably unbalanced, cross entropy loss should get the weight parameter
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(weight = torch.from_numpy(weights).float())

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.to(device)
criterion.to(device)

# Generators
training_set = mixed_dataset_train
training_generator = DataLoader(training_set, batch_size=32,shuffle=True, num_workers=0, drop_last=False)

validation_set = mixed_dataset_test
validation_generator = DataLoader(validation_set, batch_size=32, shuffle=True, num_workers=0, drop_last=False)

# Loop over epochs
for epoch in range(max_epochs):
    running_loss = 0
    # Training
    batch_number = 0
    for input_batch, input_labels in training_generator:
        batch_number += 1
        # Transfer to GPU
        input_batch, input_labels = input_batch.to(device), input_labels.to(device)
        # Model computations
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_batch.float())

        loss = criterion(outputs, input_labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 10 == 0:    # print every 10 samples
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

    # Validation
    with torch.set_grad_enabled(False):
        total_loss = 0
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            outputs = model(local_batch.float())
            loss = criterion(outputs, local_labels)
            total_loss += loss
        print(total_loss)

print('Finished Training')

torch.save(model.state_dict(), '/content/drive/MyDrive/Mila Fall 2020/Probabilistic Graphical Models/Project/Data/model')
# model.load_state_dict(torch.load(PATH))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

test_gen = torch.utils.data.DataLoader(mixed_dataset_test, batch_size=mixed_dataset_test.__len__(), shuffle=False)
for test_data, test_labels in test_gen:
    pass
    # print(test_data.shape, test_labels.shape)

# print(test_data.shape, test_labels.shape)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)/float(len(labels))

def evaluate(model, validation_data, validation_labels):

    validation_data = validation_data.to(device)

    out = model(validation_data.float())
    # print(out.shape)
    outputs = np.argmax(out.cpu().detach().numpy(), axis=1)
    # print(outputs)
    acc = accuracy(out.cpu().detach().numpy(), validation_labels.cpu().detach().numpy())
    print('Accuracy: ', acc)

    # As mentioned before, data is unbalanced, hence, the accuracy itself is not 
    # enough for evaluating the performance of the model.
    # print(outputs,local_labels.cpu().detach().numpy())
    cm = confusion_matrix(outputs.transpose(), validation_labels.detach().cpu().numpy().transpose())
    sns.set_theme()
    plt.figure()
    ax = sns.heatmap(cm)
    print('\nConfusion Matrix: ', cm)
    precision,recall,fscore,_ = precision_recall_fscore_support(validation_labels.cpu(), outputs)
    print('\nPrecision: ',precision,'\nRecall: ', recall,'\nF-score: ', fscore)

evaluate(model, test_data, test_labels)


params = model.parameters()
for par in params:
    pass
    # print(par.shape)
    # print(par.grad)

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

f = jnp.array([1,2,3,4,5,6]).astype(float)

def sq(x):
    return jnp.sum(x**2)

# derivative_f = grad(sq)
# print(derivative_f(f))

criterion.to('cpu')
print(torch.tensor([[2,3],[4,5]]))
print(torch.tensor([2,5]).shape)
print(criterion(torch.tensor([[2,3],[4,5]]).float(),torch.tensor([1,0])))
derivative_loss = grad(criterion)

print(derivative_loss(f))

# Parameters
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 6}

max_epochs = 2

# Since the labels are probably unbalanced, cross entropy loss should get the weight parameter
# reduction is set to 'none' so that loss is not returned as a scalar, rather a tensor of values for elements in a batch
criterion = nn.CrossEntropyLoss(weight = torch.from_numpy(weights).float(), reduction='none') 

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.to(device)
criterion.to(device)

# Generators
training_set = mixed_dataset_train
training_generator = DataLoader(training_set, batch_size=32,shuffle=True, num_workers=6, drop_last=False)

validation_set = mixed_dataset_test
validation_generator = DataLoader(validation_set, batch_size=32, shuffle=True, num_workers=6, drop_last=False)

# Loop over epochs
for epoch in range(max_epochs):
    running_loss = 0
    # Training
    batch_number = 0
    for input_batch, input_labels in training_generator:
        batch_number += 1
        # Transfer to GPU
        input_batch, input_labels = input_batch.to(device), input_labels.to(device)
        # Model computations
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_batch.float())
        # outputs = outputs.view(1,-1)
        # print('epoch: ', epoch, 'batch_number: ', batch_number, '\n',  outputs.shape, input_labels.shape)
        loss = criterion(outputs, input_labels)
        print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 10 == 0:    # print every 10 samples
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

    # Validation
    with torch.set_grad_enabled(False):
        total_loss = 0
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            outputs = model(local_batch.float())
            loss = criterion(outputs, local_labels)
            total_loss += loss
        print(total_loss)

print('Finished Training')

def train(x, y, method, lr, weight_decay, n_iters, verbose=False):

    for epoch in range(max_epochs):
    running_loss = 0
    # Training
    batch_number = 0
    for input_batch, input_labels in training_generator:
        batch_number += 1
        # Transfer to GPU
        input_batch, input_labels = input_batch.to(device), input_labels.to(device)
        # Model computations
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_batch.float())
        # outputs = outputs.view(1,-1)
        # print('epoch: ', epoch, 'batch_number: ', batch_number, '\n',  outputs.shape, input_labels.shape)
        loss = criterion(outputs, input_labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 10 == 0:    # print every 10 samples
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

    # Validation
    with torch.set_grad_enabled(False):
        total_loss = 0
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            outputs = model(local_batch.float())
            loss = criterion(outputs, local_labels)
            total_loss += loss
        print(total_loss)

    print('Finished Training')

    
    thetas, iters, losses = [], [0], []
    theta = torch.randn(5, requires_grad=True) * 0.1
    thetas.append(theta.data.numpy())
    
    with torch.no_grad():
        loss = loss_fn(x, theta, y)
        losses.append(loss.item())
    
    for i in range(n_iters + 1):
        lr *= 0.9995
        
        grads = []
        loss_e = 0.
        for e in range(x.shape[0]):
            loss_e = loss_fn(x[e], theta, y[e])
            grad_e = torch.autograd.grad(loss_e, theta)[0]
            grads.append(grad_e)

        grad = torch.stack(grads, dim=-1)
        
        if method == 'geom_mean':
            n_agreement_domains = len(grads)
            signs = torch.sign(grad) 
            mask = torch.abs(signs.mean(dim=-1)) == 1
            avg_grad = grad.mean(dim=-1) * mask
            prod_grad = torch.sign(avg_grad) * \
                        torch.exp(torch.sum(torch.log(torch.abs(grad) + 1e-10), dim=1)) \
                        ** (1. / n_agreement_domains)
            final_grads = prod_grad
        elif method == 'and_mask':
            signs = torch.sign(grad) 
            mask = torch.abs(signs.mean(dim=-1)) == 1
            avg_grad = grad.mean(dim=-1) * mask
            final_grads = avg_grad
        elif method == 'arithm_mean':
            avg_grad = grad.mean(dim=-1)
            final_grads = avg_grad
        else:
            raise ValueError()
            
        theta = theta - lr * final_grads
        
        # weight decay
        theta -= weight_decay * lr * theta
        
        if not i % (n_iters // 200):
            thetas.append(theta.data.numpy())
            iters.append(i)
            with torch.no_grad():
                loss = loss_fn(x, theta, y)
                losses.append(loss.item())
        
        if not i % (n_iters // 5):
            print(".", end="")
            with torch.no_grad():
                loss = loss_fn(x, theta, y)
                if verbose:
                    print(f"loss: {loss.item():.6f}, theta: {theta.data.numpy()}, it: {i}")
              
    with torch.no_grad():
        loss = loss_fn(x, theta, y)
        print(f"loss: {loss.item():.6f}, theta: {theta.data.numpy()}, it: {i}")
    return np.stack(thetas), iters, losses


print(X[0])
numpy_4d_img = X[0].dataobj
numpy_4d_img.shape 
print(numpy_4d_img)

slice_50 = index_img(X[0], 50) # grab slice number 50 into a 
slice_50_60 = index_img(X[0], slice(50, 60)) # grab slice number 50 to 60 

slice_50.shape, slice_50_60.shape

a = index_img(X[0], 181)
print(a.header)

from nilearn import plotting
from nilearn import image
data = slice_50.get_fdata()
# plotting.plot_stat_map(slice_50)

for img in image.iter_img(slice_50_60):
    # img is now an in-memory 3D img
    # print(slice_50_60)
    plotting.plot_stat_map(img, threshold=0, display_mode="z", cut_coords=1,
                           colorbar=False)
