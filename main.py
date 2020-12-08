import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import json
import torch
from models import Net 
from training import Train, evaluate, accuracy



""" Read in parameters """
params_file = 'main_parameters.json'
params = json.loads(params_file)


""" Get datasets and weights for optimization """

train_data, test_data, weights =  prepare_data(
                                                                root_dir,
                                                                subjects_events_path,
                                                                subjects_samples_pathnumber_of_subjects,
                                                                subsample,
                                                                data_format,
                                                                train_partition,
                                                                test_partition,
                                                                )



""" Train """
fc_dim = mixed_dataset_train[0][0].shape[0]
model = Net(fc_dim)

trained_model = Train(model,weights,train_data,test_data,params['optimization'])
evaluate(model, test_data, test_labels)

# model.load_state_dict(torch.load(PATH))

test_gen = torch.utils.data.DataLoader(test_data, batch_size=test_data.__len__(), shuffle=False)
for test_data, test_labels in test_gen:
    pass
    # print(test_data.shape, test_labels.shape)

# print(test_data.shape, test_labels.shape)







