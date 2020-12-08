from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


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


class Train():
  
  def __init__(self,model,weights,train_data,test_data,device,**kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)
    
  self.model = model
  self.weights = weights
  self.train_data = train_data
  self.test_data = test_data
  
  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if self.use_cuda else "cpu")
  torch.backends.cudnn.benchmark = True
  
  self.device = device

  self._train()

  def _train(self):
    
    
    """ Generators """
    training_generator = DataLoader(self.train_data, batch_size=self.batch_size,shuffle=self.shuffle, num_workers=self.num_workers, drop_last=self.drop_last)
    #TODO: make an actual validation set
    validation_set = self.test_data # why is num_workers = 0 ? 
    validation_generator = DataLoader(validation_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=self.drop_last)
    
    """ Instantiate model on device """
    criterion = nn.CrossEntropyLoss(weight = torch.from_numpy(self.weights).float())
    optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
    self.model.to(self.device)
    criterion.to(self.device)
    
    
    def optimize_sdg():
      
      print("Begin training with SGD...")
      
      """ Training """
      # Loop over epochs
      for epoch in range(self.max_epochs):
          running_loss = 0
          batch_number = 0
          
          # Begin
          for input_batch, input_labels in training_generator:
              
              batch_number += 1
              # Transfer to GPU
              input_batch, input_labels = input_batch.to(self.device), input_labels.to(self.device)
              # Model computations
              # zero the parameter gradients
              optimizer.zero_grad()
      
              # forward + backward + optimize
              outputs = self.model(input_batch.float())
      
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
                  outputs = self.model(local_batch.float())
                  loss = criterion(outputs, local_labels)
                  total_loss += loss
              print(total_loss)
      
      print('Finished training with SGD.')
      
  
    def optimize_sdg_reg_ilc():
  
      # Loop over epochs
      for epoch in range(self.max_epochs):
          running_loss = 0
          print(epoch)
          # Training
          batch_number = 0
          for input_batch, input_labels in training_generator:
              batch_number += 1
              # Transfer to GPU
              input_batch, input_labels = input_batch.to(self.device), input_labels.to(self.device)
              # Model computations
              # zero the parameter gradients
              optimizer.zero_grad()
      
              # forward + backward + optimize
              outputs = self.model(input_batch.float())
      
              if config['agreement_threshold'] > 0.0:
                  # The "batch_size" in this function refers to the batch size per env
                  # Since we treat every example as one env, we should set the parameter
                  # n_agreement_envs equal to batch size
                  mean_loss, masks = get_grads(
                      agreement_threshold=config['agreement_threshold'],
                      batch_size=1,
                      loss_fn=criterion,
                      n_agreement_envs=config['batch_size'],
                      params=optimizer.param_groups[0]['params'],
                      output=outputs,
                      target=input_labels,
                      method=config['method'],
                      scale_grad_inverse_sparsity=config['scale_grad_inverse_sparsity'],
                  )
      
                  if l1_coef > 0.0:
                      add_l1_grads(self.l1_coef, optimizer.param_groups)
      
                  if l2_coef > 0.0:
                      add_l2_grads(self.l2_coef, optimizer.param_groups)
      
              else:
                  mean_loss = criterion(outputs, input_labels)
                  mean_loss.backward()
              
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
                  local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)
      
                  # Model computations
                  outputs = self.model(local_batch.float())
                  loss = criterion(outputs, local_labels)
                  total_loss += loss
                  
              print(total_loss)
              
      print('Finished training with SGD + ILC.')
              
    
    if self.method == 'sdg': 
      return optimize_sdg()
    
    elif self.method == 'sdg-reg': 
      return optimize_sdg_reg()
    
    elif self.method == 'sdg-reg-ilc': 
      return optimize_sdg_reg_ilc()
      
    else:
      print("Select a valid training method (sdg, sdg-reg, sdg-reg-ilc")
      
  torch.save(self.model.state_dict(), self.model_output)
  print('Finished Training')
    
