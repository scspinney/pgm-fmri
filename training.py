from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc 
import os


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
    
    
    
def add_l1_grads(l1_coef, param_groups):
    for group in param_groups:
        for p in group['params']:
            assert p.grad is not None, 'We have not decided yet what to do in this case'
            grad = p.grad.data
            grad.add_(l1_coef, torch.sign(p.data))

def add_l2_grads(l2_coef, param_groups):
    for group in param_groups:
        for p in group['params']:
            assert p.grad is not None, 'We have not decided yet what to do in this case'
            grad = p.grad.data
            grad.add_(2*l2_coef, p.data)

def get_grads(agreement_threshold, batch_size, loss_fn,
              n_agreement_envs, params, output,
              target,
              method,
              scale_grad_inverse_sparsity,
              ):
    """
    Use the and mask or the geometric mean to put gradients together.
    Modifies gradients wrt params in place (inside param.grad).
    Returns mean loss and masks for diagnostics.
    Args:
        agreement_threshold: a float between 0 and 1 (tau in the paper).
            If 1 -> requires complete sign agreement across all environments (everything else gets masked out),
             if 0 it requires no agreement, and it becomes essentially standard sgd if method == 'and_mask'. Values
             in between are fractional ratios of agreement.
        batch_size: The original batch size per environment. Needed to perform reshaping, so that grads can be computed
            independently per each environment.
        loss_fn: the loss function
        n_agreement_envs: the number of environments that were stacked in the inputs. Needed to perform reshaping.
        params: the model parameters
        output: the output of the model, where inputs were *all examples from all envs stacked in a big batch*. This is
            done to at least compute the forward pass somewhat efficiently.
        method: 'and_mask' or 'geom_mean'.
        scale_grad_inverse_sparsity: If True, rescale the magnitude of the gradient components that survived the mask,
            layer-wise, to compensate for the reduce overall magnitude after masking and/or geometric mean.
    Returns:
        mean_loss: mean loss across environments
        masks: a list of the binary masks (every element corresponds to one layer) applied to the gradient.
    """

    param_gradients = [[] for _ in params]
    outputs = output.view(n_agreement_envs, batch_size, -1)
    targets = target.view(n_agreement_envs, batch_size, -1)

    outputs = outputs.squeeze(-1)
    targets = targets.squeeze(-1)

    total_loss = 0.
    for env_outputs, env_targets in zip(outputs, targets):
        env_loss = loss_fn(env_outputs, env_targets)
        total_loss += env_loss
        env_grads = torch.autograd.grad(env_loss, params,
                                           retain_graph=True)
        for grads, env_grad in zip(param_gradients, env_grads):
            grads.append(env_grad)
    mean_loss = total_loss / n_agreement_envs
    assert len(param_gradients) == len(params)
    assert len(param_gradients[0]) == n_agreement_envs

    masks = []
    avg_grads = []
    weights = []
    for param, grads in zip(params, param_gradients):
        assert len(grads) == n_agreement_envs
        grads = torch.stack(grads, dim=0)
        assert grads.shape == (n_agreement_envs,) + param.shape
        grad_signs = torch.sign(grads)
        mask = torch.mean(grad_signs, dim=0).abs() >= agreement_threshold
        mask = mask.to(torch.float32)
        assert mask.numel() == param.numel()
        avg_grad = torch.mean(grads, dim=0)
        assert mask.shape == avg_grad.shape

        if method == 'and_mask':
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            if scale_grad_inverse_sparsity:
                param.grad *= (1. / (1e-10 + mask_t))
        elif method == 'geom_mean':
            prod_grad = torch.sign(avg_grad) * torch.exp(torch.sum(torch.log(torch.abs(grads) + 1e-10), dim=0) / n_agreement_envs)
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * prod_grad
            if scale_grad_inverse_sparsity:
                param.grad *= (1. / (1e-10 + mask_t))
        else:
            raise ValueError()

        weights.append(param.data)
        avg_grads.append(avg_grad)
        masks.append(mask)

    return mean_loss, masks



class Train():
  
  def __init__(self,model,weights,train_data,test_data,**kwargs):

    self.model = model
    self.weights = weights
    self.train_data = train_data
    self.test_data = test_data
    
    for key, value in kwargs.items():
      setattr(self, key, value)
  
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda == 'True' else "cpu")
    torch.backends.cudnn.benchmark = True
    
    self.device = device
    
    print(f"Training using device: {device}")
    print(f"X shape: {train_data.subject_frames.shape}, y shape: {train_data.labels.shape}")
  
    self._train()

  def _train(self):
    
    """ Generators """
    training_generator = DataLoader(self.train_data, batch_size=self.batch_size,shuffle=self.shuffle, num_workers=self.num_workers, drop_last=(self.drop_last=='True'))
    #TODO: make an actual validation set
    validation_set = self.test_data # why is num_workers = 0 ? 
    validation_generator = DataLoader(validation_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=(self.drop_last=='True'))
    
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
          print(f"Epoch: {epoch}")
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
              print(f"training loss: {loss}")
      
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
              print(f"total validation: {total_loss}")
              
      torch.save(self.model.state_dict(), os.path.join(self.model_output, 'model-sdg.pt'))
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
      
              if self.agreement_threshold > 0.0:
                  # The "batch_size" in this function refers to the batch size per env
                  # Since we treat every example as one env, we should set the parameter
                  # n_agreement_envs equal to batch size
                  mean_loss, masks = get_grads(
                      agreement_threshold=self.agreement_threshold,
                      batch_size=self.batch_size,
                      loss_fn=criterion,
                      n_agreement_envs=len(input_labels),
                      params=optimizer.param_groups[0]['params'],
                      output=outputs,
                      target=input_labels,
                      method='and_mask',
                      scale_grad_inverse_sparsity=self.scale_grad_inverse_sparsity,
                  )
      
                  if self.l1_coef > 0.0:
                      add_l1_grads(self.l1_coef, optimizer.param_groups)
      
                  if self.l2_coef > 0.0:
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
      
      print('Finished Training')
  
      # # Loop over epochs
      # for epoch in range(self.max_epochs):
      #     running_loss = 0
      #     print(epoch)
      #     # Training
      #     batch_number = 0
      #     for input_batch, input_labels in training_generator:
      #         batch_number += 1
      #         # Transfer to GPU
      #         input_batch, input_labels = input_batch.to(self.device), input_labels.to(self.device)
      #         # Model computations
      #         # zero the parameter gradients
      #         optimizer.zero_grad()
      # 
      #         # forward + backward + optimize
      #         outputs = self.model(input_batch.float())
      # 
      #         if config['agreement_threshold'] > 0.0:
      #             # The "batch_size" in this function refers to the batch size per env
      #             # Since we treat every example as one env, we should set the parameter
      #             # n_agreement_envs equal to batch size
      #             mean_loss, masks = get_grads(
      #                 agreement_threshold=config['agreement_threshold'],
      #                 batch_size=1,
      #                 loss_fn=criterion,
      #                 n_agreement_envs=config['batch_size'],
      #                 params=optimizer.param_groups[0]['params'],
      #                 output=outputs,
      #                 target=input_labels,
      #                 method=config['method'],
      #                 scale_grad_inverse_sparsity=config['scale_grad_inverse_sparsity'],
      #             )
      # 
      #             if l1_coef > 0.0:
      #                 add_l1_grads(self.l1_coef, optimizer.param_groups)
      # 
      #             if l2_coef > 0.0:
      #                 add_l2_grads(self.l2_coef, optimizer.param_groups)
      # 
      #         else:
      #             mean_loss = criterion(outputs, input_labels)
      #             mean_loss.backward()
      #         
      #         optimizer.step()
      # 
      #         # print statistics
      #         # running_loss += loss.item()
      #         # if i % 10 == 0:    # print every 10 samples
      #         #     print('[%d, %5d] loss: %.3f' %
      #         #           (epoch + 1, i + 1, running_loss / 2000))
      #         #     running_loss = 0.0
      # 
      #     # Validation
      #     with torch.set_grad_enabled(False):
      #         total_loss = 0
      #         for local_batch, local_labels in validation_generator:
      #             # Transfer to GPU
      #             local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)
      # 
      #             # Model computations
      #             outputs = self.model(local_batch.float())
      #             loss = criterion(outputs, local_labels)
      #             total_loss += loss
      #             
      #         print(total_loss)
      #     
      #     #TODO: resolve the memory problem
      #     gc.collect()
      #     torch.cuda.empty_cache()
      # 
      # torch.save(self.model.state_dict(), os.path.join(self.model_output, 'model-sdg-reg-ilc.pt'))
      # print('Finished training with SGD + ILC.')
              
    
    if self.method == 'sdg': 
      return optimize_sdg()
    
    elif self.method == 'sdg-reg': 
      return optimize_sdg_reg()
    
    elif self.method == 'sdg-reg-ilc': 
      return optimize_sdg_reg_ilc()
      
    else:
      print("Select a valid training method (sdg, sdg-reg, sdg-reg-ilc")
      

