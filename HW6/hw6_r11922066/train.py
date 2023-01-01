import os
import pandas as pd
import numpy as np

from utils.options import args
import utils.common as utils

from importlib import import_module

import torch
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from data import dataPreparer

import warnings, math

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)

loss_history_train = []
loss_history_valid = []

def main():

    start_epoch = 0
    best_acc = 0.0
 
    # Data loading
    print('=> Preparing data..')
 
    # data loader
    
    loader = dataPreparer.Data(args, 
                               data_path=args.src_data_path, 
                               label_path=args.src_label_path)
    
    data_loader = loader.loader_train
    data_loader_valid = loader.loader_valid
    data_loader_test = loader.loader_test
    
    
    # Create model
    print('=> Building model...')

    # load training model
    model = import_module(f'model.{args.arch}').__dict__[args.model]().to(device)
    

    # Load pretrained weights
    if args.pretrained:
 
        ckpt = torch.load(os.path.join(checkpoint.ckpt_dir, args.source_file), map_location = device)
        state_dict = ckpt['state_dict']

        model.load_state_dict(state_dict)
        model = model.to(device)
        
    if args.inference_only:
        inference(args, data_loader_valid, model, args.output_file)
        return

    param = [param for name, param in model.named_parameters()]
    
    optimizer = optim.SGD(param, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma = args.lr_gamma)

    summary(model, input_size=( 3, 28, 28))
    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        
        train(args, data_loader, model, optimizer, epoch)
        
        valid_acc = valid(args, data_loader_valid, model)
   
        is_best = best_acc < valid_acc
        best_acc = max(best_acc, valid_acc)
        

        state = {
            'state_dict': model.state_dict(),
            
            'optimizer': optimizer.state_dict(),
            
            'scheduler': scheduler.state_dict(),
            
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        
    inference(args, data_loader_test, model, args.output_file)
    print(f'Best acc: {best_acc:.3f}\n')
    print(model)
    plot_confusion(args, data_loader_valid, model, args.output_file)
    plot_training(loss_history_train, loss_history_valid)
    
    


  
       
def train(args, data_loader, model, optimizer, epoch):
    losses = utils.AverageMeter()

    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()
    
    num_iterations = len(data_loader)
    
    # switch to train mode
    model.train()
        
    for i, (inputs, targets, _) in enumerate(data_loader, 1):
        
        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # train
        output = model(inputs)
        loss = criterion(output, targets)

        # optimize cnn
        loss.backward()
        optimizer.step()

        ## train weights        
        losses.update(loss.item(), inputs.size(0))
        
        ## evaluate
        prec1, _ = utils.accuracy(output, targets, topk = (1, 5))
        acc.update(prec1[0], inputs.size(0))
        
        
        if i % args.print_freq == 0:     
            loss_history_train.append(losses.avg)
            print(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                'Train acc {acc.val:.3f} ({acc.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses,
                acc = acc))
                
      
 
def valid(args, loader_valid, model):
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_valid, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
             
            preds = model(inputs)
            loss = criterion(preds, targets)
        
            # image classification results
            prec1, _ = utils.accuracy(preds, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            
            acc.update(prec1[0], inputs.size(0))

    loss_history_valid.append(losses.avg)
    print(f'Validation acc {acc.avg:.3f}\n')

    return acc.avg
    

def inference(args, loader_test, model, output_file_name):
    outputs = []
    datafiles = []
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_test, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
          
            preds = model(inputs)
    
            _, output = preds.topk(1, 1, True, True)
            
            outputs.extend(list(output.reshape(-1).cpu().detach().numpy()))
            
            datafiles.extend(list(datafile))
            
    

    output_file = dict()
    output_file['image_name'] = datafiles
    output_file['label'] = outputs
    
    output_file = pd.DataFrame.from_dict(output_file)
    output_file.to_csv(output_file_name, index = False)
  
def inference_valid(args, loader_test, model, output_file_name):
    outputs = []
    datafiles = []
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_test, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
          
            preds = model(inputs)
    
            _, output = preds.topk(1, 1, True, True)
            
            outputs.extend(list(output.reshape(-1).cpu().detach().numpy()))
            
            datafiles.extend(list(datafile))
    
    return outputs

def plot_confusion(args, loader_test, model, output_file_name):
    pred = inference_valid(args, loader_test, model, output_file_name)
    df = pd.read_csv('../digit/valid.csv')
    ans = df['label']

    label_name = np.arange(10)
    array = confusion_matrix(ans, pred, )
    plt.figure(figsize=(8, 8))
    sns.heatmap(array, annot=True, fmt='d', square=True, cmap='Blues', xticklabels=label_name, yticklabels=label_name)
    plt.xlabel('Predict', size=10)
    plt.ylabel('True', size=10)
    plt.savefig('HW6_matrix.png')
    plt.show()

def plot_training(loss_history_train, loss_history_valid):
    epoch = np.arange(len(loss_history_train)) + 1
    #steps_v = np.arange(len(loss_history_valid)) + 1
    plt.figure(figsize=(12, 6))
    plt.plot(epoch, loss_history_train, 'b', label='train')
    plt.plot(epoch, loss_history_valid, 'r', label='valid')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('cross entropy loss')
    plt.title('Learning Curve')

    plt.savefig('HW6_model.png')
    plt.show()

if __name__ == '__main__':
    main()

