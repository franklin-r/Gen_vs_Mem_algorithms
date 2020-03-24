# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:31:35 2020

@File			:   CNN.py
@author         :   Alexis ROSSI <alexis.rossi97@gmail.com>
@Description 	:	CNN of variables hidden layers and feature maps number
@Released		:	08/03/2020
@Updated		:         
    
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from random import choice, randint
from time import clock
#from time import perf_counter


# Define the CNN
class CNN(nn.Module) :
    """
    \Description : CNN architecture
    \Attributes :
        dataset     : dataset used with the network
        chromosome  : dictionnary of the hyper-parameters of the network. These  are :
            NL  : number of hidden layers
            NF  : number of feature maps
            lr  : learning rate
            mom : momentum
        layers          : list of layers
        inaccuracy      : inaccuracy of the model during the test phase
        time            : computation time during the test phase
        fitness         : fitness score of the model
        feat_maps_seq   : list of the sizes of feature maps
    """
    def __init__(self, dataset, NL, NF, lr, mom) :
        """
        \Description : Build a CNN following specific rules. Those are [A. Bakshi et al. 2019] :
            "
            (1) The CNN architecture is created by an alternative combination of the convolutional and max pooling layers 
            that are followed by an averaging pooling layer and a linear fully connected layer on the top
            (2) The minimum and maximum number of consecutive convolutional layers is 2 and 4, respectively
            (3) Two pooling layers never occur directly in sequence
            (4) We keep adding a sequence of convolutional layers and pooling layer until we reach the number of layers (NL)
            (5) A subsequence of size NF is selected from the sequence {32, 64, 128, 256, 512}. The elements of the selected
            subsequence are randomly used as the feature map value of convolutional block. NF can take the values 3, 4 or 5
            (6) If the number of feature maps (NF) were less than the number of different convolutional blocks in the 
            generated network, then one/more of the selected feature maps will be repeated randomly
            (7) For each layer the same kernel size and stride size is used
            (8) Each convolutional layer is followed by batch normalization and rectifier function (ReLU)
            "
        \Args : 
            dataset : the dataset used for the CNN
            NL : number of hidden layers
            NF : number of feature maps  
            lr  : learning rate
            mom : momentum
        \Outputs : None
        """
        
        super(CNN, self).__init__()
        
        self.dataset    = dataset                                               # Dataset used
        self.chromosome = {"NL" : NL, "NF" : NF, "lr" : lr, "mom" : mom}        # Hyper-parameters of the network
        self.layers     = nn.ModuleList()                                       # Array of layers
        self.inaccuracy = 0.0                                                   # Inaccuracy during the test phase
        self.time       = 0.0                                                   # Computation time during the test phase
        self.fitness    = 0.0                                                   # Fitness score of the model
        
        if dataset == "MNIST" :
            in_channels = 1    # Input dimension
            
        elif dataset == "CIFAR10" :
            in_channels = 3    # Input dimension
          
        else :
            raise ValueError("Invalid dataset name. Either choose 'MNIST' or 'CIFAR10'")
            
         
        
        # Determine the feature maps subsequence from {32, 64, 128, 256, 512}
        self.feat_maps_seq = [32, 64, 128, 256, 512]
        start = randint(0, len(self.feat_maps_seq) - self.chromosome["NF"])
        end = start + NF
        self.feat_maps_seq = self.feat_maps_seq[start : end]
        ind_feat_maps = 0
        
        classes    = 10            # Number of classes
        img_h      = 32            # Input image height (MNIST images are resized to 32x32)
        img_w      = 32            # Input image width  (MNIST images are resized to 32x32)
        
        # Add the layers
        while NL > 0 :
            
            # Convolutional block cannot be of size 4 if there are 7 layers left because it would make the 
            # last block to be of size 1 and it is forbidden
            if NL == 7 :
                conv_block_size = choice([2, 3])
                
            # Convolutional blocks cannot be of size 3 or 4 if there are 6 or 3 layers left because it would
            # make the last block to be of size 1 and it is forbidden
            elif NL == 6 or NL == 3 :
                conv_block_size = 2
                
            # # Convolutional block cannot be of size 2 or 3 if there are 5 layers left because it would make
            # the last block to be of size 1 and it is forbidden
            elif NL == 5 :
                conv_block_size = 4
               
            # Convolutional block cannot be of size 2 or 4 if there are 4 mayers left because it would make 
            # the last block to be of size 1 and it is forbidden
            elif NL == 4 :
                conv_block_size = 3
            
            # Can be whatever size
            # (We cannot have 3, 2 or 1 layer(s) left with the preceding cases)
            else :
                conv_block_size = choice([2, 3, 4])
                 
            # Select the feature maps size
            feat_maps = self.feat_maps_seq[ind_feat_maps] if ind_feat_maps < len(self.feat_maps_seq) \
                                                            else self.feat_maps_seq[len(self.feat_maps_seq) - 1]
            ind_feat_maps += 1
            
            # Add a convolutional block
            for i in range(conv_block_size) :
                # If it is the first layer
                if NL == self.chromosome["NL"] :
                    self.layers.append(
                            nn.Conv2d(in_channels=in_channels,
                                      out_channels=feat_maps,
                                      kernel_size= 3,
                                      stride=1))
                
                else :
                    # If the previous layer is a BatchNorm2d layer
                    if type(self.layers[len(self.layers) - 3]).__name__ == "Conv2d" :
                        self.layers.append(
                                nn.Conv2d(in_channels=self.layers[len(self.layers) - 3].out_channels,   # out_channels in the previous convolutional layer
                                          out_channels=feat_maps,
                                          kernel_size=3,
                                          stride=1))
                        
                    else :  # If the previous layer is a MaxPool2d layer
                        self.layers.append(
                                nn.Conv2d(in_channels=self.layers[len(self.layers) - 4].out_channels,   # out_channels in the previous convolutional layer
                                          out_channels=feat_maps,
                                          kernel_size=3,
                                          stride=1))
                    # end if
                # end if
                
                # Update the height and width
                img_h = (img_h - self.layers[len(self.layers) - 1].kernel_size[0]) + 1      # Not divided by the stride because it is 1
                img_w = (img_w - self.layers[len(self.layers) - 1].kernel_size[0]) + 1      # Not divided by the stride because it is 1
                
                # Add a batch normalization layer
                self.layers.append(nn.BatchNorm2d(feat_maps))      # Number of features in the  previous convolutional layer
                
                # Add a ReLU activation
                self.layers.append(nn.ReLU())
                
                NL -= 1     # Decrements the number of layers left
                
            # End for i in range(conv_block_size)
            
            # Add the max pooling layer after the convolutional block
            self.layers.append(nn.MaxPool2d(kernel_size= 2,
                                            stride=1))
            
            # Update the height and width
            img_h = (img_h - self.layers[len(self.layers) - 1].kernel_size) + 1     # Not divided by the stride because it is 1
            img_w = (img_w - self.layers[len(self.layers) - 1].kernel_size) + 1     # Not divided by the stride because it is 1
            
            NL -= 1         # Decrements the number of layers left      
        # end while NL > 0
            
        # Add an average pooling layer
        self.layers.append(nn.AvgPool2d(kernel_size=2,
                                        stride=1))  
        
        # Update the height and width
        img_h = (img_h - self.layers[len(self.layers) - 1].kernel_size) + 1      # Not divided by the stride because it is 1
        img_w = (img_w - self.layers[len(self.layers) - 1].kernel_size) + 1      # Not divided by the stride because it is 1
        
        # Add a fully connected layer
        self.layers.append(
                nn.Linear(in_features=self.layers[len(self.layers) - 5].out_channels * img_h * img_w,
                          out_features=classes,
                          bias=True))
     # end __init__()


    # Forward propagate
    def forward(self, x) :
        for i in range(len(self.layers) - 1) :  # Stop before the fully connected layer
            x = self.layers[i](x)
        # end for i
        
        # Apply to the fully connected layer
        x = x.view(x.size()[0], -1)
        x = self.layers[len(self.layers)-1](x)
        
        return x
    # end forward()      
    
    # Print the CNN
    def printCNN(self) :
        """
        \Description : Print the CNN's info
        \Args : None
        \Outputs : None
        """
        print("CNN")
        print("Number of hidden layers : {}".format(self.chromosome["NL"]))
        print("Feature maps : {}".format(self.feat_maps_seq))
        print("Learning rate : {}".format(self.chromosome["lr"]))
        print("Momentum : {}".format(self.chromosome["mom"]))
        
        print("Architecture :")
        for i in range(len(self.layers)) :
            print(type(self.layers[i]).__name__)
            
            if type(self.layers[i]).__name__ == "Conv2d" :
                print("\tin_channels = {}".format(self.layers[i].in_channels))
                print("\tout_channels = {}".format(self.layers[i].out_channels))
                print("\tkernel_size = {}".format(self.layers[i].kernel_size))
                print("\tstride = {}".format(self.layers[i].stride))
                
            elif type(self.layers[i]).__name__ == "MaxPool2d" or type(self.layers[i]).__name__ == "AvgPool2d" :
                print("\tkernel_size = {}".format(self.layers[i].kernel_size))
                print("\tstride = {}".format(self.layers[i].stride))

            elif type(self.layers[i]).__name__ == "Linear" :
                print("\tin_features = {}".format(self.layers[i].in_features))
                print("\tout_features = {}".format(self.layers[i].out_features))
    # end printCNN
                
                
    # Function to train the model
    # Inspired from : https://www.kaggle.com/vincentman0403/pytorch-v0-3-1b-on-mnist-by-lenet (consulted on 07/03/2020)
    def train_model(self, optimizer, epoch, train_loader, log_interval) :
        """
        \Description : Train the model
        \Args : 
            optimizer : optimizer during the training
            epoch : number of epochs
            train_loader : loader of the train batch
            log_interval : interval to print training status
        \Output : None
        """
        # State that the model is being trained
        self.train()
    
        # Define the loss function
        loss_func = torch.nn.CrossEntropyLoss()
    
        # Iterate over batches of data
        for batch, (data, target) in enumerate(train_loader) :
            
            # Convert data to be used on GPU
            if torch.cuda.is_available() :
                data = data.cuda()  
                target = target.cuda()
            
            # Wrap the input and target output in the 'Variable' wrapper
            data, target = Variable(data), Variable(target)
    
            # Clear the gradients, since PyTorch accumulates them
            optimizer.zero_grad()
    
            # Forward propagation
            output = self(data)
    
            loss = loss_func(output, target)
    
            # Backward propagation
            loss.backward()
    
            # Update the parameters (weights, bias)
            optimizer.step()
    
            # Print log
            if batch % log_interval == 0 :
                print('Train set, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch * len(data), len(train_loader.dataset),
                        100. * batch / len(train_loader),
                        loss.data))
    # end train_model()


    # Function to test the model
    # Inspired from : https://www.kaggle.com/vincentman0403/pytorch-v0-3-1b-on-mnist-by-lenet (consulted on 07/03/2020)
    def test_model(self, epoch, test_loader):
        """
        \Description : Test the model
        \Args : 
            epoch : number of epochs
            test_loader : loader of the test batch
        \Output : 
            Accuracy of the model
        """
        # State that you are testing the model
        self.eval()
    
        # Init loss & correct prediction accumulators
        test_loss = 0
        correct = 0
    
        # Define loss function
        loss_func = torch.nn.CrossEntropyLoss(reduction="sum")
    
        # Measure starting time
        #start = perf_counter()
        start = clock()
        
        # Iterate over data
        for data, target in test_loader:
            
            # Convert data to be used on GPU
            if torch.cuda.is_available() :
                data = data.cuda()  
                target = target.cuda()
            
            # Wrap the input and target output in the 'Variable' wrapper
            data, target = Variable(data), Variable(target)
            
            # Forward propagation
            output = self(data)
    
            # Calculate & accumulate loss
            test_loss += loss_func(output, target).data
    
            # Get the index of the max log-probability (the predicted output label)
            pred = torch.argmax(output.data, dim=1)
    
            # If correct, increment correct prediction accumulator
            correct = correct + torch.eq(pred, target.data).sum()
    
        # Measure ending time
        #end = perf_counter()
        end = clock()
        
        test_loss /= len(test_loader.dataset)
        self.inaccuracy = 100 - (100. * correct / len(test_loader.dataset))     # Inaccuracy of the model
        self.time = end - start                                                 # Time elapsed for the test
        
        # Print log
        print('\nTest set, Epoch {} , Average loss: {:.4f}, Inaccuracy: {}/{} ({:.2f}%) in {:.3f}s\n'.format(epoch,
              test_loss, 
              len(test_loader.dataset) - correct, 
              len(test_loader.dataset),
              self.inaccuracy,
              self.time))   
    # end test_model()


    # Function to evaluate the model
    # Inspired from : https://www.kaggle.com/vincentman0403/pytorch-v0-3-1b-on-mnist-by-lenet (consulted on 07/03/2020)
    def evaluate_model(self, train_loader, test_loader, epochs=10, train_batch_size=64, test_batch_size=1000) :
        """
        \Description : Evaluate the model
        \Args : 
            model : model to test
            epoch : number of epochs
            test_loader : loader of the test batch
        \Output : 
            Accuracy of the model
        """      
        
        # Optimizer for training phase
        optimizer = optim.SGD(self.parameters(), 
                              lr=self.chromosome["lr"], 
                              momentum=self.chromosome["mom"])
        
        # Evaluation of the individual
        log_interval = 100
        for epoch in range(1, epochs + 1) :
            # Training phase
            self.train_model(optimizer, epoch, train_loader, log_interval=log_interval)
            
            # Testing phase
            self.test_model(epoch, test_loader)
        # end for epoch
    # end evaluate_model()
    
# end class CNN
        
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    