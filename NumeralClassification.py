
#----------------------------------------------------------------------------------
#
#   Author:     William Bourn
#   File:       NumeralClassification.py
#   Version:    1.00
#
#   Description:
#   Training library for a Convolution Neural Network that identifies handdrawn 
#   single digit numbers.
#      
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
#   Includes
#----------------------------------------------------------------------------------

from pathlib import Path

import csv

#Numpy Library is used for manipulating data vectors and arrays
import numpy as np

#Pyplot Library is used for plotting learning characteristics
import matplotlib.pyplot as plt

#Sklearn is used for resampling and scaling data sets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

#Torch is used in the designing and implementing of a Convolutional Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler

from PIL import Image

#Command line argument parser
from argparse import ArgumentParser

#----------------------------------------------------------------------------------
#   Command Line Argument Parser
#----------------------------------------------------------------------------------

parser = ArgumentParser(description="Numeral Classfication Convolution Network Training")
parser.add_argument('--filename', '-F',
                    type = str,
                    help="Pathname and filename of the data CSV file", 
                    required = True)
parser.add_argument('--max-samples', '-S',
                    type = int,
                    help="Number of samples to take from the data CSV file", 
                    required = False,
                    default = 10000)
parser.add_argument('--target-acc', '-A',
                    type = float,
                    help="Target accuracy to be achieved by the model", 
                    required = False,
                    default = 0.85)
parser.add_argument('--max-epochs', '-E',
                    type = int,
                    help="Maximum number of epochs the function is allowed to train for", 
                    required = False,
                    default = 50)
parser.add_argument('--test-size', '-T',
                    type = float,
                    help="Percentage of samples to be used in testing", 
                    required = False,
                    default = 0.5)

#Export parameters
args = parser.parse_args()                
                
#----------------------------------------------------------------------------------
#   Classes
#----------------------------------------------------------------------------------

#MSTAR and Dataloader implementation were taken from Machine Learning Lab 2
class MSTAR(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.transform = transform
        self.enc = LabelEncoder()
        targets = self.enc.fit_transform(targets.reshape(-1,))
        self.targets = torch.LongTensor(targets)
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = int(index.item())
            
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index])
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class Convolution_Net(nn.Module):
    '''
    Convolution Neural Network Topology
    '''

    def __init__(self):
        '''
        Constructor Function
        '''
        super(Convolution_Net, self).__init__()
  
        #First Convolution:     12 feature maps, 5x5 kernel, 28x28 -> 24x24
        #                       Max pool reduction 24x24 -> 12x12
        self.first_convolution = nn.Conv2d(1,12,5)
        
        #Second Convolution:    30 feature maps, 5x5 kernel, 12x12 -> 8x8
        #                       Max pool reduction 8x8 -> 4x4
        self.second_convolution = nn.Conv2d(12,30,5)
        
        #Max pool reduction
        self.max_pool_reduction = nn.MaxPool2d(2,2)
        
        #First layer:           480 features -> 128 features
        self.first_layer = nn.Linear(480, 128)

        #Second layer:          128 featrures -> 64 features
        self.second_layer = nn.Linear(128, 64)

        #Third layer:           64 features -> 10 features      
        self.third_layer = nn.Linear(64, 10)

    def forward(self, features):
        '''
        Forwards the sample through the Convolution Neural Network to get the predicted output

        @param features:        The input features of the sample
        @type features:         np.array(float) of shape (28, 28)

        @return prediction:     The predicted output of the sample
        @rtype prediction:      np.array(float) of shape (10)

        '''
        #First Convolution
        first_convolution = self.max_pool_reduction(F.relu(self.first_convolution(features)))
        
        #Second Convolution
        second_convolution = self.max_pool_reduction(F.relu(self.second_convolution(first_convolution)))
        
        #Reshape sixteen 4x4 feature maps into 256 feature array
        first_linear = second_convolution.view(second_convolution.size(0), -1)
        
        #First Hidden Layer
        second_linear = F.relu(self.first_layer(first_linear))
        
        #Second Hidden Layer
        third_linear = F.relu(self.second_layer(second_linear))
        
        #Third Hidden Layer
        prediction = self.third_layer(third_linear)
        
        #Return predicted output
        return prediction


#----------------------------------------------------------------------------------
#   Functions
#----------------------------------------------------------------------------------

def Fetch_Data_From_CSV(filename, max_samples):
    '''
    Fetches the contents of a given CSV file and attempts to extract a set of input
    features and corresponding ground truth labels for training and testing a
    Convolutional Neural Network Model.

    @param filename:        The filepath and filename of the dataset CSV file. Each
                            row in the file corresponds to a sample, where the first
                            item in the row is the ground truth label and 784 items
                            correspond to the pixels of a 28x28 image flattened by 
                            layng rows end-on-end
    @type filename:         String

    @param max_samples:     The maximum number of samples to retrieve from the dataset
    @type max_samples:      int

    @return feature_set:    The set of input features for the parsed samples. Each 
                            feature in a sample corresponds to a single pixel in
                            a 28x28 image, where each pixel is greyscale and represented
                            by a value from 0 to 255
    @rtype feature_set:     np.array(int) of shape (samples,28,28)

    @return label_set:      The set of ground truth labels for the parsed samples
    @rtype label_set:       np.array(int) of shape (samples)

    '''
    
    print("Parsing CSV file: " + filename)

    #Create output data arrays
    feature_set = []
    label_set = []
    
    #Parse CSV file line-by-line
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)

        sample = 0
        #Iterate through the lines of the CSV file
        for row in csv_file:
            #Stop parsing lines if sufficient samples have been taken
            if(sample >= max_samples + 1):
                break
            #Ignore the first line
            if(sample == 0):
                sample += 1
                continue
            else:
                #Split the line into individual array elements
                raw_data = (row.rstrip()).split(',')

                #Convert String inputs into integer values in range 0-255
                data = []
                for datum in raw_data:
                    data.append(int(datum))

                #Split data into feature and ground truth arrays
                y = data[0]
                x = data[1:785]

                #Convert feature array to numpy and reshape into 28x28 pixel matrix
                x = np.array(x) 
                x = x.reshape(28,28)

                #Add sample to sample set
                feature_set.append(x)
                label_set.append(y)
                sample += 1
    
    print("Parsing Finished")

    #Return feature set and ground truth label set 
    return np.array(feature_set), np.array(label_set)      


def Holdout(feature_set, label_set, test_size):
    '''
    Create an independent training and data set via the Holdout method

    @param feature_set:         The set of input features for the samples
    @type feature_set:          np.array(int) of shape (samples, 28, 28)

    @param label_set:           The set of ground truths for the samples
    @type label_set:            np.array(int) of shape (samples)

    @param test_size:           The portion of the samples that should be placed in the
                                test set
    @type test_size:            float

    @return feature_set_train:  The set of input features for the training set
    @rtype feature_set_train:   np.array(int) of shape (samples, 28, 28)

    @return feature_set_test:   The set of input features for the test set
    @rtype feature_set_test:    np.array(int) of shape (samples, 28, 28)

    @return label_set_train:    The set of ground truths for the training set
    @rtype label_set_train:     np.array(int) of shape (samples)

    @return label_set_test:     The set of ground truths for the test set
    @rtype label_set_test:      np.array(int) of shape (samples)
    '''

    #Divide the feature and label sets
    feature_set_train, feature_set_test, label_set_train, label_set_test = train_test_split(feature_set, label_set, test_size = test_size)
    
    #Convert training and test sets to numpy arrays and return
    return np.array(feature_set_train), np.array(feature_set_test), np.array(label_set_train), np.array(label_set_test)


def Scale_Data(feature_set_train, feature_set_test):
    '''
    Scale the set of input features for the training and test data to a value between 0.0 and 1.0

    @param feature_set_train:       The set of input features for the training data
    @type feature_set_train         np.array(int) of shape (samples, 28, 28)
    
    @param feature_set_test:        The set of input features for the test data
    @type feature_set_test:         np.array(int) of shape (samples, 28, 28)

    @return feature_set_train:      The scaled set of input features for training data
    @rtype feature_set_train:       np.array(float) of shape (samples, 28, 28)

    @return feature_set_test:       The scaled set of input features for test data
    @rtype feature_set_test:        np.array(float) of shape (samples, 28, 28) 
    '''

    #Create a scaler for scaling values in the set from 0 to 255 to 0.0 to 1.0
    scaler = MinMaxScaler((0, 1))

    #Scale and transform training data
    feature_set_train_shape = feature_set_train.shape
    feature_set_train = scaler.fit_transform(feature_set_train.reshape(-1, 1))
    feature_set_train = feature_set_train.reshape(feature_set_train_shape)

    #Scale test data
    feature_set_test_shape = feature_set_test.shape
    feature_set_test = scaler.transform(feature_set_test.reshape(-1, 1))
    feature_set_test = feature_set_test.reshape(feature_set_test_shape)
    
    #Return scaled feature sets
    return feature_set_train, feature_set_test


def Create_Data_Sets(feature_set_train, feature_set_test, label_set_train, label_set_test):
    '''
    Create data sets for the training and test data

    @param feature_set_train:   The set of input features for the training set
    @type feature_set_train:    np.array(float) of shape (samples, 28, 28)

    @param feature_set_test:    The set of input features for the test set
    @type feature_set_test:     np.array(float) of shape (samples, 28, 28)

    @param label_set_train:     The set of ground truths for the training set
    @type label_set_train:      np.array(int) of shape (samples)

    @param label_set_test:      The set of ground truths for the test set
    @type label_set_test:       np.array(int) of shape (samples)  

    @return data_set_train:     The data set for the training data
    @rtype data_set_train:      MSTAR

    @return data_set_test:      The data set for the test data
    @rtype data_set_test:       MSTAR
    '''

    #Use the default transforms
    transform = transforms.Compose([transforms.ToTensor()])
    
    #Create the data sets
    data_set_train = MSTAR(feature_set_train, label_set_train, transform = transform)
    data_set_test = MSTAR(feature_set_test, label_set_test, transform = transform)

    #Return data sets
    return data_set_train, data_set_test


#MSTAR and Dataloader implementation were taken from Machine Learning Lab 2
def Setup_Dataloaders(data_set_train, data_set_test):
    '''
    Create dataloaders for the training and test data sets

    @param data_set_train:      The data set for the training data
    @type data_set_train:       MSTAR

    @param data_set_test:       The data set for the test data
    @type data_set_test:        MSTAR

    @return dataloader_train:   The dataloader for the training data set
    @rtype dataloader_train:    DataLoader

    @return dataloader_test:    The dataloader for the test data set
    @rtype dataloader_test:     DataLoader
    '''

    #Create data loaders
    dataloader_train = DataLoader(data_set_train, batch_size=len(data_set_train))
    dataloader_test = DataLoader(data_set_test, batch_size=len(data_set_test))
    
    #Return Dataloader
    return dataloader_train, dataloader_test


def Evaluate(model, dataloader_test):
    '''
    Perform an evaluation on the Convolution Neural Network using a given test dataloader

    @param model:               The Convolution Neural Network that is being evaluated
    @type model:                Convolution_Net

    @param dataloader_test:     The dataloader for the test data set
    @type dataloader_test:      DataLoader

    @return mean_loss:          The mean value of the loss function for the test samples
    @rtype mean_loss:           float
    
    @return mean_acc:           The mean value of the accuracy for the test samples
    @rtype mean_acc:            float
    '''
    #Prepare evaluation
    model.eval()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        #Iterate through samples in dataloarder
        for sample in dataloader_test:
            #Extract features and labels from sample
            features, labels = sample

            #Perform forward pass of features and determine the predicted output
            prediction = model(features)
            _, predicted = torch.max(prediction.data, 1)

            #Add the loss function value to the running value
            loss += criterion(prediction, labels).item()

            #Add the total size of the label and the correct number of predictions in
            #label to running value
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    #Calculate mean loss and accuracy
    mean_loss = loss / len(dataloader_test)
    mean_acc = correct / total
        
    # Return loss and accuracy
    return mean_loss, mean_acc


def Train_Epoch(model, feature_set, label_set, test_size):
    '''
    Trains and evaluates the model for a single epoch

    @param model:           The Convolution Neural Neural Network that is being trained
    @type model:            Convolution_Net

    @param feature_set:     The set of input features for the data
    @type feature_set:      np.array(float) of shape (samples, 28, 28)

    @param label_set:       The set of ground truth labels for the data
    @type label_set:        np.array(int) of shape (samples)

    @param test_size:       The portion of the samples that should be placed in the
                            test set
    @type test_size:        float

    @return loss:           The mean value of the loss function for the test samples
    @rtype loss:            float

    @return acc:            The mean value of the accuracy for the test samples
    @rtype acc:             float
    '''
    
    #Perform Holdout to obtain training and test data
    feature_set_train, feature_set_test, label_set_train, label_set_test = Holdout(feature_set, label_set, test_size)

    #Scale the feature sets
    feature_set_train, feature_set_test = Scale_Data(feature_set_train, feature_set_test)
    
    #Create data sets
    data_set_train, data_set_test = Create_Data_Sets(feature_set_train, feature_set_test, label_set_train, label_set_test)

    #Create dataloaders
    dataloader_train, dataloader_test = Setup_Dataloaders(data_set_train, data_set_test)

    #Train model on training dataloader
    for i, sample in enumerate(dataloader_train, 0):
        model.train()
        features, labels = sample
        optimizer.zero_grad()

        #Forward pass
        prediction = model(features)

        #Calculate loss function value
        loss = criterion(prediction, labels)

        #Back propagation
        loss.backward()
        optimizer.step()
    
    #Perform evaluation and return evaluation results
    loss, acc = Evaluate(model, dataloader_test)
    return loss, acc


def Train(model, target_acc, max_epochs, feature_set, label_set, test_size = 0.5):
    '''
    Trains and evaluates the model until the target accuracy is achieved

    @param model:           The Convolution Neural Neural Network that is being trained
    @type model:            Convolution_Net

    @param target_acc:      The target accuracy of the model. The model will be trained until it
                            achieves a mean target accuracy over five epochs
    @type target_acc:       float

    @param max_epochs:      The maximum number of epochs the model will train for before returning
    @type max_epochs:       int

    @param feature_set:     The set of input features for the data
    @type feature_set:      np.array(float) of shape (samples, 28, 28)

    @param label_set:       The set of ground truth labels for the data
    @type label_set:        np.array(int) of shape (samples)

    @param test_size:       The portion of the samples that should be placed in the
                            test set
    @type test_size:        float

    @return history:        Dictionary containing the historical accuracy and loss function values of
                            the model training
    @rtype history:         Dict{'loss':list, 'acc':list}
    '''

    #Create a historical record of loss function and accuracy
    history = {'loss':[], 'acc':[]}

    #Get the initial loss and accuracy values
    _, feature_set_test = Scale_Data(feature_set, feature_set)
    _, data_set_test = Create_Data_Sets(feature_set_test, feature_set_test, label_set, label_set)
    _, dataloader_test = Setup_Dataloaders(data_set_test, data_set_test)
    start_loss, start_acc = Evaluate(model, dataloader_test)
    
    history['loss'].append(start_loss)
    history['acc'].append(start_acc)

    #Iterate until mean target accuracy is reached
    mean_acc = start_acc
    epoch = 1
    while((mean_acc <= target_acc) | (len(history['acc']) < 5)):
        
        if(epoch == (max_epochs + 1)):
            break

        #Train model
        epoch_loss, epoch_acc = Train_Epoch(model, feature_set, label_set, test_size)

        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)

        #Determine new moving average accuracy
        if(len(history['acc']) >= 5):
            mean_acc = np.sum((history['acc'])[len(history['acc']) - 5:len(history['acc'])])/5
        else:
            mean_acc = np.sum(history['acc'])/len(history['acc'])
        
        print("Epoch %d" %epoch)
        print("Accuracy: %f" %epoch_acc)
        print("Loss: %f" %epoch_loss)

        epoch += 1
    
    return history


#----------------------------------------------------------------------------------
#   Main Function Call
#----------------------------------------------------------------------------------

if __name__ == "__main__":

    #Fetch data from 
    features, labels = Fetch_Data_From_CSV(args.filename, args.max_samples)   

    model = Convolution_Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = Train(model, args.target_acc, args.max_epochs, features, labels, args.test_size)

    #Plot the historical accuracy in logarithmic scale
    figure = plt.figure(figsize=(8,8))

    plt.plot(history['acc'], label = 'Historical Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xlim([0,len(history['acc'])])
    plt.ylim([0,1.0])

    