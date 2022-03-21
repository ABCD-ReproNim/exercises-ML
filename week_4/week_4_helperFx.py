
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch


def generateSpiral(n, num_spiral = 4, noise = 1.25):
    
    '''
    Function to generate some special spiral data
    
    Adapted From: https://github.com/tensorflow/playground/blob/master/src/dataset.ts
    
    Attributes:
        n          (int): Number of data points
        num_spiral (int): How swirly is the spiral
        noise      (float): How noisy should the spiral be
    
    Returns:
        X          (numpy.array): Data for training
        y          (numpy.array): Outcome
        df         (pd.DataFrame): Containing x, y, and class label
    '''
    
    labels = [0, 1] # Class label options
 
    # Create the empty dict
    d = {'x': [], 'y': [], 'class': []}
    
    # Loop through each class label and n/2 datapoints
    for ind, dT in enumerate([0, math.pi]):
        for ii in range(round(n/2)):
            r = ii/n * 5
            t = 1.75 * ii/n * num_spiral * math.pi + dT
            d['x'].append(r * math.sin(t) + np.random.uniform(-.25, .25) * noise)
            d['y'].append(r * math.cos(t) + np.random.uniform(-.25, .25) * noise)
            d['class'].append(labels[ind])

    # Create training data and outcome
    X = np.array(list(zip(d['x'], d['y'])))
    y = np.array(d['class'])
        
    return X, y, pd.DataFrame(d)


def modPlot(tr_x, tr_y, te_x, te_y, net, perf, cmap = 'PiYG', h = .01):
    
    '''
    Function to generate some special spiral data
    
    Adapted From: https://stackoverflow.com/questions/41138706/recreating-decision-boundary-plot-in-python-with-scikit-learn-and-matplotlib
    
    Attributes:
        tr_x     (np.array): Training data
        tr_y     (np.array): Training class labels
        te_x     (np.array): Test data
        te_y     (np.array): Test class labels
        net      (torch.Sequential): The trained neural network
        perf     (pd.DataFrame): Model performance
        cmap     (str): Colormap
        h        (float): Step size for meshgrid
    '''
    

    '''###############################################
    Plot performance
    ###############################################'''
    
    # Create the plot
    fig, axs = plt.subplots(1, 2, figsize = (15, 7)) 
    
    palette = {'train': 'tab:purple', 'test': 'tab:pink'}
    
    sns.lineplot(x = 'epoch', y = 'loss', data = perf, hue = 'type', 
                 palette = palette, linewidth = 5, ax = axs[0])
    axs[0].legend()
    axs[0].title.set_text('Loss')
    
    '''###############################################
    Create decision boundary
    ###############################################'''
    
        # create a mesh to plot in
    xMin = tr_x[:, 0].min() - 1 
    xMax = tr_x[:, 0].max() + 1
    yMin = tr_x[:, 1].min() - 1 
    yMax = tr_x[:, 1].max() + 1

    # Create meshgrid
    xx, yy = np.meshgrid(np.arange(xMin, xMax, h), 
                         np.arange(yMin, yMax, h))

    # Generate predictions for all points in our meshgrid and reshape
    Z = net(torch.tensor(np.c_[xx.ravel(), yy.ravel()]).to(torch.float))
    Z = Z.reshape(xx.shape)
    Z = Z.detach().numpy()
    
    
    '''###############################################
    Plot Decision boundary
    ###############################################'''
    
    # Generate meshgrid
    axs[1].pcolormesh(xx, yy, Z, cmap=cmap, alpha=.05, antialiased=True, shading='gouraud')

    # Plot also the training and test data
    axs[1].scatter(tr_x[:, 0], tr_x[:, 1], c=tr_y, cmap=cmap, edgecolor='r', alpha=.2, s=125, label='Train')
    axs[1].scatter(te_x[:, 0], te_x[:, 1], c=te_y, cmap=cmap, edgecolor='k', alpha=.8, s=125, label='Test')
    
    
    # Place limits on plot
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    axs[1].title.set_text('Decision Boundary')

    plt.legend()
    plt.suptitle('Model Evaluation') 
    plt.show()


def update_perf(perf, tr, te, e):
    
    # Store loss
    perf['loss'].append(np.mean(np.array(tr)))
    perf['loss'].append(np.mean(np.array(te)))
    
    # Store labels
    perf['type'].append('train')
    perf['type'].append('test')
    
    # Store epochs
    perf['epoch'].append(e + 1)
    perf['epoch'].append(e + 1)

    return perf