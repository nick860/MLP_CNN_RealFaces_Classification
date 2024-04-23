import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from helpers import *

def implict_representation(train_data, val_data, test_data):
    alpha = np.arange(0.1, 1.01, 0.1)
    x_train = train_data[['long', 'lat']].values
    x_val = val_data[['long', 'lat']].values
    x_test = test_data[['long', 'lat']].values

    x_with_sin_train = np.empty((x_train.shape[0], 0))
    x_with_sin_val = np.empty((x_val.shape[0], 0))
    x_with_sin_test = np.empty((x_test.shape[0], 0))

    for alpha in alpha:
        x_with_sin_train = np.concatenate((x_with_sin_train, np.sin(alpha * x_train)), axis=1)
        x_with_sin_val = np.concatenate((x_with_sin_val, np.sin(alpha * x_val)), axis=1)
        x_with_sin_test = np.concatenate((x_with_sin_test, np.sin(alpha * x_test)), axis=1)

    return torch.tensor(x_with_sin_train).float(), torch.tensor(x_with_sin_val).float(), torch.tensor(x_with_sin_test).float()

def train_model(train_data, val_data, test_data, model, lr, epochs, batch_size,do_clip_grad_norm_=False, implict = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    if implict:
        x,y,z = implict_representation(train_data, val_data, test_data)
        trainset = torch.utils.data.TensorDataset(x, torch.tensor(train_data['country'].values).long())
        valset = torch.utils.data.TensorDataset(y, torch.tensor(val_data['country'].values).long())
        testset = torch.utils.data.TensorDataset(z, torch.tensor(test_data['country'].values).long())
    else:
        trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
        valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
        testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    different_ephocs = []
    the_layers = [0, 30, 60, 90, 95, 99] # for question 5
    max_grad_norm = 0.02
    clip = 1.5
    monitoring_gradients_ep = {layer: [] for layer in the_layers}
    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        dict_grads = {layer: [] for layer in the_layers}
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            #### YOUR CODE HERE ####

            # perform a training iteration
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            counter_layers = 0
            for _, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    if counter_layers in the_layers:
                        if do_clip_grad_norm_:
                            clip_grad_norm_(m.parameters(), clip)
                        grad_magnitude = torch.norm(m.weight.grad)**2 + torch.norm(m.bias.grad)**2
                        dict_grads[counter_layers].append(grad_magnitude.item())
                    counter_layers += 1
            optimizer.step()
            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()


        for key in dict_grads.keys():
            if len(dict_grads[key]) > 0:
                monitoring_gradients_ep[key].append(np.mean(dict_grads[key]))
        

        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        # check if there is 100 hidden layers

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:

                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))
         

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, monitoring_gradients_ep

if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    output_dim = len(train_data['country'].unique())
    model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model = nn.Sequential(*model) 
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses= train_model(train_data, val_data, test_data, model, lr=0.01, epochs=50, batch_size=256)
    plt.title('Validation Losses')
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs.')
    plt.legend()
    plt.show()

    plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=False)
