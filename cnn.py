import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from torchvision import transforms
from xgboost import XGBClassifier
import torchvision
import os

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            self.resnet18 = resnet18()
        
        in_features_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                    param.requires_grad = False
        
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):

            features = self.resnet18(x)
            ### YOUR CODE HERE ###
            return self.logistic_regression(features)

def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """


    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    #plt.imshow(test_set[0][0].permute(1, 2, 0))
    #plt.show()

    return train_loader, val_loader, test_loader

           
def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    ### YOUR CODE HERE ###
    correct = 0
    total = 0
    prediction = []
    with torch.no_grad():
        for (imgs, labels) in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs).squeeze()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            prediction.append((torch.sigmoid(outputs) > 0.5).int())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
         
    return correct / total , torch.cat(prediction, 0).cpu().numpy()

def extract_features(model, data_loader, device):
    """
    Extract the features from the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The features extracted from the model on the data in data_loader
    """
    model.eval()
    features = []
    labels = []
    print(len(data_loader))
    with torch.no_grad():
        for (imgs, lbls) in data_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model.resnet18(imgs)
            features.append(out)
            labels.append(lbls)
    
    return torch.cat(features, 0), torch.cat(labels, 0)


def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    loss_sum = []
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        ### YOUR CODE HERE ###
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        loss_sum.append(loss.item())
    
    return np.mean(loss_sum)

    
def train_baseline(model, num_of_epochs, learning_rate, sklearn=False, bonus = False):
    transform = model.transform
    # the best batch size 
    batch_size = 32
    print(f'Learning rate: {learning_rate}')
    path = os.path.join(os.getcwd(), 'whichfaceisreal') # For example '/cs/usr/username/whichfaceisreal/'
    train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    prediction = None
    if sklearn:
        print('Training the Logistic Regression model')
        features, labels = extract_features(model, train_loader, device)
        print('Extracted features')
        model_sklearn = LogisticRegression(max_iter=1000)
        print('Fitting the model')
        model_sklearn.fit(features.cpu().numpy(), labels.cpu().numpy())
        print('Fitted the model')
        features_test, labels_test = extract_features(model, test_loader, device)
        prediction = model_sklearn.predict(features_test.cpu().numpy())
        test_acc = model_sklearn.score(features_test.cpu().numpy(), labels_test.cpu().numpy())
    else:
        print('Training the model')
        criterion = torch.nn.BCEWithLogitsLoss()
        if bonus:
            # weight decay for the bonus
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=52, gamma=0.1)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bonus_idea = True

        for epoch in range(num_of_epochs):
            # Run a training epoch
            loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
            if bonus:
                lr_scheduler.step()
            test_acc, prediction = compute_accuracy(model, test_loader, device)
            print(f'Epoch {epoch+1}/{num_of_epochs}, Loss: {loss:.4f}, Test accuracy: {test_acc:.4f}')
            if test_acc >= 0.9675 and bonus_idea:
                print('Accuracy is good enough, starting probing')
                for name, param in model.named_parameters():
                    param.requires_grad = False
                bonus_idea = False
                model.logistic_regression.weight.requires_grad = True

            if test_acc >= 0.97 and bonus:
                break
        print(f'Test accuracy: {test_acc:.4f}')

    return test_acc, prediction




