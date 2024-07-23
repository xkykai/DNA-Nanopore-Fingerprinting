#%%
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import os
import pywt
torch.manual_seed(123)

device = torch.device("cpu")

# Hyperparameters
learning_rate = 3e-4
epochs = 200
batch_size = 64

hidden_layer_size = 64
N_hidden_layer = 1
activation = nn.ReLU()

# Read in data
df_1 = pd.read_excel("Data/origami1/output_2500_events.xlsx", header=None)
df_1 = df_1.dropna(how="all")
origami1_data = np.array(df_1)

df_2 = pd.read_excel("Data/origami2/output_2_2500_events.xlsx", header=None)
df_2 = df_2.dropna(how="all")
origami2_data = np.array(df_2)

for i in range(len(origami1_data)):
    origami1_data[i,:] = np.nan_to_num(origami1_data[i, :], nan=origami1_data[i,1])

for i in range(len(origami2_data)):
    origami2_data[i,:] = np.nan_to_num(origami2_data[i, :], nan=origami2_data[i,1])

nrows = len(origami1_data) + len(origami2_data)
ntimes = np.amax([np.shape(origami1_data)[1], np.shape(origami2_data)[1]])
nclasses = 2

species = np.zeros((nrows, 2), bool)
for i in range(len(origami1_data)):
    species[i, 0] = 1

for i in range(len(origami2_data)):
    species[i+len(origami1_data), 1] = 1

# Normalize data
X = torch.tensor(np.concatenate((origami1_data, origami2_data))).float()
X = (X - torch.mean(X)) / torch.std(X)
X = torch.nan_to_num(X, nan=0)
X = torch.reshape(X, (X.shape[0], 1, X.shape[1]))

Y = torch.tensor(species).float()

# Split data into training, validation, and test sets (70/20/10) with stratification
X_train, X_not_train, Y_train, Y_not_train = train_test_split(X, Y, train_size=0.7, random_state=100, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_not_train, Y_not_train, train_size=2/3, random_state=100, stratify=Y_not_train)

ds_train = torch.utils.data.TensorDataset(X_train, Y_train)
ds_val = torch.utils.data.TensorDataset(X_val, Y_val)
ds_test = torch.utils.data.TensorDataset(X_test, Y_test)

trainloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(ds_val,batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(ds_test,batch_size=batch_size, shuffle=True)

class NN(nn.Module):
    '''
    Class for neural net.
    '''
    def __init__(self, input_dim, output_dim, hidden_layer_size=100, N_hidden_layers=2, activation=nn.GELU()):
        '''
        Parameters
        ----------
        input_dim: int
            input dimension (i.e., # of features in each example passed to the network)
        hidden_dim: int
            number of nodes in hidden layer
        output_dim: int
            output dimension (i.e., # of classes)
        N_hidden_layers: int
            number of hidden layers
        activation: torch.nn activation function
            activation function to use in hidden layers
        '''
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Conv1d(1, 8, 3, padding=1),
            activation,
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 5),
            activation,
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5),
            activation,
            nn.Flatten(),
            nn.Linear(128, hidden_layer_size),
            *[nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), activation) for _ in range(N_hidden_layers-1)],
            nn.Linear(hidden_layer_size, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 5 batches
        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Preallocate arrays for predictions and labels
    all_preds = torch.zeros(size, dtype=torch.long)
    all_labels = torch.zeros(size, dtype=torch.long)

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        start_idx = 0
        for X, y in dataloader:
            pred = model(X)  # Model predictions
            test_loss += loss_fn(pred, y).item()  # Accumulate test loss
            correct += (pred.argmax(-1) == y.argmax(-1)).type(torch.float).sum().item()  # Accumulate correct predictions

            # Calculate batch size
            batch_size = X.size(0)
            end_idx = start_idx + batch_size

            # Fill in the preallocated arrays with predictions and labels
            all_preds[start_idx:end_idx] = pred.argmax(-1).cpu()
            all_labels[start_idx:end_idx] = y.argmax(-1).cpu()

            start_idx = end_idx

    test_loss /= num_batches  # Average test loss
    correct /= size  # Accuracy

    # Calculate F1 score
    f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='weighted')

    return test_loss, 100 * correct, f1  # Return average loss, accuracy percentage, and F1 score

def build_confusion_matrix(model, dataloader):
    y_pred_train = []
    y_true_train = []

    # iterate over data
    for inputs, labels in dataloader:
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred_train.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true_train.extend(labels) # Save Truth

    y_true_train = [np.argmax(y) for y in y_true_train]

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true_train, y_pred_train)

    return cf_matrix


def train_model(hidden_layer_size, N_hidden_layers, activation):
    model = NN(ntimes, nclasses, hidden_layer_size, N_hidden_layers, activation)
    activation_str = str(activation).split("(")[0]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    train_accuracy = np.zeros(epochs)
    val_accuracy = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    train_f1 = np.zeros(epochs)
    val_f1 = np.zeros(epochs)
    test_f1 = np.zeros(epochs)

    WEIGHTS_DIR = f"./NN_weights/SI_1DCNN/{activation_str}_{hidden_layer_size}_{N_hidden_layer}"
    OUTPUT_DIR = f"./Output/SI_1DCNN"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    min_loss = np.inf
    max_accuracy = 0
    max_f1 = 0

    min_loss_state = model.state_dict()
    max_accuracy_state = model.state_dict()
    max_f1_state = model.state_dict()

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(trainloader, model, loss_fn, optimizer)
        train_loss[t], train_accuracy[t], train_f1[t] = test_loop(trainloader, model, loss_fn)
        val_loss[t], val_accuracy[t], val_f1[t] = test_loop(valloader, model, loss_fn)
        test_loss[t], test_accuracy[t], test_f1[t] = test_loop(testloader, model, loss_fn)

        print(f"Train Error: \n Accuracy: {(train_accuracy[t]):>0.1f}%, Avg loss: {train_loss[t]:>8f}, f1: {train_f1[t]:>8f} \n")
        print(f"Validation Error: \n Accuracy: {(val_accuracy[t]):>0.1f}%, Avg loss: {val_loss[t]:>8f}, f1: {val_f1[t]:>8f} \n")
        print(f"Test Error: \n Accuracy: {(test_accuracy[t]):>0.1f}%, Avg loss: {test_loss[t]:>8f}, f1: {test_f1[t]:>8f} \n")

        if val_loss[t] < min_loss:
            min_loss = val_loss[t]
            min_loss_state = model.state_dict()
            torch.save(model.state_dict(), f'{WEIGHTS_DIR}/model_weights_min_loss.pth')
        if val_accuracy[t] > max_accuracy:
            max_accuracy = val_accuracy[t]
            max_accuracy_state = model.state_dict()
            torch.save(model.state_dict(), f'{WEIGHTS_DIR}/model_weights_max_accuracy.pth')
        if val_f1[t] > max_f1:
            max_f1 = val_f1[t]
            max_f1_state = model.state_dict()
            torch.save(model.state_dict(), f'{WEIGHTS_DIR}/model_weights_max_f1.pth')

    min_loss = np.amin(val_loss)
    epoch_min_loss = np.argmin(val_loss)
    max_accuracy = np.amax(val_accuracy)
    epoch_max_accuracy = np.argmax(val_accuracy)
    max_f1 = np.amax(val_f1)
    epoch_max_f1 = np.argmax(val_f1)

    print("Done!")

    # Save plots
    plt.figure()
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label=f"Validation loss")
    plt.plot(test_loss, label="Test loss")
    plt.title(f"Hidden layer size: {hidden_layer_size}, {N_hidden_layers} hidden layers, activation function: {activation_str} \n Min loss: {min_loss:.4f} at epoch {epoch_min_loss}")
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_loss.png", dpi=1000)

    plt.figure()
    plt.plot(train_accuracy, label="Training accuracy")
    plt.plot(val_accuracy, label=f"Validation accuracy")
    plt.plot(test_accuracy, label="Test accuracy")
    plt.title(f"Hidden layer size: {hidden_layer_size}, {N_hidden_layers} hidden layers, activation function: {activation_str} \n Max accuracy: {max_accuracy:.2f}% at epoch {epoch_max_accuracy}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_accuracy.png", dpi=1000)

    plt.figure()
    plt.plot(train_f1, label="Training f1")
    plt.plot(val_f1, label=f"Validation f1")
    plt.plot(test_f1, label="Test f1")
    plt.title(f"Hidden layer size: {hidden_layer_size}, {N_hidden_layers} hidden layers, activation function: {activation_str} \n Max f1: {max_f1:.4f} at epoch {epoch_max_f1}")
    plt.xlabel("Epochs")
    plt.ylabel("f1")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_f1.png", dpi=1000)

    # Load best model
    model.load_state_dict(torch.load(f'{WEIGHTS_DIR}/model_weights_max_f1.pth'))
    model.eval()

    # constant for classes
    classes = ("Species 1", "Species 2")

    # Build confusion matrix for training data
    cf_matrix_train = build_confusion_matrix(model, trainloader)
    df_cm_train = pd.DataFrame(cf_matrix_train / np.sum(cf_matrix_train, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure()
    ax = sn.heatmap(df_cm_train, annot=True, cbar_kws={"label":"Probability"}, square=True)
    ax.set_title("Training")
    plt.savefig(f'{OUTPUT_DIR}/{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_training_confusion.png', dpi=1000)

    # Build confusion matrix for validation data
    cf_matrix_val = build_confusion_matrix(model, valloader)
    df_cm_val = pd.DataFrame(cf_matrix_val / np.sum(cf_matrix_val, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure()
    ax = sn.heatmap(df_cm_val, annot=True, cbar_kws={"label":"Probability"}, square=True)
    ax.set_title("Validation")
    plt.savefig(f'{OUTPUT_DIR}/{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_validation_confusion.png', dpi=1000)

    # Build confusion matrix for testing data
    cf_matrix_test = build_confusion_matrix(model, testloader)
    df_cm_test = pd.DataFrame(cf_matrix_test / np.sum(cf_matrix_test, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure()
    ax = sn.heatmap(df_cm_test, annot=True, cbar_kws={"label":"Probability"}, square=True)
    ax.set_title("Testing")
    plt.savefig(f'{OUTPUT_DIR}/{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_test_confusion.png', dpi=1000)

print(f"Training model with hidden layer size {hidden_layer_size}, {N_hidden_layer} hidden layers, and activation function {str(activation).split('(')[0]}")
train_model(hidden_layer_size, N_hidden_layer, activation)

# %%
