from utils.load_eeg import load_data_per_subject_eeg
import numpy as np
import os

## HELPER FUNCTIONS
def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

## Load MUSIC MI
def get_data_openclose(path, paradigm):
    data,labels = load_data_per_subject_eeg(path, paradigm)

    data = data[:,np.newaxis,:,:]
    onehot_labels = one_hot(labels[0],2)

    data_valid = data[len(data):]
    onehot_labels_valid = onehot_labels[len(data):]

    data_train = data[:len(data)]
    onehot_labels_train = onehot_labels[:len(onehot_labels)]
    
    return data_train,onehot_labels_train,data_valid,onehot_labels_valid

hr_file_path = "/home/hanwei/Music/P004"
paradigm = "Open"

for path, folders, files in os.walk(hr_file_path):

        for folder in folders:
            print(folder)
            for file in os.listdir(f"{path}/{folder}"):
                if file.endswith(".vhdr"):
                    print(file)

                    data_path = os.path.join(hr_file_path, folder, file)
X_train_all,y_train_onehot_all,X_test_all,y_test_all = get_data_openclose(data_path,paradigm)

all_channels = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8',
                            'CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7','FC3','C1',
                            'C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8',
                            'F6','AF8','AF4','F2']

channel_to_select = ['FC5', 'FC1', 'C3',
                                   'CP5', 'CP1', 'CP6', 'CP2', 'Cz',
                                   'C4','FC6', 'FC2', 'FC3', 'C1',
                                   'C5', 'CP3', 'CPz', 'CP4', 'C6',
                                   'C2', 'FC4']
new_channel_index =[i for cadd in channel_to_select for i,c in enumerate(all_channels) if c==cadd]
X_train_all = X_train_all[:,:,new_channel_index,:]

print(X_train_all.shape)
print(y_train_onehot_all.shape)
# print(y_train_onehot_all[:20])

## Load KU Dataset
print("CROSS DATASET AUGMENTATION")
import h5py
datapath = '/home/hanwei/pre-processed/KU_mi_smt.h5'
subj = 6

def get_cross_dataset(subj):
    dfile = h5py.File(datapath, 'r')
    dpath = 's' + str(subj)
    X = dfile[dpath]['X']
    Y = dfile[dpath]['Y']
    X = np.array(X)
    X = X[:,np.newaxis,:,:]
    Y = np.array(Y)
    Y = one_hot(Y,2)

    new_channel_index = [8,9,13,18,19,21,20,14,15,11,10,33,36,35,39,40,41,38,37,34]
    X = X[:,:,new_channel_index,:]
    Y = Y[:]
    return X,Y

X,Y = get_cross_dataset(subj)
print(X.shape)
print(Y.shape)
# print(Y[:20])

# all_channels = ['Fp1','Fz','F3','F7','FT9','FC5','FC1', 'FCz', 'C3','T7','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8',
            #                 'CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7','FC3','C1',
            #                 'C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8',
            #                 'F6','AF8','AF4','F2']

# [Fp1,Fp2,F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,TP9,CP5,CP1,CP2,CP6,TP10,P7,P3,Pz,P4,P6,PO9,O1,Oz,O2,PO10,FC3,FC4,C5,C1,C2,C6,CP3,CPz,CP4,P1,P2,POz,FT9,FTT9,TPP7,TP7,TPP9,FT10,FTT10,TPP8,TP8,TPP10,F9,F10,AF7,AF3,AF4,AF8,PO3,PO4]


## Testing

import torch
import torcheeg
from torcheeg.models import EEGNet
from torcheeg.transforms import Compose, ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulate EEG data (20 channels, 1000 time samples, 1000 samples)
num_channels = 20
num_time_samples = 1000
num_samples = 1000  # Number of EEG signal samples
eeg_data = np.random.randn(num_samples, num_channels, num_time_samples)  # Shape (num_samples, num_channels, num_time_samples)
labels = np.random.randint(0, 2, num_samples)  # Example binary classification labels (0 or 1)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(eeg_data, labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Custom dataset class for EEG data
class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels, transform=None):
        self.eeg_data = eeg_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        sample = self.eeg_data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# Define custom transforms for EEG processing
transform = Compose([ToTensor()])

# Create EEGDataset
train_dataset = EEGDataset(X_train_tensor, y_train_tensor, transform=transform)
test_dataset = EEGDataset(X_test_tensor, y_test_tensor, transform=transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model (EEGNet as an example)
model = EEGNet(num_electrodes=num_channels, num_classes=2)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, labels in train_loader:
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        
        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

# Evaluate the model on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test data: {100 * correct / total:.2f}%')