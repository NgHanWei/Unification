## Load datasets
from utils.load_eeg import load_data_per_subject_eeg
import numpy as np
import os
import torch
from sklearn.model_selection import KFold
from model import ShallowFBCSPNet
# from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
import mne
mne.set_log_level("ERROR")

## HELPER FUNCTIONS
def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

## Load MUSIC MI
def get_data_openclose(path, paradigm):
    data,labels = load_data_per_subject_eeg(path, paradigm)

    # data = data[:,np.newaxis,:,:]
    onehot_labels = one_hot(labels[0],2)

    data_valid = data[len(data):]
    onehot_labels_valid = onehot_labels[len(data):]

    data_train = data[:len(data)]
    onehot_labels_train = onehot_labels[:len(onehot_labels)]
    
    return data_train,onehot_labels_train,data_valid,onehot_labels_valid

hr_file_path = "/home/hanwei/Music/P003"
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
X_train_all = X_train_all[:,new_channel_index,:]

print(X_train_all.shape)
print(y_train_onehot_all.shape)
## End Load datasets

### Random dataset creation
n_classes = 2
classes = list(range(n_classes))
n_chans = 20
input_window_samples = 1000

# from braindecode.datasets import BaseConcatDataset, WindowsDataset
from braindecode.datautil import create_from_X_y

# Example data (n_trials, n_channels, n_times)
# Convert to a BaseConcatDataset
print(X_train_all.shape)
y_train_onehot_all = binary_labels = np.argmax(y_train_onehot_all, axis=1)
print(y_train_onehot_all.shape)

num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=1104)

for i_fold in range(0,10):
    y_index = list(kf.split(X_train_all, y_train_onehot_all))[i_fold][1]
    x_index = list(kf.split(X_train_all, y_train_onehot_all))[i_fold][0]
    X_train = X_train_all[x_index]
    y_train_onehot = y_train_onehot_all[x_index]
    X_test = X_train_all[y_index]
    y_test = y_train_onehot_all[y_index]
    # print(X_train.shape)

    dataset = create_from_X_y(
        X_train, y_train_onehot, sfreq=200,  # Sampling frequency, optional
        # window_size_samples=1125,  # Use full trial or windowed data
        drop_last_window=False
    )
    valid_dataset = create_from_X_y(
        X_test, y_test, sfreq=200,  # Sampling frequency, optional
        # window_size_samples=1125,  # Use full trial or windowed data
        drop_last_window=False
    )
    ## Random dataset creation

    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True

    seed = 1104
    set_random_seeds(seed=seed, cuda=cuda)

    import torch.nn.functional as F
    import torch.nn as nn
    # class CustomEEGNet(nn.Module):
    #     def __init__(self, n_channels, n_classes, input_size):
    #         super().__init__()
    #         self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 5), padding=(0, 2))  # Temporal convolution
    #         self.bn1 = nn.BatchNorm2d(16)
    #         self.conv2 = nn.Conv2d(16, 32, kernel_size=(n_channels, 1))  # Spatial convolution
    #         self.bn2 = nn.BatchNorm2d(32)
    #         self.fc1 = nn.Linear(32 * (input_size - 0), 64)  # Fully connected layer
    #         self.fc2 = nn.Linear(64, n_classes)  # Output layer

    #     def forward(self, x):
    #         x = x.unsqueeze(1)  # Add a channel dimension (B, C, H, W)
    #         x = F.relu(self.bn1(self.conv1(x)))
    #         x = F.relu(self.bn2(self.conv2(x)))
    #         x = x.view(x.shape[0], -1)  # Flatten
    #         x = F.relu(self.fc1(x))
    #         x = self.fc2(x)  # No activation (will use softmax in loss)
    #         print(x.shape)
    #         return x

    # model = CustomEEGNet(n_chans,
    #     n_classes,
    #     input_window_samples)
    
    # print(model)

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        n_times=input_window_samples,
        final_conv_length='auto',
    )

    # Display torchinfo table describing the model
    # print(model)

    # Send model to GPU
    if cuda:
        model = model.cuda()

    # We found these values to be good for the shallow network:
    lr = 0.0625 * 0.01
    weight_decay = 0

    # For deep4 they should be:
    # lr = 1 * 0.01
    # weight_decay = 0.5 * 0.001

    batch_size = 32
    n_epochs = 40

    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_dataset),  # using valid_set for validation
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
        classes=classes,
    )
    # Model training for the specified number of epochs. `y` is None as it is
    # already supplied in the dataset.

    _ = clf.fit(dataset, y=None, epochs=n_epochs)

    score = clf.score(valid_dataset,y=y_test)
    print(f"Model Score: {score}")