from braindecode.datasets import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

## Load datasets
from utils.load_eeg import load_data_per_subject_eeg
import numpy as np

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
## End Load datasets

from numpy import multiply

from braindecode.preprocessing import (Preprocessor,
                                       exponential_moving_standardize,
                                       preprocess)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV
factor = 1e6

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors, n_jobs=-1)

from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)

splitted = windows_dataset.split('session')
train_set = splitted['0train']  # Session train
valid_set = splitted['1test']  # Session evaluation

import torch

from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
# Set random seed to be able to roughly reproduce results
# Note that with cudnn benchmark set to True, GPU indeterminism
# may still make results substantially different between runs.
# To obtain more consistent results at the cost of increased computation time,
# you can set `cudnn_benchmark=False` in `set_random_seeds`
# or remove `torch.backends.cudnn.benchmark = True`
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

# Display torchinfo table describing the model
print(model)

# Send model to GPU
if cuda:
    model = model.cuda()

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier

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
    train_split=predefined_split(valid_set),  # using valid_set for validation
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
_ = clf.fit(train_set, y=None, epochs=n_epochs)