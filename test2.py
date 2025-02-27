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
# from braindecode.datasets import BaseConcatDataset, WindowsDataset
from braindecode.datasets import create_from_X_y
mne.set_log_level("ERROR")

print("WORKING")

seed = 1104
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seeds(seed=seed, cuda=cuda)

## HELPER FUNCTIONS
def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

## Load MUSIC MI
def get_data_openclose(path, paradigm, phasetype):
    data,labels = load_data_per_subject_eeg(path, paradigm,phasetype)

    # data = data[:,np.newaxis,:,:]
    onehot_labels = one_hot(labels[0],2)

    data_valid = data[len(data):]
    onehot_labels_valid = onehot_labels[len(data):]

    data_train = data[:len(data)]
    onehot_labels_train = onehot_labels[:len(onehot_labels)]
    
    return data_train,onehot_labels_train,data_valid,onehot_labels_valid

## Obtain data path
def get_data_path(hr_file_path):
    for path, folders, files in os.walk(hr_file_path):

            for folder in folders:
                print(folder)
                for file in os.listdir(f"{path}/{folder}"):
                    if file.endswith(".vhdr"):
                        print(file)

                        data_path = os.path.join(hr_file_path, folder, file)
    return data_path

## Target subject for eval
hr_file_path = "/home/hanwei/Music/MBCI001"
data_path = get_data_path(hr_file_path)
X_train_all,y_train_onehot_all,X_test_all,y_test_all = get_data_openclose(data_path,"","motiv")

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
y_train_onehot_all = binary_labels = np.argmax(y_train_onehot_all, axis=1)
if "Phase2" in hr_file_path:
    X_train_all = X_train_all[2:]
    y_train_onehot_all = y_train_onehot_all[2:]

print("TARGET SUBJECT DATA")
print(X_train_all.shape)
print(y_train_onehot_all.shape)
## End Load datasets

## Subject Independent
def subj_independent(new_files):

    X_train = []
    y_train = []

    for new_file in new_files:
        eeg_path = get_data_path(new_file)
        X_train_new_2,y_train_onehot_new_2,X_test_all,y_test_all = get_data_openclose(eeg_path,"","motiv")
        y_train_onehot_new_2 = binary_labels = np.argmax(y_train_onehot_new_2, axis=1)
        X_train_new_2 = X_train_new_2[:,new_channel_index,:]

        if len(X_train) == 0:
            X_train = X_train_new_2
            y_train = y_train_onehot_new_2
        else:
            X_train = np.concatenate((X_train, X_train_new_2), axis=0)
            y_train = np.concatenate((y_train, y_train_onehot_new_2), axis=0)


    return X_train,y_train

def list_folders(directory):
    return [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

# getting folders of every subject
folder_path = "/home/hanwei/Music/"
folders = list_folders(folder_path)
if hr_file_path in folders:
    folders.remove(hr_file_path)
print(folders)

### Initialize model and dataset
n_classes = 2
classes = list(range(n_classes))
n_chans = 20
input_window_samples = 1000

if "Phase2" in hr_file_path:
    num_folds = 1
else:
    num_folds = 10
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=1104)
total_score = 0
score_array = np.zeros(num_folds)
# import tensorflow as tf
# tf.keras.utils.set_random_seed(1104)

for i_fold in range(0,num_folds):
    ## Split Data By Folds, for Phase 1 
    if "Phase2" in hr_file_path:
        print("Phase 2 Data Split")
        ## For Phase 2 Only
        X_train = X_train_all[:40]
        y_train_onehot = y_train_onehot_all[:40]
        X_test = X_train_all[40:]
        y_test = y_train_onehot_all[40:]

    else:
        y_index = list(kf.split(X_train_all, y_train_onehot_all))[i_fold][1]
        x_index = list(kf.split(X_train_all, y_train_onehot_all))[i_fold][0]
        X_train = X_train_all[x_index]
        y_train_onehot = y_train_onehot_all[x_index]
        X_test = X_train_all[y_index]
        y_test = y_train_onehot_all[y_index]
        print(X_train.shape)


    ## Selection for additional data concat
    ## Choose best folders
    print("SUBJ SELECTION")
    from select_subj import subj_selector
    selector = subj_selector(folders,X_train,y_train_onehot)
    # selected_folders = selector.compare_runs()
    selected_folders = ["/home/hanwei/Music/MBCI005","/home/hanwei/Music/P003","/home/hanwei/Music/P005"]
    if hr_file_path in selected_folders:
        selected_folders.remove(hr_file_path)
    print(selected_folders)
    print("DONE")
    ## Obtaining data to be concatenated
    if len(selected_folders) != 0:
        X_train_new, y_train_onehot_new = subj_independent(selected_folders)

        ## Concat additional data
        # X_train = np.concatenate((X_train, X_train_new), axis=0)
        # y_train_onehot = np.concatenate((y_train_onehot, y_train_onehot_new), axis=0)

    print(X_train.shape)
    print(y_train_onehot.shape)
    print(X_test.shape)
    print(y_test.shape)

    ## Create datasets
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

    # model = ShallowFBCSPNet(
    #     n_chans,
    #     n_classes,
    #     n_times=input_window_samples,
    #     n_filters_time=30,
    #     n_filters_spat=30,
    #     final_conv_length='auto',
    # )
    from braindecode.models import Deep4Net
    n_fake_chans = n_chans
    n_fake_targets = n_classes
    model = Deep4Net(
        n_chans=n_fake_chans,
        n_outputs=n_fake_targets,
        n_times=input_window_samples,
        n_filters_time=25,
        n_filters_spat=25,
        stride_before_pool=True,
        n_filters_2=n_fake_chans * 2,
        n_filters_3=n_fake_chans * 4,
        n_filters_4=n_fake_chans * 8,
        final_conv_length='auto',
    )


    from braindecode.models import ShallowFBCSPNet
    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        n_times=input_window_samples,
        n_filters_time=30,
        n_filters_spat=30,
        final_conv_length='auto',
    )

    # from braindecode.models import ATCNet
    # model = ATCNet(
    #     n_outputs=n_classes,
    #     n_chans=n_chans,
    #     input_window_seconds=4,
    #     n_times=input_window_samples,
    #     # conv_block_n_filters=4, 
    #     # conv_block_kernel_length_1=16,
    #     # final_fc_length='auto',
    # )
    # print(model)

    # from model2 import ShallowFBCSPNetTransformer
    # model = ShallowFBCSPNetTransformer(n_chans, n_classes, input_window_samples)

    # Display torchinfo table describing the model
    # print(model)

    # Send model to GPU
    if cuda:
        model = model.cuda()

    # We found these values to be good for the shallow network:
    lr = 0.0625 * 0.5 * 0.5
    weight_decay = 0.05

    batch_size = 32
    n_epochs = 150

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
    total_score += score
    score_array[i_fold] = score
    print(f"Model Score: {score}")

print(f"Final Score: {total_score/num_folds}")
print(score_array)