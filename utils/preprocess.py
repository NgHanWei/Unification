import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy import io
class preprocess_data:
    def __init__(self, args):
        self.sampling_frequency = args['sampling_frequency']
        self.decimate_factor = args['decimate_factor']
        self.chan_set = args['chan_set']
        self.plot_flag = args['plot_flag']
        self.verbose = args['verbose']

    def filter_data(self, data):
        fs = self.sampling_frequency
        low_cut_hz, low_step_hz, high_cut_hz, high_step_hz, Apass, Astop = 40, 45, 0.3, 0.5, 3, 40

        nyq_freq = 0.5 * fs
        low = low_cut_hz / nyq_freq
        low_step = low_step_hz / nyq_freq
        high = high_cut_hz / nyq_freq
        high_step = high_step_hz / nyq_freq

        # log.info('\nlow pass filtering ... %d Hz (check plot)', low_cut_hz)

        if self.verbose: print('\nlow pass filtering ...', low_cut_hz, 'Hz (check plot)')
        N, Wn = signal.cheb2ord(low, low_step, Apass, Astop)
        sos_low = signal.cheby2(N, Astop, Wn, output='sos')
        filt_data_1 = signal.sosfiltfilt(sos_low, data, axis=-1)

        # log.info('\nhigh pass filtering ... %d Hz (check plot)', high_cut_hz)
        if self.verbose: print('\nhigh pass filtering ...', high_cut_hz, 'Hz (check plot)')
        N, Wn = signal.cheb2ord(high_step, high, Apass, Astop)
        sos_high = signal.cheby2(N, Astop, Wn, btype='high', output='sos')
        filt_data_2 = signal.sosfiltfilt(sos_high, filt_data_1, axis=-1)

        if self.plot_flag:
            plt.figure()
            w, h = signal.sosfreqz(sos_low)
            plt.subplot(2, 1, 1).plot(w * fs / 2 / np.pi, 20 * np.log10(abs(h)))
            w, h = signal.sosfreqz(sos_high)
            plt.subplot(2, 1, 2).plot(w * fs / 2 / np.pi,
                                      20 * np.log10(np.maximum(np.abs(h), 1e-5)))  # to avoid trigger warning
            plt.title('Chebyshev Type II frequency responses')
            plt.show()

            plt.figure()
            plt.subplot(3, 1, 1).plot(data[8, :])
            plt.subplot(3, 1, 2).plot(filt_data_1[8, :])
            plt.subplot(3, 1, 3).plot(filt_data_2[8, :])
            plt.title('Filtering data')
            plt.show()

        return filt_data_2

    def decimate_data(self, data):
        deci_data = signal.decimate(data, self.decimate_factor, axis=-1, zero_phase=True)  # set true for filtfilt

        # log.info('\ndownsampling factor ... %d (check plot)', self.decimate_factor)
        if self.verbose: print('\ndownsampling factor ...', self.decimate_factor, ' (check plot)')

        time_index = np.asarray(range(data.shape[-1])) / self.sampling_frequency

        if self.plot_flag:
            plt.figure()
            plt.subplot(2, 1, 1).plot(time_index, data[8, :])
            plt.subplot(2, 1, 2).plot(signal.decimate(time_index, self.decimate_factor), deci_data[0, :])
            plt.title('Downsampling data')
            plt.show()

        return deci_data

    def car_data(self, data):
        # n_chan = data.shape[1]
        # car_data = data + np.tile((-1 / (1 + n_chan)) * np.sum(data, axis=1)[:, None, :], (1, n_chan, 1))
        # Calculate the mean across channels for each time point
        mean_across_channels = np.mean(data, axis=0)
        # Subtract the mean from each channel's data
        car_data = data - mean_across_channels
        # log.info('\nre-referencing ... car (check plot)')
        if self.verbose: print('\nre-referencing ...', 'car (check plot)')

        if self.plot_flag:
            plt.figure()
            plt.subplot(2, 1, 1).plot(data[8, :])
            plt.subplot(2, 1, 2).plot(car_data[8, :])
            plt.title('Re-referenced data')
            plt.show()

        return car_data

    def chansel_data(self, data, mat_chan_set):
        full_chan_set = [str(c).replace(" ", "") for c in mat_chan_set]

        # full_chan_set=[]
        # [full_chan_set.append(str(c[0][0])) for c in mat_chan_set]

        if self.chan_set == 'all':
            channel_to_select = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1',
                                 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7',
                                 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                                 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4',
                                 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5',
                                 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                                 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6',
                                 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4',
                                 'FT8', 'F6', 'AF8', 'AF4', 'F2']
        if self.chan_set == 'SMA20':
            channel_to_select = ['FC5', 'FC1', 'C3',
                                 'CP5', 'CP1', 'CP6', 'CP2', 'Cz',
                                 'C4', 'FC6', 'FC2', 'FC3', 'C1',
                                 'C5', 'CP3', 'CPz', 'CP4', 'C6',
                                 'C2', 'FC4']
        new_channel_index = [i for i, c in enumerate(full_chan_set) for cadd in channel_to_select if c == cadd]

        # log.info('\nselected channels ... %s', [full_chan_set[c] for c in new_channel_index])
        if self.verbose:  print('\nselected channels ...', [full_chan_set[c] for c in new_channel_index])

        return data[new_channel_index,:]

    def normalize_data(self, data):

        # log.info('\nsample normalization ... ')
        if self.verbose:  print('\nsample normalization ...')

        mean = np.tile(np.mean(data, axis=-1)[:, :, None], (1, 1, data.shape[-1]))
        std = np.tile(np.std(data, axis=-1)[:, :, None], (1, 1, data.shape[-1]))

        return (data - mean) / (std)

    def split_shift(self, data, label=None):
        # log.info('\ndata segmentation ... %d  (samples) %d (shift) ', self.seg_len, self.seg_shift)
        if self.verbose:  print('\ndata segmentation ...', self.seg_len, ' (samples) ', self.seg_shift, ' (shift) ')
        data = data[:, None, :, :]  # add dimension for segments
        segdata = []
        for i in range(0, data.shape[-1] - self.seg_len + 1, int(self.seg_shift)):
            segdata.append(data[:, :, :, i:i + self.seg_len])

        return np.hstack(segdata), np.tile(label, (1, len(segdata)))

    def save_data(self, data, label):
        save_preproc_data = dict({'x': [], 'y': []})
        save_preproc_data['x'] = data
        save_preproc_data['y'] = label
        io.savemat(self.save_data_path, save_preproc_data)
        # log.info('\nsaving pre-processed data in ... %s', self.save_data_path)
        if self.verbose:  print('\nsaving pre-processed data in ... ', self.save_data_path)

    def process(self, data):
        # load data
        # filter
        # print('start preprocessing')
        mat_chan_set = [
            'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'VEOG', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1',
            'Oz',
            'O2',
            'P4', 'P8', 'HEOG', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
            'AFz',
            'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6',
            'P2',
            'CPz',
            'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2'
        ]
        data = data * 1e6 #convert to uV
        data = self.filter_data(data)

        # decimate
        data = self.decimate_data(data)
        # update sampling frequency

        # car
        data = self.car_data(data)

        # channel selection
        data = self.chansel_data(data, mat_chan_set)

        return data

""" 
Copyright (C) 2022 King Saud University, Saudi Arabia 
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Hamdi Altaheri 
"""

# Dataset BCI Competition IV-2a is available at 
# http://bnci-horizon-2020.eu/database/data-sets

import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# We need the following function to load and preprocess the High Gamma Dataset
# from preprocess_HGD import load_HGD_data

#%%
def load_data_LOSO (data_path, subject, dataset): 
    """ Loading and Dividing of the data set based on the 
    'Leave One Subject Out' (LOSO) evaluation approach. 
    LOSO is used for  Subject-independent evaluation.
    In LOSO, the model is trained and evaluated by several folds, equal to the 
    number of subjects, and for each fold, one subject is used for evaluation
    and the others for training. The LOSO evaluation technique ensures that 
    separate subjects (not visible in the training data) are usedto evaluate 
    the model.
    
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available at 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9/14]
            Here, the subject data is used  test the model and other subjects data
            for training
    """
    
    X_train, y_train = [], []
    for sub in range (0,9):
        path = data_path+'s' + str(sub+1) + '/'
        
        if (dataset == 'BCI2a'):
            X1, y1 = load_BCI2a_data(path, sub+1, True)
            X2, y2 = load_BCI2a_data(path, sub+1, False)
        elif (dataset == 'CS2R'):
            X1, y1, _, _, _  = load_CS2R_data_v2(path, sub, True)
            X2, y2, _, _, _  = load_CS2R_data_v2(path, sub, False)
        # elif (dataset == 'HGD'):
        #     X1, y1 = load_HGD_data(path, sub+1, True)
        #     X2, y2 = load_HGD_data(path, sub+1, False)
        
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
                   
        if (sub == subject):
            X_test = X
            y_test = y
        elif (X_train == []):
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test


#%%
def load_BCI2a_data(data_path, subject, training, all_trials = True):
    """ Loading and Dividing of the data set based on the subject-specific 
    (subject-dependent) approach.
    In this approach, we used the same training and testing dataas the original
    competition, i.e., 288 x 9 trials in session 1 for training, 
    and 288 x 9 trials in session 2 for testing.  
   
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts 
    """
    
    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6*48     
    window_Length = 7*250 
    
    # Define MI trial window 
    fs = 250          # sampling rate
    t1 = int(1.5*fs)  # start time_point
    t2 = int(6*fs)    # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path+'A0'+str(subject)+'T.mat')
    else:
        a = sio.loadmat(data_path+'A0'+str(subject)+'E.mat')
    a_data = a['data']
    for ii in range(0,a_data.size):
        a_data1 = a_data[0,ii]
        a_data2= [a_data1[0,0]]
        a_data3= a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0,a_trial.size):
             if(a_artifacts[trial] != 0 and not all_trials):
                 continue
             data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
             class_return[NO_valid_trial] = int(a_y[trial])
             NO_valid_trial +=1        
    

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return-1).astype(int)

    return data_return, class_return



#%%
import json
from mne.io import read_raw_edf
from dateutil.parser import parse
import glob as glob
from datetime import datetime

def load_CS2R_data_v2(data_path, subject, training, 
                      classes_labels =  ['Fingers', 'Wrist','Elbow','Rest'], 
                      all_trials = True):
    """ Loading training/testing data for the CS2R motor imagery dataset
    for a specific subject        
   
        Parameters
        ----------
        data_path: string
            dataset path
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        classes_labels: tuple
            classes of motor imagery returned by the method (default: all) 
    """
    
    # Get all subjects files with .edf format.
    subjectFiles = glob.glob(data_path + 'S_*/')
    
    # Get all subjects numbers sorted without duplicates.
    subjectNo = list(dict.fromkeys(sorted([x[len(x)-4:len(x)-1] for x in subjectFiles])))
    # print(SubjectNo[subject].zfill(3))
    
    if training:  session = 1
    else:         session = 2
    
    num_runs = 5
    sfreq = 250 #250
    mi_duration = 4.5 #4.5

    data = np.zeros([num_runs*51, 32, int(mi_duration*sfreq)])
    classes = np.zeros(num_runs * 51)
    valid_trails = 0
    
    onset = np.zeros([num_runs, 51])
    duration = np.zeros([num_runs, 51])
    description = np.zeros([num_runs, 51])

    #Loop to the first 4 runs.
    CheckFiles = glob.glob(data_path + 'S_' + subjectNo[subject].zfill(3) + '/S' + str(session) + '/*.edf')
    if not CheckFiles:
        return 
    
    for runNo in range(num_runs): 
        valid_trails_in_run = 0
        #Get .edf and .json file for following subject and run.
        EDFfile = glob.glob(data_path + 'S_' + subjectNo[subject].zfill(3) + '/S' + str(session) + '/S_'+subjectNo[subject].zfill(3)+'_'+str(session)+str(runNo+1)+'*.edf')
        JSONfile = glob.glob(data_path + 'S_'+subjectNo[subject].zfill(3) + '/S'+ str(session) +'/S_'+subjectNo[subject].zfill(3)+'_'+str(session)+str(runNo+1)+'*.json')
    
        #Check if EDFfile list is empty
        if not EDFfile:
          continue
    
        # We use mne.read_raw_edf to read in the .edf EEG files
        raw = read_raw_edf(str(EDFfile[0]), preload=True, verbose=False)
        
        # Opening JSON file of the current RUN.
        f = open(JSONfile[0],) 
    
        # returns JSON object as a dictionary 
        JSON = json.load(f) 
    
        #Number of Keystrokes Markers
        keyStrokes = np.min([len(JSON['Markers']), 51]) #len(JSON['Markers']), to avoid extra markers by accident
        # MarkerStart = JSON['Markers'][0]['startDatetime']
           
        #Get Start time of marker
        date_string = EDFfile[0][-21:-4]
        datetime_format = "%d.%m.%y_%H.%M.%S"
        startRecordTime = datetime.strptime(date_string, datetime_format).astimezone()
    
        currentTrialNo = 0 # 1 = fingers, 2 = Wrist, 3 = Elbow, 4 = rest
        if(runNo == 4): 
            currentTrialNo = 4
    
        ch_names = raw.info['ch_names'][4:36]
             
        # filter the data 
        raw.filter(4., 50., fir_design='firwin')  
        
        raw = raw.copy().pick_channels(ch_names = ch_names)
        raw = raw.copy().resample(sfreq = sfreq)
        fs = raw.info['sfreq']

        for trail in range(keyStrokes):
            
            # class for current trial
            if(runNo == 4 ):               # In Run 5 all trials are 'reset'
                currentTrialNo = 4
            elif (currentTrialNo == 3):    # Set the class of current trial to 1 'Fingers'
                currentTrialNo = 1   
            else:                          # In Runs 1-4, 1st trial is 1 'Fingers', 2nd trial is 2 'Wrist', and 3rd trial is 'Elbow', and repeat ('Fingers', 'Wrist', 'Elbow', ..)
                currentTrialNo = currentTrialNo + 1
                
            trailDuration = 8
            
            trailTime = parse(JSON['Markers'][trail]['startDatetime'])
            trailStart = trailTime - startRecordTime
            trailStart = trailStart.seconds 
            start = trailStart + (6 - mi_duration)
            stop = trailStart + 6

            if (trail < keyStrokes-1):
                trailDuration = parse(JSON['Markers'][trail+1]['startDatetime']) - parse(JSON['Markers'][trail]['startDatetime'])
                trailDuration =  trailDuration.seconds + (trailDuration.microseconds/1000000)
                if (trailDuration < 7.5) or (trailDuration > 8.5):
                    print('In Session: {} - Run: {}, Trail no: {} is skipped due to short/long duration of: {:.2f}'.format(session, (runNo+1), (trail+1), trailDuration))
                    if (trailDuration > 14 and trailDuration < 18):
                        if (currentTrialNo == 3):   currentTrialNo = 1   
                        else:                       currentTrialNo = currentTrialNo + 1
                    continue
                
            elif (trail == keyStrokes-1):
                trailDuration = raw[0, int(trailStart*int(fs)):int((trailStart+8)*int(fs))][0].shape[1]/fs
                if (trailDuration < 7.8) :
                    print('In Session: {} - Run: {}, Trail no: {} is skipped due to short/long duration of: {:.2f}'.format(session, (runNo+1), (trail+1), trailDuration))
                    continue

            MITrail = raw[:32, int(start*int(fs)):int(stop*int(fs))][0]
            if (MITrail.shape[1] != data.shape[2]):
                print('Error in Session: {} - Run: {}, Trail no: {} due to the lost of data'.format(session, (runNo+1), (trail+1)))
                return
            
            # select some specific classes
            if ((('Fingers' in classes_labels) and (currentTrialNo==1)) or 
            (('Wrist' in classes_labels) and (currentTrialNo==2)) or 
            (('Elbow' in classes_labels) and (currentTrialNo==3)) or 
            (('Rest' in classes_labels) and (currentTrialNo==4))):
                data[valid_trails] = MITrail
                classes[valid_trails] =  currentTrialNo
                
                # For Annotations
                onset[runNo, valid_trails_in_run]  = start
                duration[runNo, valid_trails_in_run] = trailDuration - (6 - mi_duration)
                description[runNo, valid_trails_in_run] = currentTrialNo
                valid_trails += 1
                valid_trails_in_run += 1
                         
    data = data[0:valid_trails, :, :]
    classes = classes[0:valid_trails]
    classes = (classes-1).astype(int)

    return data, classes, onset, duration, description


#%%
def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test


#%%
def get_data(path, subject, dataset = 'BCI2a', classes_labels = 'all', LOSO = False, isStandard = True, isShuffle = True):
    
    # Load and split the dataset into training and testing 
    if LOSO:
        """ Loading and Dividing of the dataset based on the 
        'Leave One Subject Out' (LOSO) evaluation approach. """ 
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject, dataset)
    else:
        """ Loading and Dividing of the data set based on the subject-specific 
        (subject-dependent) approach.
        In this approach, we used the same training and testing data as the original
        competition, i.e., for BCI Competition IV-2a, 288 x 9 trials in session 1 
        for training, and 288 x 9 trials in session 2 for testing.  
        """
        if (dataset == 'BCI2a'):
            path = path + 's{:}/'.format(subject+1)
            X_train, y_train = load_BCI2a_data(path, subject+1, True)
            X_test, y_test = load_BCI2a_data(path, subject+1, False)
        elif (dataset == 'CS2R'):
            X_train, y_train, _, _, _ = load_CS2R_data_v2(path, subject, True, classes_labels)
            X_test, y_test, _, _, _ = load_CS2R_data_v2(path, subject, False, classes_labels)
        # elif (dataset == 'HGD'):
        #     X_train, y_train = load_HGD_data(path, subject+1, True)
        #     X_test, y_test = load_HGD_data(path, subject+1, False)
        else:
            raise Exception("'{}' dataset is not supported yet!".format(dataset))

    # shuffle the data 
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train,random_state=42)
        X_test, y_test = shuffle(X_test, y_test,random_state=42)

    # Prepare training data     
    N_tr, N_ch, T = X_train.shape 
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)
    # Prepare testing data 
    N_tr, N_ch, T = X_test.shape 
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)    
    
    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot
