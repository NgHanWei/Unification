import mne
from utils.eventinfo_dual import *
from utils.preprocess import preprocess_data

def load_data_per_subject_eeg(path, paradigm):

    subject_path = path

    raw = mne.io.read_raw_brainvision(subject_path)
    # raw = mne.io.read_raw_eeglab(subject_path, preload=True)

    eeg, times = raw[:]  # full data
    # print(eeg.shape)
    event = mne.events_from_annotations(raw)  # event list
    fsamp = int(raw.info['sfreq'])

    ## 50 subjects corrected/raw
    ## 'hr' extype eventinfo.py change to close and open readings only
    extype = 'hr'
    ## XIMING
    extype = 'motiv'

    if extype == 'hr':
        ev = eventlist_HRmi(event, fsamp)  # eventlist_HRrest1
        taskdur = 4  # task duration in seconds
        # evrest = eventlist_HRrest(event,fsamp)
        # evrest1 = eventlist_HRrest1(event,fsamp)
    elif extype == 'roc':
        ev = eventlist_ROC(event, fsamp, paradigm)  # eventlist_ROC
        taskdur = 4  # task duration in seconds
    elif extype == 'motiv':
        ev = eventlist_grazmi(event, fsamp)
        taskdur = 4  # task duration in seconds
    elif extype == 'cirg':
        ev = eventlist_cirgmi(event, fsamp)
        taskdur = 4  # task duration in seconds

    rawdata = dict({'eeg': [], 'event': []})
    rawdata['eeg'] = eeg
    rawdata['event'] = event

    sampdur = int(fsamp * taskdur)
    # prepdur = int(fsamp * 0)

    data = dict({'X': [], 'y': [], 'yclass': []})
    # print(ev)
    for i, t in enumerate(ev['code']):
        if (ev['label'][i] != 'non'):
            data['X'].append(eeg[:, ev['sampstart'][i]:ev['sampstart'][i] + sampdur])
            # data['X'].append(eeg[:,ev['sampstart'][i]-prepdur :ev['sampstart'][i]+ sampdur])
            data['y'].append(t)
            data['yclass'].append(ev['label'][i])

    # print(len(data['X']))
    # for i in range(1,len(data['X'])):
    #     print(data['X'][i].shape)
    # print(len(data['y']))
    data['X'] = np.stack((data['X'][:200]),axis=0)

    data['s'] = fsamp
    data['c'] = raw.info['ch_names']
    data['rawfile'] = subject_path  # 2 is close, 1 is open, 0 is rest

    X_Data = np.array(data['X'])
    Y_Data = [np.array(data['y'])[:200]]

    count = 0
    for data_point in Y_Data[0]:
        # print(data_point)
        if data_point == 2:
            Y_Data[0][count] = 1
        count += 1

    ### No preprocessing
    # X_Data = np.delete(X_Data, (21), axis=1)
    # X_Data = np.delete(X_Data, (10), axis=1)
    # X_Data = X_Data[:,:,::4]
    # X_Data_array = X_Data

    ### Preprocess
    preprocess_args = {
        'sampling_frequency': 1000,
        'decimate_factor': 4,
        'chan_set': 'all',
        'plot_flag': False,
        'verbose': False
    }
    preprocess_instance = preprocess_data(args=preprocess_args)

    # print(X_Data.shape)

    X_Data_array = []
    for i in range(0, len(X_Data)):
        X_Data_new = preprocess_instance.process(X_Data[i])
        X_Data_new = X_Data_new[np.newaxis, :, :]
        if len(X_Data_array) == 0:
            X_Data_array = X_Data_new
        else:
            X_Data_array = np.concatenate((X_Data_array, X_Data_new), axis=0)
        # print(X_Data_array.shape)

    return X_Data_array, Y_Data

def load_data_per_subject_mat_fbcnet(path):
        import os
        import numpy as np
        from scipy.io import loadmat

        #file = 'CDsub' + str(sub) + '.mat'
        #file = 'preprocessedNRMAT' + '.mat'
        # file = 'CORRECTEDRESTOPENHR61CH' + str(sub) + '.mat'# testHRsub%resttaskHR61CH_9Bandsub  resttaskHR44CH newRESTOPENHR60CHLOWbands

        # file = 'RESTOPENHR61CH_PREPROCESSED_NTR' + str(sub) + '.mat'
        # file = 'CORRECTEDRESTCLOSEHR61CH' + str(sub) + '.mat'

        # print(file)
        # file = 'CORRECTEDOPENCLOSEHR61CH' + str(sub) + '.mat'
        #restOPENHR60CH9bands WORKED restOPENHR60CH9bands newRESTOPENHR60CHLOWbands
        # if sub < 9:restOPENHR60CH9bands0 restOPENHR60CHLOWbands
        # file = 'sub' + '0'+ str(sub) + '.mat'
        # if sub > 9:
        # file = 'sub'+ str(sub) + '.mat'    
        # subject_path = os.path.join(self.arg.get('data_dir'), file)        

        # print('loading data from subject ' + str(sub+1) + ' ..')

        data = loadmat(path)['data']

        # print(data[0][0][0].shape)
        # print(data[0][0][1])

        data_sub = data[0][0][0]
        data_label = data[0][0][1]

        data_sub = np.moveaxis(data_sub,2,0)

        return data_sub,data_label