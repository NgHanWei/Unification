import os
import numpy as np
from utils.load_eeg import load_data_per_subject_eeg

class EEGDataLoader:
    def __init__(self, hr_file_path):
        self.hr_file_path = hr_file_path
        self.all_channels = [
            'Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8',
            'CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7','FC3','C1',
            'C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8',
            'F6','AF8','AF4','F2'
        ]
        self.channel_to_select = [
            'FC5', 'FC1', 'C3', 'CP5', 'CP1', 'CP6', 'CP2', 'Cz', 'C4','FC6', 'FC2', 'FC3', 'C1',
            'C5', 'CP3', 'CPz', 'CP4', 'C6', 'C2', 'FC4'
        ]
        self.new_channel_index = [i for cadd in self.channel_to_select for i, c in enumerate(self.all_channels) if c == cadd]
    
    def one_hot(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    
    def get_data_path(self):
        for path, folders, files in os.walk(self.hr_file_path):
            for folder in folders:
                for file in os.listdir(os.path.join(path, folder)):
                    if file.endswith(".vhdr"):
                        return os.path.join(self.hr_file_path, folder, file)
        return None
    
    def get_data_openclose(self, path, paradigm, phasetype):
        data, labels = load_data_per_subject_eeg(path, paradigm, phasetype)
        onehot_labels = self.one_hot(labels[0], 2)
        return data[:len(data), self.new_channel_index, :], np.argmax(onehot_labels[:len(onehot_labels)], axis=1)
    
    def load_data(self):
        data_path = self.get_data_path()
        if data_path is None:
            raise FileNotFoundError("No .vhdr file found in the specified directory.")
        
        phasetype = "motiv_2" if "Phase2" in self.hr_file_path else "motiv"
        X_train_all, y_train_onehot_all = self.get_data_openclose(data_path, "", phasetype)
        
        X_train_segments, X_train_next = self.apply_sliding_window(X_train_all)
        
        return X_train_segments, X_train_next, y_train_onehot_all
    
    def apply_sliding_window(self, data, window_size=500, step_size=10, pred_length=50):
        trials, channels, timepoints = data.shape
        segments = []
        next_timepoints = []
        
        for start in range(0, timepoints - window_size - pred_length, step_size):
            segment = data[:, :, start:start + window_size]
            next_timepoint = data[:, :, start + window_size:start + window_size + pred_length]
            segments.append(segment)
            next_timepoints.append(next_timepoint)
        
        return np.array(segments).reshape(-1, channels, window_size), np.array(next_timepoints).reshape(-1, channels, pred_length)
# Example usage:
# loader = EEGDataLoader("/home/hanwei/Music/MBCI006")
# X_train_all, y_train_onehot_all = loader.load_data()
