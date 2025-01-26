import numpy as np
import scipy.signal as signal
import os
from scipy.io import loadmat
from scipy import io     
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from networks import *
# from deepCNN import *
from run_model import *
from util import *

class preprocess_data:
    def __init__(self,args):
        self.logfilename = args['log']
        self.data_path = args['data_path'] #os.path.join(args['data_path'], fname)     
        self.sampling_frequency = args['sampling_frequency']
        self.decimate_factor = args['decimate_factor']
        self.resolution = args['resolution']
        self.chan_set = args['chan_set']
        self.plot_flag = False
        self.unique_labels = args['labels']
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
        
        if self.verbose: print2(['\nlow pass filtering ...', low_cut_hz, 'Hz (check plot)'],self.logfilename)
        N, Wn = signal.cheb2ord(low, low_step, Apass, Astop)
        sos_low = signal.cheby2(N, Astop, Wn, output = 'sos')   
        filt_data_1 = signal.sosfiltfilt(sos_low, data, axis=-1)        
        
        # log.info('\nhigh pass filtering ... %d Hz (check plot)', high_cut_hz)
        if self.verbose: print2(['\nhigh pass filtering ...', high_cut_hz, 'Hz (check plot)'],self.logfilename)
        N, Wn = signal.cheb2ord(high_step, high, Apass, Astop)
        sos_high = signal.cheby2(N, Astop, Wn, btype = 'high', output = 'sos')         
        filt_data_2 = signal.sosfiltfilt(sos_high, filt_data_1, axis=-1)
        
        if self.plot_flag:
            plt.figure()
            w, h = signal.sosfreqz(sos_low)       
            plt.subplot(2,1,1).plot(w*fs/2/np.pi, 20 * np.log10(abs(h)))
            w, h = signal.sosfreqz(sos_high)
            plt.subplot(2,1,2).plot(w*fs/2/np.pi, 20*np.log10(np.maximum(np.abs(h), 1e-5))) #to avoid trigger warning
            plt.title('Chebyshev Type II frequency responses')       
            plt.show()  
              
            
            plt.figure()
            plt.subplot(3,1,1).plot(data[0,0])
            plt.subplot(3,1,2).plot(filt_data_1[0,0])
            plt.subplot(3,1,3).plot(filt_data_2[0,0])
            plt.title('Filtering data')    
            plt.show()  
        
        return filt_data_2
    
    def decimate_data(self, data):
        deci_data = signal.decimate(data, self.decimate_factor, axis=-1, zero_phase=True) #set true for filtfilt
        
        # log.info('\ndownsampling factor ... %d (check plot)', self.decimate_factor)
        if self.verbose: print2(['\ndownsampling factor ...', self.decimate_factor, ' (check plot)'],self.logfilename)
        
        time_index = np.asarray(range(data.shape[-1]))/self.sampling_frequency
        
        if self.plot_flag:
            plt.figure()
            plt.subplot(2,1,1).plot(time_index,data[0,0])
            plt.subplot(2,1,2).plot(signal.decimate(time_index,self.decimate_factor),deci_data[0,0])
            plt.title('Downsampling data')    
            plt.show()  
        
        return deci_data
    
    
    
    def car_data(self, data):
        n_chan = data.shape[1]      
        car_data = data + np.tile((-1/(1+n_chan))*np.sum(data,axis=1)[:,None,:],(1,n_chan,1))
        
        # log.info('\nre-referencing ... car (check plot)')
        if self.verbose: print2(['\nre-referencing ...', 'car (check plot)'],self.logfilename)
        
        if self.plot_flag:
            plt.figure()
            plt.subplot(2,1,1).plot(data[0,0])
            plt.subplot(2,1,2).plot(car_data[0,0])
            plt.title('Re-referenced data')    
            plt.show()  
        
        return car_data
    
    def chansel_data(self, data, mat_chan_set):
        full_chan_set = [str(c).replace(" ","") for c in mat_chan_set]
        
        # full_chan_set=[]
        for ic,c in enumerate(full_chan_set): # Rename following Distal UE-HR protocol for 4 subjects (SUB-21 ,22,23,24)
            if c=='Iz': full_chan_set[ic] = 'FT9' 
        
        if self.chan_set ==  'all':
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
                                   'C4','FC6', 'FC2', 'FC3', 'C1',
                                   'C5', 'CP3', 'CPz', 'CP4', 'C6',
                                   'C2', 'FC4'] 
        if self.chan_set == 'SMA32':
            channel_to_select = ['Fp1','Fz','F3','F7','FT7','FC5','FC1',
                                 'C3','T7','CP5','CP1','Pz','P3',
                                 'P7','O1','Oz','O2','P4','P8',
                                 'CP6','CP2','Cz','C4','T8',
                                 'FT8','FC6','FC2','F4','F8','Fp2']
        
        # new_channel_index =[i for i,c in enumerate(full_chan_set) for cadd in channel_to_select if c==cadd]
        new_channel_index =[i for cadd in channel_to_select for i,c in enumerate(full_chan_set) if c==cadd]
        
        # log.info('\nselected channels ... %s', [full_chan_set[c] for c in new_channel_index])
        if self.verbose:  print2(['\nselected channels ...', [full_chan_set[c] for c in new_channel_index]],self.logfilename)
        
        return data[:,new_channel_index]
        
        
    def normalize_data(self, data):
        
        # log.info('\nsample normalization ... ')
        if self.verbose:  print2('\nsample normalization ...',self.logfilename)
        
        mean = np.tile(np.mean(data, axis = -1)[:,:,None],(1,1,data.shape[-1]))
        std = np.tile(np.std(data, axis = -1)[:,:,None],(1,1,data.shape[-1]))
        
        return (data - mean)/(std)     
    
    def split_shift(self, data, label = None):
        # log.info('\ndata segmentation ... %d  (samples) %d (shift) ', self.seg_len, self.seg_shift)
        if self.verbose:  print2(['\ndata segmentation ...', self.seg_len ,' (samples) ', self.seg_shift, ' (shift) '],self.logfilename)
        data = data[:,None,:,:] # add dimension for segments
        segdata = []
        for i in range(0, data.shape[-1]-self.seg_len+1, int(self.seg_shift)): 
            segdata.append(data[:,:,:,i:i+self.seg_len])
        
        return np.hstack(segdata), np.tile(label,(1,len(segdata)))
    
    def save_data_mat(self,data, label):
        save_preproc_data = dict({'x': [],'y': []})
        save_preproc_data['x'] = data
        save_preproc_data['y'] = label
        io.savemat(self.save_data_path, save_preproc_data) 
        # log.info('\nsaving pre-processed data in ... %s', self.save_data_path)
        if self.verbose:  print2(['\nsaving pre-processed data in ... ', self.save_data_path],self.logfilename)
        if self.verbose:  print2(['\nsaving pre-processed data size ... ', data.shape],self.logfilename)
     
    def save_data_hdf(self,data, label):
        save_preproc_data = h5py.File(self.save_data_path, 'w')
        save_preproc_data['x'] = data
        save_preproc_data['y'] = label
        save_preproc_data.close()
        if self.verbose:  print2(['\nsaving pre-processed data in ... ', self.save_data_path],self.logfilename)
        if self.verbose:  print2(['\nsaving pre-processed data size ... ', data.shape],self.logfilename)
    
    def process(self, fname):
        # load data
        file_path = os.path.join(self.data_path, fname)
        # log.info('\nloading data from ... %s', file_path)
        
        if self.verbose:  print2(['\nloading data from ...', file_path],self.logfilename)
        if os.path.splitext(file_path)[1] =='.mat':
            data = loadmat(file_path)['x'] # N x Nchan x Ntime 
            label = loadmat(file_path)['y']
            mat_chan_set = loadmat(file_path)['c'] #load from header or wherever
        elif os.path.splitext(file_path)[1] =='.hdf':
            f = h5py.File(file_path, "r") 
            data = np.array(f['x']) # N x Nchan x Ntime 
            label = np.array(f['y'])
            mat_chan_set = [c.decode("ascii") for c in np.array(f['c'])] #np.array(f['c']
            f.close()
        
        # log.info('\nadjust resolution factor ... %d', self.resolution)
        if self.verbose:  print2(['\nadjust resolution factor ...', self.resolution],self.logfilename)
        # resolution adjust
        data = data * self.resolution
        
        # filter
        data = self.filter_data(data)   
        
        # decimate
        data = self.decimate_data(data)  
        # update sampling frequency
        
        # car
        data = self.car_data(data)
        
        # channel selection          
        data = self.chansel_data(data, mat_chan_set)
        
        # save data
        if os.path.splitext(file_path)[1] =='.mat':
            self.save_data_path = os.path.join(self.data_path, os.path.split(fname)[0], 'preproc-' + os.path.split(fname)[1]) 
            self.save_data_mat(data, label)
        elif os.path.splitext(file_path)[1] =='.hdf':
            self.save_data_path = os.path.join(self.data_path, 'preproc-' + fname)        
            self.save_data_hdf(data, label)
    
class model_training:
    def __init__(self,args):        
        self.method = args['method']
        self.device = 'cuda'
        self.data_augmentation = True
        self.need_voting = True
        
        self.logfilename = args['log']
        
        self.fsamp = args['sampling_frequency']/args['decimate_factor']
        self.seg_len = int(args['segment_length']*self.fsamp)
        self.seg_shift = int(args['segment_shift']*self.fsamp)
        
        
        self.model_name =  os.path.join(args['modelpath'], args['modelname']) #args['model_name']
        self.model_training_params = args['model_training_params']
        
        self.seg_per_sample = 1
        self.verbose = args['verbose']
        
        self.label_desc = args["label_description"]
        
        # if mode_key == 'training':
        #     self.model_path = args['model_path']
        #     self.model_name = os.path.join(self.model_path, 'end_epoch_model.pt') 
        # elif mode_key == 'evaluation':
            
    
    def prepare_data(self):
        # load data
        # log.info('\nloading data from ... %s', self.data_path)
        if self.verbose:  print2(['\nloading data from ...', self.data_path],self.logfilename)
        if os.path.splitext(self.data_path)[1] =='.mat':
            data = loadmat(self.data_path)['x'] # N x Nchan x Ntime 
            label = loadmat(self.data_path)['y']
        elif os.path.splitext(self.data_path)[1] =='.hdf':
            f = h5py.File(self.data_path, "r") 
            data = np.array(f['x']) # N x Nchan x Ntime 
            label = np.array(f['y'])
            f.close()
        
        # if self.verbose:  print2('\nloading data from ...', self.data_path)
        # data = loadmat(self.data_path)['x'] # Ntrials x Nchan x Ntime preprocessed data
        # label = loadmat(self.data_path)['y'] # Ntrials x 1
        
        # log.info('\ninput sample size ... %d', len(label))
        if self.verbose:   print2(['\ninput sample size ...', len(label)],self.logfilename)        
        
        if self.data_augmentation:
           data, label = preprocess_data.split_shift(self, data, label)
           self.seg_per_sample = data.shape[1]
           # stack data
           data = np.concatenate(data)
           label = np.concatenate(label)
        else:
           label = np.concatenate(label)
        # normalize for DL
        # data = preprocess_data.normalize_data(self, data)       #self.normalize_data(data)    
        
        # reshape and convert for DL
        data = data[:,None,:,:] # insert channel dimension
        if self.method == 'DeepCNNhbm':data = np.moveaxis(data,[1,2,3],[3,1,2])
        data = data.astype('float32') # memory saving :-o
        data = torch.from_numpy(data).float().to(self.device)   # samples x 1 x channels x time; samples = Ntrials x Nseg
        label = torch.from_numpy(label).long().to(self.device) 
        
        # log.info('\nfinal sample size ... %d', len(label))
        if self.verbose:  print2(['\nfinal sample size ...', len(label)],self.logfilename)
        
        return data, label
    
    def split_balanced(self,label,fold): #only works for binary
        # label = label.cpu().numpy()
        np.random.seed(0)        
        index0 = np.where(label[:,0]==np.unique(label)[0])[0]
        np.random.shuffle(index0)
        index1 = np.where(label[:,0]==np.unique(label)[1])[0]
        np.random.shuffle(index0)
        num_trial_class = np.min((len(index0),len(index1)))
        index=[]
        
        for j in range(fold):            
            itest = np.arange(j,num_trial_class,fold)
            itrain = np.delete(np.arange(num_trial_class),itest)
            ival = itrain[np.arange(0,len(itrain),fold-1)]
            itrain = np.delete(itrain,np.arange(0,len(itrain),fold-1))
            
            index.extend([[np.concatenate((index0[itest], index1[itest]))]+
                         [np.concatenate((index0[ival], index1[ival]))]+
                         [np.concatenate((index0[itrain], index1[itrain]))]])
        return index
    
    def train(self, data_path):     
        self.data_path = data_path
        train_data, train_label = self.prepare_data()   
        if self.verbose:  print2(['\ninput dimension ...', train_data.shape[-2:]],self.logfilename)  
        args = {    
             "input_shape": train_data.shape[-2:],
             "sampling_rate": self.fsamp, 
             "model": self.method,
             'verbose': self.verbose,
             'log':self.logfilename,
             'model_training_params':self.model_training_params}
        
        if self.method == 'DeepCNNhbm':args['input_shape']=train_data.shape[-3:-1]
        # training
        tm = run_model(args)
        # tm.get_result(train_data, train_label, train_data, train_label, train_data, train_label, 1, 1, 'ho', 'try')
        
        model = tm.train_only(train_data, train_label)        # returns model at the end of 200 epochs
        # save model
        torch.save(model, self.model_name)     
        io.savemat(self.model_name[:-3] + '.mat',self.label_desc)
        # evaluate trained model
        # prob, output, loss, acc = tm.eval_only(self.model_name, train_data, train_label)
        
    
    def eval(self, data_path):   
        self.data_path = data_path
        eval_data, eval_label = self.prepare_data()   
        if self.verbose:  print2(['\ninput dimension ...', eval_data.shape[-2:]],self.logfilename)  
        args = {    
             "input_shape": eval_data.shape[-2:],
             "sampling_rate": self.fsamp, 
             "model": self.method,
             'verbose': self.verbose,
             'log':self.logfilename,
             'model_training_params':self.model_training_params}
        if self.method == 'DeepCNNhbm':args['input_shape']=eval_data.shape[-3:-1]
        # evaluation
        tm = run_model(args)
        est_label_prob, output, loss, acc = tm.eval_only(self.model_name, eval_data, eval_label)  
        import statistics as stat
        if self.need_voting:
            est_label = output.argmax(1)
            
            est_label_trial = np.reshape(est_label,(-1,self.seg_per_sample))        
            
            trial_est_label_prob = np.mean(np.reshape(est_label_prob,(-1,self.seg_per_sample,2)),1)
            vote = 'hard'
            if vote == 'hard':
                vote_label = trial_est_label_prob.argmax(1)
            elif vote =='soft':
                vote_label = [stat.mode(d) for d in est_label_trial] #stat.mode(est_label)    
                
            true_label = eval_label.reshape((-1,self.seg_per_sample))[:,0].cpu().tolist()
        else:
            est_label = output.argmax(1)
            vote_label = est_label
            true_label = eval_label.cpu().tolist()
            trial_est_label_prob = est_label_prob
            
        correct = (vote_label == true_label).sum()
        acc = correct.item() / len(vote_label)
        
        # log.info('\nevaluation-voting acc: %.4f', acc)
        if self.need_voting: print2(['\nevaluation-voting acc: {:.4f}'.format(acc)],self.logfilename)
        else: print2(['\nevaluation acc: {:.4f}'.format(acc)],self.logfilename)        
        
        return np.expand_dims(vote_label,1), acc, self.label_desc_decode(vote_label), trial_est_label_prob

    def label_desc_decode(self, label):
        oplabel=[]
        for l in label:
            oplabel.append(self.label_desc['desc'][self.label_desc['oh'].index(l)])
        return oplabel
    
# import sys

# class Logger(object):
#     def __init__(self, logfilename):
#         self.terminal = sys.stdout
#         self.log = open(logfilename, "a")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)  

#     def flush(self):
#         #this flush method is needed for python 3 compatibility.
#         #this handles the flush command by doing nothing.
#         #you might want to specify some extra behavior here.
#         pass    
