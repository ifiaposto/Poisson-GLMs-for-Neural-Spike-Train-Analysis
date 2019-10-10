# Copyright 2019
# Author: Ifigeneia Apostolopoulou iapostol@andrew.cmu.edu, ifiaposto@gmail.com.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
import numpy as np
import pandas as pd


## This file creates the training and testing files to be used in the regression:
## It finds the last time bin so that a total number of 'nof_spikes' spikes across all neurons has occured up to that time bin (included).
## Subsequently, it finds the next time bin which contains the next 'nof_spikes' spikes (immediately following the spike train used for training).

#number of neurons in the spike train
nof_neurons=int(sys.argv[1])
#discretization interval - time bin size
Delta=float(sys.argv[2])
#nof spikes to be used for training and for testing
nof_spikes=int(sys.argv[3])



#path to the directory with the preprocessed files
aux_files_path="aux_files/"
#keep only up to the time which contains nof_spikes spikes

#find for each neuron the total number of spikes, up to each time
for n in range(1,nof_neurons+1):
    spike_train = pd.read_csv(aux_files_path+"counts_"+str(Delta)+"_neuron_"+str(n)+".csv")
    spike_train['nof_spikes_neuron_'+str(n)] = spike_train.counts.cumsum()
    del spike_train['counts']
    spike_train.to_csv(aux_files_path+"counts_neuron_total_"+str(n)+".csv",index=False)

#merge in one dataframe the spike counts across all neurons
spike_train = pd.read_csv(aux_files_path+"counts_neuron_total_"+str(1)+".csv")
for n in range(2,nof_neurons+1):
    spike_train_n = pd.read_csv(aux_files_path+"counts_neuron_total_"+str(n)+".csv")
    spike_train=pd.merge(spike_train, spike_train_n, on=['time'])

spike_train.to_csv(aux_files_path+"counts_neuron_total_all"+".csv",index=False)
spike_train_counts=spike_train
del spike_train_counts['time']
#sum the total number of spikes across all neurons up to each time
spike_train_counts=spike_train_counts.sum(axis = 1, skipna = True).to_list()
#find the first entry (time) which gives larger number of spikes than nof_spikes
res = list(filter(lambda i: i > nof_spikes, spike_train_counts))[0]
r=spike_train_counts.index(res)


for n in range(1,nof_neurons+1):
    #the first r rows shoud be counted in for the training
    spike_train = pd.read_csv(aux_files_path+"counts_"+str(Delta)+"_neuron_"+str(n)+".csv")
    cropped_spike_train=spike_train.head(r)
    cropped_spike_train.to_csv(aux_files_path+"counts_neuron_"+str(Delta)+"_"+str(n)+"_"+str(nof_spikes)+".csv", index=False)


spike_test = pd.read_csv(aux_files_path+"counts_neuron_total_all"+".csv")
#skip the first r rows, they are used for the training
spike_test = spike_test.iloc[r:]
spike_test_counts=spike_test
del spike_test_counts['time']
#find the next rows that contain another batch of nof_spikes spikes (it may be r!=r2)
spike_test_counts=spike_test_counts.sum(axis = 1, skipna = True).to_list()
res = list(filter(lambda i: i > 2*nof_spikes, spike_test_counts))[0]
r2=spike_test_counts.index(res)

for n in range(1,nof_neurons+1):
    spike_test = pd.read_csv(aux_files_path+"counts_"+str(Delta)+"_neuron_"+str(n)+".csv")
    #skip the first r rows, they are used for the training
    spike_test = spike_test.iloc[r:]
    #get the next r2 rows
    spike_test = spike_test.head(r2)
    spike_test.to_csv(aux_files_path+"test_counts_neuron_"+str(Delta)+"_"+str(n)+"_"+str(nof_spikes)+".csv", index=False)





