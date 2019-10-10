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

## This file prepares auxiliary files needed for the poisson regression:
## It creates the covariates of the regression by merging the counts of spikes of all neurons that happened 'Q' time bins in the past.
## The dependent variable is the spike counts in each time bin.

#number of neurons in the spike train
nof_neurons=int(sys.argv[1])
#discretization interval - time bin size
Delta=float(sys.argv[2])
#nof spikes to be used for training and for testing
nof_spikes=int(sys.argv[3])
#order of mutual regression, how many time bins in the past to be included in the regression
Q=int(sys.argv[4])


aux_files_path="aux_files/"

#process the training time-series
for n in range(1,nof_neurons+1):
    #load the spike counts of neuron
    spike_mutual_regression=pd.read_csv(aux_files_path+"counts_neuron_"+str(Delta)+"_"+str(n)+"_"+str(nof_spikes)+".csv")
    spike_mutual_regression.rename(index=str, columns={"counts":"counts0"}, inplace=True)
    spike_mutual_regression=spike_mutual_regression.reset_index()
    nof_entries=spike_mutual_regression.shape[0]
    #add effect from neuron n2
    for n2 in range(1,nof_neurons+1):
        spike_train_n = pd.read_csv(aux_files_path+"counts_neuron_"+str(Delta)+"_"+str(n2)+"_"+str(nof_spikes)+".csv")
        del spike_train_n['time']
        #add effect r units back in the past
        for r in range(1,Q+1):
            spike_train_r=spike_train_n.shift(periods=r)
            spike_mutual_regression=pd.concat([spike_mutual_regression, spike_train_r], axis=1)
            spike_mutual_regression.rename(columns={"counts":"counts_neuron"+str(n2)+"_r_"+str(r)}, inplace=True)

    #consider only the entries with full history
    spike_mutual_regression=spike_mutual_regression.tail(nof_entries-Q)
    spike_mutual_regression=spike_mutual_regression.drop(['index'], axis=1)
    spike_mutual_regression =  spike_mutual_regression.sort_values(by=['time'])
    spike_mutual_regression.to_csv(aux_files_path+"mutual_regression_neuron_"+str(Delta)+"_"+str(n)+"_"+str(nof_spikes)+"_"+str(Q)+".csv", index=False)



#process the testing time-series
for n in range(1,nof_neurons+1):
    #load the spike counts of neuron
    spike_mutual_regression=pd.read_csv(aux_files_path+"test_counts_neuron_"+str(Delta)+"_"+str(n)+"_"+str(nof_spikes)+".csv")
    spike_mutual_regression.rename(index=str, columns={"counts":"counts0"}, inplace=True)
    spike_mutual_regression=spike_mutual_regression.reset_index()
    nof_entries=spike_mutual_regression.shape[0]
    #add effect from neuron n2
    for n2 in range(1,nof_neurons+1):
        spike_test_n = pd.read_csv(aux_files_path+"test_counts_neuron_"+str(Delta)+"_"+str(n2)+"_"+str(nof_spikes)+".csv")
        del spike_test_n['time']
        #add effect r units back in the past
        for r in range(1,Q+1):
            spike_test_r=spike_test_n.shift(periods=r)
            spike_mutual_regression=pd.concat([spike_mutual_regression, spike_test_r], axis=1)
            spike_mutual_regression.rename(columns={"counts":"counts_neuron"+str(n2)+"_r_"+str(r)}, inplace=True)

    #consider only the entries with full history
    spike_mutual_regression=spike_mutual_regression.tail(nof_entries-Q)
    spike_mutual_regression=spike_mutual_regression.drop(['index'], axis=1)
    spike_mutual_regression =  spike_mutual_regression.sort_values(by=['time'])
    spike_mutual_regression.to_csv(aux_files_path+"test_mutual_regression_neuron_"+str(Delta)+"_"+str(n)+"_"+str(nof_spikes)+"_"+str(Q)+".csv", index=False)





