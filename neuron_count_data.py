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

import numpy as np
import pandas as pd
import math
import sys

## This file discretizes the spike trains of the neurons:
## Count the number of spikes that occured in each time bin of the discretized time interval.


#number of neurons in the spike train
nof_neurons=int(sys.argv[1])
#discretization interval - time bin size
Delta=float(sys.argv[2])

data_path="data/"
aux_files_path="aux_files/"

neuron_id=1
#discretize the spike trains
for n in range(neuron_id,nof_neurons+1):
    #load the spike trains of neuron
    spike_train = pd.read_csv(data_path+"neuron_"+str(n)+".csv" , header=None)
    time=0
    spike_train.loc[-1]=[0]#to start dividing the time starting from 0 in equal-sized  bins
    
    spike_train=spike_train.mul(1e-3)
    last_spike=spike_train.max()[0]
    nof_time_bins=math.ceil(last_spike/Delta)
    spike_train.loc[-2]=[nof_time_bins*Delta]
    df=pd.cut(spike_train[0],nof_time_bins,labels=False,retbins=True)
    spike_train['time']=pd.cut(spike_train[0],nof_time_bins, labels=False)
    spike_train.drop(spike_train.tail(1).index,inplace=True)#remove the first row inserted previously
    spike_train.drop(spike_train.tail(1).index,inplace=True)#remove the first row inserted previously
    
    spike_train=spike_train.groupby(['time']).count().reset_index().rename(columns={0:'counts'})
    
    #fill-in the time-intervals that do not appear in the original spike train, as bins with 0 counts
    spike_train.index = spike_train['time']
    spike_train=spike_train.reindex(np.arange(0, spike_train.time.max() + 1)).fillna(0)
    del spike_train['time']
    spike_train['time'] = spike_train.index
    
    #rescale time
    spike_train['time']=spike_train['time'].mul(Delta)
    spike_train= spike_train[['time','counts']]
    print('processed neuron %d' % (n))
   
    

    spike_train.to_csv(aux_files_path+"counts_"+str(Delta)+"_neuron_"+str(n)+".csv", index=False)




