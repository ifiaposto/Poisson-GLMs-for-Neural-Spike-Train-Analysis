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
import statsmodels.api as sm
import sys
import math
import csv


## This file fits Poisson-GLMs  for spike trains.
## In the results directory it prints the training and testing discrete-time loglikelihood achieved and its continuous-time approximation.
## It also returns a csv/ neuron that contains the learned parameters.


aux_files_path="aux_files/"
res_files_path="results/"


#number of neurons in the spike train
nof_neurons=int(sys.argv[1])
#discretization interval - time bin size
Delta=float(sys.argv[2])
#nof spikes to be used for training and for testing
nof_spikes=int(sys.argv[3])
#order of mutual regression, how many time bins in the past to be included in the regression
Q=int(sys.argv[4])

sys.stdout = open(res_files_path+'fit_glm_'+str(Delta)+'_'+str(Q)+'_'+str(nof_spikes)+'.txt', 'w')

#discretize the spike trains
logl=0
test_logl=0
for n in range(1,nof_neurons+1):
    #load the spike counts of neuron
    spike_train = pd.read_csv(aux_files_path+"mutual_regression_neuron_"+str(Delta)+"_"+str(n)+"_"+str(nof_spikes)+"_"+str(Q)+".csv")
    
    #the dependent variable
    endog=spike_train['counts0'].tolist()
    
    del spike_train['time']
    del spike_train['counts0']
    
    #the covariates, list of lists, external list corresponds to datapoints/samples, internal to covariates per sample
    exog=[]
    nof_entries=spike_train.shape[0]
    for row in range(0, nof_entries):
        s=spike_train.iloc[row:row+1,:].values.tolist()
        exog.extend(s)

    #add bias term
    exog = sm.add_constant(exog, prepend=False)

    #poisson glm with log link function
    glm_poisson = sm.GLM(endog, exog, family=sm.families.Poisson(link=sm.genmod.families.links.log))
    print('fit neuron')
    print(n)
    res = glm_poisson.fit()
    #get the training loglikelihood for fitting th current neuron and add to the total loglikelihood
    with open(res_files_path+'neuron_'+str(n)+'_'+str(Delta)+'.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(res.params)
    
    logl=logl+res.llf
    print(res.summary())
    train_logl=glm_poisson.loglike(res.params)
    print('train logl')
    print(train_logl)



    #get loglikelihood for held-out data
    #load the spike counts of neuron
    spike_test = pd.read_csv(aux_files_path+"test_mutual_regression_neuron_"+str(Delta)+"_"+str(n)+"_"+str(nof_spikes)+"_"+str(Q)+".csv")

    #get the dependent variable
    test_endog=spike_test['counts0'].tolist()

    del spike_test['time']
    del spike_test['counts0']

    #get the covariates
    test_exog=[]
    nof_entries=spike_test.shape[0]
    for row in range(0, nof_entries):
        s=spike_test.iloc[row:row+1,:].values.tolist()
        test_exog.extend(s)

    test_exog = sm.add_constant(test_exog, prepend=False)
    #initialize the model with the data
    glm_poisson_test = sm.GLM(test_endog, test_exog, family=sm.families.Poisson(link=sm.genmod.families.links.log))

    print('test neuron')
    #compute loglikelihood of the held-out data with the learned params
    test_logl_n=glm_poisson_test.loglike(res.params)
    test_logl=test_logl+test_logl_n
    print('test logl')
    print(test_logl_n)
print('total discrete training logl for glms')
print(logl)
print('continuous time apporximation of log')
#rectify logl to make it similar to the continuous time logl
print(logl-nof_spikes*math.log(Delta))


print('total discrete testing logl for glms')
print(test_logl)
#rectify test logl to make it similar to the continuous time logl
print('continuous time apporximation of log')
print(test_logl-nof_spikes*math.log(Delta))




