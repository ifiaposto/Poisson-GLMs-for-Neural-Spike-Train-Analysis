# Poisson-GLMs-for-Neural-Spike-Train-Analysis

This repository provides a Python implementation for analyzing multi-neuronal spike trains with Poisson Generalized Linear Models.

1) The dataset used in this repository can be downloaded here:

    Tim Blanche.   Multi-neuron recordings in primary visual cortex: 
    
    http://crcns.org/data-sets/vc/pvc-3,3702016.
    
2) For a formal description of the Poisson-GLM framework demonstrated in this repository, you may refer to:

    Truccolo, W., Eden, U.T., Fellows, M.R., Donoghue, J.P. and Brown, E.N., 2005. 
    A point process framework for relating neural spiking activity to spiking history, neural ensemble, and extrinsic covariate effects. Journal of neurophysiology, 93(2), pp.1074-1089

    https://www.ncbi.nlm.nih.gov/pubmed/15356183
    
## Setting-up the tutorial
1. Clone the repo

    ```
    git clone https://github.com/ifiaposto/Poisson-GLMs-for-Neural-Spike-Train-Analysis.git
    ```
 2. Install the requirements
    
    ```
    pip install -r requirements.txt
    ```
## Running the tutorial

### Discretization of the neuronal data


```
python3 neuron_count_data.py <nof neurons> <time bin size>
```
Example:
    
```
python3 neuron_count_data.py 25 1
```

### Create the training and testing dataset



```
python3 neuron_count_data.py <nof neurons> <time bin size> <nof_spikes>
```
Example:
    
```
python3 neuron_data_crop_train_test.py 25 1 4000
```

### Create regression covariates


```
python3 neuron_count_data.py <nof neurons> <time bin size> <nof_spikes> <degree of regression>
```
Example:
    
```
python3 neuron_mutual_regression.py 25 1 4000 1
```

### Fit the Poisson-GLM

```
python3 fit_spike_trains_mutual_regression.py <nof neurons> <time bin size> <nof_spikes> <degree of regression>
```
Example:
    
```
python3 fit_spike_trains_mutual_regression.py 25 1 4000 1
```





