# inter-areal-interactions
Analysis of inter-areal interactions in mouse visual cortex with Neuropixels data
Author: Ramakrishnan Iyer
Date: 10/06/2019
Copyright: RamIyer@2019

This repo contains code for characterization of inter-areal interactions in mouse visual cortex using population coupling/peer prediction methods. The method is based on predicting the activity of single neurons/populations using activities of other simultaneously recorded neuronal populations. In general, the model is:

Y = F(kc.Xc + ks.Xs) 

where,
Y  : single neuron/target population activity
Xc : (possibly transformed representation of) activity of other simultaneously recorded cells. Y is excluded from Xc
Xs : Stimulus input array
F : non-linearity (currently F is the identity transform)

Please note that this repo is work in progress and will be updated regularly. 

To do:
1. Generalize model to include non-linear F.
2. Stimulus input matrix currently exists for static grating responses. Will be generalized to build up for other stimuli. 
3. Include externally recorded behavioral co-variates.
4. Perform more extensive testing on PLS, RRR, PCR fitting routines.
5. Include hierarchical clustering transform on recorded activities.


