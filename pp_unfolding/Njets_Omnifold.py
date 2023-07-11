#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
print(sys.path)

import numpy as np
import pandas as pd
import uproot as ur

from keras.layers import Dense, Input
from keras.models import Model

import omnifold as of
import os
import tensorflow as tf

from matplotlib import pyplot as plt
from IPython.display import Image
pd.set_option('display.max_columns', None) # to see all columns of df.head()

np.set_printoptions(edgeitems=15)


# In[3]:


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
print(tf.test.is_built_with_cuda())


# ___
# ___

# |   |   |   |
# |---|---|---|
# |Synthetic Generator-Level Sim   | $\theta_{0,G}$  | Truth-Level Sim  |
# |Synthetic Reconstruction-Level Sim   | $\theta_{0,S}$   | Full Reco-Level Sim  |
# |Natural Reconstruction  | $\theta_\mathrm{unknown,S}$  | Observed Detector Data  |
# |Natural Truth   |$\theta_\mathrm{unknown,G}$   | Nature  |
# 

# # Omnifold for EEC unfolding, (energy_weight, R_L, jet_pt)

# #### we assume all imported root files consist of `TTree preprocessed` consiting of three columns `energy_weight`, `R_L`, and `jet_pt`

# ### Define variables to unfold

# In[ ]:


obs_features = ["obs_jet_pt"]
gen_features = ["gen_jet_pt"]

labels = ["N_jets"]


# ### Import "natural reco": ALICE measured (testing: PYTHIA8 generator-level)
# ### Import "natural truth": ALICE truth (testing: PYTHIA8 generator-level, unused during actuall unfolding) 

# In[ ]:


natural_file = "preprocess_tr_eff_sigmap2.root"
natural_tree = ur.open("%s:preprocessed"%(natural_file))
natural_df = natural_tree.arrays(library="pd") #open the TTree as a pandas data frame


# ### Take a quick look at the data

# In[ ]:


print(natural_df.describe())


# In[ ]:


print(natural_df.head(10))


# ### Import "synthetic simulation", both generated and reconstructed level.

# In[ ]:


synthetic_file = "preprocess_tr_eff_cross_ref.root"
synth_tree = ur.open("%s:preprocessed"%(synthetic_file))
synth_df = synth_tree.arrays(library="pd")


# In[ ]:


print(synth_df.tail(10)) #look at some entries


# ### define 4 main datasets

# In[ ]:


theta_unknown_S = natural_df[obs_features].to_numpy() #Reconstructed Data
theta_unknown_G = natural_df[gen_features].to_numpy() #Nature, which unfolded data approaches

theta0_S = synth_df[obs_features].to_numpy() #Simulated, synthetic reco-level
theta0_G = synth_df[gen_features].to_numpy() #Generated, synthetic truth-level


_, theta_unknown_unique_ids = np.unique(theta_unknown_G, return_index=True)
theta_unknown_unique_ids = np.sort(theta_unknown_unique_ids)

theta_unknown_S = theta_unknown_S[theta_unknown_unique_ids]
theta_unknown_G = theta_unknown_G[theta_unknown_unique_ids]

_, theta0_unique_ids = np.unique(theta0_G, return_index=True)
theta0_unique_ids = np.sort(theta0_unique_ids)

theta0_S = theta0_S[theta0_unique_ids]
theta0_G = theta0_G[theta0_unique_ids]



# ### Ensure the samples have the same number of entries

# In[ ]:


N_Events = min(np.shape(theta0_S)[0],np.shape(theta_unknown_S)[0])-1
#N_Events = 100000


# Synthetic
theta0_G = theta0_G[:N_Events]
theta0_S = theta0_S[:N_Events]

theta0 = np.stack([theta0_G, theta0_S], axis=1)

# Natural
theta_unknown_G = theta_unknown_G[:N_Events]
theta_unknown_S = theta_unknown_S[:N_Events]



# In[ ]:


N = len(obs_features)

binning = np.linspace(20,40,100)

plt.hist(theta0_G[:,0],binning,color='blue',alpha=0.5,label="MC, true")
plt.hist(theta0_S[:,0],binning,histtype="step",color='black',ls=':',label="MC, reco")
plt.hist(theta_unknown_G[:,0],binning,color='orange',alpha=0.5,label="Data, true")
plt.hist(theta_unknown_S[:,0],binning,histtype="step",color='black',label="Data, reco")

plt.title(labels[0])
plt.xlabel(labels[0])
plt.ylabel("Events")
plt.legend(frameon=False)

plt.savefig("Njets_pre_training.png")
plt.close()


# In[ ]:


inputs = Input((len(obs_features), ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
model_dis = Model(inputs=inputs, outputs=outputs)


# In[ ]:


N_Iterations = 5

""" run to evaluate new data, calculate new weights """
myweights = of.omnifold_tr_eff(theta0,theta_unknown_S,N_Iterations,model_dis,dummyval=-9999)

print(myweights)
print(myweights.shape)

of.save_object(myweights, "./myweights_Njets_sigmap2.p")

""" run to load in saved weights """
#myweights = of.load_object("./myweights_Njets_sigmap2.p")



# In[ ]:


binning = np.linspace(20,40,100)

for iteration in range(N_Iterations):

    plt.hist(theta0_G[:,0],binning,color='blue',alpha=0.5,label="MC, true")
    plt.hist(theta_unknown_G[:,0],binning,color='orange',alpha=0.5,label="Data, true")
    plt.hist(theta0_G[:,0],weights=myweights[iteration, 0, :],bins=binning,color='black',histtype="step",label="OmniFolded",lw=2)

    plt.title(labels[0])
    plt.xlabel(labels[0])
    plt.ylabel("Events")
    plt.legend(frameon=False)
    
    plt.savefig("Njets_post_training_" + str(iteration) + ".png")
    plt.close()
    
    
# print out number of jets information

Njets_omnifolded = np.sum(myweights[-1, 0, :])

print("number of MC true jets   = " + str(len(theta0_G[:,0])))
print("number of MC reco jets   = " + str(len(np.unique(theta0_S[:,0]))))
print("number of Data true jets = " + str(len(theta_unknown_G[:,0])))
print("ESTIMATED BY OMNIFOLD    = " + str(Njets_omnifolded))
print("number of Data reco jets = " + str(len(np.unique(theta_unknown_S[:,0]))))