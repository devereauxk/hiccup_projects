#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.path)


# In[2]:


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


# In[3]:


#gpus = tf.config.experimental.list_physical_devices('GPU')
#print(gpus)


# # [Omnifold Paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.182001) Background

# |   |   |   |
# |---|---|---|
# |Synthetic Generator-Level Sim   | $\theta_{0,G}$  | Truth-Level Sim  |
# |Synthetic Reconstruction-Level Sim   | $\theta_{0,S}$   | Full Reco-Level Sim  |
# |Natural Reconstruction  | $\theta_\mathrm{unknown,S}$  | Observed Detector Data  |
# |Natural Truth   |$\theta_\mathrm{unknown,G}$   | Nature  |
# 

# ___
# ___

# # Omnifold for EEC unfolding, (energy_weight, R_L, jet_pt)

# #### we assume all imported root files consist of `TTree preprocessed` consiting of three columns `energy_weight`, `R_L`, and `jet_pt`

# ### Define variables to unfold

# In[4]:


obs_features = ["obs_energy_weight", "obs_R_L", "obs_jet_pt"]
gen_features = ["gen_energy_weight", "gen_R_L", "gen_jet_pt"]

labels = ["energy weight", "$R_L$", "jet $p_T$"]

dummyval = [-9999 for i in range(3)]


# ### Import "natural reco": ALICE measured (testing: PYTHIA8 generator-level)
# ### Import "natural truth": ALICE truth (testing: PYTHIA8 generator-level, unused during actuall unfolding) 

# In[5]:


natural_file = "preprocess_tr_eff.root"
natural_tree = ur.open("%s:preprocessed"%(natural_file))
natural_df = natural_tree.arrays(library="pd") #open the TTree as a pandas data frame
#natural_df = natural_reco_df[natural_reco_df["obs_hfs_pt"] != 0] #remove empty entries


# ### Take a quick look at the data

# In[6]:


natural_df.describe()


# In[7]:


natural_df.head(10)


# In[8]:


# particle pt relative error
part_pt_tree = ur.open("%s:particle_pt"%(natural_file))
part_pt_df = part_pt_tree.arrays(library="pd")
part_pt_df = part_pt_df[part_pt_df['obs_pt'] != -9999]
part_pt = part_pt_df['gen_pt']
part_pt_smeared = part_pt_df['obs_pt']

binning = np.linspace(0, 4, 100)
plt.hist(part_pt, binning, alpha=0.5, label='true')
plt.hist(part_pt_smeared, binning, alpha=0.5, label='smeared')
plt.legend()
plt.xlabel('particle pt')
plt.show()

binning = np.linspace(-0.05, 0.05, 100)
plt.hist( (part_pt_smeared - part_pt) / part_pt, binning)
plt.title('particle pt relative error')
plt.show()


# ### Jet pt resolution

# In[9]:


# jet pt relative error
jet_pt_tree = ur.open("%s:jet_pt"%(natural_file))
jet_pt_df = jet_pt_tree.arrays(library="pd")
jet_pt = jet_pt_df['gen_pt']
jet_pt_smeared = jet_pt_df['obs_pt']

binning = np.linspace(20, 40, 100)
plt.hist(jet_pt, binning, alpha=0.5, label='true')
plt.hist(jet_pt_smeared, binning, alpha=0.5, label='smeared')
plt.legend()
plt.xlabel('jet pt')
plt.show()

binning = np.linspace(-0.2, 0.2, 100)
plt.hist( (jet_pt_smeared - jet_pt) / jet_pt, binning)
plt.title('jet pt relative error')
plt.show()


# ### Import "synthetic simulation", both generated and reconstructed level.

# In[10]:


synthetic_file = "preprocess_tr_eff.root"
synth_tree = ur.open("%s:preprocessed"%(synthetic_file))
synth_df = synth_tree.arrays(library="pd")
#synth_df = synth_df[synth_df["obs_hfs_pt"] != 0]


# In[11]:


synth_df.tail(10) #look at some entries


# ### define 4 main datasets

# In[12]:


theta_unknown_S = natural_df[obs_features].to_numpy() #Reconstructed Data
theta_unknown_G = natural_df[gen_features].to_numpy() #Nature, which unfolded data approaches

theta0_S = synth_df[obs_features].to_numpy() #Simulated, synthetic reco-level
theta0_G = synth_df[gen_features].to_numpy() #Generated, synthetic truth-level

obs_thrown = synth_df['obs_thrown'].to_numpy() # binary if pair DOESN'T pass efficiency cut


# ### Ensure the samples have the same number of entries

# In[13]:


N_Events = min(np.shape(theta0_S)[0],np.shape(theta_unknown_S)[0])-1

# Synthetic
theta0_S = theta0_S[:N_Events]
theta0_G = theta0_G[:N_Events]

theta0 = np.stack([theta0_S, theta0_G], axis=1)

# Natural
theta_unknown_S = theta_unknown_S[:N_Events]
theta_unknown_G = theta_unknown_G[:N_Events]

obs_thrown = obs_thrown[:N_Events]


# In[14]:


N = len(obs_features)

binning = [np.logspace(-5,0,100),np.logspace(-4,0,100),np.linspace(20,40,100)]

fig, axes = plt.subplots(1, 3, figsize=(15,4))

obs_i = 0

for i,ax in enumerate(axes.ravel()):
    if (i >= N): break
    _,_,_=ax.hist(theta0_G[:,i][obs_thrown!=1],binning[i],color='blue',alpha=0.5,label="MC, true")
    _,_,_=ax.hist(theta0_S[:,i][obs_thrown!=1],binning[i],histtype="step",color='black',ls=':',label="MC, reco")
    _,_,_=ax.hist(theta_unknown_G[:,i][obs_thrown!=1],binning[i],color='orange',alpha=0.5,label="Data, true")
    _,_,_=ax.hist(theta_unknown_S[:,i][obs_thrown!=1],binning[i],histtype="step",color='black',label="Data, reco")

    ax.set_title(labels[i])
    ax.set_xlabel(labels[i])
    ax.set_ylabel("Events")
    ax.legend(frameon=False)
    
    if obs_i in [0, 1]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        
    obs_i += 1
    
fig.tight_layout()


# In[15]:


inputs = Input((len(obs_features), ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
model_dis = Model(inputs=inputs, outputs=outputs)


# In[20]:


print(theta0_G.shape)
print(theta0_S.shape)
print(theta_unknown_S.shape)

print(theta0_G[theta0_S[:,0]==-9999])
print(theta0_G[theta0_S[:,0]==-9999].shape)


# In[ ]:


N_Iterations = 2
myweights = of.omnifold_tr_eff(theta0,theta_unknown_S,N_Iterations,model_dis,dummyval=-9999)


# In[ ]:


for iteration in range(N_Iterations):
    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    obs_i = 0

    for i,ax in enumerate(axes.ravel()):
        if (i >= N): break
        _,_,_=ax.hist(theta0_G[:,i],binning[i],color='blue',alpha=0.5,label="MC, true")
        _,_,_=ax.hist(theta_unknown_G[:,i],binning[i],color='orange',alpha=0.5,label="Data, true")
        _,_,_=ax.hist(theta0_G[:,i],weights=myweights[iteration, 0, :],bins=binning[i],color='black',histtype="step",label="OmniFolded",lw="2")

        ax.set_title(labels[i])
        ax.set_xlabel(labels[i])
        ax.set_ylabel("Events")
        ax.legend(frameon=False)

        if obs_i in [0, 1]:
            ax.set_xscale('log')
            ax.set_yscale('log')

        obs_i += 1
    
    fig.tight_layout()


# ___
# ___

# ### true vs smeared EEC calculation

# In[ ]:


binning = np.logspace(-4, np.log10(0.4), 100)

_,_,_=plt.hist(theta0_G[:,1], binning, weights=theta0_G[:,0], color='blue', alpha=0.5, label="MC, true") # true
_,_,_=plt.hist(theta0_S[:,1], binning, weights=theta0_S[:,0], color='orange', alpha=0.5, label="data, smeared") # smeared
_,_,_=plt.hist(theta0_G[:,1] * myweights[-1, 0, :], binning, weights=theta0_G[:,0] * myweights[-1, 0, :], color='black', histtype="step", label="OmniFolded", lw="2") # omnifolded

plt.xscale('log')
plt.yscale('log')

plt.xlabel("$R_L$")
plt.ylabel("cs")
plt.title("EEC calculation")
plt.legend(frameon=False, loc='upper left')


# In[ ]:


myweights[-1]


# In[ ]:


myweights


# In[ ]:


pass_reco = np.random.binomial(1,0.9,30)


# In[ ]:


pass_reco
