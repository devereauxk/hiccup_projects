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


obs_features = ["obs_energy_weight", "obs_R_L", "obs_jet_pt"]
gen_features = ["gen_energy_weight", "gen_R_L", "gen_jet_pt"]

labels = ["energy weight", "$R_L$", "jet $p_T$"]


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


# In[ ]:


# particle pt relative error
part_pt_tree = ur.open("%s:particle_pt"%(natural_file))
part_pt_df = part_pt_tree.arrays(library="pd")
part_pt = part_pt_df['gen_pt']
part_pt_smeared = part_pt_df['obs_pt']

binning = np.linspace(0, 4, 100)
plt.hist(part_pt, binning, alpha=0.5, label='true')
plt.hist(part_pt_smeared, binning, alpha=0.5, label='smeared')
plt.legend()
plt.xlabel('particle pt')
plt.savefig("part_pt.png")
plt.close()

binning = np.linspace(-0.05, 0.05, 100)
plt.hist( (part_pt_smeared - part_pt) / part_pt, binning)
plt.title('particle pt relative error')
plt.savefig("part_pt_err.png")
plt.close()


# ### Jet pt resolution

# In[ ]:


# jet pt relative error
jet_pt_tree_nat = ur.open("%s:jet_pt"%(natural_file))
jet_pt_df_nat = jet_pt_tree_nat.arrays(library="pd")
jet_pt_nat = jet_pt_df_nat['gen_pt']
jet_pt_nat_smeared = jet_pt_df_nat['obs_pt']

jet_pt_tree_synth = ur.open("%s:jet_pt"%(synthetic_file))
jet_pt_df_synth = jet_pt_tree_synth.arrays(library="pd")
jet_pt_synth = jet_pt_df_synth['gen_pt']
jet_pt_synth_smeared = jet_pt_df_synth['obs_pt']

binning = np.linspace(20, 40, 100)
plt.hist(jet_pt_synth,binning,color='blue',alpha=0.5,label="MC, true")
plt.hist(jet_pt_synth_smeared,binning,histtype="step",color='black',ls=':',label="MC, reco")
plt.hist(jet_pt_nat,binning,color='orange',alpha=0.5,label="Data, true")
plt.hist(jet_pt_nat_smeared,binning,histtype="step",color='black',label="Data, reco")
plt.legend()
plt.xlabel('jet pt')
plt.savefig("jet_pt.png")
plt.close()

binning = np.linspace(-0.2, 0.2, 100)
plt.hist( (jet_pt_nat_smeared - jet_pt_nat) / jet_pt_nat, binning)
plt.title('jet pt relative error')
plt.savefig("jet_pt_err.png")
plt.close()


# ### define 4 main datasets

# In[ ]:


theta_unknown_S = natural_df[obs_features].to_numpy() #Reconstructed Data
theta_unknown_G = natural_df[gen_features].to_numpy() #Nature, which unfolded data approaches

theta0_S = synth_df[obs_features].to_numpy() #Simulated, synthetic reco-level
theta0_G = synth_df[gen_features].to_numpy() #Generated, synthetic truth-level

obs_thrown = synth_df['obs_thrown'].to_numpy() # binary if pair DOESN'T pass efficiency cut


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


obs_thrown = obs_thrown[:N_Events]



""" halfway split of events 
halfway = round(N_Events / 2)

# Synthetic
theta0_G = theta0_G[:halfway]
theta0_S = theta0_S[:halfway]

theta0 = np.stack([theta0_G, theta0_S], axis=1)

# Natural
theta_unknown_G = theta_unknown_G[halfway:N_Events]
theta_unknown_S = theta_unknown_S[halfway:N_Events]
"""



# In[ ]:


N = len(obs_features)

binning = [np.logspace(-5,0,100),np.logspace(-4,0,100),np.linspace(20,40,100)]

fig, axes = plt.subplots(1, 3, figsize=(15,4))

obs_i = 0

for i,ax in enumerate(axes.ravel()):
    if (i >= N): break
    _,_,_=ax.hist(theta0_G[:,i],binning[i],color='blue',alpha=0.5,label="MC, true")
    _,_,_=ax.hist(theta0_S[:,i],binning[i],histtype="step",color='black',ls=':',label="MC, reco")
    _,_,_=ax.hist(theta_unknown_G[:,i],binning[i],color='orange',alpha=0.5,label="Data, true")
    _,_,_=ax.hist(theta_unknown_S[:,i],binning[i],histtype="step",color='black',label="Data, reco")

    ax.set_title(labels[i])
    ax.set_xlabel(labels[i])
    ax.set_ylabel("Events")
    ax.legend(frameon=False)
    
    if obs_i in [0, 1]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        
    obs_i += 1
    
fig.tight_layout()
fig.savefig("pre_training.png")
plt.close()


# In[ ]:


inputs = Input((len(obs_features), ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
model_dis = Model(inputs=inputs, outputs=outputs)


# In[ ]:


N_Iterations = 2

""" run to evaluate new data, calculate new weights """
"""
myweights = of.omnifold_tr_eff(theta0,theta_unknown_S,N_Iterations,model_dis,dummyval=-9999)

print(myweights)
print(myweights.shape)

of.save_object(myweights, "./myweights_sigmap2.p")
"""


""" run to load in saved weights """
myweights = of.load_object("./myweights_sigmap2.p")


# In[ ]:


binning = [np.logspace(-5,0,100),np.logspace(-4,0,100),np.linspace(20,40,100)]

for iteration in range(N_Iterations):
    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    obs_i = 0

    for i,ax in enumerate(axes.ravel()):
        if (i >= N): break
        _,_,_=ax.hist(theta0_G[:,i],binning[i],color='blue',alpha=0.5,label="MC, true")
        _,_,_=ax.hist(theta_unknown_G[:,i],binning[i],color='orange',alpha=0.5,label="Data, true")
        _,_,_=ax.hist(theta0_G[:,i],weights=myweights[iteration, 0, :],bins=binning[i],color='black',histtype="step",label="OmniFolded",lw=2)

        ax.set_title(labels[i])
        ax.set_xlabel(labels[i])
        ax.set_ylabel("Events")
        ax.legend(frameon=False)

        if obs_i in [0, 1]:
            ax.set_xscale('log')
            ax.set_yscale('log')

        obs_i += 1
    
    fig.tight_layout()
    fig.savefig("post_training_" + str(iteration) + ".png")
    plt.close()


# ___
# ___

# ### true vs smeared EEC calculation

# In[ ]:

plt.clf()
binning = np.logspace(-4, np.log10(0.4), 100)

_,_,_=plt.hist(theta0_G[:,1], binning, weights=theta0_G[:,0], color='blue', alpha=0.5, label="MC, true")
_,_,_=plt.hist(theta0_S[:,1], binning, weights=theta0_S[:,0], histtype="step", color='black',ls=':',label="MC, reco")
_,_,_=plt.hist(theta_unknown_G[:,1], binning, weights=theta_unknown_G[:,0], color='orange',alpha=0.5,label="Data, true")
_,_,_=plt.hist(theta_unknown_S[:,1], binning, weights=theta_unknown_S[:,0], histtype="step",color='black',label="Data, reco")

_,_,_=plt.hist(theta0_G[:,1], binning, weights=theta0_G[:,0] * myweights[-1, 0, :], color='black', histtype="step", label="OmniFolded", lw=2) # omnifolded


plt.xscale('log')
plt.yscale('log')
plt.xlim(3E-3, 0.5)
plt.ylim(1E-1,1E3)

plt.xlabel("$R_L$")
plt.ylabel("cs")
plt.title("EEC calculation")
plt.legend(frameon=False, loc='upper left')
plt.savefig("unfolded_EEC.png")
plt.close()



# ### true vs smeared EEC calculation, normalized by number of jets

# In[ ]:


# calculate number of jets in each sample
Njets_theta0_G = len(np.unique(theta0_G[:,2]))
Njets_theta0_S = len(np.unique(theta0_S[:,2]))
Njets_theta_unknown_G = len(np.unique(theta_unknown_G[:,2]))
Njets_theta_unknown_S = len(np.unique(theta_unknown_S[:,2]))
Njets_unfolded = (Njets_theta0_G / Njets_theta0_S) * Njets_theta_unknown_S

print("number of MC true jets   = " + str(Njets_theta0_G))
print("number of MC reco jets   = " + str(Njets_theta0_S))
print("number of Data true jets = " + str(Njets_theta_unknown_G))
print("ESTIMATED BY ME          = " + str(Njets_unfolded))
print("number of Data reco jets = " + str(Njets_theta0_S))

plt.clf()
binning = np.logspace(-4, np.log10(0.4), 100)

_,_,_=plt.hist(theta0_G[:,1], binning, weights=theta0_G[:,0]/Njets_theta0_G, color='blue', alpha=0.5, label="MC, true")
_,_,_=plt.hist(theta0_S[:,1], binning, weights=theta0_S[:,0]/Njets_theta0_S, histtype="step", color='black',ls=':',label="MC, reco")
_,_,_=plt.hist(theta_unknown_G[:,1], binning, weights=theta_unknown_G[:,0]/Njets_theta_unknown_G, color='orange',alpha=0.5,label="Data, true")
_,_,_=plt.hist(theta_unknown_S[:,1], binning, weights=theta_unknown_S[:,0]/Njets_theta_unknown_S, histtype="step",color='black',label="Data, reco")

_,_,_=plt.hist(theta0_G[:,1], binning, weights=theta0_G[:,0] * myweights[-1, 0, :] / Njets_unfolded, color='black', histtype="step", label="OmniFolded", lw=2) # omnifolded


plt.xscale('log')
plt.yscale('log')
plt.xlim(3E-3, 0.5)
plt.ylim(1E-5,1E-1)

plt.xlabel("$R_L$")
plt.ylabel("cs / number of jets")
plt.title("EEC calculation")
plt.legend(frameon=False, loc='upper left')
plt.savefig("unfolded_EEC_normbyjets.png")
plt.close()

N_truth_jets = len(np.unique(theta0_G[:,2]))
N_det_jets = len(np.unique(theta0_S[:,2]))

print("number of truth jets: " + str(N_truth_jets))
print("number of det   jets: " + str(N_det_jets))
print("percentage = " + str(N_det_jets / N_truth_jets))

N_truth_pairs = len(theta0_G)
N_det_pairs = len(np.unique(theta0_S[:,0]))

print("number of truth pairs: " + str(N_truth_pairs))
print("number of det   pairs: " + str(N_det_pairs))
print("percentage = " + str(N_det_pairs / N_truth_pairs))