# usage:
# horovodrun --gloo -np 4 -H localhost:4 python unfold.py

import sys, os
print(sys.path)

import numpy as np
import pandas as pd
import uproot as ur

import omnifold as of
from omnifold import Multifold
import tensorflow as tf
import tensorflow.keras.backend as K
import horovod.tensorflow.keras as hvd

from matplotlib import pyplot as plt
from IPython.display import Image
pd.set_option('display.max_columns', None) # to see all columns of df.head()
np.set_printoptions(edgeitems=7) 


hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
print(tf.test.is_built_with_cuda())

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


print(80*'#')
print("Total hvd size: {}, rank: {}, local size: {}, local rank: {}".format(hvd.size(), hvd.rank(), hvd.local_size(), hvd.local_rank()))
print(80*'#')

verbose = 2 if hvd.rank() == 0 else 0


# # Omnifold for EEC unfolding, (energy_weight, R_L, jet_pt)

obs_features = ["obs_energy_weight", "obs_R_L", "obs_jet_pt"]
gen_features = ["gen_energy_weight", "gen_R_L", "gen_jet_pt"]

labels = ["energy weight", "$R_L$", "jet $p_T$"]


# ### Import "natural reco": ALICE measured (testing: PYTHIA8 generator-level)
# ### Import "natural truth": ALICE truth (testing: PYTHIA8 generator-level, unused during actuall unfolding) 

natural_file = "preprocess_tr_eff_sigmap2.root"
natural_tree = ur.open("%s:preprocessed"%(natural_file))
natural_df = natural_tree.arrays(library="pd") #open the TTree as a pandas data frame


# ### Take a quick look at the data

if verbose > 1:
    print("natural data")
    print(natural_df.describe())
    print(natural_df.head(10))


# ### Import "synthetic simulation", both generated and reconstructed level.
synthetic_file = "preprocess_tr_eff_cross_ref.root"
synth_tree = ur.open("%s:preprocessed"%(synthetic_file))
synth_df = synth_tree.arrays(library="pd")

if verbose > 1:
    print("synthetic data")
    print(synth_df.tail(10)) #look at some entries


# TODO before this, synth_df and natural_df have to be constructed identically across hvd

# ### define 4 main datasets

# global vars

theta_unknown_S = natural_df[obs_features].to_numpy() #Reconstructed Data
theta_unknown_G = natural_df[gen_features].to_numpy() #Nature, which unfolded data approaches

theta0_S = synth_df[obs_features].to_numpy() #Simulated, synthetic reco-level
theta0_G = synth_df[gen_features].to_numpy() #Generated, synthetic truth-level


# limit event size
# NOTE: using too low a number of events will yeild noisy/poor unfolding
# N_Events = min(np.shape(theta0_S)[0],np.shape(theta_unknown_S)[0])-1

N_Events = 200000
theta0_G = theta0_G[:N_Events]
theta0_S = theta0_S[:N_Events]
theta_unknown_S = theta_unknown_S[:int(0.7*N_Events)]
theta_unknown_G = theta_unknown_G[:int(0.7*N_Events)]


# learning sets partitions for horovod usage
nevts = theta0_S.shape[0]

data_vars = theta_unknown_S[hvd.rank()::hvd.size()]
mc_reco = theta0_S[hvd.rank():nevts:hvd.size()]
mc_gen = theta0_G[hvd.rank():nevts:hvd.size()]

print("HVD data vars " + str(data_vars.shape))
print(data_vars)
print("HVD mc reco " + str(mc_reco.shape))
print(mc_reco)
print("HVD mc gen " + str(mc_gen.shape))
print(mc_gen)


# produce plots pre-training

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



# TODO implement some preprocessing


# PERFORM UNFOLDING

# |---|---|---|
# |Synthetic Generator-Level Sim   | $\theta_{0,G}$  | Truth-Level Sim  |
# |Synthetic Reconstruction-Level Sim   | $\theta_{0,S}$   | Full Reco-Level Sim  |
# |Natural Reconstruction  | $\theta_\mathrm{unknown,S}$  | Observed Detector Data  |
# |Natural Truth   |$\theta_\mathrm{unknown,G}$   | Nature  |

""" run to evaluate new data, calculate new weights """
K.clear_session()
mfold = Multifold(
        nevts=nevts,
        mc_gen=theta0_G,
        mc_reco=theta0_S,
        data=theta_unknown_S,
        config_file='config_omnifold.json',
        verbose = 1  
    )

mfold.Preprocessing(weights_mc=None, weights_data=None)
mfold.Unfold()
myweights = mfold.GetWeights()

of.save_object(myweights, "./myweights_try.p")


""" run to load in saved weights """
#myweights = of.load_object("./myweights.p")


print("multifolded weights")
print(myweights)
print(myweights.shape)


# print unfolded results for each dimension

opt = of.LoadJson('config_omnifold.json')
N_Iterations = opt['General']['NITER']

binning = [np.logspace(-5,0,100),np.logspace(-4,0,100),np.linspace(20,40,100)]

for iteration in range(N_Iterations):
    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    obs_i = 0

    for i,ax in enumerate(axes.ravel()):
        if (i >= N): break
        _,_,_=ax.hist(theta0_G[:,i],binning[i],color='blue', alpha=0.5, label="MC, true")
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
    fig.savefig("post_training_" + str(iteration) + "_try.png")
    plt.close()

# ### true vs smeared EEC calculation

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

# calculate number of jets in each sample
Njets_theta0_G = len(np.unique(theta0_G[:,2]))
Njets_theta0_S = len(np.unique(theta0_S[:,2]))
Njets_theta_unknown_G = len(np.unique(theta_unknown_G[:,2]))
Njets_theta_unknown_S = len(np.unique(theta_unknown_S[:,2]))
Njets_unfolded = (Njets_theta0_G / Njets_theta0_S) * Njets_theta_unknown_S

print("number of MC reco jets    = " + str(Njets_theta0_S))
print("number of MC true jets    = " + str(Njets_theta0_G))
print("number of Data reco jets  = " + str(Njets_theta0_S))
print("number of Data truth jets = " + str(Njets_unfolded) + " (ESTIMATED BY ME)")


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

print("number of truth sim jets: " + str(N_truth_jets))
print("number of det sim   jets: " + str(N_det_jets))
print("percentage = " + str(N_det_jets / N_truth_jets))

N_truth_pairs = len(theta0_G)
N_det_pairs = len(np.unique(theta0_S[:,0]))

print("number of truth sim pairs: " + str(N_truth_pairs))
print("number of det sim   pairs: " + str(N_det_pairs))
print("percentage = " + str(N_det_pairs / N_truth_pairs))