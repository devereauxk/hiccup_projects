# horovod usage:
# horovodrun --gloo -np 4 -H localhost:4 python unfold.py
# single gpu usage:
# python -u unfold.py --level=jet --save_weights=./myweights_400k_jet200_150_3kings_temp.p --suffix=_400k_jet200_150_3kings_temp --preprocess

import sys, os
print(sys.path)

import numpy as np
import pandas as pd
import uproot as ur
import argparse

import omnifold as of
from omnifold import Multifold
from omnifold import flatten
import tensorflow as tf
import tensorflow.keras.backend as K
import horovod.tensorflow.keras as hvd

from matplotlib import pyplot as plt
from IPython.display import Image
pd.set_option('display.max_columns', None) # to see all columns of df.head()
np.set_printoptions(edgeitems=3) 


hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
print(tf.test.is_built_with_cuda())

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
    
parser = argparse.ArgumentParser()

parser.add_argument('--level', default='pair', help='[pair/jet/event]')
parser.add_argument('--config', default='config_omnifold.json', help='Basic config file containing general options')
parser.add_argument('--save_weights', default=None, help='directory to save weights in')
parser.add_argument('--load_weights', default=None, help='directory of weights to load from')
parser.add_argument('--suffix', default="", help='suffix to add the end of weights plot names')
parser.add_argument('--nevents', type=float,default=-1, help='Number of events to load; if None, max amount used')
parser.add_argument('--preprocess', action='store_true', default=False,help='Scale/normalize input data to improve training efficacy')
parser.add_argument('--closure', action='store_true', default=False,help='Train omnifold for a closure test using simulation')
parser.add_argument('--verbose', action='store_true', default=False,help='Display additional information during training')

flags = parser.parse_args()

print("unfolding on : " + flags.level + "-level data")
print("config file : " + flags.config)
print("closure test : " + str(flags.closure))

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

#natural_file = "preprocess_sigma2_100k_treffoff.root"
natural_file = "preprocess_sigma2_400k.root"
#natural_file = "preprocess_tr_eff_sigmap2.root"
natural_tree = ur.open("%s:preprocessed"%(natural_file))
natural_df = natural_tree.arrays(library="pd") #open the TTree as a pandas data frame


# ### Take a quick look at the data

if verbose > 1:
    print("natural data")
    print(natural_df.describe())
    print(natural_df.head(10))


# ### Import "synthetic simulation", both generated and reconstructed level.
#synthetic_file = "preprocess_sigma335_100k_treffoff.root"
synthetic_file = "preprocess_sigma335_400k.root"
#synthetic_file = "preprocess_tr_eff_cross_ref.root"
synth_tree = ur.open("%s:preprocessed"%(synthetic_file))
synth_df = synth_tree.arrays(library="pd")

if verbose > 1:
    print("synthetic data")
    print(synth_df.tail(10)) #look at some entries


# TODO before this, synth_df and natural_df have to be constructed identically across hvd

# ### define 4 main datasets

# TODO add in pt hat support

all_features = gen_features + obs_features
theta_unknown = natural_df[all_features].to_numpy()
theta0 = synth_df[all_features].to_numpy()



############# RL cut
RL_min = 5 * (10**-3)

for row in theta_unknown:    
    if row[4] < RL_min:
        row[3:] = [-9999 for i in range(3)]
        
for row in theta0:    
    if row[4] < RL_min:
        row[3:] = [-9999 for i in range(3)]



############################# PREPROCESSING ################################
# applies logp1 to enegry weight and RL and minmax [0,1] scalling to jet pt
if flags.preprocess:
    
    print(80*'#')
    print("preprocessing...\n")

    log1p_transform = lambda x : -1*np.log(x) / 10
    log1p_inverse = lambda x : np.exp(-10*x)
    sqrt_transform = lambda x : np.sqrt(x)
    sqrt_inverse = lambda x : x**2
    scale_transform = lambda x : (x - 20) / (40 - 20)
    scale_inverse = lambda x : (40 - 20) * x + 20
    transforms = [log1p_transform, sqrt_transform, scale_transform]
    inverses = [log1p_inverse, sqrt_inverse, scale_inverse]

    print(theta_unknown)
    for row in theta_unknown:
        # gen features
        row[0] = log1p_transform(row[0])
        row[1] = sqrt_transform(row[1])
        row[2] = scale_transform(row[2])

        # obs features
        if row[3] >= 0:
            row[3] = log1p_transform(row[3])
            row[4] = sqrt_transform(row[4])     # obs RL
            row[5] = scale_transform(row[5])

    for row in theta0:
        # gen features
        row[0] = log1p_transform(row[0])
        row[1] = sqrt_transform(row[1])
        row[2] = scale_transform(row[2])

        # obs features
        if row[3] >= 0:
            row[3] = log1p_transform(row[3])
            row[4] = sqrt_transform(row[4])
            row[5] = scale_transform(row[5])


    print(30*'#')
    print(theta_unknown)
    
else:
    # identity transforms
    print(80*'#')
    print("skipping preprocessing...\n")
    
    transforms = [lambda x:x for i in range(3)]
    inverses = transforms


        
############################ MASKING/PADDING ####################################
print(80*'#')
print("masking/padding\n")

# make guarantee dummyval in each set is set to = -1
theta_unknown[theta_unknown < 0] = -1
theta0[theta0 < 0] = -1

print(theta0)
    
if flags.level == 'pair':    
    theta_unknown_G = theta_unknown[:, 0:3]
    theta_unknown_S = theta_unknown[:, 3:7]

    theta0_G = theta0[:, 0:3]
    theta0_S = theta0[:, 3:7]
    
    pad_length = 1
    

elif flags.level == 'jet':
    # pad lenght needs to be found manually for now
    # pad_length = np.min([max_length, np.max([split.shape[0] for split in arr])])
    pad_length = 600

    theta_unknown = of.split_by_index(theta_unknown, 2)
    
    jet_sizes = np.array([jet.shape[0] for jet in theta_unknown])
    print("theta_unknown jet pairs avg : " + str(np.mean(jet_sizes)))
    print("theta_unknown jet pairs std : " + str(np.std(jet_sizes)))
    print("theta_unknown jet pairs max : " + str(np.max(jet_sizes)))
    
    theta_unknown = of.pad_out_splits(theta_unknown, dummyval=-1, pad_length=pad_length)
    
    print("theta_unknown jets lost from being longer than pad length : " + str(np.sum([size > pad_length for size in jet_sizes]) / jet_sizes.shape[0]))

    theta0 = of.split_by_index(theta0, 2)
    
    jet_sizes = np.array([jet.shape[0] for jet in theta0])
    print("theta_unknown jet pairs avg : " + str(np.mean(jet_sizes)))
    print("theta_unknown jet pairs std : " + str(np.std(jet_sizes)))
    print("theta_unknown jet pairs max : " + str(np.max(jet_sizes)))
    
    theta0 = of.pad_out_splits(theta0, dummyval=-1, pad_length=pad_length)
    
    print("theta0 jets lost from being longer than pad length : " + str(np.sum([size > pad_length for size in jet_sizes]) / jet_sizes.shape[0]))
    
    print(80*'#')
    print(theta0)

    # TEMPORARY remove instances where reco jet is not found
    # theta_unknown = theta_unknown[theta_unknown[:,0,4] != -1]
    # theta0 = theta0[theta0[:,0,4] != -1]
    
    theta_unknown_G = theta_unknown[:, :, 0:2]
    theta_unknown_S = theta_unknown[:, :, 3:5]

    theta0_G = theta0[:, :, 0:2]
    theta0_S = theta0[:, :, 3:5]
    

    """
elif flags.level == 'event':
    pad_length = 600
    
    theta_unknown = of.split_by_index(theta_unknown, 4)
    theta_unknown = of.pad_out_splits(theta_unknown, dummyval=-1, pad_length=pad_length)

    theta0 = of.split_by_index(theta0, 7)
    theta0 = of.pad_out_splits(theta0, dummyval=-1, pad_length=pad_length)
    
    print(80*'#')
    print(theta0)

    theta_unknown_G = theta_unknown[:, :, 0:3]
    theta_unknown_S = theta_unknown[:, :, 3:7]

    theta0_G = theta0[:, :, 0:3]
    theta0_S = theta0[:, :, 3:7]
    """
    
    
else:
    print("--level arguement not recognized")
    exit()
    
print(80*'#')
print(theta0_S)


########################## PARTITIONING FOR HOROVID #############################
# limit event size
# NOTE: using too low a number of events will yeild noisy/poor unfolding
# N_Events = min(np.shape(theta0_S)[0],np.shape(theta_unknown_S)[0])-1
if flags.nevents > 0:
    print("trancating input data sets... ")
    print("\t mc set size : " + str(flags.nevents) + " pairs/jets/events")
    print("\t natural set size : " + str(int(0.7*flags.nevets)) + " pairs/jets/events")
    N_Events = flags.nevents # 2000 # 200000
    theta0_G = theta0_G[:N_Events]
    theta0_S = theta0_S[:N_Events]
    theta_unknown_S = theta_unknown_S[:int(0.7*N_Events)]
    theta_unknown_G = theta_unknown_G[:int(0.7*N_Events)]
else:
    print("using full available data...")
    print("\t mc set size : " + str(theta0_S.shape[0]) + " pairs/jets/events")
    print("\t natural set size : " + str(theta_unknown_S.shape[0]) + " pairs/jets/events")

nevts = theta0_S.shape[0]

# horovod partitioning
data_vars = theta_unknown_S[hvd.rank()::hvd.size()]
mc_reco = theta0_S[hvd.rank():nevts:hvd.size()]
mc_gen = theta0_G[hvd.rank():nevts:hvd.size()]

print("HVD data vars " + str(data_vars.shape))
print(data_vars)
print("HVD mc reco " + str(mc_reco.shape))
print(mc_reco)
print("HVD mc gen " + str(mc_gen.shape))
print(mc_gen)


############################### PRE-TRAINING PLOTS #############################

# distributions
N = len(obs_features)

binning = [np.logspace(-5,0,100),np.logspace(-4,0,100),np.linspace(20,40,100)]

fig, axes = plt.subplots(1, 2, figsize=(15,4))

for i,ax in enumerate(axes.ravel()):
    if (i >= N): break
    
    tiv = lambda x : inverses[i](flatten(x, level=flags.level))
    
    _,_,_=ax.hist(tiv(theta0_G)[:,i],binning[i],color='blue',alpha=0.5,label="MC, true")
    _,_,_=ax.hist(tiv(theta0_S)[:,i],binning[i],histtype="step",color='black',ls=':',label="MC, reco")
    _,_,_=ax.hist(tiv(theta_unknown_G)[:,i],binning[i],color='orange',alpha=0.5,label="Data, true")
    _,_,_=ax.hist(tiv(theta_unknown_S)[:,i],binning[i],histtype="step",color='black',label="Data, reco")

    ax.set_title(labels[i])
    ax.set_xlabel(labels[i])
    ax.set_ylabel("Events")
    ax.legend(frameon=False)
    
    if i in [0, 1]:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
fig.tight_layout()
fig.savefig("pre_training"+flags.suffix+".png")
plt.close()

# preprocessing efficacy

binning = [np.linspace(-0.2,1.2,100),np.linspace(-0.2,1.2,100),np.linspace(-0.2,1.2,100)]

fig, axes = plt.subplots(1, 2, figsize=(15,4))

for i,ax in enumerate(axes.ravel()):
    if (i >= N): break
    
    tiv = lambda x : flatten(x, level=flags.level)
    
    _,_,_=ax.hist(tiv(theta0_G)[:,i],binning[i],color='blue',alpha=0.5,label="MC, true")
    _,_,_=ax.hist(tiv(theta0_S)[:,i],binning[i],histtype="step",color='black',ls=':',label="MC, reco")
    _,_,_=ax.hist(tiv(theta_unknown_G)[:,i],binning[i],color='orange',alpha=0.5,label="Data, true")
    _,_,_=ax.hist(tiv(theta_unknown_S)[:,i],binning[i],histtype="step",color='black',label="Data, reco")

    ax.set_title(labels[i])
    ax.set_xlabel(labels[i])
    ax.set_ylabel("Events")
    ax.legend(frameon=False)
    
fig.tight_layout()
fig.savefig("preprocessing_result"+flags.suffix+".png")
plt.close()


############################### PERFORM UNFOLDING ###############################

# |---|---|---|
# |Synthetic Generator-Level Sim   | $\theta_{0,G}$  | Truth-Level Sim  |
# |Synthetic Reconstruction-Level Sim   | $\theta_{0,S}$   | Full Reco-Level Sim  |
# |Natural Reconstruction  | $\theta_\mathrm{unknown,S}$  | Observed Detector Data  |
# |Natural Truth   |$\theta_\mathrm{unknown,G}$   | Nature  |

""" run to evaluate new data, calculate new weights """
if flags.save_weights is not None:
    print("performing multifold to get weights ... ")
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

    of.save_object(myweights, flags.save_weights)
    
elif flags.load_weights is not None:
    """ run to load in saved weights """
    print("loading in saved weights from " + flags.load_weights)
    myweights = of.load_object(flags.load_weights)
else:
    exit()

print(80*'#')
print("multifolded weights")
print(myweights)
print(myweights.shape)

flattened_weights = of.flatten_weights(myweights, theta0_G, level=flags.level, dim=2)
flattened_weights_reco = of.flatten_weights(myweights, theta0_S, level=flags.level, dim=2)

print(80*'#')
print("flattened weights")
print(flattened_weights)
print(flattened_weights.shape)

opt = of.LoadJson(flags.config)
N_Iterations = myweights.shape[0] # opt['General']['NITER']


############################## UNFOLDED DISTRIBUTIONS ############################
# individual distros

for iteration in range(N_Iterations):
    fig, axes = plt.subplots(3, 2, figsize=(15,11))
    
    # ROW 1: raw distributions
    binning = [np.logspace(-5,0,100),np.logspace(-4,0,100),np.linspace(20,40,100)]
    for i in range(2):
        ax = axes[0, i]

        tiv = lambda x : inverses[i](flatten(x, level=flags.level))

        _,_,_=ax.hist(tiv(theta0_G)[:,i],binning[i],color='blue', alpha=0.5, label="MC, true")
        _,_,_=ax.hist(tiv(theta_unknown_G)[:,i],binning[i],color='orange',alpha=0.5,label="Data, true")
        
        _,bins,_=ax.hist(tiv(theta0_G)[:,i],weights=flattened_weights[iteration, 1, :],bins=binning[i],color='black',histtype="step",label="MuliFolded",lw=1)
        
        ax.set_title(labels[i])
        ax.set_xlabel(labels[i])
        ax.set_ylabel("Events")
        ax.legend(frameon=False)

        if i in [0, 1]:
            ax.set_xscale('log')
            ax.set_yscale('log')
            
    # ROW 2: residual plots
    for i in range(2):
        ax = axes[1, i]

        tiv = lambda x : inverses[i](flatten(x, level=flags.level))
        
        # detector-level
        unfolded_hist, _ = np.histogram(tiv(theta0_S)[:,i], binning[i], weights=flattened_weights_reco[iteration, 0, :])
        true_hist, _ = np.histogram(tiv(theta_unknown_S)[:,i], binning[i])
        err_hist = np.divide(unfolded_hist-true_hist, true_hist, out=np.zeros_like(unfolded_hist-true_hist), where=true_hist!=0)
                             
        ax.step(binning[i][1:],err_hist,where="pre",color='black',label="MuliFolded (reco)",lw=1)
        
        # particle-level
        unfolded_hist, _ = np.histogram(tiv(theta0_G)[:,i], binning[i], weights=flattened_weights[iteration, 1, :])
        true_hist, _ = np.histogram(tiv(theta_unknown_G)[:,i], binning[i])
        err_hist = np.divide(unfolded_hist-true_hist, true_hist, out=np.zeros_like(unfolded_hist-true_hist), where=true_hist!=0)
                             
        ax.step(binning[i][1:],err_hist,where="pre",color='red',label="MuliFolded (true)",lw=1)
        ax.hlines(y=0, xmin=binning[i][0], xmax=binning[i][len(binning[i])-1], color='grey', linestyle='--')

        ax.set_xlabel(labels[i])
        ax.set_ylabel("relative error")
        ax.legend(frameon=False)
        
        if i in [0, 1]:
            ax.set_xscale('log')
            ax.set_ylim([-0.5, 0.5])
        else:
            ax.set_ylim([-1, 1])
            
    # ROW 3: distributions, preprocessed inputs (gaussians)
    binning = [np.linspace(-0.2,1.2,100),np.linspace(-0.2,1.2,100),np.linspace(-0.2,1.2,100)]
    for i in range(2):
        ax = axes[2, i]

        tiv = lambda x : flatten(x, level=flags.level)
        
        # detector-level (step 1+1b)
        _,_,_=ax.hist(tiv(theta0_S)[:,i],binning[i],histtype="step",color='black',ls=':',label="MC, reco")
        _,_,_=ax.hist(tiv(theta_unknown_S)[:,i],binning[i],color='orange',alpha=0.5,label="Data, reco")
        _,_,_=ax.hist(tiv(theta0_S)[:,i],weights=flattened_weights_reco[iteration, 0, :],bins=binning[i],color='black',histtype="step",label="MuliFolded (reco)",lw=1)
        
        # particle-level (step 2)
        _,_,_=ax.hist(tiv(theta0_G)[:,i],binning[i], histtype="step",color='green',ls=':', label="MC, true")
        _,_,_=ax.hist(tiv(theta_unknown_G)[:,i],binning[i], color='green',alpha=0.2, label="Data, true")
        _,_,_=ax.hist(tiv(theta0_G)[:,i],weights=flattened_weights[iteration, 1, :],bins=binning[i],color='red',histtype="step",label="MuliFolded (true)",lw=1)
        
        ax.set_xlabel(labels[i])
        ax.set_ylabel("Events")
        ax.legend(frameon=False)
    
    
    fig.tight_layout()
    fig.savefig("post_training_" + str(iteration) + flags.suffix + ".png")
    plt.close()
    
    
    
############################## WEIGHTS DISTRIBUTIONS ############################
fig = plt.figure()
for iteration in range(N_Iterations-1):
    
    # step 1 (sand step 1b) weights
    plt.plot(np.sort((myweights[iteration+1, 0, :] - myweights[iteration, 0, :])/myweights[iteration,0,:]), label="iteration "+str(iteration)+", step 1")

    plt.title("step 1 learned weight")
    plt.xlabel("sorted pairs/jets")
    plt.ylabel("$(w_{n+1} - w_n) / w_n$")
    plt.legend(frameon=False)
        
fig.tight_layout()
fig.savefig("weights_step1_" + flags.suffix + ".png")
plt.close()

fig = plt.figure()
for iteration in range(N_Iterations-1):
    
    # step 2 weights
    plt.plot(np.sort((myweights[iteration+1, 1, :] - myweights[iteration, 1, :])/myweights[iteration,1,:]), label="iteration "+str(iteration)+", step 2")

    plt.title("step 2 learned weight")
    plt.xlabel("sorted pairs/jets")
    plt.ylabel("$(w_{n+1} - w_n) / w_n$")
    plt.legend(frameon=False)
        
fig.tight_layout()
fig.savefig("weights_step2_" + flags.suffix + ".png")
plt.close()


plt.rcParams["font.family"] = "serif"

for iteration in range(N_Iterations):
    plt.hist(myweights[iteration, 1, :], bins=50, range=(0, 4), histtype='step', stacked=True, fill=False, label="iteration "+str(iteration)+", step 2")

plt.xlabel('weight value')
plt.ylabel('counts')
plt.legend(loc='upper right', fontsize=10, frameon=False)
plt.title("step 2 weight distribution")
axes = plt.gca()
fig = plt.gcf()
fig.savefig("weights_hist_" + flags.suffix + ".png")
plt.close()


    

############################## EEC DISTRIBUTIONS ############################
w_iter = 0 # iteration of weights to use, for highest set to =-1, for first set to 0

# ### true vs smeared EEC calculation
fig, axes = plt.subplots(2, 1, figsize=(5,7))
binning = np.logspace(-4, np.log10(0.4), 100)

tiv_weight = lambda x : inverses[0](flatten(x, level=flags.level))
tiv_RL = lambda x : inverses[1](flatten(x, level=flags.level))

# EEC distribution
ax = axes[0]

_,_,_=ax.hist(tiv_RL(theta0_G)[:,1], binning, weights=tiv_weight(theta0_G)[:,0], color='blue', alpha=0.5, label="MC, true")
_,_,_=ax.hist(tiv_RL(theta_unknown_G)[:,1], binning, weights=tiv_weight(theta_unknown_G)[:,0], color='orange',alpha=0.5,label="Data, true")
_,bins,_=ax.hist(tiv_RL(theta0_G)[:,1], binning, weights=tiv_weight(theta0_G)[:,0]*flattened_weights[w_iter, 1, :], color='black',histtype="step",label="MuliFolded",lw=1)

ax.set_title("EEC unfolded")
ax.set_xlabel("$R_L$")
ax.set_ylabel("EEC")
ax.legend(frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')

# residual EEC
ax = axes[1]
        
unfolded_hist, _ = np.histogram(tiv_RL(theta0_G)[:,1], binning, weights=tiv_weight(theta0_G)[:,0]*flattened_weights[w_iter, 1, :])
true_hist, _ = np.histogram(tiv_RL(theta_unknown_G)[:,1], binning, weights=tiv_weight(theta_unknown_G)[:,0])
err_hist = np.divide(unfolded_hist-true_hist, true_hist, out=np.zeros_like(unfolded_hist-true_hist), where=true_hist!=0)

ax.step(binning[1:],err_hist,where="pre",color='red',label="MuliFolded",lw=1)
ax.hlines(y=0, xmin=binning[0], xmax=binning[len(binning)-1], color='grey', linestyle='--')

ax.set_xlabel("$R_L$")
ax.set_ylabel("relative error")
ax.legend(frameon=False)
ax.set_xscale('log')
ax.set_ylim([-0.5, 0.5])

fig.tight_layout()
fig.savefig("unfolded_EEC_" + flags.suffix + ".png")
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
fig, axes = plt.subplots(2, 1, figsize=(5,7))

# EEC distribution
ax = axes[0]

_,_,_=ax.hist(tiv_RL(theta0_G)[:,1], binning, weights=tiv_weight(theta0_G)[:,0]/Njets_theta0_G, color='blue', alpha=0.5, label="MC, true")
_,_,_=ax.hist(tiv_RL(theta_unknown_G)[:,1], binning, weights=tiv_weight(theta_unknown_G)[:,0]/Njets_theta_unknown_G, color='orange',alpha=0.5,label="Data, true")
_,bins,_=ax.hist(tiv_RL(theta0_G)[:,1], binning, weights=tiv_weight(theta0_G)[:,0]*flattened_weights[w_iter, 1, :]/Njets_unfolded, color='black',histtype="step",label="MuliFolded",lw=1)


ax.set_title("EEC unfolded")
ax.set_xlabel("$R_L$")
ax.set_ylabel("cs / number of jets")
ax.legend(frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')

# residual EEC
ax = axes[1]
        
unfolded_hist, _ = np.histogram(tiv_RL(theta0_G)[:,1], binning, weights=tiv_weight(theta0_G)[:,0]*flattened_weights[w_iter, 1, :]/Njets_theta0_G)
true_hist, _ = np.histogram(tiv_RL(theta_unknown_G)[:,1], binning, weights=tiv_weight(theta_unknown_G)[:,0]/Njets_theta_unknown_G)
err_hist = np.divide(unfolded_hist-true_hist, true_hist, out=np.zeros_like(unfolded_hist-true_hist), where=true_hist!=0)

ax.step(binning[1:],err_hist,where="pre",color='red',label="MuliFolded",lw=1)
ax.hlines(y=0, xmin=binning[0], xmax=binning[len(binning)-1], color='grey', linestyle='--')

ax.set_xlabel("$R_L$")
ax.set_ylabel("relative error")
ax.legend(frameon=False)
ax.set_xscale('log')
ax.set_ylim([-0.5, 0.5])

fig.tight_layout()
fig.savefig("unfolded_EEC_normbyjets_" + flags.suffix + ".png")
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
