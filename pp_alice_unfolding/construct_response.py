# WANT TO BE ABLE TO WRITE SMTH GENERALIZABLE TO OTHER THINGS
# DO THIS BY GENERALIZING THE UNFOLD FILE, THIS ONE HAS TO BE EEC-SPECIFIC

# usage
# python3 construct_response.py --mc_file ../pp_unfolding/preprocess_sigma335_400k.root --data_file ../pp_unfolding/preprocess_sigma2_400k.root --closure True

# imports
import numpy as np
import pandas as pd
import uproot as ur
import ROOT
import argparse
ROOT.gSystem.Load("libRooUnfold.so")

verbose = 1

# define binnings
n_bins = [9, 9, 9] # WARNING RooUnfold seg faults if too many bins used
binnings = [np.logspace(-5,0,10),np.logspace(-4,0,10),np.linspace(20,40,10)]

gen_features = ["gen_energy_weight", "gen_R_L", "gen_jet_pt"]
obs_features = ["obs_energy_weight", "obs_R_L", "obs_jet_pt"]
labels = ["energy weight", "$R_L$", "jet $p_T$"]

def construct_response(n_mc_file="preprocessed_mc.root", n_out="preunfold.root"):

    print("constructing response matrix from mc_file ... ")

    fout = ROOT.TFile(n_out, 'UPDATE')
    fout.cd()

    h3_reco = ROOT.TH3D("reco", "reco", n_bins[0], binnings[0], n_bins[1], binnings[1], n_bins[2], binnings[2])
    h3_gen = ROOT.TH3D("gen", "gen", n_bins[0], binnings[0], n_bins[1], binnings[1], n_bins[2], binnings[2])

    h1_reco = ROOT.TH1D("reco1D", "reco1D", n_bins[2], binnings[2])
    h1_gen = ROOT.TH1D("gen1D", "gen1D", n_bins[2], binnings[2])

    response = ROOT.RooUnfoldResponse(h3_reco, h3_gen)
    response1D = ROOT.RooUnfoldResponse(h1_reco, h1_gen)

    synth_tree = ur.open("%s:preprocessed"%(n_mc_file))
    synth_df = synth_tree.arrays(library="pd")

    if verbose > 1:
        print("synthetic data")
        print(synth_df.tail(10)) #look at some entries

    all_features = gen_features + obs_features
    theta0 = synth_df[all_features].to_numpy()

    for row in theta0:

        [weight_gen, rl_gen, jetpt_gen, weight_obs, rl_obs, jetpt_obs] = [i for i in row]

        if weight_obs >= 0: # if sucessful measurement
            response.Fill(weight_obs, rl_obs, jetpt_obs, weight_gen, rl_gen, jetpt_gen)
            response1D.Fill(jetpt_obs, jetpt_gen)

            h3_reco.Fill(weight_obs, rl_obs, jetpt_obs)
            h1_reco.Fill(jetpt_obs)
        else:
            response.Miss(weight_gen, rl_gen, jetpt_gen)
            response1D.Miss(jetpt_gen)

            h3_gen.Fill(weight_gen, rl_gen, jetpt_gen)
            h1_gen.Fill(jetpt_gen)

    response.Write()
    response1D.Write()
    h3_reco.Write()
    h3_gen.Write()
    h1_reco.Write()
    h1_gen.Write()

    fout.Write()
    fout.Close()
    print('[i] written ', fout.GetName())


def constructed_data_hist(n_data_file="preprocessed_data.root", n_out="preunfold.root", closure=False):

    print("constructing raw data histogram from natural data file ... ")
    if closure: print("(clusure test) true natural histograms also prepared")
    
    fout = ROOT.TFile(n_out, 'UPDATE')
    fout.cd()

    h3_raw = ROOT.TH3D("raw", "raw", n_bins[0], binnings[0], n_bins[1], binnings[1], n_bins[2], binnings[2])
    h1_raw =  ROOT.TH1D("raw1D", "raw1D", n_bins[2], binnings[2])

    if closure:
        h3_true = ROOT.TH3D("true", "true", n_bins[0], binnings[0], n_bins[1], binnings[1], n_bins[2], binnings[2])
        h1_true =  ROOT.TH1D("true1D", "true1D", n_bins[2], binnings[2])

    natural_tree = ur.open("%s:preprocessed"%(n_data_file))
    natural_df = natural_tree.arrays(library="pd")

    if verbose > 1:
        print("synthetic data")
        print(natural_df.tail(10)) #look at some entries

    if closure:
        all_features = gen_features + obs_features
    else:
        all_features = obs_features
    theta_unknown = natural_df[all_features].to_numpy()

    for row in theta_unknown:
        if closure:
            [weight_gen, rl_gen, jetpt_gen, weight_obs, rl_obs, jetpt_obs] = row
        else:
            [weight_obs, rl_obs, jetpt_obs] = row

        h3_raw.Fill(weight_obs, rl_obs, jetpt_obs)
        h1_raw.Fill(jetpt_obs)

        if closure:
            h3_true.Fill(weight_gen, rl_gen, jetpt_gen)
            h1_true.Fill(jetpt_gen)

    h3_raw.Write()
    h1_raw.Write()
    if closure:
        h3_true.Write()
        h1_true.Write()

    fout.Write()
    fout.Close()
    print('[i] written ', fout.GetName())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mc_file', default='preprocessed_mc.root')
    parser.add_argument('--data_file', default='preprocessed_data.root')
    parser.add_argument('--output_file', default='preunfold.root')
    parser.add_argument('--closure', type=bool, default=False)

    flags = parser.parse_args()

    construct_response(flags.mc_file, flags.output_file)
    constructed_data_hist(flags.data_file, flags.output_file, flags.closure)
