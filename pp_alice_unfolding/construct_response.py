# WANT TO BE ABLE TO WRITE SMTH GENERALIZABLE TO OTHER THINGS
# DO THIS BY GENERALIZING THE UNFOLD FILE, THIS ONE HAS TO BE EEC-SPECIFIC

# WARNING METHODS APPEND TO OUTPUT FILE, WHEN RUNNING CONFIRM THIS IS INTENDED BEHAVIOR

# usage - run with the "pyjetty_load" environment
# CLOSURE (pythia)
# python3 construct_response.py --mc_file ../pp_unfolding/preprocess_sigma335_400k.root --data_file ../pp_unfolding/preprocess_sigma2_400k.root --output_file ./preunfold_closure_new.root --closure True
# CLOSURE (fullsim)
# python3 construct_response.py --mc_file /global/cfs/cdirs/alice/kdevero/pp_alice_unfolding/AnalysisResults/mc-13794540/merged_1star.root --data_file /global/cfs/cdirs/alice/kdevero/pp_alice_unfolding/AnalysisResults/mc-13794540/merged_1star.root --output_file ./preunfold_closurefullsim_redoredo.root --closure True
# FULL SIM / DATA
# python3 construct_response.py --mc_file /global/cfs/cdirs/alice/kdevero/pp_alice_unfolding/AnalysisResults/mc-13794540/merged_1star.root --data_file /global/cfs/cdirs/alice/kdevero/pp_alice_unfolding/AnalysisResults/data-13796056/merged.root --output_file ./preunfold_fr.root


# imports
import numpy as np
import ROOT
import argparse
ROOT.gSystem.Load("libRooUnfold.so")

verbose = 1

# define binnings
n_bins = [20, 20, 6] # WARNING RooUnfold seg faults if too many bins used
binnings = [np.logspace(-5,0,n_bins[0]+1), \
            np.logspace(-2.09,0,n_bins[1]+1), \
            np.array([5, 20, 40, 60, 80, 100, 150]).astype(float) ]
# using 8E-3 as the lower rl bin: solve 10^x = 8*10^(-3) 

def construct_response(n_mc_file="preprocessed_mc.root", n_out="preunfold.root"):

    print("constructing response matrix from mc_file ... ")

    if verbose > 1: print("binnings : " + str(binnings))

    h3_reco = ROOT.TH3D("reco", "reco", n_bins[0], binnings[0], n_bins[1], binnings[1], n_bins[2], binnings[2])
    h3_gen = ROOT.TH3D("gen", "gen", n_bins[0], binnings[0], n_bins[1], binnings[1], n_bins[2], binnings[2])

    h1_reco = ROOT.TH1D("reco1D", "reco1D", n_bins[2], binnings[2])
    h1_gen = ROOT.TH1D("gen1D", "gen1D", n_bins[2], binnings[2])

    response = ROOT.RooUnfoldResponse(h3_reco, h3_gen)
    response1D = ROOT.RooUnfoldResponse(h1_reco, h1_gen)

    print("loading synthetic tree...")
    fin = ROOT.TFile.Open(n_mc_file)
    synth_tree = fin.preprocessed
    print("reading events ...")

    n_rows = synth_tree.GetEntries()
    i = 0
    last_jetpt_obs = -1
    for row in synth_tree:

        i += 1
        if i % int(n_rows / 50) == 0: print("{} / {}".format(i, n_rows))

        [weight_gen, rl_gen, jetpt_gen, weight_obs, rl_obs, jetpt_obs, pt_hat_weight] = [row.gen_energy_weight, row.gen_R_L, row.gen_jet_pt, row.obs_energy_weight, row.obs_R_L, row.obs_jet_pt, row.pt_hat_weight]
        # remember to weigh everything with pt_hat_weight!

        h3_gen.Fill(weight_gen, rl_gen, jetpt_gen, pt_hat_weight)

        # if sucessful measurement, assumes missed are given some negative value for the energy weight
        if weight_obs >= 0:
            response.Fill(weight_obs, rl_obs, jetpt_obs, weight_gen, rl_gen, jetpt_gen, pt_hat_weight)
            h3_reco.Fill(weight_obs, rl_obs, jetpt_obs, pt_hat_weight)
        else:
            response.Miss(weight_gen, rl_gen, jetpt_gen, pt_hat_weight)
        
        # only fill 1D jetpt unfolding matrix once per jet
        # assumes perfect jet matching - is that valid ?!
        if jetpt_obs >= 0 and jetpt_obs != last_jetpt_obs:
            response1D.Fill(jetpt_obs, jetpt_gen, pt_hat_weight)
            h1_reco.Fill(jetpt_obs, pt_hat_weight)
            h1_gen.Fill(jetpt_gen, pt_hat_weight)
            last_jetpt_obs = jetpt_obs
    
    fin.Close()

    fout = ROOT.TFile(n_out, 'UPDATE')
    fout.cd()

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
    if closure: print("(closure test) true natural histograms also prepared")

    h3_raw = ROOT.TH3D("raw", "raw", n_bins[0], binnings[0], n_bins[1], binnings[1], n_bins[2], binnings[2])
    h1_raw =  ROOT.TH1D("raw1D", "raw1D", n_bins[2], binnings[2])

    h2_raw_eec = ROOT.TH2D("raw_eec", "raw_eec", n_bins[1], binnings[1], n_bins[2], binnings[2])

    if closure:
        h3_true = ROOT.TH3D("true", "true", n_bins[0], binnings[0], n_bins[1], binnings[1], n_bins[2], binnings[2])
        h1_true =  ROOT.TH1D("true1D", "true1D", n_bins[2], binnings[2])

        h2_true_eec = ROOT.TH2D("true_eec", "true_eec", n_bins[1], binnings[1], n_bins[2], binnings[2])

    print("loading natural tree...")
    fin = ROOT.TFile.Open(n_data_file)
    natural_tree = fin.preprocessed
    print("reading events ...")

    n_rows = natural_tree.GetEntries()
    i = 0
    last_jetpt_obs = -1
    for row in natural_tree:

        i += 1
        if i % int(n_rows / 50) == 0: print("{} / {}".format(i, n_rows))

        if closure:
            [weight_gen, rl_gen, jetpt_gen, weight_obs, rl_obs, jetpt_obs, pt_hat_weight] = [row.gen_energy_weight, row.gen_R_L, row.gen_jet_pt, row.obs_energy_weight, row.obs_R_L, row.obs_jet_pt, row.pt_hat_weight]

            h3_raw.Fill(weight_obs, rl_obs, jetpt_obs, pt_hat_weight)
            h2_raw_eec.Fill(rl_obs, jetpt_obs, weight_obs * pt_hat_weight)

            h3_true.Fill(weight_gen, rl_gen, jetpt_gen, pt_hat_weight)
            h2_true_eec.Fill(rl_gen, jetpt_gen, weight_gen * pt_hat_weight)
        else:
            [weight_obs, rl_obs, jetpt_obs] = [row.obs_energy_weight, row.obs_R_L, row.obs_jet_pt]

            h3_raw.Fill(weight_obs, rl_obs, jetpt_obs)
            h2_raw_eec.Fill(rl_obs, jetpt_obs, weight_obs)
        
        # only fill 1D jetpt unfolding matrix once per jet
        if jetpt_obs >= 0 and jetpt_obs != last_jetpt_obs:
            last_jetpt_obs = jetpt_obs

            if closure:
                h1_raw.Fill(jetpt_obs, pt_hat_weight)
                h1_true.Fill(jetpt_gen, pt_hat_weight)
            else:
                h1_raw.Fill(jetpt_obs)

    fin.Close()

    fout = ROOT.TFile(n_out, 'UPDATE')
    fout.cd()

    h3_raw.Write()
    h1_raw.Write()
    h2_raw_eec.Write()
    if closure:
        h3_true.Write()
        h1_true.Write()
        h2_true_eec.Write()

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

    print("closure ; ", flags.closure)

    # construct_response(flags.mc_file, flags.output_file)
    constructed_data_hist(flags.data_file, flags.output_file, flags.closure)
