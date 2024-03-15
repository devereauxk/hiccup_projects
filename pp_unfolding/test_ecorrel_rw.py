#!/usr/bin/env python3

from pyjetty.mputils.mputils import logbins
import operator as op
import itertools as it
import sys
import os
import argparse
import math
from tqdm import tqdm
from heppy.pythiautils import configuration as pyconf
import pythiaext
import pythiafjext
import pythia8
import fjtools
import ecorrel
import fjcontrib
import fjext
import fastjet as fj
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
# ROOT.gSystem.AddDynamicPath('$HEPPY_DIR/external/roounfold/roounfold-current/lib')
# ROOT.gSystem.Load('libRooUnfold.dylib')
# ROOT.gSystem.AddDynamicPath('$PYJETTY_DIR/cpptools/lib')
# ROOT.gSystem.Load('libpyjetty_rutilext')
# _test = ROOT.RUtilExt.Test()
from tqdm import tqdm
import argparse
import os
from array import array
import numpy as np
import uproot as ur
import yaml

# usage
# python test_ecorrel_rw.py --nev 400000 --output "./preprocess_sigma335_400k.root"

dummyval = -9999

with open("./scaleFactors.yaml", 'r') as stream:
	pt_hat_yaml = yaml.safe_load(stream)
pt_hat_lo = pt_hat_yaml["bin_lo"]
pt_hat_hi = pt_hat_yaml["bin_hi"]


# load track efficiency tree
tr_eff_file = ur.open("tr_eff.root")
tr_eff = tr_eff_file["tr_eff"].to_numpy()

def smear_track(part, sigma=0.01):
	pt = part.perp() * (1 + np.random.normal(0, sigma))
	px = pt * np.cos(part.phi())
	py = pt * np.sin(part.phi())
	pz = part.pz() * (1 + np.random.normal(0, sigma))
	E = np.sqrt(part.m2() + pt**2 + pz**2)
	smeared_part = fj.PseudoJet(px, py, pz, E)
	smeared_part.set_user_index(part.user_index())
	return smeared_part

def do_keep_track(part):
    bin_index = np.digitize(part.perp(), tr_eff[1]) - 1
    keep_prob = tr_eff[0][bin_index]
    return np.random.choice([0, 1], p=[1 - keep_prob, keep_prob])


class EEC_pair:
	def __init__(self, _index1, _index2, _weight, _r, _pt):
		self.index1 = _index1
		self.index2 = _index2
		self.weight = _weight
		self.r = _r
		self.pt = _pt

	def is_equal(self, pair2):
		return (self.index1 == pair2.index1 and self.index2 == pair2.index2) \
			or (self.index1 == pair2.index2 and self.index2 == pair2.index1)
	
	def __str__(self):
		return "EEC pair with (index1, index2, weight, RL, pt) = (" + \
			str(self.index1) + ", " + str(self.index2) + ", " + str(self.weight) + \
			", " + str(self.r) + ", " + str(self.pt) + ")"


def get_args_from_settings(ssettings):
	sys.argv = sys.argv + ssettings.split()
	parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly')
	pyconf.add_standard_pythia_args(parser)
	parser.add_argument('--output', default="test_ecorrel_rw.root", type=str)
	parser.add_argument('--user-seed', help='pythia seed', default=1111, type=int)
	parser.add_argument('--tr_eff_off', action='store_true', default=False)
	args = parser.parse_args()
	return args


def main():
	mycfg = []
	mycfg.append("StringPT:sigma=0.335") # for producing slightly different data sets, default is 0.335 GeV
	ssettings = "--py-ecm 5020 --py-pthatmin 5"
	args = get_args_from_settings(ssettings)
	pythia_hard = pyconf.create_and_init_pythia_from_args(args, mycfg)
	
	max_eta_hadron = 0.9  # ALICE
	jet_R0 = 0.4
	max_eta_jet = max_eta_hadron - jet_R0
	#parts_selector = fj.SelectorPtMin(0.15) & fj.SelectorAbsEtaMax(max_eta_hadron)
	#jet_selector = fj.SelectorPtMin(20) & fj.SelectorPtMax(40) & fj.SelectorAbsEtaMax(max_eta_jet)
	parts_selector = fj.SelectorAbsEtaMax(max_eta_hadron)
	jet_selector = fj.SelectorPtMin(5) & fj.SelectorAbsEtaMax(max_eta_jet)  

	# print the banner first
	fj.ClusterSequence.print_banner()
	print()
	# set up our jet definition and a jet selector
	jet_def = fj.JetDefinition(fj.antikt_algorithm, jet_R0)
	print(jet_def)

	fout = ROOT.TFile(args.output, 'recreate')
	fout.cd()

	# TTree output definition
	preprocessed = ROOT.TTree("preprocessed", "true and smeared obs")
	[gen_energy_weight, gen_R_L, gen_jet_pt] = [array('d', [0]) for i in range(3)]
	[obs_energy_weight, obs_R_L, obs_jet_pt] = [array('d', [0]) for i in range(3)]
	[pt_hat_weight, event_n] = [array('d', [0]) for i in range(2)]

	preprocessed.Branch("gen_energy_weight", gen_energy_weight, "gen_energy_weight/D")
	preprocessed.Branch("gen_R_L", gen_R_L, "gen_R_L/D")
	preprocessed.Branch("gen_jet_pt", gen_jet_pt, "gen_jet_pt/D")
	preprocessed.Branch("obs_energy_weight", obs_energy_weight, "obs_energy_weight/D")
	preprocessed.Branch("obs_R_L", obs_R_L, "obs_R_L/D")
	preprocessed.Branch("obs_jet_pt", obs_jet_pt, "obs_jet_pt/D")
	preprocessed.Branch("pt_hat_weight", pt_hat_weight, "pt_hat_weight/D")
	preprocessed.Branch("event_n", event_n, "event_n/D")
    
	# debug tree definitions
	particle_pt_tree = ROOT.TTree("particle_pt", "true and smeared particle-level")
	[gen_pt, obs_pt] = [array('d', [0]) for i in range(2)]
	particle_pt_tree.Branch("gen_pt", gen_pt, "gen_pt/D")
	particle_pt_tree.Branch("obs_pt", obs_pt, "obs_pt/D")

	jet_pt_tree = ROOT.TTree("jet_pt", "true and smeared particle-level")
	[gen_jet_pt_debug, obs_jet_pt_debug] = [array('d', [0]) for i in range(2)]
	jet_pt_tree.Branch("gen_pt", gen_jet_pt_debug, "gen_pt/D")
	jet_pt_tree.Branch("obs_pt", obs_jet_pt_debug, "obs_pt/D")

	event_n[0] = 0
	for n in tqdm(range(args.nev)):
		if not pythia_hard.next():
				continue
		
		# count events
		event_n[0] += 1
		
		# find weight from yaml file for this pthat
		pthat = pythia_hard.info.pTHat()
		pt_hat_bin = len(pt_hat_lo) # 1-indexed
		for i in range(1,len(pt_hat_lo)):
			if pthat >= pt_hat_yaml[i] and pthat < pt_hat_yaml[i+1]:
				pt_hat_bin = i
				break
		pt_hat_weight[0] = pt_hat_yaml[pt_hat_bin]
		
		#======================================
		#            Particle level
		#======================================
		parts_pythia_p = pythiafjext.vectorize_select(pythia_hard, [pythiafjext.kFinal], 0, True)
		parts_pythia_p_selected = parts_selector(parts_pythia_p)

		# assign an event-level index to each particle (zero-indexed)
        # AND produce a second, smeared set of particles
		i = 0
		parts_pythia_p_smeared = fj.vectorPJ()
		for part in parts_pythia_p_selected:
			part.set_user_index(i)
			gen_pt[0] = part.perp()
            
			# smearing + track efficiency
			obs_pt[0] = -9999
			if args.tr_eff_off or do_keep_track(part):
				smeared_part = smear_track(part, 0.01)
				parts_pythia_p_smeared.push_back(smeared_part)
				obs_pt[0] = smeared_part.perp()
                
			particle_pt_tree.Fill()
			i += 1
            
		############################# TRUTH PAIRS ################################
		# truth level EEC pairs
		truth_pairs = []

		# truth jet reconstruction
		jets_p = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_p_selected)))

		for j in jets_p:
			jet_pt = j.perp() # jet pt

   			#push constutents to a vector in python
			_v = fj.vectorPJ()
			_ = [_v.push_back(c) for c in j.constituents()]

			# n-point correlator with all charged particles
			max_npoint = 2
			weight_power = 1
			dphi_cut = -9999
			deta_cut = -9999
			cb = ecorrel.CorrelatorBuilder(_v, jet_pt, max_npoint, weight_power, dphi_cut, deta_cut) # constructued for every jet

			EEC_cb = cb.correlator(2)

			EEC_weights = EEC_cb.weights() # cb.correlator(npoint).weights() constains list of weights
			EEC_rs = EEC_cb.rs() # cb.correlator(npoint).rs() contains list of RL
			EEC_indicies1 = EEC_cb.indices1() # contains list of 1st track in the pair (index should be based on the indices in _v)
			EEC_indicies2 = EEC_cb.indices2() # contains list of 2nd track in the pair

			for i in range(len(EEC_rs)):
				event_index1 = _v[int(EEC_indicies1[i])].user_index()
				event_index2 = _v[int(EEC_indicies2[i])].user_index()
				truth_pairs.append(EEC_pair(event_index1, event_index2, EEC_weights[i], EEC_rs[i], jet_pt))

		############################# SMEARED PAIRS ################################
		# smeared EEC pairs
		smeared_pairs = []

		# smeared jet reconstruction
		jets_p_smeared = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_p_smeared)))

		for j in jets_p_smeared:
			jet_pt = j.perp() # jet pt

   			#push constutents to a vector in python
			_v = fj.vectorPJ()
			_ = [_v.push_back(c) for c in j.constituents()]

			# n-point correlator with all charged particles
			max_npoint = 2
			weight_power = 1
			dphi_cut = -9999
			deta_cut = -9999
			cb = ecorrel.CorrelatorBuilder(_v, jet_pt, max_npoint, weight_power, dphi_cut, deta_cut) # constructued for every jet

			EEC_cb = cb.correlator(2)

			EEC_weights = EEC_cb.weights() # cb.correlator(npoint).weights() constains list of weights
			EEC_rs = EEC_cb.rs() # cb.correlator(npoint).rs() contains list of RL
			EEC_indicies1 = EEC_cb.indices1() # contains list of 1st track in the pair (index should be based on the indices in _v)
			EEC_indicies2 = EEC_cb.indices2() # contains list of 2nd track in the pair

			for i in range(len(EEC_rs)):
				event_index1 = _v[int(EEC_indicies1[i])].user_index()
				event_index2 = _v[int(EEC_indicies2[i])].user_index()
				smeared_pairs.append(EEC_pair(event_index1, event_index2, EEC_weights[i], EEC_rs[i], jet_pt))
				
		########################## TTree output generation #########################
		# composite of truth and smeared pairs, fill the TTree preprocessed
		for t_pair in truth_pairs:

			gen_energy_weight[0] = t_pair.weight
			gen_R_L[0] = t_pair.r
			gen_jet_pt[0] = t_pair.pt

			match_found = False
			for s_pair in smeared_pairs:
				if s_pair.is_equal(t_pair):
					obs_energy_weight[0] = s_pair.weight
					obs_R_L[0] = s_pair.r
					obs_jet_pt[0] = s_pair.pt
					preprocessed.Fill()
					match_found = True
					break
			if not match_found:
				obs_energy_weight[0] = dummyval
				obs_R_L[0] = dummyval
				obs_jet_pt[0] = dummyval
				preprocessed.Fill()

		# fill jet resolution debug ttree
		for s_jet in jets_p_smeared:
			for t_jet in jets_p:
				delta_R = np.sqrt( (s_jet.eta() - t_jet.eta())**2 + (s_jet.phi() - t_jet.phi())**2 )
				if delta_R <= 0.6:
					gen_jet_pt_debug[0] = t_jet.perp()
					obs_jet_pt_debug[0] = s_jet.perp()
					jet_pt_tree.Fill()
					break

	pythia_hard.stat()

	# write TTree to output file
	preprocessed.Write()
	preprocessed.Scan()
	particle_pt_tree.Write()
	particle_pt_tree.Scan()
	jet_pt_tree.Write()
	jet_pt_tree.Scan()

	# output file you want to write to
	fout.Write()
	fout.Close()
	print('[i] written ', fout.GetName())


if __name__ == '__main__':
	main()
