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


def smear_track(part, sigma=0.01):
	smeared_pt = part.perp() * (1 + np.random.normal(0, sigma))
	smeared_pz = part.pz() * (1 + np.random.normal(0, sigma))
	smeared_E = np.sqrt(part.m2() + smeared_pt**2 + smeared_pz**2)
	smeared_eta = 0.5 * np.log((smeared_E + smeared_pz) / (smeared_E - smeared_pz))
	part.reset_PtYPhiM(smeared_pt, smeared_eta, part.phi(), smeared_E)


def get_args_from_settings(ssettings):
	sys.argv = sys.argv + ssettings.split()
	parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly')
	pyconf.add_standard_pythia_args(parser)
	parser.add_argument('--output', default="test_ecorrel_rw.root", type=str)
	parser.add_argument('--user-seed', help='pythia seed',
											default=1111, type=int)
	args = parser.parse_args()
	return args


def main():
	mycfg = []
	ssettings = "--py-ecm 5020 --py-pthatmin 20"
	args = get_args_from_settings(ssettings)
	pythia_hard = pyconf.create_and_init_pythia_from_args(args, mycfg)
	
	max_eta_hadron = 0.9  # ALICE
	jet_R0 = 0.4
	max_eta_jet = max_eta_hadron - jet_R0
	parts_selector = fj.SelectorAbsEtaMax(max_eta_hadron)
	jet_selector = fj.SelectorPtMin(20) & fj.SelectorPtMax(40) & fj.SelectorAbsEtaMax(max_eta_jet) 
	pfc_selector0 = fj.SelectorPtMin(0.)
	pfc_selector1 = fj.SelectorPtMin(1.)

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

	preprocessed.Branch("gen_energy_weight", gen_energy_weight, "gen_energy_weight/D")
	preprocessed.Branch("gen_R_L", gen_R_L, "gen_R_L/D")
	preprocessed.Branch("gen_jet_pt", gen_jet_pt, "gen_jet_pt/D")
	preprocessed.Branch("obs_energy_weight", obs_energy_weight, "obs_energy_weight/D")
	preprocessed.Branch("obs_R_L", obs_R_L, "obs_R_L/D")
	preprocessed.Branch("obs_jet_pt", obs_jet_pt, "obs_jet_pt/D")

 
	for n in tqdm(range(args.nev)):
		if not pythia_hard.next():
				continue
		
		#======================================
		#            Particle level
		#======================================
		parts_pythia_p = pythiafjext.vectorize_select(pythia_hard, [pythiafjext.kFinal], 0, True)
		parts_pythia_p_selected = parts_selector(parts_pythia_p)

		# assign an event-level index to each particle (zero-indexed)
		for i in range(len(parts_pythia_p_selected)):
			parts_pythia_p_selected[i].set_user_index(i)

		# produce a second, smeared set of particle
		parts_pythia_p_smeared = []
		for part in parts_pythia_p_selected:
			parts_pythia_p_smeared.append(smear_track(part, 0.01))

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

			print(EEC_indicies1)
			for i in range(len(EEC_rs)):
				event_index1 = _v[EEC_indicies1[i]].user_index()
				event_index2 = _v[EEC_indicies2[i]].user_index()
				truth_pairs.append(EEC_pair(event_index1, event_index2, EEC_weights[i], EEC_rs[i], jet_pt))

		############################# SMEARED PAIRS ################################
		# smeared EEC pairs
		smeared_pairs = truth_pairs

		"""
		# smeared jet reconstruction
		jets_p = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_p_smeared)))

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
				event_index1 = _v[EEC_indicies1[i]].user_index()
				event_index2 = _v[EEC_indicies2[i]].user_index()
				smeared_pairs.append(EEC_pair(event_index1, event_index2, EEC_weights[i], EEC_rs[i], jet_pt))
			"""
				
		########################## TTree output generation #########################
		# composite of truth and smeared pairs, fill the TTree preprocessed
		for s_pair in smeared_pairs:
			for t_pair in truth_pairs:
				if s_pair.is_equal(t_pair):
					gen_energy_weight[0] = t_pair.weight
					gen_R_L[0] = t_pair.r
					gen_jet_pt[0] = t_pair.pt 
					obs_energy_weight[0] = s_pair.weight
					obs_R_L[0] = s_pair.r
					obs_jet_pt[0] = s_pair.pt
					preprocessed.Fill()
					break

	pythia_hard.stat()

	# write TTree to output file
	preprocessed.Write()

	# output file you want to write to
	fout.Write()
	fout.Close()
	print('[i] written ', fout.GetName())


if __name__ == '__main__':
	main()
