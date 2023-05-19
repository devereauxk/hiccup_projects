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
import array


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
 
	for n in tqdm(range(args.nev)):
		if not pythia_hard.next():
				continue
		
		#======================================
		#            Particle level
		#======================================
		parts_pythia_p = pythiafjext.vectorize_select(pythia_hard, [pythiafjext.kFinal], 0, True)
		parts_pythia_p_selected = parts_selector(parts_pythia_p)
		jets_p = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_p_selected))) 

		for j in jets_p:
			# j.perp() return jet pt

   			# alternative: push constutents to a vector in python
			_v = fj.vectorPJ()
			_ = [_v.push_back(c) for c in j.constituents()]

			# n-point correlator with all charged particles
			max_npoint = 2
			weight_power = 1
			dphi_cut = -9999
			deta_cut = -9999
			cb = ecorrel.CorrelatorBuilder(_v, j.perp(), max_npoint, weight_power, dphi_cut, deta_cut) # constructued for every jet
   
			npoint = 2 # FIX ME: double check what npoint to set to access the two point correlators
			if cb.correlator(npoint).rs().size() > 0:
				# cb.correlator(npoint).rs() contains list of RL
				# cb.correlator(npoint).weights() constains list of weights
				# j.perp() is jet pt
				# corr_builder.correlator(ipoint).indices1() contains list of 1st track in the pair (index should be based on the indices in _v)
				# corr_builder.correlator(ipoint).indices2() contains list of 2nd track in the pair


	pythia_hard.stat()

	# output file you want to write to
	fout.Write()
	fout.Close()
	print('[i] written ', fout.GetName())


if __name__ == '__main__':
	main()
