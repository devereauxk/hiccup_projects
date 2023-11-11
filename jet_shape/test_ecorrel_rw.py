#!/usr/bin/env python3
# python3 test_ecorrel_rw.py --nev 100 --output "test_preprocess.root" > log.txt

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
    
def get_delta_R(pj1, pj2):
	return np.sqrt( (pj1.eta() - pj2.eta())**2 + (pj1.phi() - pj2.phi())**2)

def get_args_from_settings(ssettings):
	sys.argv = sys.argv + ssettings.split()
	parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly')
	pyconf.add_standard_pythia_args(parser)
	parser.add_argument('--output', default="test_ecorrel_rw.root", type=str)
	parser.add_argument('--user-seed', help='pythia seed', default=1111, type=int)
	parser.add_argument('--tr_eff_off', action='store_true', default=False)
	parser.add_argument('--no_smear', action='store_false', default=True)
	args = parser.parse_args()
	return args


def main():
	mycfg = []
	mycfg.append("StringPT:sigma=0.335") # for producing slightly different data sets, default is 0.335 GeV
	ssettings = "--py-ecm 5020 --py-pthatmin 20"
	args = get_args_from_settings(ssettings)
	pythia_hard = pyconf.create_and_init_pythia_from_args(args, mycfg)
	
	max_eta_hadron = 0.9  # ALICE
	jet_R0 = 0.4
	max_eta_jet = max_eta_hadron - jet_R0
	parts_selector = fj.SelectorPtMin(0.15) & fj.SelectorAbsEtaMax(max_eta_hadron)
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

	# output histogram defintions
	nbins = 30
	lbins = logbins(1.e-3, 1, nbins)
    
	n_pt_bins = 4
	jet_pt_lo = [20, 25, 30, 20]
	jet_pt_hi = [25, 30, 40, 40]
    
	h_jetshape_pt = []
	for i in range(len(jet_pt_lo)):
		h_jetshape = ROOT.TH1D('h_jetshape_pt_{}'.format(i), 'h_jetshape_pt_{}'.format(i), nbins, lbins)
		h_jetshape.Sumw2()
		h_jetshape_pt.append(h_jetshape)
    

    # PYTHIA EVENT-BY-EVENT GENERATION
	for n in tqdm(range(args.nev)):
		if not pythia_hard.next():
				continue
		
		# find weight from yaml file for this pthat
		pthat = pythia_hard.info.pTHat()
		pt_hat_bin = len(pt_hat_lo) # 1-indexed
		for i in range(1,len(pt_hat_lo)):
			if pthat >= pt_hat_yaml[i] and pthat < pt_hat_yaml[i+1]:
				pt_hat_bin = i
				break
		
		#======================================
		#            Particle level
		#======================================
		parts_pythia_p = pythiafjext.vectorize_select(pythia_hard, [pythiafjext.kFinal], 0, True)
        
        # add in charged particle selector at fj level
		charged_selector = fj.SelectorIsCharged()
        
		parts_pythia_p_selected = charged_selector(parts_selector(parts_pythia_p))
        
		# apply realistic ALICE detector effects (smearing) if no_smear is not true
		if not args.no_smear:
			parts_pythia_p_smeared = fj.vectorPJ()
			for part in parts_pythia_p_selected:
                # smearing + track efficiency
				if args.tr_eff_off or do_keep_track(part):
					smeared_part = smear_track(part, 0.01)
					parts_pythia_p_smeared.push_back(smeared_part)
			
			parts_pythia_p_selected = parts_pythia_p_smeared
            
		############################# PAIR CONSTITUENTS AND JET AXIS ################################
		# jet reconstruction
		jets_p = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_p_selected)))

		for j in jets_p:
			jet_pt = j.perp() # jet pt

   			# push constutents to a vector in python
			_v = fj.vectorPJ()
			_ = [_v.push_back(c) for c in j.constituents()]
			delta_Rs = [get_delta_R(const, j) for const in _v]
            
            # fill histograms
			for i in n_pt_bins:
				if jet_pt_lo[i] < jet_pt and jet_pt < jet_pt_hi[i]:
					for delta_R in delta_Rs:
						h_jetshape_pt[i].Fill(delta_R)
                        
        #################################################################################

	pythia_hard.stat()

	# write histograms to output file
	for h in h_jetshape_pt:
		intg = h.GetEntries()
		h.Scale(1/intg)
		h.Write()

	# output file you want to write to
	fout.Write()
	fout.Close()
	print('[i] written ', fout.GetName())


if __name__ == '__main__':
	main()
