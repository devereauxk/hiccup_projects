#!/usr/bin/env python3
# python3 pythia8_rho.py --nev 1000000 --no_smear --jetR 0.4 --py-ecm 5020 --py-pthatmin 5 --jet_ptmin 5 --jet_ptmax 120 --output "pt5_R0p4_s5p02.root"

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
import array

# load track efficiency tree
"""
tr_eff_file = ur.open("tr_eff.root")
tr_eff = tr_eff_file["tr_eff"].to_numpy()
"""

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

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

def get_args_from_settings():
	parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly')
	pyconf.add_standard_pythia_args(parser)
	#parser.add_argument('--py-ecm', default=14000, type=float)
	parser.add_argument('--output', default="test_ecorrel_rw.root", type=str)
	parser.add_argument('--user-seed', help='pythia seed', default=1111, type=int)
	parser.add_argument('--tr_eff_off', action='store_true', default=False)
	parser.add_argument('--no_smear', action='store_false', default=True)
	parser.add_argument('--jetR', default=0.4, type=float)
	parser.add_argument('--jet_ptmin', default=100, type=float)
	parser.add_argument('--jet_ptmax', default=120, type=float)
	args = parser.parse_args()
	return args


class Process_Pythia_JetTrk:

	def __init__(self):
		self.observable_list = ['jet_pt_JetPt', 'trk_pt_TrkPt', 'jet-trk_shape_RL_TrkPt_JetPt', 'jet-trk_shape_RL_z_JetPt', 
    'jet-trk_ptprofile_RL_TrkPt_JetPt', 'jet-trk_ptprofile_RL_z_JetPt']
		
		self.args = get_args_from_settings()
		self.jetR = self.args.jetR
		self.max_eta_hadron = 0.9 # ALICE

		self.initialize_user_output_objects()

	def initialize_user_output_objects(self):
		# histogram definitions
		jetR = self.jetR
		obs_label = "0.15"

		for observable in self.observable_list:
			if observable == 'jet_pt_JetPt':
				name = 'h_{}_R{}_{}'.format(observable, jetR, obs_label)
				jetpt_bins = linbins(0,200,200)
				h = ROOT.TH1D(name, name, 200, jetpt_bins)
				h.GetXaxis().SetTitle('p_{T jet}')
				h.GetYaxis().SetTitle('Counts')
				setattr(self, name, h)

			if observable == "trk_pt_TrkPt":
				name = 'h_{}_R{}_{}'.format(observable, jetR, obs_label)
				trkpt_bins = linbins(0,20,200)
				h = ROOT.TH1D(name, name, 200, trkpt_bins)
				h.GetXaxis().SetTitle('p_{T,ch trk}')
				h.GetYaxis().SetTitle('Counts')
				setattr(self, name, h)

			if "_RL_TrkPt_JetPt" in observable:
				name = 'h_{}_R{}_{}'.format(observable, jetR, obs_label)
				RL_bins = linbins(0,jetR,50)
				trkpt_bins = linbins(0,20,200)
				jetpt_bins = linbins(0,200,200)
				h = ROOT.TH3D(name, name, 50, RL_bins, 200, trkpt_bins, 200, jetpt_bins)
				h.GetXaxis().SetTitle('#Delta R')
				h.GetYaxis().SetTitle('p_{T,ch trk}')
				h.GetZaxis().SetTitle('p_{T jet}')
				setattr(self, name, h)

			if "_RL_z_JetPt" in observable:
				name = 'h_{}_R{}_{}'.format(observable, jetR, obs_label)
				RL_bins = linbins(0,jetR,50)
				z_bins = logbins(1.e-5, 1., 200)
				jetpt_bins = linbins(0,200,200)
				h = ROOT.TH3D(name, name, 50, RL_bins, 200, z_bins, 200, jetpt_bins)
				h.GetXaxis().SetTitle('#Delta R')
				h.GetYaxis().SetTitle('z')
				h.GetZaxis().SetTitle('p_{T jet}')
				setattr(self, name, h)

	def process(self):
		mycfg = []
		mycfg.append("StringPT:sigma=0.335") # for producing slightly different data sets, default is 0.335 GeV
		pythia_hard = pyconf.create_and_init_pythia_from_args(self.args, mycfg)
		
		jetR = self.jetR
		obs_label = "0.15"

		max_eta_jet = self.max_eta_hadron - jetR
		parts_selector = fj.SelectorPtMin(0.15) & fj.SelectorAbsEtaMax(self.max_eta_hadron)
		
		jet_ptmin = self.args.jet_ptmin
		jet_ptmax = self.args.jet_ptmax
		jet_selector = fj.SelectorAbsEtaMax(max_eta_jet) & fj.SelectorPtMin(jet_ptmin) & fj.SelectorPtMax(jet_ptmax)

		# print the banner first
		fj.ClusterSequence.print_banner()
		print()
		# set up our jet definition and a jet selector
		jet_def = fj.JetDefinition(fj.antikt_algorithm, self.jetR)
		print(jet_def)

		fout = ROOT.TFile(self.args.output, 'recreate')
		fout.cd()

		# PYTHIA EVENT-BY-EVENT GENERATION
		for n in tqdm(range(self.args.nev)):
			if not pythia_hard.next():
					continue
			
			#======================================
			#            Particle level
			#======================================
			parts_pythia_p = pythiafjext.vectorize_select(pythia_hard, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)
			parts_pythia_p_selected = parts_selector(parts_pythia_p)

			############################# PAIR CONSTITUENTS AND JET AXIS ################################
			# jet reconstruction
			jets_p = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_p_selected)))

			for jet in jets_p:

				c_select = fj.sorted_by_pt(jet.constituents())
				
				# fill histograms
				hname = 'h_{}_R{}_{}'
				for observable in self.observable_list:
					
					if observable == 'jet_pt_JetPt':
						print(hname.format(observable, jetR, obs_label))
						getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet.perp())

					if observable == "trk_pt_TrkPt":
						for c in c_select:
							getattr(self, hname.format(observable, jetR, obs_label)).Fill(c.perp())
							
					if 'jet-trk' in observable:
						h = getattr(self, hname.format(observable, jetR, obs_label))
						
						for c in c_select:
							rl = jet.delta_R(c)

							if observable == "jet-trk_shape_RL_TrkPt_JetPt":
								h.Fill(rl, c.perp(), jet.perp())
							elif observable == "jet-trk_ptprofile_RL_TrkPt_JetPt":
								h.Fill(rl, c.perp(), jet.perp(), c.perp())
							elif observable == "jet-trk_shape_RL_z_JetPt":
								h.Fill(rl, c.perp()/jet.perp(), jet.perp())
							elif observable == "jet-trk_ptprofile_RL_z_JetPt":
								h.Fill(rl, c.perp()/jet.perp(), jet.perp(), c.perp())

			#################################################################################

		pythia_hard.stat()

		for attr in dir(self):
			obj = getattr(self, attr)
        
			#print(str(attr) + " " + str(type(obj)))

			# Write all ROOT histograms and trees to file
			types = (ROOT.TH1, ROOT.THnBase, ROOT.TTree, ROOT.TH2, ROOT.TH3)
			if isinstance(obj, types):
				obj.Write()

		# output file you want to write to
		fout.Write()
		fout.Close()
		print('[i] written ', fout.GetName())


if __name__ == '__main__':
	process_obj = Process_Pythia_JetTrk()
	process_obj.process()
