#!/usr/bin/env python3
# python3 analyze_events.py --R0 0.4 --py-pthatmin 100 --jet_ptmin 100 --jet_ptmax 110 --input <input> --output <output>

from pyjetty.mputils.mputils import logbins, linbins
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
from tqdm import tqdm
import argparse
import os
from array import array
import numpy as np
import uproot as ur
import yaml
import pandas as pd
import particle

def get_args_from_settings(ssettings):
	sys.argv = sys.argv + ssettings.split()
	parser = argparse.ArgumentParser(description='jetscape jet shape analysis with fastjet on the fly')
	pyconf.add_standard_pythia_args(parser)
	parser.add_argument('--input', default="/global/cfs/cdirs/alice/kdevero/jetscape_hiccup/5020_PP_Colorless/JetscapeHadronListBin100_110_01", type=str)
	parser.add_argument('--output', default="test.root", type=str)
	parser.add_argument('--R0', default=0.4, type=float)
	parser.add_argument('--jet_ptmin', default=100, type=float)
	parser.add_argument('--jet_ptmax', default=120, type=float)
	args = parser.parse_args()
	return args


def main():
	args = get_args_from_settings("")
	
	max_eta_hadron = 0.9  # ALICE
	jet_R0 = args.R0   # R=0.4 standard
	max_eta_jet = max_eta_hadron - jet_R0
	parts_selector = fj.SelectorPtMin(0.15) & fj.SelectorAbsEtaMax(max_eta_hadron)
	
	jet_ptmin = args.jet_ptmin
	jet_ptmax = args.jet_ptmax
	jet_selector = fj.SelectorAbsEtaMax(max_eta_jet) & fj.SelectorPtMin(jet_ptmin) & fj.SelectorPtMax(jet_ptmax)

	# print the banner first
	fj.ClusterSequence.print_banner()
	print()
	# set up our jet definition and a jet selector
	jet_def = fj.JetDefinition(fj.antikt_algorithm, jet_R0)
	print(jet_def)

	fout = ROOT.TFile(args.output, 'recreate')
	fout.cd()
	
	nbins = int(80.)
	lbins = logbins(1.e-5, 1., nbins)
	n_rlbins = 25
	rlbins = linbins(0,jet_R0,25)

	h_jetpt = ROOT.TH1D('h_jetpt', 'h_jetpt', 232, 20, 1200)
	h_trkpt = ROOT.TH1D('h_trkpt', 'h_trkpt', 200, 0, 80)
	h_jetshape = ROOT.TH3D('h_jetshape', 'h_jetshape_rl_pTtrk_z', n_rlbins, rlbins, 200, linbins(0,80,200), nbins, lbins)
	h_ptprofile = ROOT.TH1D('h_ptprofile', 'h_ptprofile', nbins, lbins)
	h_ptshape = ROOT.TH3D('h_ptshape', 'h_ptshape_rl_pTtrk_z', n_rlbins, rlbins, 200, linbins(0,80,200), nbins, lbins)
	
	h_jetshape.Sumw2()
	h_ptshape.Sumw2()

    # JETSCAPE OUTPUT EVENT-BY-EVENT I/O
	input_file_hadrons = args.input
	df_event_chunk = pd.read_parquet(input_file_hadrons)
	n_event_max = df_event_chunk.shape[0]
	
	for i,event in df_event_chunk.iterrows():
		if i % (n_event_max / 20) == 0:
			print("event " + str(i) + " / " + str(n_event_max))
		
		#======================================
		#            Particle level
		#======================================
		
		particle_ID = event['particle_ID']
		status = event['status']
		E = event['E'].tolist()
		px = event['px'].tolist()
		py = event['py'].tolist()
		pz = event['pz'].tolist()
		n_parts = len(particle_ID)
		
		### obtain list of pj objects containing charged final state particles in the event
		
		fjparts = fj.vectorPJ()
		for part_i in range(n_parts):
			if particle.Particle.from_pdgid(particle_ID[part_i]).charge != 0:
				psj = fj.PseudoJet(px[part_i], py[part_i], pz[part_i], E[part_i])
				fjparts.push_back(psj)
            
		############################# PAIR CONSTITUENTS AND JET AXIS ################################
		# jet reconstruction
		jets_p = fj.sorted_by_pt(jet_selector(jet_def(parts_selector(fjparts))))

		for j in jets_p:
			jet_pt = j.perp() # jet pt
			
			h_jetpt.Fill(jet_pt)
			
			# fill histogram
			for c in j.constituents():
				rl = j.delta_R(c)
				h_jetshape.Fill(rl, c.perp(), c.perp()/j.perp())
				h_trkpt.Fill(c.perp())
				h_ptprofile.Fill(c.perp()/j.perp())
				
				h_ptshape.Fill(rl, c.perp(), c.perp()/j.perp(), c.perp())
        #################################################################################

	# write histograms to output file
	h_jetshape.Write()
	h_jetpt.Write()
	h_trkpt.Write()
	h_ptprofile.Write()
	h_ptshape.Write()

	# output file you want to write to
	fout.Write()
	fout.Close()
	print('[i] written ', fout.GetName())


if __name__ == '__main__':
	main()
