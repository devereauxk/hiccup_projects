#!/usr/bin/env python3
# python3 pythia8_rho.py --nev 100000 --no_smear --R0 0.4 --py-pthatmin 100 --jet_ptmin 100 --jet_ptmax 120 --output "pt100_R0p4_temptemp2.root"

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

# load track efficiency tree
tr_eff_file = ur.open("tr_eff.root")
tr_eff = tr_eff_file["tr_eff"].to_numpy()

def get_args_from_settings(ssettings):
	sys.argv = sys.argv + ssettings.split()
	parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly')
	pyconf.add_standard_pythia_args(parser)
	parser.add_argument('--output', default="test_ecorrel_rw.root", type=str)
	parser.add_argument('--user-seed', help='pythia seed', default=1111, type=int)
	parser.add_argument('--tr_eff_off', action='store_true', default=False)
	parser.add_argument('--no_smear', action='store_false', default=True)
	parser.add_argument('--R0', default=0.4, type=float)
	parser.add_argument('--jet_ptmin', default=100, type=float)
	parser.add_argument('--jet_ptmax', default=120, type=float)
	args = parser.parse_args()
	return args


def main():
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
	
	nbins = int(50.)
	lbins = logbins(1.e-3, 1., nbins)

	h_jetpt = ROOT.TH1D('h_jetpt', 'h_jetpt', 232, 20, 1200)
	h_trkpt = ROOT.TH1D('h_trkpt', 'h_trkpt', 200, 0, 20)
	h_jetshape = ROOT.TH3D('h_jetshape', 'h_jetshape', 25, linbins(0,jet_R0, 25), 80, linbins(0,20,80), nbins, lbins)
	#h_jetshape = ROOT.TH1D('h_jetshape', 'h_jetshape', 25, 0, jet_R0)
	h_ptprofile = ROOT.TH1D('h_ptprofile', 'h_ptprofile', nbins, lbins)
	h_jetshape.Sumw2()

    # JETSCAPE OUTPUT EVENT-BY-EVENT I/O
	input_file_hadrons = "/rstorage/jetscape/JETSCAPE-AA-events/skim/497764/v3/5020_PbPb_0-5_0.30_2.0_1/JetscapeHadronListBin9_11_01"
	df_event_chunk = pd.read_parquet(input_file_hadrons)
    n_event_max = df_event_chunk.shape[0]
	
	for i,event in df_event_chunk.iterrows():
		print(i, event)
		
		exit()
		
		#======================================
		#            Particle level
		#======================================
		parts_pythia_p = pythiafjext.vectorize_select(pythia_hard, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)
		parts_pythia_p_selected = parts_selector(parts_pythia_p)
            
		############################# PAIR CONSTITUENTS AND JET AXIS ################################
		# jet reconstruction
		jets_p = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_p_selected)))

		for j in jets_p:
			jet_pt = j.perp() # jet pt
			
			h_jetpt.Fill(jet_pt)
			
			# fill histogram
			for c in j.constituents():
				h_jetshape.Fill(j.delta_R(c), c.perp(), c.perp()/j.perp())
				#h_jetshape.Fill(j.delta_R(c))
				h_trkpt.Fill(c.perp())
				h_ptprofile.Fill(c.perp()/j.perp())
        #################################################################################

	pythia_hard.stat()

	# write histograms to output file
	intg = h_jetshape.GetEntries()
	if intg > 0:
		h_jetshape.Scale(1/intg)
	h_jetshape.Write()
	h_jetpt.Write()
	h_trkpt.Write()
	h_ptprofile.Write()

	# output file you want to write to
	fout.Write()
	fout.Close()
	print('[i] written ', fout.GetName())


if __name__ == '__main__':
	main()
