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
	#mycfg.append("HadronLevel:all = off") # so that we can look at the parton level easily
	#mycfg.append("StringPT:sigma = 0.1")
	# mycfg.append("TimeShower:pTmin = 2")
	# mycfg.append("HadronVertex:kappa = 5")
	# mycfg.append("StringFragmentation:stopMass = 2")
	# mycfg.append("StringZ:aLund = 0.1")
	ssettings = "--py-ecm 5020 --py-pthatmin 20"
	args = get_args_from_settings(ssettings)
	pythia_hard = pyconf.create_and_init_pythia_from_args(args, mycfg)
	
	max_eta_hadron = 0.9  # ALICE
	jet_R0 = 0.4
	max_eta_jet = max_eta_hadron - jet_R0
	parts_selector = fj.SelectorAbsEtaMax(max_eta_hadron)
	jet_selector = fj.SelectorPtMin(20) & fj.SelectorPtMax(40) & fj.SelectorAbsEtaMax(max_eta_jet) # original: 500, 550
	#pfc_selector0 = fj.SelectorPtMin(0.)
	#pfc_selector1 = fj.SelectorPtMin(1.)

	# print the banner first
	fj.ClusterSequence.print_banner()
	print()
	# set up our jet definition and a jet selector
	jet_def = fj.JetDefinition(fj.antikt_algorithm, jet_R0)
	print(jet_def)

	nbins = int(30.)
	lbins = logbins(1.e-3, 1., nbins)
	fout = ROOT.TFile(args.output, 'recreate')
	fout.cd()

	hec0_p = []
	hec1_p = []
	for i in range(1):
		h = ROOT.TH1D('hec0_p_{}'.format(i+2), 'hec0_p_{}'.format(i+2), nbins, lbins)
		h.Sumw2()
		hec0_p.append(h)
		h = ROOT.TH1D('hec1_p_{}'.format(i+2), 'hec1_p_{}'.format(i+2), nbins, lbins)
		h.Sumw2()
		hec1_p.append(h)
	hjpt_p = ROOT.TH1D('hjpt_p', 'hjpt_p', 600, 0, 600)

	hec0_ch = []
	hec1_ch = []
	for i in range(1):
		h = ROOT.TH1D('hec0_ch_{}'.format(i+2), 'hec0_ch_{}'.format(i+2), nbins, lbins)
		h.Sumw2()
		hec0_ch.append(h)
		h = ROOT.TH1D('hec1_ch_{}'.format(i+2), 'hec1_ch_{}'.format(i+2), nbins, lbins)
		h.Sumw2()
		hec1_ch.append(h)
	hjpt_h = ROOT.TH1D('hjpt_h', 'hjpt_h', 600, 0, 600)
	hz_h = ROOT.TH1D('hz_h', 'hz_h', 100, 0, 1) 
 
	for n in tqdm(range(args.nev)):
		if not pythia_hard.next():
				continue
		
		#======================================
		#            Parton level
		#======================================
		parts_pythia_p = pythiafjext.vectorize_select(pythia_hard, [pythiafjext.kFinal], 0, True)
		# print("Parton size:",len(parts_pythia_p))
		# for ip in range(len(parts_pythia_p)):
		# 	print("Parton info",pythiafjext.getPythia8Particle(parts_pythia_p[ip]).id(),pythiafjext.getPythia8Particle(parts_pythia_p[ip]).status(),pythiafjext.getPythia8Particle(parts_pythia_p[ip]).px(),pythiafjext.getPythia8Particle(parts_pythia_p[ip]).py(),pythiafjext.getPythia8Particle(parts_pythia_p[ip]).pz())
		parts_pythia_p_selected = parts_selector(parts_pythia_p)
		jets_p = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_p_selected))) 

		for j in jets_p:
			hjpt_p.Fill(j.perp())

   			# alternative: push constutents to a vector in python
			_v = fj.vectorPJ()
			_ = [_v.push_back(c) for c in j.constituents()]

			# n-point correlator with all charged particles
			max_npoint = 2
			weight_power = 1
			dphi_cut = -9999
			deta_cut = -9999
			cb = ecorrel.CorrelatorBuilder(_v, j.perp(), max_npoint, weight_power, dphi_cut, deta_cut)
   
   			# select particles with 1 GeV cut
			_v1 = fj.vectorPJ()
			_ = [_v1.push_back(c) for c in pfc_selector1(j.constituents())]

			# n-point correlator with charged particles pt > 1
			cb1 = ecorrel.CorrelatorBuilder(_v1, j.perp(), max_npoint, weight_power, dphi_cut, deta_cut)
   
			for i in range(1):
				if cb.correlator(i+2).rs().size() > 0:
					hec0_p[i].FillN(	cb.correlator(i+2).rs().size(), 
                   					array.array('d', cb.correlator(i+2).rs()), 
                     				array.array('d', cb.correlator(i+2).weights()))
				if cb1.correlator(i+2).rs().size() > 0:
					hec1_p[i].FillN(	cb1.correlator(i+2).rs().size(),
                   					array.array('d', cb1.correlator(i+2).rs()), 
                     				array.array('d', cb1.correlator(i+2).weights()))

		#======================================
		#            Hadron level
		#======================================
		hstatus = pythia_hard.forceHadronLevel()
		if not hstatus:
			continue
		
		# print(pythia_hard.event)

		parts_pythia_h = pythiafjext.vectorize_select(pythia_hard, [pythiafjext.kFinal], 0, True)
		# print("Particle size:",len(parts_pythia_h))
		# for ip in range(len(parts_pythia_h)):
		# 	print("Particle info",pythiafjext.getPythia8Particle(parts_pythia_h[ip]).id(),pythiafjext.getPythia8Particle(parts_pythia_h[ip]).status(),pythiafjext.getPythia8Particle(parts_pythia_h[ip]).px(),pythiafjext.getPythia8Particle(parts_pythia_h[ip]).py(),pythiafjext.getPythia8Particle(parts_pythia_h[ip]).pz())
		parts_pythia_h_selected = parts_selector(parts_pythia_h)

		jets_h = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_h_selected))) 
		if len(jets_h) < 1:
				continue

		# print('njets:', hjpt_h.Integral())
		
		for j in jets_h:
			hjpt_h.Fill(j.perp())
			# note: the EEC,... takes vector<PseudoJet> while PseudoJet::constituents() returns a tuple in python
			# so we use a helper function (in SWIG only basic types typles handled easily...)
			# vconstits = ecorrel.constituents_as_vector(j)
			# eecs_alt = ecorrel.EEC(vconstits, scale=j.perp())   
			# e3cs_alt = ecorrel.E3C(vconstits, scale=j.perp())
			# e4cs_alt = ecorrel.E4C(vconstits, scale=j.perp())

   			# alternative: push constutents to a vector in python
			_v = fj.vectorPJ()
			_ = [_v.push_back(c) for c in j.constituents()]

			# select only charged constituents
			_vc = fj.vectorPJ()
			_ = [_vc.push_back(c) for c in j.constituents()
                            if pythiafjext.getPythia8Particle(c).isCharged()]

			for c in _vc:
				delta_R = j.delta_R(c)
				z = c.perp()*math.cos(delta_R)/j.perp() # longitudinal
				hz_h.Fill(z)

			# n-point correlator with all charged particles
			max_npoint = 2
			weight_power = 1
			dphi_cut = -9999
			deta_cut = -9999
			cb = ecorrel.CorrelatorBuilder(_vc, j.perp(), max_npoint, weight_power, dphi_cut, deta_cut)
   
   			# select only charged constituents with 1 GeV cut
			_vc1 = fj.vectorPJ()
			_ = [_vc1.push_back(c) for c in pfc_selector1(j.constituents())
                            if pythiafjext.getPythia8Particle(c).isCharged()]
			# n-point correlator with charged particles pt > 1
			cb1 = ecorrel.CorrelatorBuilder(_vc1, j.perp(), max_npoint, weight_power, dphi_cut, deta_cut)
   
			for i in range(1):
				if cb.correlator(i+2).rs().size() > 0:
					hec0_ch[i].FillN(	cb.correlator(i+2).rs().size(), 
                   					array.array('d', cb.correlator(i+2).rs()), 
                     				array.array('d', cb.correlator(i+2).weights()))
				if cb1.correlator(i+2).rs().size() > 0:
					hec1_ch[i].FillN(	cb1.correlator(i+2).rs().size(),
                   					array.array('d', cb1.correlator(i+2).rs()), 
                     				array.array('d', cb1.correlator(i+2).weights()))

	njets = hjpt_h.Integral()
	if njets == 0:
		njets = 1.

	fout.cd()

	hjpt_p.Write()
	for hg in [hec0_p, hec1_p]:
		for i in range(1):
			hg[i].Sumw2()
			intg = hg[i].Integral()
			if intg > 0:
				hg[i].Scale(1./intg)
			if i > 0:
				fout.cd()
				hc = hg[i].Clone(hg[i].GetName() + '_ratio_to_EEC')
				hc.Sumw2()
				hc.Divide(hg[0])  
				hc.Write()
	
	hjpt_h.Write()
	hz_h.Write()
	for hg in [hec0_ch, hec1_ch]:
		for i in range(1):
			hg[i].Sumw2()
			intg = hg[i].Integral()
			if intg > 0:
				hg[i].Scale(1./intg)
			if i > 0:
				fout.cd()
				hc = hg[i].Clone(hg[i].GetName() + '_ratio_to_EEC')
				hc.Sumw2()
				hc.Divide(hg[0])  
				hc.Write()

	pythia_hard.stat()

	fout.Write()
	fout.Close()
	print('[i] written ', fout.GetName())


if __name__ == '__main__':
	main()
