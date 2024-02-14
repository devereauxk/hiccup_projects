# WANT TO BE ABLE TO WRITE SMTH GENERALIZABLE TO OTHER THINGS
# DO THIS BY GENERALIZING THE UNFOLD FILE, THIS ONE HAS TO BE EEC-SPECIFIC


# DOES NOT WORK ON PERLY !!!!!!!!!!!!!! weird roounfold dependencies ... no 3D support
# usage - run with the "roounfold_load" environment
# CLOSURE
# python3 unfold.py --input_file preunfold.root --output_file unfolded.root --iter 3


# imports
import numpy as np
import pandas as pd
import uproot as ur
import ROOT
import argparse
ROOT.gSystem.Load("libRooUnfold.so")

ROOT.gROOT.SetBatch(True)

def unfold(infile="preunfold.root", outfile="unfolded.root", iter=9):

    errorTreatment = ROOT.RooUnfold.kCovariance

    response = ur.open("%s:reco_gen"%(infile))
    response1D = ur.open("%s:reco1D_gen1D"%(infile))

    h3_raw = ur.open("%s:raw"%(infile))
    h1_raw = ur.open("%s:raw1D"%(infile))

    fout = ROOT.TFile(outfile, "RECREATE")

    for i in range(1,iter+1):
        # unfold the 3D histogram
        unfold = ROOT.RooUnfoldBayes(response, h3_raw, i)
        hunf = unfold.Reco(errorTreatment)
        hfold = response.ApplyToTruth(hunf, "")

        htempUnf = hunf.Clone("htempUnf")
        htempUnf.SetName("Baysian_Unfoldediter{}".format(i))

        htempFold = hfold.Clone("htempFold")
        htempFold.SetName("Baysian_Foldediter{}".format(i))

        htempUnf.Write()
        htempFold.Write()

    fout.Write()
    fout.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', default='preunfold.root')
    parser.add_argument('--output_file', default='unfolded.root')
    parser.add_argument('--iter', type=int, default=9)

    flags = parser.parse_args()

    unfold(flags.input_file, flags.output_file, flags.iter)

