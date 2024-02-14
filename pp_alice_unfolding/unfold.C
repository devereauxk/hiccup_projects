#include <iostream>
#include <TFile.h>
#include <TH1.h>
#include <TH3.h>
#include <TSystem.h>
#include <TString.h>
#include </global/cfs/cdirs/alice/kdevero/RooUnfold/RooUnfold/build/RooUnfoldBayes.h>
#include </global/cfs/cdirs/alice/kdevero/RooUnfold/RooUnfold/build/RooUnfoldResponse.h>

// usage
// CLOSURE
// root -q "unfold.C(\"preunfold_closure_new.root\", \"unfolded_closure_new.root\", 8)"
// FULLSIM / DATA
// root -q "unfold.C(\"preunfold_fr.root\", \"unfolded_fr.root\", 8)"


void unfold(const TString& infile="preunfold.root", const TString& outfile="unfolded.root", int iter=9) {
    // Set ROOT to batch mod
    gROOT->SetBatch(true);

    // Error treatment for unfolding
    RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

    // Open inputs
    TFile* f = new TFile(infile);
    RooUnfoldResponse* response = (RooUnfoldResponse*) f->Get("reco_gen");
    RooUnfoldResponse* response1D = (RooUnfoldResponse*) f->Get("reco1D_gen1D");
    TH3D* h3_raw = (TH3D*) f->Get("raw");
    TH1D* h1_raw = (TH1D*) f->Get("raw1D");

    // TODO add in h1 unfolding

    // Create output file
    TFile* fout = new TFile(outfile, "RECREATE");

    for (int i = 1; i <= iter; ++i) {
        // Unfold the 3D histogram
        RooUnfoldBayes unfold(response, h3_raw, i);
        RooUnfoldBayes unfold1D(response1D, h1_raw, i);
        TH3* hunf = (TH3*)unfold.Hunfold(errorTreatment);
        TH1* hunf1D = (TH1*)unfold1D.Hunfold(errorTreatment);
        //TH3* hfold = (TH3*)response->ApplyToTruth(hunf, "");

        cout<<"unfolded"<<endl;

        // Clone and name histograms
        TH3* htempUnf = (TH3*)hunf->Clone(TString::Format("Baysian_Unfoldediter%d", i));
        TH1* htempUnf1D = (TH1*)hunf1D->Clone(TString::Format("Baysian_Unfolded1Diter%d", i));
        //TH3* htempFold = (TH3*)hfold->Clone(TString::Format("Baysian_Foldediter%d", i));

        // Write histograms to output file
        htempUnf->Write();
        htempUnf1D->Write();

        cout<<"written"<<endl;
    }

    // Write output file
    fout->Write();
    fout->Close();
}

int main(int argc, char* argv[]) {
    TString input_file = "preunfold.root";
    TString output_file = "unfolded.root";
    int iter = 9;

    // Parse command line arguments if provided
    if (argc > 1) {
        input_file = argv[1];
    }
    if (argc > 2) {
        output_file = argv[2];
    }
    if (argc > 3) {
        iter = std::atoi(argv[3]);
    }

    // Run unfolding
    unfold(input_file, output_file, iter);

    return 0;
}
