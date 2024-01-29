#include <iostream>
#include <TFile.h>
#include <TH1.h>
#include <TH3.h>
#include <TSystem.h>
#include <TString.h>
//#include <RooUnfoldBayes.h>
//#include <RooUnfoldResponse.h>

gSystem->Load("libRooUnfold");

// usage
// root -q "unfold.C(\"preunfold.root\", \"unfolded.root\", 3)"

void unfold(const TString& infile="preunfold.root", const TString& outfile="unfolded.root", int iter=9) {
    // Set ROOT to batch mode
    gROOT->SetBatch(true);

    // Load the RooUnfold library
    //gSystem->Load("libRooUnfold");

    // Error treatment for unfolding
    RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

    // Open input files
    TFile* responseFile = TFile::Open(infile + ":reco_gen");
    TFile* response1DFile = TFile::Open(infile + ":reco1D_gen1D");
    TFile* h3_rawFile = TFile::Open(infile + ":raw");
    TFile* h1_rawFile = TFile::Open(infile + ":raw1D");

    // Create output file
    TFile* fout = new TFile(outfile, "RECREATE");

    for (int i = 1; i <= iter; ++i) {
        // Unfold the 3D histogram
        RooUnfoldResponse* response = (RooUnfoldResponse*)responseFile->Get("response");
        TH3* h3_raw = (TH3*)h3_rawFile->Get("h3_raw");
        RooUnfoldBayes unfold(response, h3_raw, i);
        TH3* hunf = (TH3*)unfold.Hreco(errorTreatment);
        TH3* hfold = (TH3*)response->ApplyToTruth(hunf, "");

        // Clone and name histograms
        TH3* htempUnf = (TH3*)hunf->Clone(TString::Format("Baysian_Unfoldediter%d", i));
        TH3* htempFold = (TH3*)hfold->Clone(TString::Format("Baysian_Foldediter%d", i));

        // Write histograms to output file
        htempUnf->Write();
        htempFold->Write();
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
