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
// root -q "unfold.C(\"preunfold_closure_new.root\", \"unfolded_closure_new.root\", 5)"
// CLOSURE WITH FULLSIM
// root -q "unfold.C(\"preunfold_closurefullsim.root\", \"unfolded_closurefullsim.root\", 5)"
// FULLSIM / DATA
// root -q "unfold.C(\"preunfold_closurefullsim.root\", \"preunfold_fr.root\", \"unfolded_fr.root\", 5)"


void unfold(const TString& rm_file="preunfold.root", const TString& data_file="preunfold_data.root", const TString& outfile="unfolded.root", int iter=9) {
    // Set ROOT to batch mod
    gROOT->SetBatch(true);

    // Error treatment for unfolding
    RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

    // Open inputs
    TFile* f = new TFile(rm_file);
    RooUnfoldResponse* response = (RooUnfoldResponse*) f->Get("reco_gen");
    RooUnfoldResponse* response1D = (RooUnfoldResponse*) f->Get("reco1D_gen1D");
    TFile* f_data = new TFile(data_file);
    TH3D* h3_raw = (TH3D*) f_data->Get("raw");
    TH1D* h1_raw = (TH1D*) f_data->Get("raw1D");
    // do not close these, if think its because the program reads directly from the files and does NOT copy it into memory

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

    // Make TH2Ds out of response matricies, add to output file
    TH2D* matrix = (TH2D*) response->Hresponse();
    matrix->SetName("matrix");
    matrix->Write();

    TH2D* matrix1D = (TH2D*) response1D->Hresponse();
    matrix1D->SetName("matrix1D");
    matrix1D->Write();

    // Write output file
    fout->Write();
    fout->Close();
}

int main(int argc, char* argv[]) {
    TString rm_file = "preunfold.root";
    TString data_file = rm_file;
    TString output_file = "unfolded.root";
    int iter = 5;

    // Parse command line arguments if provided
    if (argc > 1) {
        rm_file = argv[1];
    }
    if (argc > 2) {
        data_file = argv[2];
    }
    if (argc > 3) {
        output_file = argv[3];
    }
    if (argc > 4) {
        iter = std::atoi(argv[4]);
    }

    // Run unfolding
    unfold(rm_file, data_file, output_file, iter);

    return 0;
}
