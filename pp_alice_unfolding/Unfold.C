#if !(defined(__CINT__) || defined(__CLING__)) || defined(__ACLIC__)
#include "RooUnfoldResponse.h"
#include "RooUnfoldBayes.h"
#include "sstream"
#endif

void readin_data(TString preprocessed_file="preprocessed_mc.root")
{
  
}

void Unfold(std::string file_mc, std::string file_data, std::string date)
{
  RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

  TFile* f_mc = new TFile(file_mc.c_str());  
  RooUnfoldResponse* response = (RooUnfoldResponse*)f_mc->Get("smeared_true");
  RooUnfoldResponse* response1D = (RooUnfoldResponse*)f_mc->Get("h1_smeared_h1_true");

  TFile* f_data = new TFile(file_data.c_str());
  TH3D* h3_raw = (TH3D*)f_data->Get("raw");
  
  std::stringstream ss;
  ss << "Unfold_nom_" << date << ".root";
  TFile* fout = new TFile(ss.str().c_str(), "recreate");
  ss.str("");

  h3_raw->Write();
  
  /* TODO 
  TH3D* h3_purity = (TH3D*)h3_reco_match->Clone("h3_purity");
  h3_purity->Divide(h3_reco);
  h3_purity->Write();
  */
  TH3D* h3_raw_corr = (TH3D*)h3_raw->Clone("h3_raw_corr");
  // h3_raw_corr->Multiply(h3_purity);
  h3_raw_corr->Write();

  for(int jar=1;jar<10;jar++){
    Int_t iter=jar;
    cout<<"iteration"<<iter<<endl;
    cout<<"==============Unfold====================="<<endl;
    //unfold the 3D splitting histogram
    RooUnfoldBayes   unfold(response, h3_raw_corr, iter);    // OR
    TH3D* hunf= (TH3D*) unfold.Hreco(errorTreatment);
    //FOLD BACK
    TH1* hfold = response->ApplyToTruth (hunf, "");

    //unfold the 1D histogram for per jet normalization
    RooUnfoldBayes   unfold1D(response1D, h1_raw, iter);    // OR
    TH1D* hunf1D= (TH1D*) unfold1D.Hreco(errorTreatment);
    //FOLD BACK
    TH1* hfold1D = response1D->ApplyToTruth (hunf1D, "");
    
    TH3D *htempUnf=(TH3D*)hunf->Clone("htempUnf");
    htempUnf->SetName(Form("Bayesian_Unfoldediter%d",iter));

    TH1D *htempUnf1D=(TH1D*)hunf1D->Clone("htempUnf1D");
    htempUnf1D->SetName(Form("Bayesian_Unfolded1Diter%d",iter));

    TH3D *htempFold=(TH3D*)hfold->Clone("htempFold");
    htempFold->SetName(Form("Bayesian_Foldediter%d",iter));

    TH3D *htempFold1D=(TH3D*)hfold1D->Clone("htempFold1D");
    htempFold1D->SetName(Form("Bayesian_Folded1Diter%d",iter));


    htempUnf->Write();
    htempFold->Write();
    htempUnf1D->Write();
    htempFold1D->Write();
  }
  fout->Close();
}
  
  
  
