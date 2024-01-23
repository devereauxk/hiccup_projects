
//=====================================================================-*-C++-*-
// File and Version Information:
//      $Id: RooUnfoldExample.cxx 279 2011-02-11 18:23:44Z T.J.Adye $
//
// Description:
//      Simple example usage of the RooUnfold package using toy MC.
//
// Authors: Tim Adye <T.J.Adye@rl.ac.uk> and Fergus Wilson <fwilson@slac.stanford.edu>
//
//==============================================================================


#if !(defined(__CINT__) || defined(__CLING__)) || defined(__ACLIC__)
#include <iostream>
using std::cout;
using std::endl;

#include "TRandom.h"
#include "TH1F.h"

#include "TFile.h"
#include "TVectorD.h"

#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TRandom.h"
#include "TPostScript.h"
#include "TH2D.h"
#include "TFile.h"
#include "TLine.h"
#include "TNtuple.h"
#include "TProfile.h"

#include "RooUnfoldResponse.h"
#include "RooUnfoldBayes.h"
//#include "RooUnfoldTestHarness2D.h"
#endif

//==============================================================================
// Global definitions
//==============================================================================

const Double_t cutdummy= -99999.0;

//==============================================================================
// Gaussian smearing, systematic translation, and variable inefficiency
//==============================================================================

TH2D* CorrelationHistShape (const TMatrixD& cov,const char* name, const char* title,
		       Int_t na, Int_t nb, Int_t kbin)
{
 
   TH2D* h= new TH2D (name, title, nb, 0, nb, nb, 0, nb);
 
     	  for(int l=0;l<nb;l++){
                 for(int n=0;n<nb;n++){
                int index1=kbin+na*l;
                int index2=kbin+na*n;
   	        Double_t Vv=cov(index1,index1)*cov(index2,index2);
   		 if (Vv>0.0) h->SetBinContent(l+1,n+1,cov(index1,index2)/sqrt(Vv));


  		  }}
  return h;
}

TH2D* CorrelationHistPt (const TMatrixD& cov,const char* name, const char* title,
		       Int_t na, Int_t nb, Int_t kbin)
{
 
   TH2D* h= new TH2D (name, title, na, 0, na, na, 0, na);
 
     	  for(int l=0;l<na;l++){
                 for(int n=0;n<na;n++){

                int index1=l+na*kbin;
                int index2=n+na*kbin;
   	        Double_t Vv=cov(index1,index1)*cov(index2,index2);
   		 if (Vv>0.0)h->SetBinContent(l+1,n+1,cov(index1,index2)/sqrt(Vv));


  		  }}
  return h;
}


   





void Normalize2D(TH2* h)
{
   Int_t nbinsYtmp = h->GetNbinsY();
   const Int_t nbinsY = nbinsYtmp;
   Double_t norm[nbinsY];
   for(Int_t biny=1; biny<=nbinsY; biny++)
     {
       norm[biny-1] = 0;
       for(Int_t binx=1; binx<=h->GetNbinsX(); binx++)
     {
       norm[biny-1] += h->GetBinContent(binx,biny);
     }
     }

   for(Int_t biny=1; biny<=nbinsY; biny++)
     {
       for(Int_t binx=1; binx<=h->GetNbinsX(); binx++)
     {
       if(norm[biny-1]==0)  continue;
       else
         {
  h->SetBinContent(binx,biny,h->GetBinContent(binx,biny)/norm[biny-1]);
  h->SetBinError(binx,biny,h->GetBinError(binx,biny)/norm[biny-1]);
         }
     }
     }
}







TH2D* CorrelationHist (const TMatrixD& cov,const char* name, const char* title,
		       Double_t lo, Double_t hi,Double_t lon,Double_t hin)
{
  Int_t nb= cov.GetNrows();
  Int_t na= cov.GetNcols();
  cout<<nb<<" "<<na<<endl;
  TH2D* h= new TH2D (name, title, nb, 0, nb, na, 0, na);
  h->SetAxisRange (-1.0, 1.0, "Z");
  for(int i=0; i < na; i++)
  for(int j=0; j < nb; j++) {
  Double_t Viijj= cov(i,i)*cov(j,j);
      if (Viijj>0.0) h->SetBinContent (i+1, j+1, cov(i,j)/sqrt(Viijj));
    }
  return h;
}

//==============================================================================
// Example Unfolding
//==============================================================================

void RooSimple3DPbPb_match(TString cFiles2="files1.txt", std::string tag = "", std::string date = "", int pthard = 1)
{
#ifdef __CINT__
  gSystem->Load("libRooUnfold");
#endif
  Int_t difference=1;
  Int_t Ppol=0;
  cout << "==================================== pick up the response matrix for background==========================" << endl;
  ///////////////////parameter setting
  RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;   
 
  //Get the tree for data
  TString fnamedata;
  fnamedata="../ppData_Dec1.root";
  TFile *inputdata;
  inputdata=TFile::Open(fnamedata);
  inputdata->ls();
  TTree *data=(TTree*)inputdata->Get("JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_scheme_TCRawTree_Data_NoSub_Incl");
  data->Show(0);

  //get the histogram for reweighting the prior
  TFile* f_rw = new TFile("../../Lund/Reweight_Nov23.root");
  TH3D* h3_rw = (TH3D*)f_rw->Get("h3_smooth");
  
  //***************************************************

  float pThard_val[21] = {5., 7., 9., 12., 16., 21., 28., 36., 45., 57., 70., 85., 99., 115., 132., 150., 169., 190., 212., 235., 1000.};
  
  //reco jet pT bins
 Double_t xbins[5];
 xbins[0]=20;
 if (tag == "truncup") xbins[0] = 22; //change the pt cut-off for truncation systematic
 xbins[1]=40;
 xbins[2]=60;
 xbins[3]=80;
 xbins[4]=120;

 //true jet pT bins
  Double_t tbins[8];
  tbins[0]=0;
  tbins[1]=20;
  tbins[2]=40;
  tbins[3]=60;
  tbins[4]=80;
  tbins[5]=120;
  tbins[6]=160;

  //reco splitting dR bins
  Double_t ybins[8];
  ybins[0] = 0.;
  ybins[1] = 0.2;
  ybins[2] = 0.4;
  ybins[3] = 0.6;
  ybins[4] = 0.8;
  ybins[5] = 1.0;
  ybins[6] = 1.2;
  ybins[7] = 1.4;

  //true splitting dR bins
  Double_t ybinst[9];
  ybinst[0] = 0.;
  ybinst[1] = 0.2;
  ybinst[2] = 0.4;
  ybinst[3] = 0.6;
  ybinst[4] = 0.8;
  ybinst[5] = 1.0;
  ybinst[6] = 1.2;
  ybinst[7] = 1.4;
  ybinst[8] = 2.5;

  //reco splitting kT bins
  Double_t zbins[12];
  zbins[0]=-1;
  zbins[1]=-0.75;
  zbins[2]=-0.5;
  zbins[3]=-0.3;
  zbins[4]=-0.1;
  zbins[5] = 0.;
  zbins[6] = 0.1;
  zbins[7] = 0.3;
  zbins[8] = 0.5;
  zbins[9] = 0.75;
  zbins[10] = 1;
  zbins[11] = 1.5;

  //true splitting kT bins
  Double_t zbinst[14];
  zbinst[0]=-3.0;
  zbinst[1]=-1.0;
  zbinst[2]=-0.75;
  zbinst[3]=-0.5;
  zbinst[4]=-0.3;
  zbinst[5]=-0.1;
  zbinst[6] = 0.0;
  zbinst[7] = 0.1;
  zbinst[8] = 0.3;
  zbinst[9] = 0.5;
  zbinst[10] = 0.75;
  zbinst[11] = 1.0;
  zbinst[12] = 1.5;
  zbinst[13] = 3.0;

  //the raw correlation
  TH3D *h3_raw(0);
  h3_raw=new TH3D("r","raw",7,ybins,11,zbins, 4, xbins);
  TH3D *h3_raw2(0);
  h3_raw2=new TH3D("raw2","raw2",8,ybinst,13,zbinst, 6, tbins);
  TH1D* h1_raw = new TH1D("h1_raw", "h1_raw", 4, xbins);

  //detector measure level
  TH3D *h3_smeared(0);
  h3_smeared=new TH3D("smeared","smeared",7,ybins,11,zbins, 4, xbins);
  TH3D *h3_smeared2(0);
  h3_smeared2=new TH3D("smeared2","smeared2",8,ybinst,13, zbinst, 6, tbins);
  TH1D* h1_smeared = new TH1D("h1_smeared", "h1_smeared", 4, xbins);

  //true match level
  TH3D *h3_true(0);
  h3_true=new TH3D("true","true",8,ybinst, 13, zbinst, 6, tbins);
  TH3D *h3_true_match(0);
  h3_true_match=new TH3D("true_match","true_match",8,ybinst, 13, zbinst, 6, tbins);
  TH1D* h1_true(0);
  h1_true = new TH1D("h1_true", "h1_true", 6, tbins);
  //full true correlation
  TH3D *h3_fulleff(0);
  h3_fulleff=new TH3D("truef","truef",8,ybinst, 13, zbinst, 6, tbins);
  TH1D* h1_fulleff(0);
  h1_fulleff= new TH1D("h1_fulleff", "h1_fulleff", 6, tbins);
  TH3D *h3_fulleff_match(0);
  h3_fulleff_match=new TH3D("truef_match","truef_match",8, ybinst, 13, zbinst, 6, tbins);
  TH1D* h1_fulleff_match = new TH1D("h1_fulleff_match", "h1_fulleff_match", 6, tbins);

  //full reco correlation
  TH3D *h3_fullreco(0);
  h3_fullreco=new TH3D("recof","recof", 7, ybins, 11, zbins, 4, xbins);
  TH1D* h1_fullreco = new TH1D("h1_fullreco", "h1_fullreco", 4, xbins);
  //match reco correlation
  TH3D *h3_reco(0);
  h3_reco=new TH3D("reco","reco",7,ybins, 11, zbins, 4, xbins);
  TH1D* h1_reco = new TH1D("h1_reco", "h1_reco", 4, xbins);

  //delta pT bins for JES
  Double_t delta_bins[221];
  for (int i = 0; i <= 220; i++) {
    delta_bins[i] = -6.0 + (11./220.)*i;
  }
  
  TH3D* h3_dz = new TH3D("h3_dz", "h3_dz", 6, tbins, 11, zbins, 220, delta_bins);
  TH3D* h3_dtheta = new TH3D("h3_dtheta", "h3_dtheta", 8, 0, 160, 25, 0, 2.5, 200, -3, 6);
  TH3D* h3_z = new TH3D("h3_z", "h3_z", 6, tbins, 11, zbins, 11, zbins);
  TH3D* h3_theta = new TH3D("h3_theta", "h3_theta", 6, tbins, 7, ybins, 7,ybins);
  TH3D* h3_n = new TH3D("h3_n", "h3_n", 8, 0, 160, 15, 0, 15, 15, 0, 15);

  TH3D* h3_z_1 = new TH3D("h3_z_1", "h3_z_1", 7, ybins, 11, zbins, 11, zbins);
  TH3D* h3_z_2 = new TH3D("h3_z_2", "h3_z_2", 7, ybins, 11, zbins, 11, zbins);
  TH3D* h3_z_3 = new TH3D("h3_z_3", "h3_z_3", 7, ybins, 11, zbins, 11, zbins);
  TH3D* h3_theta_1 = new TH3D("h3_theta_1", "h3_theta_1", 11, zbins, 7, ybins, 7,ybins);
  TH3D* h3_theta_2 = new TH3D("h3_theta_2", "h3_theta_2", 11, zbins, 7, ybins, 7,ybins);
  TH3D* h3_theta_3 = new TH3D("h3_theta_3", "h3_theta_3", 11, zbins, 7, ybins, 7,ybins);
  
  TH2D *hcovariance(0);
  hcovariance=new TH2D("covariance","covariance",10,0.,1.,10,0,1.);

  h3_smeared->Sumw2();
  h3_smeared2->Sumw2();
  h3_true_match->Sumw2();
  h3_reco->Sumw2();		   
  h3_raw->Sumw2();
  h3_raw2->Sumw2();
  h3_fulleff_match->Sumw2();
  h1_raw->Sumw2();
  h1_smeared->Sumw2();
  h1_true->Sumw2();
  h1_fullreco->Sumw2();

  h3_dz->Sumw2();
  h3_dtheta->Sumw2();
  h3_z->Sumw2();
  h3_theta->Sumw2();
  h3_n->Sumw2();

  std::stringstream ss;
  
  Double_t ptJet,ptJetMatch, zg, zgMatch, rg, rgMatch, ktg, ktgMatch;
  std::vector<std::vector<double> > *zgData = new std::vector<std::vector<double> >(0);
  std::vector<std::vector<double> > *rgData = new std::vector<std::vector<double> >(0);
  std::vector<std::vector<double> > *ktgData = new std::vector<std::vector<double> >(0);
    
  Int_t nEv=0;; 
  //so mcr is correctly normalized to one, not the response.       
  cout<<"cucu"<<endl;
  nEv=data->GetEntries(); 
  cout<<"entries"<<nEv<<endl;
  data->SetBranchAddress("ptjet", &ptJet); 
  data->SetBranchAddress("z", &zgData);
  data->SetBranchAddress("kt", &ktgData); 
  data->SetBranchAddress("angle", &rgData); 
  
  float pTmin = 20.;
  //change the pTmin for truncation systematic
  if (tag == "truncup") pTmin = 22;
  if (tag == "truncdown") pTmin = 18;
  float matchR = 0.1;
  //change the matching radius for matching systematic
  if (tag == "matchup") matchR = 0.15;
  if (tag == "matchdown") matchR = 0.05;
  
  for(int iEntry=0; iEntry< nEv; iEntry++){
   data->GetEntry(iEntry);
   if(ptJet>120 || ptJet<pTmin) continue;
   
   for (int i = 0; i < zgData->at(0).size(); i++)
     {
       float lnz = std::log(ktgData->at(0).at(i));
       float lnr = std::log(0.4/rgData->at(0).at(i));
       //the 3D distribution that will be unfolding for the splittings
       h3_raw->Fill(lnr,lnz,ptJet);
       h3_raw2->Fill(lnr,lnz,ptJet);
     }
   //the 1D distribution that will be unfolding for the normalization
   h1_raw->Fill(ptJet);
   }

  //used for getting the pThard scaling correct
   float  mytrials[20] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

   ifstream infile2;
   infile2.open(cFiles2.Data());
   char filename2[300];
   RooUnfoldResponse response;
   response.Setup(h3_smeared,h3_true);
   RooUnfoldResponse response1D;
   response1D.Setup(h1_smeared,h1_true);

   while(infile2>>filename2){
     int pthardbin=0;
     TFile *input=TFile::Open(filename2);
     if (!input || input->IsZombie()) { cout <<"The file could not be opened!"; delete input; continue;}
     TList *list=(TList*) input->Get("JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_scheme_TCRaw_PythiaDef_NoSub_Incl");
     TList *list2=(TList*) list->FindObject("EventCuts");
     TH1D *hcent=(TH1D*)input->FindObject("Centrality_selected");
     TProfile *hcross=(TProfile*)list->FindObject("fHistXsection");
     TH1D *htrials=(TH1D*)list->FindObject("fHistTrials");
     TH1D *hpthard=(TH1D*)list->FindObject("fHistPtHard");
     TH1D *hnevent=(TH1D*)list->FindObject("fHistEventCount");
     for(Int_t i=1;i<=htrials->GetNbinsX();i++){
       if(htrials->GetBinContent(i)!=0) pthardbin=i;}
     if(pthardbin==1) cout<<"shit"<<htrials->Integral(1,1)<<endl;
     mytrials[pthardbin-1]=mytrials[pthardbin-1]+htrials->Integral(pthardbin,pthardbin);
     delete hcent;
     delete hcross;
     delete htrials;
     delete hpthard;
     delete hnevent;
     input->Close();
     delete input;
   }

   //make sure the number of trials got calculated correctly 
   for(Int_t k=0;k<20;k++){
     cout<<mytrials[k]<<endl;}

   ifstream infile3;
   infile3.open(cFiles2.Data());
   char filename3[300];

   while(infile3>>filename3){
     int pthardbin=0;
  
     TFile *input=TFile::Open(filename3);
     if (!input || input->IsZombie()) { cout <<"The file could not be opened!"; delete input; continue;}
     input->ls();

     TList *list=(TList*) input->Get("JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_scheme_TCRaw_PythiaDef_NoSub_Incl");
     list->ls();
     TH1F *hcent=(TH1F*)list->FindObject("Centrality_selected");
     TProfile *hcross=(TProfile*)list->FindObject("fHistXsection");
     TH1F *htrials=(TH1F*)list->FindObject("fHistTrials");
     TH1F *hpthard=(TH1F*)list->FindObject("fHistPtHard");
     TH1F *hnevent=(TH1F*)list->FindObject("fHistEventCount");

     for(Int_t i=1;i<=htrials->GetNbinsX();i++){
       if(htrials->GetBinContent(i)!=0) pthardbin=i;}
     
     double scalefactor=(hcross->Integral(pthardbin,pthardbin))/mytrials[pthardbin-1];
     std::cout << "pt hard bin: " << pthardbin-1 << std::endl;
     if (isnan(scalefactor)) std::cout << "already bad" << std::endl;
     if (isnan(htrials->Integral(pthardbin,pthardbin))) std::cout << "the trials is bad" << std::endl;

     //get the tree of the matched splittings
     TFile *input2=TFile::Open(filename3);
     TTree *mc=(TTree*)input2->Get("JetSubstructure_Jet_AKTChargedR040_tracks_pT0150_E_scheme_TCRawTree_PythiaDef_NoSub_Incl");
     mc->Show(0);

     //get the matched 3D histogram (or evaluate it on the fly from the tree)
     TH3D* h1 = (TH3D*)list->FindObject("fHtrueMatch");
     ss << "h1_pthard" << pthardbin;
     h1->SetName(ss.str().c_str());
     h1->Scale(scalefactor);
     //here we are evaluating it on the fly later on from the tree
    // h3_true_match->Add(h1);
     ss.str("");

     //get the full true level 3D histogram (can't be generated from the tree of matched splits)
     TH3D* h2 = (TH3D*)list->FindObject("fHtrueAll");
     ss << "h2_pthard" << pthardbin;
     h2->SetName(ss.str().c_str());
     h2->Scale(scalefactor);
     h3_fulleff->Add(h2);
     ss.str("");
     
     //get the matched reco level 3D histogram (can also fill on he fly from tree)
     TH3D* h3 = (TH3D*)list->FindObject("fHrecoMatch");
     ss << "h3_pthard" << pthardbin;
     h3->SetName(ss.str().c_str());
     h3->Scale(scalefactor);
     //if doing the truncation systematic need to fill on the fly
     if (tag != "truncup") h3_reco->Add(h3);
     ss.str("");
     
     //get the full reco level 3D histogram 
    TH3D* h4 = (TH3D*)list->FindObject("fHrecoAll");
    ss << "h4_pthard" << pthardbin;
    h4->SetName(ss.str().c_str());
    h4->Scale(scalefactor);
    if (tag != "truncup") h3_fullreco->Add(h4);
    //if doing truncation systematic need to fill a rebinned histogram
    else
      {
	for (int i = 1; i <= h4->GetNbinsX(); i++)
	  {
	    for (int j = 1; j <= h4->GetNbinsY(); j++)
	      {
		for (int k = 1; k <= h4->GetNbinsZ(); k++)
		  {
		    int b = h3_fullreco->GetBin(i, j, k);
		    h3_fullreco->AddBinContent(b, h4->GetBinContent(i, j, k));
		  }
	      }
	  }
      }
    ss.str("");
    
    //get the 1D matched true level histogram
    TH1D* h5 = (TH1D*)list->FindObject("fHtrueMatch1D");
    ss << "h5_pthard" << pthardbin;
    h5->SetName(ss.str().c_str());
    h5->Scale(scalefactor);
    h1_true->Add(h5);
    ss.str("");

    //get the 1D full true level histogram
    TH1D* h6 = (TH1D*)list->FindObject("fHtrueAll1D");
    ss << "h6_pthard" << pthardbin;
    h6->SetName(ss.str().c_str());
    h6->Scale(scalefactor);
    h1_fulleff->Add(h6);
    ss.str("");
    
    Int_t nEv=mc->GetEntries();
    if (isnan(scalefactor)) std::cout << "already bad 1" << std::endl;
    
    mc->SetBranchAddress("ptjet", &ptJet); 
    mc->SetBranchAddress("ptjet_part", &ptJetMatch);
    mc->SetBranchAddress("lnkt", &ktg); 
    mc->SetBranchAddress("lnkt_part", &ktgMatch);
    mc->SetBranchAddress("lnR", &rg); 
    mc->SetBranchAddress("lnR_part", &rgMatch);
    if (isnan(scalefactor)) std::cout << "already bad 2" << std::endl;

    double ptJetconst = 0.;
    for(int iEntry=0; iEntry< nEv; iEntry++){
      if (isnan(scalefactor)) {std::cout << "already bad loop before" << std::endl; exit(0);}
      mc->GetEntry(iEntry);
      //These are for when we use Herwig to remove the outliers 
      //if (ptJetMatch > 4.*pThard_val[pthardbin-1]) continue;
      //if (ptJet < 20.) continue;

      //fill the 1D response for unfolding the jet pT spectra for the normalization
      float dpt = ptJet - ptJetconst;
      dpt = std::sqrt(dpt*dpt);
      if ((dpt > 0.) && (ptJet<=120) && (ptJet>=pTmin))
	{
	  ptJetconst = ptJet;
	  h1_fullreco->Fill(ptJet, scalefactor);
	  h1_reco->Fill(ptJet, scalefactor);
	  h1_smeared->Fill(ptJet, scalefactor);
	  response1D.Fill(ptJet, ptJetMatch, scalefactor);
	}

      //true level cuts
      if (ptJetMatch <0. || ptJetMatch > 160.) continue;
      if (ktgMatch < -3. || ktgMatch > 3) continue;
      if (rgMatch < 0. || rgMatch > 2.5) continue;
      float lnr_part = rgMatch;
      float lnz_part = ktgMatch;

      //reweighting the response to the prior
      double weightresp = scalefactor;
      int binx = -1;
      if (lnr_part < 0) binx = 1;
      else if (lnr_part > 1.4) binx = h3_rw->GetNbinsX();
      else binx = h3_rw->GetXaxis()->FindBin(lnr_part);
      int biny = -1;
      if (lnz_part < -1.0) biny = 1;
      else if (lnz_part > 1.5) biny = h3_rw->GetNbinsY();
      else biny = h3_rw->GetYaxis()->FindBin(lnz_part);
      int binz = -1;
      if (ptJetMatch < pTmin) binz = 1;
      else if (ptJetMatch > 120) binz = h3_rw->GetNbinsZ();
      else binz = h3_rw->GetZaxis()->FindBin(ptJetMatch);
      double rw_fac = h3_rw->GetBinContent(binx, biny, binz);
      if (tag == "rw") weightresp*=rw_fac;
            
      float lnz_det = ktg;
      float lnr_det = rg;
      h3_smeared2->Fill(lnr_det,lnz_det,ptJet,weightresp);
      ///////ONLY COMMENTER OUT FOR REWEIGHTING////
      if(ptJet>120 || ptJet<pTmin) continue;
      h3_dtheta->Fill(ptJetMatch, lnr_part, (lnr_det-lnr_part)/lnr_part, scalefactor);
      ///////ONLY COMMENTER OUT FOR REWEIGHTING////                                                                                                   
      if (lnz_det < -1. || lnz_det > 1.5) continue;
      if (lnr_det < 0. || lnr_det > 1.4) continue;
      //fill the response an the true level distribution for the numerator in the eff correction
      h3_smeared->Fill(lnr_det,lnz_det,ptJet,weightresp);
      h3_true_match->Fill(lnr_part,lnz_part,ptJetMatch,weightresp);
      response.Fill(lnr_det,lnz_det,ptJet,lnr_part,lnz_part,ptJetMatch,weightresp);
      h3_dz->Fill(ptJetMatch, lnz_part, (lnz_det-lnz_part)/lnz_part, weightresp);
      h3_z->Fill(ptJetMatch, lnz_part, lnz_det, weightresp);
      h3_theta->Fill(ptJetMatch, lnr_part, lnr_det, weightresp);

      //fill the binned response 
      if ((ptJetMatch > 20.) && (ptJetMatch < 120.)) {
	if ((lnr_det > 0) && (lnr_det < 0.4)) h3_z_1->Fill(lnr_part, lnz_part, lnz_det, weightresp);
	else if ((lnr_det > 0.4) && (lnr_det < 0.8)) h3_z_2->Fill(lnr_part, lnz_part, lnz_det, weightresp);
	else if ((lnr_det > 0.8) && (lnr_det < 1.4)) h3_z_3->Fill(lnr_part, lnz_part, lnz_det, weightresp);
	
	if ((lnz_det > -1) && (lnz_det < 0)) h3_theta_1->Fill(lnz_part, lnr_part, lnr_det, weightresp);
	else if ((lnz_det > 0) && (lnz_det < 0.5)) h3_theta_2->Fill(lnz_part, lnr_part, lnr_det, weightresp);
	else if ((lnz_det > 0.5) && (lnz_det < 1.5)) h3_theta_3->Fill(lnz_part, lnr_part, lnr_det, weightresp);
      }
    }
    delete h2;
    delete h4;
    delete h6;
    delete hcent;
    delete hcross;
    delete htrials;
    delete hpthard;
    delete hnevent;
    delete mc;
    input->Close();
    input2->Close();
    delete input;
    delete input2;
   }
    
   TH1F *htrueptd=(TH1F*) h3_fulleff->ProjectionX("trueptd",1,-1);
   TH1F *htruept=(TH1F*) h3_fulleff->ProjectionY( "truept",1,-1); 
    
  //////////efficiencies done////////////////////////////////////
 TH1F * effok=(TH1F *)h3_true->ProjectionX("effok",2,2);
 TH1F * effok1=(TH1F *)h3_fulleff->ProjectionX("effok2",2,2);
 effok->Divide(effok1);
 effok->SetName("correff20-40");
 
 TH1F * effok3=(TH1F *)h3_true->ProjectionX("effok3",3,3);
  TH1F * effok4=(TH1F *)h3_fulleff->ProjectionX("effok4",3,3);
 effok3->Divide(effok4);
  effok3->SetName("correff40-60"); 

  TH1F * effok5=(TH1F *)h3_true->ProjectionX("effok5",4,4);
  TH1F * effok6=(TH1F *)h3_fulleff->ProjectionX("effok6",4,4);
 effok5->Divide(effok6);
 effok5->SetName("correff60-80");

  TH1D * effok9=(TH1D *)h3_true->ProjectionX("effok9",4, 4, 4, 4);
 TH1D * effok10=(TH1D *)h3_fulleff->ProjectionX("effok10",4, 4, 4, 4);
 effok9->Divide(effok10);
 effok9->SetName("correff60-80_wide");

   TH1F * effok7=(TH1F *)h3_true->ProjectionX("effok7",5,6);
  TH1F * effok8=(TH1F *)h3_fulleff->ProjectionX("effok8",5,6);
 effok7->Divide(effok8);
 effok7->SetName("correff80-120"); 
 
 
 TFile *fout=new TFile (Form("../../Lund/Unfold3DLund_%s_%i_%s.root", tag.c_str(), pthard, date.c_str()),"RECREATE");
  fout->cd();
 effok->Write(); 
 effok3->Write();
 effok5->Write();
 effok7->Write();
 effok9->Write();
 h3_raw->SetName("raw");
 h3_raw->Write();
 h3_raw2->Write();
 h3_smeared->SetName("smeared");
 h3_smeared->Write();
 h3_smeared2->Write();
 htrueptd->Write();
 h3_true->SetName("true");
 h3_true->Write();
 h3_true_match->Write();
 h3_fulleff->Write();
 h3_fulleff_match->Write();
 h3_reco->Write();
 h3_fullreco->Write();
 h1_raw->Write();
 h1_smeared->Write();
 h1_true->Write();
 h1_fulleff->Write();
 h1_reco->Write();
 h1_fullreco->Write();		       
 h3_dz->Write();
 h3_dtheta->Write();
 h3_z->Write();
 h3_theta->Write();
 h3_n->Write();
 h3_z_1->Write();
 h3_z_2->Write();
 h3_z_3->Write();
 h3_theta_1->Write();
 h3_theta_2->Write();
 h3_theta_3->Write();
 response.Write();
 response1D.Write();
 
 TH3D* h3_purity = (TH3D*)h3_reco->Clone("h3_purity");
 h3_purity->Divide(h3_fullreco);
 h3_purity->Write();
 TH3D* h3_efficiency = (TH3D*)h3_true_match->Clone("h3_efficiency");
 h3_efficiency->Divide(h3_fulleff);
 h3_efficiency->Write();

 //correct the raw distribution for the purity
 TH3D* h3_raw_corr = (TH3D*)h3_raw->Clone("h3_raw_corr");
 h3_raw_corr->Multiply(h3_purity);
 h3_raw_corr->Write();
 
}




     
	    
	  
     


     
     }

     fout->Close();
}

#ifndef __CINT__
int main () { RooSimple3DPbPb_match(); return 0; }  // Main program when run stand-alone
#endif
