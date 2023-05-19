R__LOAD_LIBRARY(libeicsmear);
R__LOAD_LIBRARY(libfastjet);

#include "fastjet/ClusterSequence.hh"
#include <iostream>
using namespace fastjet;
using namespace std;
const int verbosity = 0;

TRandom3 randGen(0);

Double_t energy_weight;
Double_t R_L;
Double_t jet_pt;

TTree preprocessed("preprocessed", "preprocessed");

double calculate_distance(PseudoJet p0, PseudoJet p1)
{
  float dphiabs = fabs(p0.phi() - p1.phi());

  if (dphiabs > TMath::Pi()) dphiabs = 2*TMath::Pi() - dphiabs;

  float dy = p0.eta() - p1.eta();
  return sqrt(dy*dy + dphiabs*dphiabs);
}

class Correlator_Builder
{
  private:
    vector<PseudoJet> particle_list;
    int mult;
    vector<vector<double>> pair_list;
    double jet_eta;
    double weight_pow;

  public:
    Correlator_Builder(vector<PseudoJet> _particle_list, float _jet_pt, float _jet_eta, double _weight_pow)
    {
      particle_list = _particle_list;
      mult = particle_list.size();
      jet_pt = _jet_pt;
      jet_eta = _jet_eta;
      weight_pow = _weight_pow; // power of the pt1 * pt2 / pt^2jet factor used in calculation
    }

    void make_pairs()
    {
      // produces mult X mult upper triangular matrix with elements being the
      // distance between partices i and j
      for (int i = 0; i < mult; i++)
      {
        vector<double> inner_list(mult, 0.0);
        for (int j = 0; j < mult; j++)
        {
          inner_list[j] = calculate_distance(particle_list[i], particle_list[j]);
        }
        pair_list.push_back(inner_list);
      }
    }

    void construct_EEC()
    {
      for (int i = 0; i < mult; i++)
      {
        for (int j = 0; j < mult; j++)
        {
          double dist12 = pair_list[i][j]; // no change in boost
          double eec_weight = pow((particle_list[i].pt() * particle_list[j].pt()) / pow(jet_pt, 2), weight_pow);

          // filling TTree
          energy_weight = eec_weight;
          R_L = dist12;
          preprocessed.Fill();
        }
      }
    }
};

TLorentzVector smear(TLorentzVector part, Double_t track_resolution=0.01)
{
  Double_t mean = 0.0;

  Double_t Pt = part.GetPt() * (1 + randGen.Gaus(mean, sigma));
  Double_t Pz = part.GetPz() * (1 + randGen.Gaus(mean, sigma));
  Double_t E = TMath::Sqrt(part.M2() + pow(Pt, 2) + pow(Pz, 2));
  Double_t Eta = 0.5 * TMath::Log((E + Pz) / (E - Pz));

  part.SetPtEtaPhiM(Pt, Eta, part.Phi(), E);
  part.SetPz(Pz);

  return part;
}


void preprocess(const char* inFile = "merged.root", const char* outFile = "hists_eec.root", int gen_type = 1, double eec_weight_power = 1, int do_smear=0)
{
  // inFile assumed to contain data whether (measured or generated) in the lab frame, i.e. requiring no boosting
  // outfile produced contains a table in the form of a TTree with three columns (energy weight, RL, jet pT)
  // intended to be used by a MultiFold/OmniFold framework to unfold the data and produce a detector-corrected
  // EEC measurement
  // as of now only supports PYTHIA8 generated data

  cout << "Generator Type: ";
  if (gen_type==1) cout << "Pythia8" << endl;

  // TTree definition
  preprocessed.Branch("energy_weight", &energy_weight, "energy_weight/D");
  preprocessed.Branch("R_L", &R_L, "R_L/D");
  preprocessed.Branch("jet_pt", &jet_pt, "jet_pt/D");

  // reads file and fills in TTree
  erhic::EventHepMC *event(NULL); //pythia8

  TFile *f = new TFile(inFile);

  //Get EICTree Tree
  TTree *tree = (TTree*)f->Get("EICTree");

  Int_t nevt = tree->GetEntries();
  cout<<"-------------------------------"<<endl;
  cout<<"Total Number of Events = "<<nevt<<endl<<endl;

  //Access event Branch
  tree->SetBranchAddress("event",&event); //Note &event, even with event being a pointer

  int total_jets = 0;

  //Loop Over Events
  for(Int_t ievt = 0; ievt < nevt; ievt++)
  {
    tree->GetEntry(ievt);

    if (ievt%1000==0) cout<<"Processing event = "<<ievt<<"/"<<nevt<<endl;

    // Q2-cut
    if (event->GetQ2() < 10) continue;

    TLorentzVector part;
    Double_t Px, Py, Pz, Mass;

    // particle enumeration, addition to jet reco setup, and total pt calculation
    erhic::ParticleMC* particle;
    vector<PseudoJet> jet_constits;
    for (int ipart = 0; ipart < event->GetNTracks(); ++ipart)
    {
      particle = event->GetTrack(ipart);

      // get particle kinematics
      Px = particle->GetPx();
      Py = particle->GetPy();
      Pz = particle->GetPz();
      Mass = particle->GetM();
      part.SetXYZM(Px, Py, Pz, Mass);

      // smear, if desired
      if (do_smear == 1)
      {
        // throw out track with probability = 5%
        if (randGen.Uniform(0.0, 1.0) <= 0.05) continue;

        // smear track Pt and Pz with resolution 1%
        part = smear(part, 0.01);
      }

      // use all fsp particles w/ < 3.5 eta, not including scattered electron, for jet reconstruction
      if (particle->GetStatus()==1 && fabs(particle->GetEta())<3.5 && particle->Id()!=11)
      {
        PseudoJet constit = PseudoJet(part.Px(),part.Py(),part.Pz(),part.E());
        constit.set_user_index(ipart);
        jet_constits.push_back(constit);
      }

    }

    // jet reconstruction
    JetDefinition R1jetdef(antikt_algorithm, 1.0);
    ClusterSequence cs(jet_constits, R1jetdef);
    vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets());

    // jet processing
    for (unsigned ijet = 0; ijet < jets.size(); ijet++)
    {
      // cuts on jet kinematics, require jet_pt >= 5GeV, |jet_eta| <= 2.5
      if (jets[ijet].pt() < 5 || fabs(jets[ijet].eta()) > 2.5) continue;

      total_jets++;

      // cuts on jet constituent kinematics, require consitituents_pt >= 0.5GeV, |consitituents_eta| <= 3.5
      // take only charged constituents for eec calculation
      vector<PseudoJet> constituents = jets[ijet].constituents();
      vector<PseudoJet> charged_constituents;
      for (unsigned iconstit = 0; iconstit < constituents.size(); iconstit++)
      {
        if (constituents[iconstit].pt() < 0.5 || fabs(constituents[iconstit].eta()) > 3.5) continue;

        int ip = constituents[iconstit].user_index();
        float charge = event->GetTrack(ip)->Id().Info()->Charge();
        //cout<<"constituent pt:"<<constituents[iconstit].pt()<<" track pt:"<<event->GetTrack(ip)->GetPt()<<" charge:"<<charge<<endl;
        if (charge != 0) charged_constituents.push_back(constituents[iconstit]);
      }

      if (charged_constituents.size() < 1) continue;

      // eec calculation
      Correlator_Builder cb(charged_constituents, jets[ijet].pt(), jets[ijet].eta(), eec_weight_power);
      cb.make_pairs();
      cb.construct_EEC();
    }

  }

  cout<<"total num jets = "<<total_jets<<endl;
  f->Close();

  // create output file
  TFile* fout = new TFile(outFile,"recreate");

  // write TTree and file
  preprocessed.Write();
  fout->Write();
  fout->Close();
  
  cout<<"preprocessed entries: "<<preprocessed.GetEntries()<<endl;

}
