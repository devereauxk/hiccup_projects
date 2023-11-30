I thought its best to write to you both on the sherpa things…
9:40
Hi Kyle, CC Raymond
this is about running on SHERPA
see https://github.com/matplo/pyjetty/blob/master/pyjetty/sandbox/hepmc_jetreco_example.py
you will need to git pull --rebase to update (for example)
you may need to install some additional python packages:
do that within the virtual environment:
python -m pip install pyhepmc particle
to run - an example: ./hepmc_jetreco_example.py -i /rstorage/ploskon/eecmc/sherpa2x/jetpt20/sherpa_LHC_jets_20.hepmc --hepmc 3 --output somefile.root
please note that some analysis parameters are hard coded - use this code as an example how to work with the hepmc files - copy and modify as needed
we have two types of Sherpa generations on hiccup
with the default fragmentation/hadronization:
/rstorage/ploskon/eecmc/sherpa2x/jetpt20/sherpa_LHC_jets_20.hepmc - file good for analysis of jet pT>20
/rstorage/ploskon/eecmc/sherpa2x/jetpt40/sherpa_LHC_jets_40.hepmc - same for pT>40
/rstorage/ploskon/eecmc/sherpa2x/jetpt60/sherpa_LHC_jets_60.hepmc …
with the Lund fragmentation/hadronization - pythia6 tuned to LEP data
/rstorage/ploskon/eecmc/sherpa2x/jetpt20_lund/sherpa_LHC_jets_20.0.hepmc
/rstorage/ploskon/eecmc/sherpa2x/jetpt40_lund/sherpa_LHC_jets_40.0.hepmc
/rstorage/ploskon/eecmc/sherpa2x/jetpt60_lund/sherpa_LHC_jets_60.0.hepmc
comments:
these files are large but perhaps you find a space for it on perlmutter… (perhaps checkin with Raymond how he wants to do things / IF on perlmutter) - indeed, this will take some time to analyze (single process) but go ahead just launch the jobs