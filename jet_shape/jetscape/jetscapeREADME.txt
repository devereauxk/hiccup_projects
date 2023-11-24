Thanks Mateusz! Regarding the jetscape files:
- The main output files are in: /rstorage/jetscape/JETSCAPE-AA-events/skim/497764/v3 . There are many centralities, sqrt_s. Hopefully self explanatory, but let me know if they’re not.
- Inside of each directory, there are many files of the form JetscapeHadronListBin{pt_hat_low}_{pt_hat_high}_{index}  (eg. JetscapeHadronListBin80_90_08). Each index has up to 5000 events.
- Note that the files are missing file extensions, but they are stored in the parquet file format. The data are stored event-wise, which is to say, agged (eg. TTree-like). It’s in the following format:
ipython
In [1]: import awkward as ak

In [2]: arr = ak.from_parquet("/rstorage/jetscape/JETSCAPE-AA-events/skim/497764/v3/5020_PbPb_0-5_0.30_2.0_1/JetscapeHadronListBin9_11_01")

In [3]: arr.type.show()
5000 * {
    event_plane_angle: float32,
    hydro_event_id: uint16,
    particle_ID: var * int32,
    status: var * int8,
    E: var * float32,
    px: var * float32,
    py: var * float32,
    pz: var * float32
}
- It can be analyzed using this codebase: https://github.com/jdmulligan/JETSCAPE-analysis. It’s based on heppy. As an example, a number of ALICE analyses are implemented here. The simplest may be to edit one of these existing analyses, but as you like.
    - Alternatively, you can just take a snippet like this: https://github.com/jdmulligan/JETSCAPE-analysis/blob/c6e3d984f4936ab88665db3fe19b4c28591421cf/jetscape_analysis/analysis/analyze_events_base_PHYS.py#L119-L133 . You read the parquet file via pandas and then iterate over events.
    - The package that I mentioned above, awkward is for numpy like operations. I use it very heavily, but it’s probably not the best for working with heppy, but just including as information.