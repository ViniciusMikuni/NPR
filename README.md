# Neural Positive Reweighting Studies for Les Houches 2023

## Dataset
Initial LHE files used for the study are found at: https://cernbox.cern.ch/remote.php/dav/public-files/K5YTc2g22oxBjJL/ttbb-events-oldPDF-phase1-2-3-4.lhe

## Preprocessing

The data is converted to an h5py file for convenience using the follwing commands:

```bash
python preprocessing.py
```

## Training and evaluation of the results

The training script to reproduce the plots are found in the Jupyter notebook:

```bash
ttbb.ipynb
```

