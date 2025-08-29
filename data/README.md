# Data for Digit Recognition (LeNet)

This folder contains only small reference files.  
The full MNIST dataset is **not included in this repository** due to size limits.

### Included
- `train_label.txt` — Training labels
- `test_label.txt` — Testing labels
- `DIGIT.npy` — Preprocessed digit dataset (sample)

### Not Included
- Raw MNIST image data (`train/`, `test/`, `MNIST/`)  
  These files are stored externally (OneDrive/Google Drive).  
  Update the path in `src/mnist.py` or `src/data.py` to point to your dataset location.

### Usage
When running `train.py` or `test.py`, make sure the MNIST dataset is available locally and paths in the code are updated.
