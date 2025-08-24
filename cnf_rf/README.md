# cnf_rf

Complex-valued Continuous Normalizing Flow for RF-IQ bursts. This package
provides training and sampling utilities built on top of PyTorch and
`torchdiffeq`.

Example IQ bursts can be downloaded from the
[RF-Diffusion dataset](https://github.com/mobicom24/RF-Diffusion/releases/download/dataset_model/dataset.zip).
Extract the archive so that `--data_dir` points to the folder containing the
class subdirectories of `.npy` files.

## Usage

### Training
```
python -m cnf_rf.train_cnf --data_dir ./rf_iq_npys --epochs 200
```

### Sampling
```
python -m cnf_rf.sample_cnf --checkpoint ckpt.pt --mode sample --num 16
```

### Log-likelihood
```
python -m cnf_rf.sample_cnf --checkpoint ckpt.pt --mode ll --file test.npy
```
