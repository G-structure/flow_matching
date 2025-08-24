# CNF for Complex RF Bursts

This package provides training and sampling utilities for a complex-valued Continuous Normalizing Flow built on top of `flow_matching`.

## Training

```bash
python -m cnf_rf.train_cnf --data_dir ./rf_iq_npys --epochs 200
```

## Sampling / Likelihood

```bash
python -m cnf_rf.sample_cnf --checkpoint ckpt.pt --mode sample --num 16
python -m cnf_rf.sample_cnf --checkpoint ckpt.pt --mode ll --file test.npy
```
