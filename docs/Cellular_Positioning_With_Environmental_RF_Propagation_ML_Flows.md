# Cellular Positioning with Environmental RF Propagation and ML Flows

This document sketches a plan for locating cellular transmitters using machine learning flows informed by an environmental RF propagation model.

## Overview

We aim to leverage a single AoA SDR and a learned environmental model to estimate the 3‑D position of a cellular device. The method combines simulated data, flow‑based neural networks, and environmental context captured via Gaussian splats.

## Pipeline

1. **Environment Representation**
   - Use the Gaussian splat city model described in `Environmental_Sim.md`.
   - Each splat provides geometry and material properties for RF propagation.
2. **RF Propagation Simulation**
   - Implement a differentiable or precomputed ray‑tracing engine.
   - Generate channel responses (path gain, delay, Doppler) from any point to the SDR.
3. **Data Generation**
   - Sample transmitter locations and device parameters.
   - Propagate through the environment to create synthetic IQ samples and AoA estimates.
   - Collect pairs `(AoA, environment context) -> position` for training.
4. **ML Flow Model**
   - Train a conditional continuous normalizing flow or diffusion model that maps AoA observations and environmental features to a posterior distribution over locations.
   - Conditioning can include time, frequency band, and device type.
5. **Inference**
   - Given an observed AoA and current environment snapshot, run the flow model to obtain a location distribution.
   - Select the MAP estimate or compute uncertainty bounds.

## Research Questions

- How many synthetic samples are required for city‑scale coverage?
- What level of environment detail is necessary for accurate localization?
- Can the flow model generalize to unseen urban layouts or moving obstacles?

## Next Steps

1. Implement the environment simulator and verify propagation accuracy against analytic benchmarks.
2. Design the conditional flow architecture for combining AoA and environmental features.
3. Produce an initial dataset and run baseline training experiments.
4. Evaluate localization error metrics across the simulated city and iterate on model and simulator fidelity.

