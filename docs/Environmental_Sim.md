# Environmental Simulation Plan

This document outlines the steps to build an accurate simulation environment that enables training a model to localize a transmitter using only a single angle‑of‑arrival (AoA) software defined radio (SDR). The simulation focuses on reproducing the RF propagation characteristics of the city of San Francisco.

## Objectives

1. Model the urban environment as a **3‑D Gaussian splat** representation capturing buildings, streets, and reflective surfaces.
2. Simulate **RF emissions of cellular devices** including power, carrier frequencies, modulation schemes, and burst behaviour.
3. Produce synthetic datasets of AoA measurements paired with ground‑truth transmitter locations for supervised learning.

## Environment Model

1. **Geometry Acquisition**
   - Obtain public 3‑D city models (e.g. from SF Open Data or OpenStreetMap).
   - Convert meshes into a Gaussian splat representation where each splat encodes position, orientation, and material reflectivity.
2. **Propagation Engine**
   - Implement a ray‑tracing or radio‑wave solver that interacts with the splats.
   - Handle multi‑path, reflections, and occlusions. Store path length, phase, and attenuation.
3. **Dynamic Elements**
   - Allow moving objects (vehicles, pedestrians) represented by time‑varying splats.
   - Support different weather profiles affecting attenuation (fog, rain).

## RF Emission Model

1. **Device Types**
   - Support typical cellular transmitters (4G/5G handsets, base stations).
   - Parameterize power levels, antenna patterns, frequency bands, and burst timing.
2. **Traffic Generation**
   - Use realistic call/data session patterns to trigger transmissions.
   - Include background interference and noise floor.

## Data Generation Pipeline

1. Randomly sample transmitter positions within San Francisco.
2. Simulate RF propagation from each transmitter to the AoA SDR.
3. Record received IQ data and compute AoA measurements.
4. Store samples with metadata: time, true position, environment snapshot.

## Training & Evaluation

1. Train ML models (e.g., CNF, diffusion) on the synthetic dataset to regress transmitter location.
2. Validate using a held‑out set and analyze sensitivity to noise and environment changes.
3. Prepare interfaces for fine‑tuning with real measurements once available.

## Next Steps

- Prototype the Gaussian splat city model and basic ray tracer.
- Define cellular emission parameters and traffic statistics.
- Generate a small pilot dataset for pipeline verification.
- Iterate on environment realism before scaling to full San Francisco coverage.

