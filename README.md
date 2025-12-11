# NaturalGasGP
This project implements a Gaussian Process (GP) model to reconstruct the natural gas
forward curve (Henry Hub). The goal is to produce a smooth and realistic term structure,
handle illiquid or missing maturities, and quantify uncertainty using GP variance.

## Goals
- Build synthetic natural gas forward curves with seasonality
- Fit Gaussian Process models with RBF kernel
- Extend the GP with periodic kernel to capture yearly cycles
- Apply the model to real Henry Hub data
- Visualize uncertainty bands and reconstructed curves
- Compare GP reconstruction with classical interpolation

## How to Run
Scripts inside `experiments/` show how to execute each step.

## Author
Mehdi – Quantitative Research Project