# Overoptimism in Clustering

This is the repo for our paper "Overoptimism in Clustering".

## Overoptimism

We used optuna to find setting for popular synthetic datasets where the ROCK algorithm performed relativly best in comparison to DBSCAN, Kmeans, Mean-Shift and Spectral Clustering.
To simulate a reasearcher picking out the best datasets we performed the following formal optimization task:

<img src="https://render.githubusercontent.com/render/math?math=\text{argmax}_{D \in \mathcal{D}} \left\{ \frac{1}{10} \sum_{i = 1}^{10} AMI\left(Rock(D^i), y_{D^i}\right) - \right.\\ - \left. \text{max}_{C \in \mathcal{C}} \frac{1}{10} \sum_{i = 1}^{10} AMI\left(C(D^i), y_{D^i}\right) \right\}">

The code to each of the three datasets we tried can be found here:

- [two moons](./notebooks/Optimizations/Overoptimism_Two_Moons.ipynb)
- [blobs (with different densities)](./notebooks/Optimizations/Overoptimism_Den_Blobs.ipynb)
- [rings / circles](./notebooks/Optimizations/Overoptimism_Rings.ipynb)

Results for each of our runs can be found in csv files and the corresponding optuna study databases. 

Analysis of our results:

[Optuna_Analysis]((./notebooks/Optimizations/Optuna_Results_Analysis.ipynb))

## Analysis of each single parameter

To illustrate the effects of overoptimism in our paper we then used the three settings 

## Sensitivity towards random seed

## Hyperparameters
### ROCK 
### DBSCAN
