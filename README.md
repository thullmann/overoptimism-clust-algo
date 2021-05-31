# Overoptimism in Clustering

When researchers publish new cluster algorithms, they usually demonstrate the strengths of their novel approaches by comparing the algorithms' performance with existing competitors. However, such studies are likely to be optimistically biased towards the new algorithms, as the authors have a vested interest in presenting their method as favorably as possible in order to increase their chances of getting published. 

Therefore, the superior performance of newly introduced cluster algorithms is over-optimistic and might not be confirmed in independent benchmark studies performed by neutral and unbiased authors. We present an illustrative study to illuminate the mechanisms by which authors -- consciously or unconsciously -- paint their algorithms' performance in an over-optimistic light.  

Using the recently published cluster algorithm Rock as an example, we demonstrate how optimization of the used data sets or data characteristics, of the algorithm's parameters and of the choice of the competing cluster algorithms leads to Rock's performance appearing better than it actually is. Our study is thus a cautionary tale that illuminates how easy it can be for researchers to claim apparent 'superiority' of a new cluster algorithm. We also discuss possible solutions on how to avoid the problems of over-optimism.

## Experiments

We used the optuna<sup>[[1]](#optuna)</sup> hyperparameter optimization framework in order to find the parameter configuration for three popular synthetic datasets (Two Moons, Blobs and Rings) that yields the largest performance difference between Rock and the best of the competitors (which is not necessarily the parameter configuration that yields the best **absolute** performance of Rock). The competing algorithms we chose to compare Rock to are DBSCAN, Kmeans, Mean-Shift and Spectral Clustering. The implementation of Rock we used can be found [here](./rock.py). For the competing methods the used the implementations from the sklearn.cluster<sup>[[2]](#cluster)</sup> module.

To simulate a reasearcher picking out the best datasets we performed the following formal optimization task using the TPE Sampler<sup>[[3]](#sampler)</sup> from optuna:

<img src="https://render.githubusercontent.com/render/math?math=\text{argmax}_{D \in \mathcal{D}} \left\{ \frac{1}{10} \sum_{i = 1}^{10}  \Big( AMI\left(Rock(D^i), y_{D^i}\right) - \text{max}_{C \in \mathcal{C}} \sum_{i = 1}^{10} AMI\left(C(D^i), y_{D^i}\right) \Big) \right\}">

For each dataset we created a jupyter notebook which you can find behind the following links:
- [**Two Moons**](./notebooks/Optimizations/Overoptimism_Two_Moons.ipynb)
- [**Blobs** (with different densities)](./notebooks/Optimizations/Overoptimism_Den_Blobs.ipynb)
- [**Rings**](./notebooks/Optimizations/Overoptimism_Rings.ipynb)

Results for each of our runs can be found in the corresponding csv files and optuna study databases. 
We provide a notebook that loads these results and creates figures used in the paper [**here**]((./notebooks/Optimizations/Optuna_Results_Analysis.ipynb)).

Single Examples for each dataset:

## Analysis 
After determining the optimal values for the data parameters, we analyzed the performance of Rock for non-optimal parameter values. That is, for each dataset and single data parameter in turn, the parameter was varied over a list of values, while the other data parameters were kept fixed at their optimal values. 

1. We kept the jitter value and varied the [**number of samples**]() for the Two Moons Dataset. 
2. We kept the number of samples and varied the [**jitter**]() for the Two Moons Dataset. 
3. We varied the [**number of features**]() for the blobs dataset 

### Random Seed

In the experiments given so far, we always considered the AMI averaged over ten random seeds. In the final step of the analysis for this section, we specifically study the influence of individual random seeds. We take the Two Moons dataset as an example, with a data parameter setting which is not optimal for Rock, but for which DBSCAN performs very well. We generate 100 datasets with these characteristics by setting 100 different random seeds, to check whether there exist particular seeds for which Rock does perform well, leading to over-optimization potential. This experiment can be found in [**this notebook**]().

## Hyperparameters
### ROCK 
### DBSCAN

---
<a name="optuna">[1]</a> https://optuna.org/  
<a name="cluster">[2]</a> https://scikit-learn.org/stable/modules/clustering.html  
<a name="sampler">[3]</a> https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
