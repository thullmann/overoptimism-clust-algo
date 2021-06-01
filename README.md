# Overoptimism in Clustering

When researchers publish new cluster algorithms, they usually demonstrate the strengths of their novel approaches by comparing the algorithms' performance with existing competitors. However, such studies are likely to be optimistically biased towards the new algorithms, as the authors have a vested interest in presenting their method as favorably as possible in order to increase their chances of getting published. 

Therefore, the superior performance of newly introduced cluster algorithms is over-optimistic and might not be confirmed in independent benchmark studies performed by neutral and unbiased authors. We present an illustrative study to illuminate the mechanisms by which authors -- consciously or unconsciously -- paint their algorithms' performance in an over-optimistic light.  

Using the recently published cluster algorithm Rock as an example, we demonstrate how optimization of the used data sets or data characteristics, of the algorithm's parameters and of the choice of the competing cluster algorithms leads to Rock's performance appearing better than it actually is. Our study is thus a cautionary tale that illuminates how easy it can be for researchers to claim apparent 'superiority' of a new cluster algorithm. We also discuss possible solutions on how to avoid the problems of over-optimism.

## Reproduce

First of, in order to exactly reproduce our results you should create a vritual python environment and install our requirements. 
Starting out, if you do not have python installed you can go to the official python website and download python version ???.
Next go to your version of our git in your command line and run the following command to create the virtual envrionment and activate it:

``` 
python -m venv your_environment_name 
```
Activate your virtual environment.  
*Windows*:
```
your_environment_name\Scripts\activate
```
*Mac OS / Linux*
```
source your_environment_name/bin/activate
```
Then install the requirements using our [requirements.txt](./requirements.txt)
(This ensure you using excactly the same versions of the python libraries as we did)
```
pip install -r requirements.txt
```
Lastly you can start the jupyter notebooks by calling
```
jupyter notebook
```

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

## Analysis 
After determining the optimal values for the data parameters, we analyzed the performance of Rock for non-optimal parameter values. That is, for each dataset and single data parameter in turn, the parameter was varied over a list of values, while the other data parameters were kept fixed at their optimal values. 

1. We kept the jitter value and varied the [**number of samples**](./notebooks/Comparisons/Two_Moons_Analysis-num_samples.ipynb) for the Two Moons Dataset. 
2. We kept the number of samples and varied the [**jitter**](./notebooks/Comparisons/Two_Moons_Analysis_jitter.ipynb) for the Two Moons Dataset. 
3. We varied the [**number of features**](./notebooks/Comparisons/Den_Blobs_Analysis.ipynb) for the blobs dataset 

### Random Seed

In the experiments given so far, we always considered the AMI averaged over ten random seeds. In the final step of the analysis for this section, we specifically study the influence of individual random seeds. We take the Two Moons dataset as an example, with a data parameter setting which is not optimal for Rock, but for which DBSCAN performs very well. We generate 100 datasets with these characteristics by setting 100 different random seeds, to check whether there exist particular seeds for which Rock does perform well, leading to over-optimization potential. This experiment can be found in [**this notebook**](./notebooks/Comparisons/Analysis_two_moons_100_seed.ipynb).

## Hyperparameters

Lastly we varied ([**Rock's hyperparameter t_max**](./notebooks/Optimizations/ROCK_Hyperparameter_Search.ipynb) (maximum number of iterations). Here we considered the **absolute** performance of Rock, given researchers would also strive to maximize the absolute performance of their novel algorithm. As exemplary datasets, we again consider Two Moons, Blobs and Rings, and additionally four real datasets frequently used for performance evaluation: Digits, Wine, Iris and Breast Cancer as provided by sci-kit<sup>[[4]](#optuna)</sup>. The data parameter settings for the three synthetic datasets (number of samples, amount of jitter etc.) correspond to the optimal settings from the Optimizations above. We used a single random seed to generate the illustrative synthetic datasets.

In a next step, using the Two Moons dataset as an example, we compare the AMI performances over ten random seeds with hyperparameter optimization (HPO) for [**Rock**](./notebooks/Optimizations/Two_Moons_ROCK_Hyperparameter_Search.ipynb) and [**DBSCAN**](./notebooks/Optimizations/Two_Moons_DBSCAN_Hyperparameter_Search.ipynb). We again used the TPE for the HPO of DBSCAN (here, the TPE was not intended to model a researcher's behavior, but was used as a classical HPO method). The comparison illustrates the effect of neglecting parameter optimization for competing algorithms. 

---
<a name="optuna">[1]</a> https://optuna.org/  
<a name="cluster">[2]</a> https://scikit-learn.org/stable/modules/clustering.html  
<a name="sampler">[3]</a> https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html  
<a name="sampler">[4]</a> https://scikit-learn.org/stable/datasets/toy_dataset.html
