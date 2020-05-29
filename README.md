# Graph distance
Calculate distance between graphs. The following distances are supported:

|      Distance             |                      Description                                                         |
|:-------------------------:|:----------------------------------------------------------------------------------------:|
| spectral                  | This is the original python sunbeam distance                                        |
| relaxed_nbc           | Using nonbacktracking nonbacktracking eigenvalues                                    | 
| wasserstein_kde_dist  | Wasserstein distance between estimated distributions of nonbacktracking eigenvalues  | 
| distance_gr_wass      | Gromov-Wasserstein distance between nonbacktracking eigenvalue vectors               | 



## 🚀 Running code


* __Run on your local machine__
   * Clone this repository on your local machine. `git clone https://github.com/liubaoryol/graph_distance.git`
   * Install requirements: `pip install -r requirements.txt`
   * Open a terminal with the path where you cloned this repository `C:Users/desktop/graph_distance$ python`
   * Import `neuro_umap` library as follows 
   `>>> from neuro_umap import nbeigs_calculate, distance_gr_wass`
   * Example:
   `>>> eigs=nbeigs_calculate(graphs,'2D')`
   `>>> distance_gr_wass(eigs)`
       

## References
Motivated on the following articles:

 * Torres, L., Suárez-Serrato, P. & Eliassi-Rad, T.  <br/>
 [Non-backtracking Cycles: Length Spectrum
Theory and Graph Mining Applications](https://link.springer.com/article/10.1007/s41109-019-0147-y), <br/> 
   Appl Netw Sci 4, 41 (2019)
   
 * Achard, S., Delon-Martin, C., et al., <br/>
 [Hubs of brain functional networks are radically
reorganized in comatose patients](https://www.researchgate.net/publication/233775192_Hubs_of_brain_functional_networks_are_radically_reorganized_in_comatose_patients),  <br/>
   PNAS 109, 50  (2012)
   

