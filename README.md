# Graph distance
A python module to calculate distance between graphs. The code helps calculate distance between graphs using their topological properties. The following distances are supported:

| **pydistance**              | **Description**            | 
|:-------------------------:|:----------------------------------------------------------------------------------------:|
| **spectral**              | **This is the original python sunbeam distance**                                         |
| **relaxed_nbc**           | **Using nonbacktracking nonbacktracking eigenvalues**                                    | 
| **wasserstein_kde_dist**  | **Wasserstein distance between estimated distributions of nonbacktracking eigenvalues**  | 
| **distance_gr_wass**      | **Gromov-Wasserstein distance between nonbacktracking eigenvalue vectors**               | 



## 🚀 Running code


* __Run on your local machine__
   * Clone this repository on your local machine.
   * Open a terminal with the path where you clone this repository.
   * Create a virtual environment,(see this [link](https://gist.github.com/mevangelista-alvarado/8ee2fd663e7446e543fc04eacce0f303))
   
   * Install the requirements:
      ```
      (venv_name) C:Users/dekstop/graph_distance$ pip install
      ```
   * Instantiate the directory where your graph data is into the variable `files` (or any other preferred name)
   * Run command `graph_distance(files,distance_type)`



## References
Some of the functions in this module have been used to obtain the results in these articles:

 * Torres, L., Suárez-Serrato, P. & Eliassi-Rad, T.  <br/>
 [Non-backtracking Cycles: Length Spectrum
Theory and Graph Mining Applications](https://link.springer.com/article/10.1007/s41109-019-0147-y), <br/> 
   Appl Netw Sci 4, 41 (2019)
   
 * Achard, S., Delon-Martin, C., et al., <br/>
 [Hubs of brain functional networks are radically
reorganized in comatose patients](https://www.researchgate.net/publication/233775192_Hubs_of_brain_functional_networks_are_radically_reorganized_in_comatose_patients),  <br/>
   PNAS 109, 50  (2012)
   

