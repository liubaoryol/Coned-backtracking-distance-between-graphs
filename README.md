# Graph distance
The code helps calculate distance between graphs using their topological properties. 

Four distances are available: 
- Spectral
- Wasserstein
- Gromov-Wasserstein
- Sunbeam

which are based on the article *"Non-backtracking cycles: length spectrum theory and graph mining applications"* by Leo Torres, Pablo Suárez-Serrato & Tina Eliassi-Rad


To run the code:

1. Download repository
2. Copy-paste the code in `neuro_umap.py` on a terminal 
3. Instantiate the directory where your graph data is into the variable `files` (or any other preferred name)
4. Run command `graph_distance(files,distance_type)`
