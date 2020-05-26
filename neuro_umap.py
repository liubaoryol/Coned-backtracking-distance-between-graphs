import networkx as nx
import os 
import sunbeam
import numpy as np
import importlib
import seaborn as sb
import umap
import matplotlib.pyplot as plt
import ot
import scipy as sp
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D


def files_to_graphs(files, coned = False, count = False):
	graphs=[]
	cols = []
	for f in files:
		G = nx.read_edgelist(os.path.join(dir_comacontrol,f))
		if coned:
			G.add_node("cone")
			node_list=nodes_of_degree1(G)
			for each in node_list:
				G.add_edge(*(each,"cone"))
		graphs.append(G)
		#Counting number of observations per class
		w=[(key in f) for key in Kcolores].index(True)
		counter[Kcolores[w]]+=1 
		cols.append(colores[Kcolores[w]])
	if count:
		return graphs, cols, counter
	return graphs, cols

#Creating graph instances of files and calculating non backtracking eigenvalues
def nbeigs_calculate(graphs,eig_format,n_eigs = "max"):
	"""
	INPUT:
	eig_format: complex,1D,2D
	n_eigs: maximum number of eigenvalues to calculate. Default is maximum possible
	count: boolean to count number of instances per category
	OUTPUT:
	eigss: nonbacktracking eigenvalues
	graphs: networkx instances
	cols: color labels representing the categories
	counter: number of categories. Only if count=True
	"""
	eigss=[]
	for G in graphs:
		#calculate maximum possible number of eigenvalues
		if n_eigs=="max":
			core=sunbeam.shave(G)
			n_eigs=len(core.node)*2-2

		eigs = sunbeam.nbeigs(G, n_eigs,fmt=eig_format)
		eigss.append(eigs)
	
	return eigss


#Create Erdos-Renyi and Watts-Strogatz graphs for reference
def synthetic_graphs(n):
	eigss_erdos,eigss_watts=[],[]
	cols_erdos,cols_watts=[],[]
	for j in range(n):
		G_erdos = nx.erdos_renyi_graph(90,0.1)
		eigs_erdos = sunbeam.nbeigs(G, 80,fmt="1D") 
		eigss_erdos.append(eigs)
		cols_erdos.append(colores['ER'])
		G_watts=nx.watts_strogatz_graph(90, 10, 0.5, seed=None)
		eigs_watts = sunbeam.nbeigs(G, 80,fmt="1D") 
		eigss_watts.append(eigs)
		cols_watts.append(colores['WS'])
	return eigss_erdos,cols_erdos,eigss_watts,cols_watts


def nodes_of_degree1(G):
	lista = []
	for node in G.nodes:
		if G.degree(node)==1:
			lista.append(node)
	return lista


def relaxed_nbc(graphs,cols=None,dim_len_spec=5,coned = False,classes=["patient","control"]):
	"""
	DESCRIPTION: Calculates the distance between the RELAXED length spectrum of each graph
	
	INPUT:
	
	dist_type: euclidean
	graphs1 <----young
	graphs2 <---- old
	
	OUTPUT: heatmap of distances
	"""
	if cols!=None:
		graphs1,graphs2=[],[]
		#Sorting graphs for visualization purposes
		for i in range(len(graphs)):
			if cols[i]=="blue":
				graphs1.append(graphs[i].copy())
			else:
				graphs2.append(graphs[i].copy())
			
		
		sortedgraphs=graphs1 + graphs2
		sortedlabels = [classes[0] for j in range(len(graphs1))]+[classes[1] for j in range(len(graphs2))]
	else: 
		sortedgraphs=graphs
		sortedlabels=["temp label" for j in range(len(graphs))]
	
	
	print("Calculating distance matrix...")
	distances = np.zeros([len(graphs),len(graphs)])
	for i in range(len(graphs)):
		for j in range(len(graphs)):
			try:
				distances[i,j]=sunbeam.dist(sortedgraphs[i],sortedgraphs[j],dim_len_spec)
			except:
				print("Exception occured at "+str(i)+"-"+str(j) + " positions due to eigenvalue calculation")
				distances[i,j]=None
				continue
	
	distances = np.tril(distances)
	distances[distances==0.]=None
	for i in range(len(distances)):
		if not distances[i,i] >=0.:
			distances[i,i]=0.
	
	if coned:
		title= "Relaxed nonbacktracking spectrum distance between coned graphs"
	else: 
		title= "Relaxed nonbacktracking spectrum distance between graphs"
	
	heat_map = sb.heatmap(distances,xticklabels=sortedlabels,yticklabels=sortedlabels)
	plt.title(title)
	plt.show()





def distance_gr_wass(eig_data,cols=None,coned=False,classes=["patient","control"]):
	"""
	DESCRIPTION: calculates gromov wasserstein distance between the feature vectors of each graph. 
	The feature vector consists of eigenvalues of nonbacktracking matrix
	
	dist_type: gromov-wasserstein
	eigs1 <----young
	eigs2 <---- old
	OUTPUT: heatmap of distances
	"""
	#Sorting graphs for visualization purposes
	if cols!=None:
		eigs1,eigs2=[],[]
		for i in range(len(graphs)):
			if cols[i]=="blue":
				eigs1.append(eig_data[i])
			else:
				eigs2.append(eig_data[i])

		sortedeigs=eigs1 + eigs2
		sortedlabels = [classes[0] for j in range(len(eigs1))]+[classes[1] for j in range(len(eigs2))]
	else: 
		sortedgraphs=graphs
		sortedlabels=["temp label" for j in range(len(graphs))]
	
	print("Calculating distance matrix...")
	distances = np.zeros([len(eig_data),len(eig_data)])
	for i in range(len(eig_data)):
		for j in range(len(eig_data)):
			#Construction of dissimilarity function between bins of each histogram/feature vector
			C1 = sp.spatial.distance.cdist(sortedeigs[i],sortedeigs[i])
			C2 = sp.spatial.distance.cdist(sortedeigs[j],sortedeigs[j])
			C1 /= C1.max()
			C2 /= C2.max()
			n_samples=len(sortedeigs[i])
			p=ot.unif(n_samples)
			n_samples=len(sortedeigs[j])
			q=ot.unif(n_samples)
			gw0, log0 = ot.gromov.gromov_wasserstein2(
			    C1, C2, p,q, 'square_loss', verbose=True, log=True)
			distances[i,j]=abs(gw0)
		
	distances = np.tril(distances)
	distances[distances==0.]=None
	for i in range(len(distances)):
		if not distances[i,i] >=0.:
			distances[i,i]=0.
		
	if coned:
		title= "Gromov-Wasserstein distances between coned graphs"
	else: 
		title= "Gromov-Wasserstein distances between graphs"
	
	heat_map = sb.heatmap(distances,xticklabels=sortedlabels,yticklabels=sortedlabels)
	plt.title(title)
	plt.show()



def spectral_distance(eigs,cols = None,coned = False,classes=["patient","control"]):
	"""
	DESCRIPTION: Calculates the distances between truncated feature vectors of nonbacktracking eigenvalues
	dist_type: euclidean
	graphs1 <----young
	graphs2 <---- old
	"""
	#Sorting eigs for visualization purposes
	if cols == None:
		eigs1,eigs2=[],[]
		for i in range(len(graphs)):
			if cols[i]=="blue":
				eigs1.append(eigs[i])
			else:
				eigs2.append(eigs[i])
			
		
		sortedeigs=eigs1 + eigs2
		sortedlabels = [classes[0] for j in range(len(eigs1))]+[classes[1] for j in range(len(eigs2))]
	else: 
		sortedeigs=eigs
		sortedlabels=["temp label" for j in range(len(graphs))]
	
	print("Calculating distance matrix...")
	distances = np.zeros([len(eigs),len(eigs)])
	for i in range(len(eigs)):
		for j in range(len(eigs)):
			n=min(len(sortedeigs[i]),len(sortedeigs[j]))
			distances[i,j]=np.linalg.norm(sortedeigs[i][:n]-sortedeigs[j][:n])
	distances = np.tril(distances)
	distances[distances==0.]=None

	for i in range(len(distances)):
		if not distances[i,i] >=0.:
			distances[i,i]=0.
		
	
	if coned:
		title= "Spectral distance between coned graphs"
	else: 
		title= "Spectral distance between graphs"
	
	heat_map = sb.heatmap(distances,xticklabels=sortedlabels,yticklabels=sortedlabels)
	plt.title(title)
	plt.show()


#Wasserstein distance between estimated density distributions of eigenvalues.
def wasserstein_kde_dist(eigs,cols,dist_type="sample",bw = 2, ker = 'gaussian',sample_size=60,coned=False,classes=["patient","control"]):
	"""
	dist_type---sample,grid
	bw --- bandwidth of KDE
	ker--- kernel of KDE
	"""
	if cols!=None:
		#Sorting eigs for visualization purposes
		eigs1,eigs2=[],[]
		for i in range(len(eigs)):
			if cols[i]=="blue":
				eigs1.append(eigs[i])
			else:
				eigs2.append(eigs[i])
		eigs_data=eigs1 + eigs2 
		sortedlabels = [classes[0] for j in range(len(eigs1))]+[classes[1] for j in range(len(eigs2))]
		
	#Create a list of Kernel density objects associated to each connectome observation
	model = [KernelDensity(bandwidth=bw, kernel=ker) for eigs in eigs_data]
	#Setting limits for the Wasserstein grid evaluation, for each connectome. Limits are determined by maximum and minimum real & imaginary part of observed eigenvalue
	limits_x=[]
	limits_y=[]
	#Generating new samples of same dimension from each fitted KDE models on nonbacktracking eigenvalues
	samples=[]
	for j in range(len(eigs_data)):
		model[j].fit(eigs_data[j])
		limits_x.append([np.min(eigs_data[j][:,0]),np.max(eigs_data[j][:,0])])
		limits_y.append([np.min(eigs_data[j][:,1]),np.max(eigs_data[j][:,1])])
		samples.append(model[j].sample(sample_size))

	#Calculating distance matrix between samples generated by distribution estimation of original eigenvalue observations
	if dist_type=="sample":
		distances = np.zeros([len(eigs_data),len(eigs_data)])
		for i in range(len(eigs_data)):
			for j in range(len(eigs_data)):
				C = sp.spatial.distance.cdist(samples[i],samples[j])
				C /= C.max()
				#We assign weight 1/n to each eigenvalue for a uniform distribution
				distances[i,j] = ot.emd2(ot.unif(sample_size), ot.unif(sample_size), C)
				
			
		distances = np.tril(distances)
		distances[distances==0.]=None

	#Calculating distance matrix of distribution
	if dist_type=="grid":
		grid_size=np.int(np.floor(np.sqrt(sample_size)))
		print("Calculating distances between distributions on grid")
		limitx=[np.max(limits_x[:][0]),np.min(limits_x[:][1])]
		limity=[np.max(limits_y[:][0]),np.min(limits_y[:][1])]
		n1=np.linspace(*limitx, grid_size)
		n2=np.linspace(*limity, grid_size)
		grid=np.zeros([len(model),grid_size,grid_size])
		for n in range(len(model)):
			for i in range(len(n1)):
				for j in range(len(n2)):
					grid[n,i,j]=model[n].score_samples([[n1[i],n2[j]]])
		
		distances = np.zeros([len(eigs_data),len(eigs_data)])
		for i in range(len(model)):
			for j in range(len(model)):
				vector_i=[np.array([n1[l],n1[k],grid[i,l,k]]) for l in range(len(n1)) for k in range(len(n2))]
				vector_j=[np.array([n1[l],n1[k],grid[j,l,k]]) for l in range(len(n1)) for k in range(len(n2))]
				C = sp.spatial.distance.cdist(vector_i,vector_j)
				C /= C.max()
				distances[i,j]=ot.emd2(ot.unif(len(C)),ot.unif(len(C)),C)
		distances = np.tril(distances)
		distances[distances==0.]=None

	for i in range(len(distances)):
		if not distances[i,i] >=0.:
			distances[i,i]=0.
	if coned:
		title= "Wasserstein-KDE distance between coned graphs"
	else: 
		title= "Wasserstein-KDE distance between graphs"

	heat_map = sb.heatmap(distances,xticklabels=sortedlabels,yticklabels=sortedlabels)
	plt.title(title)
	plt.show()





#Putting it all together
def graph_distance(files,dist_type,n_eigs = 180,dim_len_spec=1,count = False,sample_size=60,coned = False):
	graphs, cols = files_to_graphs(files)	
	eigss=nbeigs_calculate(graphs, "2D", n_eigs, count, coned)

	if dist_type=="relaxed_nbc":
		relaxed_nbc(graphs,cols,dim_len_spec,coned)

	if dist_type=="gromov-wasserstein":
		distance_gr_wass(eigss,cols,coned)

	if dist_type=="wasserstein-grid":
		wasserstein_kde_dist(eigss,cols,dist_type="grid",sample_size=sample_size,coned=coned)

	if dist_type=="wasserstein-sample":
		wasserstein_kde_dist(eigss,cols,dist_type="sample",sample_size=sample_size,coned=coned)

	if dist_type=="spectral":
		spectral_distance(eigss,cols,coned)






#Define categories of graph data
dir_oldyoung="edge_list_graphs" 
dir_comacontrol="edge_list_graphs_coma"
#category1="Young"
#category2="Old"
category1="Patient"
category2="Control"

#Color identifiers for each category
colores={category1:'blue',category2:'red'}
Kcolores=list(colores.keys())

counter={category2:0,category1:0} 

dic={} 
dic[category1]=0
dic[category2]=1




#files = os.listdir(dir_oldyoung)
files = os.listdir(dir_comacontrol)

embedding=umap.UMAP(n_components=2,n_neighbors=25,spread=2,metric=dist_eigs,verbose=True)
H = embedding.fit_transform(data,y=cols)
fig = plt.figure()
plt.scatter(H[:,0],H[:,1],c=cols,s=5)
plt.show()


embedding=umap.UMAP(n_components=3,n_neighbors=25,spread=1,metric=dist_eigs,verbose=True)
H = embedding.fit_transform(data,y=cols)
fig = plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(H[:,0],H[:,1],H[:,2],c=cols,s=5)
plt.show()









"""
nx.draw(G_coma[1], with_labels=True, font_weight='bold')
plt.subplot(224)
nx.draw(G_old[1], with_labels=True, font_weight='bold')
plt.show()
np.savetxt("coma_props.csv",coma_props,delimiter=",",fmt="%s")
nx.degree_centrality(Graphs[0])
nx.eigenvector_centrality(Graphs[0])
"""

