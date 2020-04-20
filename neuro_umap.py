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




#Aqui se definen las categorias, la ruta de carpetas, y todo sobre los datos
dir_oldyoung="edge_list_graphs" 
category1="Young"
category2="Old"
category3="WS"
category4="ER"


colores={category1:'blue',category2:'red',category3:'green',category4:'yellow'}
Kcolores=list(colores.keys())
# Kcolores =  ['Control', 'Old', 'Patient', 'Young']
counter={category2:0,category1:0,category3:0,category4:0} # cuenta cuantos de cada categoria 'Old':0,"Young":0,


dic={} # columnas por categoria
dic[category1]=0
dic[category2]=1
dic[category3]=2
dic[category4]=3




#	min_len = min(eigs1.shape[0], eigs2.shape[0])
#	eigs1 = eigs1[:min_len].T.flatten()
#	eigs2 = eigs2[:min_len].T.flatten()



files = os.listdir(dir_oldyoung)

#Creando los grafos, y calculando los eigenvalores
def preprocessing(files,eig_format,n_eigs = 180,count = False,coned = False):
	"""
	eig_format: complex,1D,2D
	n_eigs: maximum number of eigenvalues to calculate
	count: boolean to count distribution of data
	"""
	eigss=[]
	cols = []
	graphs=[]
	for f in files[:]:
		G = nx.read_edgelist(os.path.join(dir_oldyoung,f))
		if coned:
			G.add_node("cone")
			node_list=nodes_of_degree1(G)
			for each in node_list:
				G.add_edge(*(each,"cone"))
		graphs.append(G)
		#Eigenvalues of the backpropagation matrix
		eigs = sunbeam.nbeigs(G, n_eigs,fmt=eig_format)

		#Se verifica cual de las palabras en el diccionario está en el nombre del archivo
		w=[(key in f) for key in Kcolores].index(True)
		#Se cuenta al counter el grafo correspondiente
		counter[Kcolores[w]]+=1 
		eigss.append(eigs)
		cols.append(colores[Kcolores[w]])
	if count: 
		return eigss,graphs, cols, counter
	else:
		return eigss,graphs, cols



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


#HEAT MAP OF SUNBEAM +EUCLIDEAN DISTANCES BETWEEN GRAPHS
def distance_sunbeam(graphs,cols,dim_len_spec=1,coned = False):
	"""
	dist_type: euclidean
	graphs1 <----young
	graphs2 <---- old
	"""
	graphs1,graphs2=[],[]
	labels=[]
	for i in range(len(graphs)):
		if cols[i]=="blue":
			graphs1.append(graphs[i].copy())
		else:
			graphs2.append(graphs[i].copy())
		
	
	sortedgraphs=graphs1 + graphs2
	labels = ["young" for j in range(len(graphs1))]+["old" for j in range(len(graphs2))]

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
		title= "Sunbeam distance between coned graphs"
	else: 
		title= "Sunbeam distance between graphs"
	
	heat_map = sb.heatmap(distances,xticklabels=labels,yticklabels=labels)
	plt.title(title)
	plt.show()


#HEAT MAP OF SUNBEAM +GROMOV-WASSERSTEIN DISTANCES BETWEEN GRAPHS


def distance_gromov(complex_eig_data,cols,coned=False):
	"""
	dist_type: gromov-wasserstein
	eigs1 <----young
	eigs2 <---- old
	"""
	complex_eigs1,complex_eigs2=[],[]
	labels=[]
	for i in range(len(graphs)):
		if cols[i]=="blue":
			complex_eigs1.append(complex_eig_data[i])
		else:
			complex_eigs2.append(complex_eig_data[i])
	
	eigs1 = [np.array((c.real, c.imag)).T for c in complex_eigs1]
	eigs2 = [np.array((c.real, c.imag)).T for c in complex_eigs2]

	sortedeigs=eigs1 + eigs2
	sortedeigs_complex = np.array(complex_eigs1+complex_eigs2)
	labels = ["young" for j in range(len(complex_eigs1))]+["old" for j in range(len(complex_eigs2))]
	print("Calculating distance matrix...")
	distances = np.zeros([len(complex_eig_data),len(complex_eig_data)])
	for i in range(len(complex_eig_data)):
		for j in range(len(complex_eig_data)):
			C1 = sp.spatial.distance.cdist(sortedeigs[i],sortedeigs[i])
			C2 = sp.spatial.distance.cdist(sortedeigs[j],sortedeigs[j])
			C1 /= C1.max()
			C2 /= C2.max()
			gw0, log0 = ot.gromov.gromov_wasserstein2(
			    C1, C2, sortedeigs_complex[i], sortedeigs_complex[j], 'square_loss', verbose=True, log=True)
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
	
	heat_map = sb.heatmap(distances,xticklabels=labels,yticklabels=labels)
	plt.title(title)
	plt.show()



def spectral_distance(eigs,cols,coned = False):
	"""
	dist_type: euclidean
	graphs1 <----young
	graphs2 <---- old
	"""
	eigs1,eigs2=[],[]
	labels=[]
	for i in range(len(graphs)):
		if cols[i]=="blue":
			eigs1.append(eigs[i])
		else:
			eigs2.append(eigs[i])
		
	
	sortedeigs=eigs1 + eigs2
	labels = ["young" for j in range(len(eigs1))]+["old" for j in range(len(eigs2))]

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
	
	heat_map = sb.heatmap(distances,xticklabels=labels,yticklabels=labels)
	plt.title(title)
	plt.show()



def wasserstein_kde_dist(eigs,cols,dist_type="sample",coned=False):
	"""
	dist_type=sample,grid
	"""
	eigs1,eigs2=[],[]
	for i in range(len(eigs)):
		if cols[i]=="blue":
			eigs1.append(eigs[i])
		else:
			eigs2.append(eigs[i])
	eigs_data=eigs1 + eigs2 #ordered set of eigenvalues
	labels = ["young" for j in range(len(eigs1))]+["old" for j in range(len(eigs2))]

	model = [KernelDensity(bandwidth=2, kernel='gaussian') for eigs in eigs_data]
	limits_x=[]
	limits_y=[]
	samples=[]
	for j in range(len(eigs_data)):
		model[j].fit(eigs_data[j])
		limits_x.append([np.min(eigs_data[j][:,0]),np.max(eigs_data[j][:,0])])
		limits_y.append([np.min(eigs_data[j][:,1]),np.max(eigs_data[j][:,1])])
		samples.append(model[j].sample(30))

	if dist_type=="sample":
		distances = np.zeros([len(eigs_data),len(eigs_data)])
		for i in range(len(eigs_data)):
			for j in range(len(eigs_data)):
				C = sp.spatial.distance.cdist(samples[i],samples[j])
				C /= C.max()
				#We assign weight 1 to each eigenvalue
				gw0 = ot.emd2(np.ones(30), np.ones(30), C)
				distances[i,j]=gw0
			
		distances = np.tril(distances)
		distances[distances==0.]=None
	if dist_type=="grid":
		limitx=[np.max(limits_x[:][0]),np.min(limits_x[:][1])]
		limity=[np.max(limits_y[:][0]),np.min(limits_y[:][1])]
		n1=np.linspace(*limitx, 80)
		n2=np.linspace(*limity, 80)
		grid=np.zeros([len(model),80,80])
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
				print("length of vector i and vector j is "+str(len(vector_i))+str(len(vector_j))+" respectively")
				distances[i,j]=ot.emd2(np.ones(len(C)),np.ones(len(C)),C)
		distances = np.tril(distances)
		distances[distances==0.]=None

	for i in range(len(distances)):
		if not distances[i,i] >=0.:
			distances[i,i]=0.
	if coned:
		title= "Wasserstein-KDE distance between coned graphs"
	else: 
		title= "Wasserstein-KDE distance between graphs"

	heat_map = sb.heatmap(distances,xticklabels=labels,yticklabels=labels)
	plt.title(title)
	plt.show()

def graph_distance(files,dist_type,n_eigs = 180,dim_len_spec=1,count = False,coned = False):
	eigss,graphs,cols=preprocessing(files, "2D", n_eigs, count, coned)

	if dist_type=="sunbeam":
		distance_sunbeam(graphs,cols,dim_len_spec,coned)

	if dist_type=="gromov-wasserstein":
		complex_eigs,graphs,cols=preprocessing(files, "complex", n_eigs, count, coned)
		distance_gromov(complex_eigs,cols,coned)

	if dist_type=="wasserstein-grid":
		wasserstein_kde_dist(eigss,cols,dist_type="grid",coned=coned)

	if dist_type=="wasserstein-sample":
		wasserstein_kde_dist(eigss,cols,dist_type="sample",coned=coned)

	if dist_type=="spectral":
		spectral_distance(eigs,cols,coned)




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

