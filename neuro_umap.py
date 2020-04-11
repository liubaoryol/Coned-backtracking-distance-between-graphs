import networkx as nx
import os 
import sunbeam
import numpy as np
import importlib
import seaborn as sb
import umap
import matplotlib.pyplot as plt
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


data=[]
files = os.listdir(dir_oldyoung)

#Creando los grafos, y calculando los eigenvalores
def preprocessing(files,eig_format,n_eigs = 180,count = False):
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
		G_watts=nx.watts_strogatz_graph(90, 10, 0.5, seed=None) data
		eigs_watts = sunbeam.nbeigs(G, 80,fmt="1D") 
		eigss_watts.append(eigs)
		cols_watts.append(colores['WS'])
	return eigss_erdos,cols_erdos,eigss_watts,cols_watts


##WASSERSTEIN DISTANCE
#Converting to fft
fft_data = [np.fft.fft(data[i]) for i in range(len(data))]
distances = np.zeros([len(data),len(data)])
for i in range(len(fft_data)):
	for j in range(len(fft_data)):
		distances[i,j]=ot.wasserstein_1d(abs(fft_data[i]),abs(fft_data[j]))

distances +=1
distances = np.tril(distances)
distances -=1
distances[distances==-1.]=None


dics = {"blue":"young","red":"old"}
labels = [dics[res] for res in sorted(cols)]

heat_map = sb.heatmap(distances,xticklabels=labels,yticklabels=labels)
plt.title("Wasserstein distance between fft data")
plt.show()


def nodes_of_degree1(G):
	lista = []
	for node in G.nodes:
		if G.degree(node)==1:
			lista.append(node)
	return lista

def add_cone(G):
	G.add_node("cone")
	node_list = nodes_of_degree1(G)
	for each in node_list:
		G.add_edge(*(each,'cone'))

#HEAT MAP OF SUNBEAM +EUCLIDEAN DISTANCES BETWEEN GRAPHS
def distance(graphs,cols,dim_len_spec=1,coned = False):
	"""
	dist_type: euclidean
	graphs1 <----young
	graphs2 <---- old
	"""
	graphs1,graphs2=[],[]
	labels=[]
	for i in range(len(graphs)):
		if cols[i]=="blue":
			graphs1.append(graphs[i])
		else:
			graphs2.append(graphs[i])
		
	
	sortedgraphs=graphs1 + graphs2
	labels = ["young" for j in range(len(graphs1))]+["old" for j in range(len(graphs2))]
	
	if coned:
		for G in sortedgraphs:
			add_cone(G)
	
	distances = np.zeros([len(graphs),len(graphs)])
	for i in range(len(graphs)):
		for j in range(len(graphs)):
			distances[i,j]=sunbeam.dist(sortedgraphs[i],sortedgraphs[j],dim_len_spec)
	
	distances = np.tril(distances)
	distances[distances==0.]=None
	
	if coned:
		title= "Sunbeam euclidean distance between coned graphs"
	else: 
		title= "Sunbeam euclidean distance between graphs"
	
	heat_map = sb.heatmap(distances,xticklabels=labels,yticklabels=labels)
	plt.title(title)
	plt.show()


#HEAT MAP OF SUNBEAM +GROMOV-WASSERSTEIN DISTANCES BETWEEN GRAPHS
import ot
import scipy as sp

def distance_gromov(complex_eig_data,cols,coned=False):
	"""
	dist_type: gromov-wasserstein
	eigs1 <----young
	eigs2 <---- old
	"""
	complex_eigs1,complex_eigs2=[],[]
	for i in range(len(graphs)):
		if cols[i]=="blue":
			eigs1.append(eig_data[i])
			complex_eigs1.append(complex_eig_data[i])
		else:
			complex_eigs2.append(complex_eig_data[i])
	
	eigs1 = np.array([(c.real, c.imag) for c in complex_eigs1]).T.flatten()
	eigs2 = np.array([(c.real, c.imag) for c in complex_eigs2]).T.flatten()

	sortedeigs=eigs1+ eigs2
	sortedeigs_complex = complex_eigs1+complex_eigs2



	distances = np.zeros([len(data),len(data)])
	for i in range(len(data)):
		for j in range(len(data)):
			C1 = sp.spatial.distance.cdist(sortedeigs[i],sortedeigs[i])
			C2 = sp.spatial.distance.cdist(sortedeigs[j],sortedeigs[j])
			C1 /= C1.max()
			C2 /= C2.max()
			gw0, log0 = ot.gromov.gromov_wasserstein2(
			    C1, C2, sortedeigs_complex[i], sortedeigs_complex[j], 'square_loss', verbose=True, log=True)
			distances[i,j]=abs(gw0)
		
	distances = np.tril(distances)
	distances[distances==0.]=None
		
	if coned:
		title= "Sunbeam Gromov-Wasserstein distances between coned graphs"
	else: 
		title= "Sunbeam Gromov-Wasserstein distances between graphs"
	
	heat_map = sb.heatmap(distances,xticklabels=labels,yticklabels=labels)
	plt.title(title)
	plt.show()



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

