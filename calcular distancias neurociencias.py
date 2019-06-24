import networkx as nx
import os 
import sunbeam
import numpy as np
import importlib
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

importlib.import_module("sunbeam")

#We will work with saga networks

dir_oldyoung="edge_list_graphs_coma" #"edge_list_graphs"
category1="Control"
category2="Patient"

colores={category1:'blue',category2:'red'}
Kcolores=list(colores.keys())
# Kcolores =  ['Control', 'Old', 'Patient', 'Young']
counter={category2:0,category1:0} # cuenta cuantos de cada categoria 'Old':0,"Young":0,


dic={} # columnas por categoria
dic[category1]=0
dic[category2]=1


#Calculating the maximum number of eigenvalues
def neigs(G):
	core=sunbeam.shave(G)
	dist=len(core.node)*2-2
	return dist

#distance between G1 y G2
#@numba.njit()
def dist(G1,G2):
	d1=sunbeam.nbdist(G1,G2,min(neigs(G1),neigs(G2)))
	return d1

def fine_tune(graph,topk):
        """Return fine-tuned eigenvalues."""
        eigs = sunbeam.nbeigs(graph, topk, fmt='1D')
        vals = np.abs(eigs)**eta
        eigs = eigs * vals
        eigs = np.array([(c.real * sigma, c.imag / sigma) for c in eigs])
        return eigs

#@numba.njit()
def dist_eigs(eigs1,eigs2):
	return np.linalg.norm(eigs1 - eigs2)


#	min_len = min(eigs1.shape[0], eigs2.shape[0])
#	eigs1 = eigs1[:min_len].T.flatten()
#	eigs2 = eigs2[:min_len].T.flatten()
	return np.linalg.norm(eigs1 - eigs2)

#data=[[],[]]
data=[]
files = os.listdir(dir_oldyoung)
cols = []
#Creando los grafos, y calculando los eigenvalores
for f in files[:]:
	#plt.figure(dirs.index(dir))
	#Se verifica cual de las palabras en el diccionario está en el nombre del archivo
	w=[(key in f) for key in Kcolores].index(True)
	#Se cuenta al counter el grafo correspondiente
	counter[Kcolores[w]]+=1
	#Se agrega a los colores el color correspondiente al tipo de grafo
	#cols.append(colores[Kcolores[w]])
	G = nx.read_edgelist(os.path.join(dir_oldyoung,f))
	#Se usa la funcion neigs() del archivo calcular distancias
	eigs = sunbeam.nbeigs(G, 80,fmt="1D") 
	data.append(eigs)
	cols.append(colores[Kcolores[w]])
#	for elem in eigs:
#		#Se agrega a los datos[key] el eigenvalor
#		data[dic[Kcolores[w]]].append(elem)
#		#De nuevo se agrega al counter la cantidad de eigenvalores
#		counter[Kcolores[w]]+=1



#numero total de eigenvalores
cm=sum([len(data[i]) for i in range(len(data)) ])
D=np.zeros((cm,2))
k=0


while k<cm:
	for i in range(len(data[0])):
		elem=data[0][i]
		D[k][0],D[k][1]=elem[0],elem[1]
		cols.append(colores[category1])
		k+=1
	for i in range(len(data[1])):
		elem=data[1][i]
		D[k][0],D[k][1]=elem[0],elem[1]
		cols.append(colores[category2])
		k+=1


embedding = umap.UMAP(a=None, angular_rp_forest=False, b=None, init='spectral',
     learning_rate=3.0, local_connectivity=1.0,
     metric=<function dist_eigs at 0x7f6295517b70>, metric_kwds=None,
     min_dist=0, n_components=2, n_epochs=30, n_neighbors=20,
     negative_sample_rate=5, random_state=None, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=42, verbose=True)


embedding=umap.UMAP(n_components=2,n_neighbors=25,spread=1,metric=dist_eigs,verbose=True)
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





fig = plt.figure()
plt.scatter(H[:,0],H[:,1],c=cols,s=5)
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

