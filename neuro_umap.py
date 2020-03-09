import networkx as nx
import os 
import sunbeam
from sunbeam import *
import numpy as np
import importlib
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





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
cols = []
graphs=[]
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
	graphs.append(G)
	#Se usa la funcion neigs() del archivo calcular distancias
	eigs = sunbeam.nbeigs(G, 80,fmt="complex") 
	data.append(eigs)
	cols.append(colores[Kcolores[w]])
#	for elem in eigs:
#		#Se agrega a los datos[key] el eigenvalor
#		data[dic[Kcolores[w]]].append(elem)
#		#De nuevo se agrega al counter la cantidad de eigenvalores
#		counter[Kcolores[w]]+=1


for j in range(20):
	G = nx.erdos_renyi_graph(90,0.1)
	eigs = sunbeam.nbeigs(G, 80,fmt="1D") 
	data.append(eigs)
	cols.append(colores['ER'])
	G=nx.watts_strogatz_graph(90, 10, 0.5, seed=None) data
	eigs = sunbeam.nbeigs(G, 80,fmt="1D") 
	data.append(eigs)
	cols.append(colores['WS'])

#numero total de eigenvalores
cm=sum([len(data[i]) for i in range(len(data)) ])
D=np.zeros((cm,2))
k=0


graphs_of_young,graphs_of_old=[],[]
for i in range(len(graphs)):
	if cols[i]=="blue":
		graphs_of_young.append(graphs[i])
	else:
		graphs_of_old.append(graphs[i])

sortedgraphs=graphs_of_young+ graphs_of_old

distances = np.zeros([len(data),len(data)])
for i in range(len(data)):
	for j in range(len(data)):
		distances[i,j]=dist(sortedgraphs[i],sortedgraphs[j])


heat_map = sb.heatmap(distances,xticklabels=sorted(cols),yticklabels=sorted(cols),annot=True)


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

