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
def dist(G1,G2):
	d1=sunbeam.nbdist(G1,G2,min(neigs(G1),neigs(G2)))
	return d1


data=[[],[]]
files = os.listdir(dir_oldyoung)

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
	eigs = sunbeam.nbeigs(G, 80, fmt='2D') 
	for elem in eigs:
		#Se agrega a los datos[key] el eigenvalor
		data[dic[Kcolores[w]]].append(elem)
		#De nuevo se agrega al counter la cantidad de eigenvalores
		counter[Kcolores[w]]+=1



#numero total de eigenvalores
cm=sum([len(data[i]) for i in range(len(data)) ])
D=np.zeros((cm,2))
k=0

cols=[]

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


embedding = umap.UMAP(n_components=3,n_neighbors=15, min_dist=0)

H = embedding.fit_transform(D)
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

