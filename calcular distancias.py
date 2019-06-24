import networkx as nx
import os 
import sunbeam
import numpy as np
import importlib
import umap
import matplotlib.pyplot as plt

importlib.import_module("sunbeam")


#We will work with saga networks
dir_saga = 'saga_networks/'
entries = os.listdir(dir_saga)

def sagaGraphGen(k,cols):
	nets = open( os.path.join(dir_saga,entries[k]),"r") #Reading egils_edges_friendly.csv files
	nets = nets.readlines()[1:]
	nets = ''.join(nets)
	nets= nets.replace(',',' ')
	nets= nets.replace('\n',' ')
	nets=nets.split()
	#nets = list(map(int,nets.split()))
	nets = [nets[x:x+2] for x in range(0,len(nets),cols)]
	nets = [[int(j) for j in i] for i in nets]
	G = nx.Graph() 
	G.add_edges_from(nets)
	return G


#Calculating the maximum number of eigenvalues
def neigs(G):
	core=sunbeam.shave(G)
	dist=len(core.node)*2-2
	return dist

#distance between G1 y G2
def dist(G1,G2):
	d1=sunbeam.nbdist(G1,G2,min(neigs(G1),neigs(G2)))
	return d1

G1=sagaGraphGen(0,3) #egils_edges_complete
G2=sagaGraphGen(1,2) # descent
G3=sagaGraphGen(3,5) #gisli_hostile
G4=sagaGraphGen(4,3) #descent sparse
G5=sagaGraphGen(5,2) #friendly
#G6=sagaGraphGen(6,4) #gisli_friendly
G7=sagaGraphGen(7,2) #unfriendly
#colores={'Complete':'purple','Descent':'yellow','Gisli_hostile':'orange','Sparse':'black','Friendly':'red','Unfriendly':'green'}
colores={'Descent':'yellow','Friendly':'red','Unfriendly':'green'}
#counter={'Old':0,"Young":0,"Patient":0,"Control":0} # cuenta cuantos de cada categoria

Kcolores=list(colores.keys()) 
Kcolores.sort()  # lista con labels



counter=neigs(G2)+neigs(G5)+neigs(G7)#+neigs(G4)+neigs(G5)+neigs(G7)
n1,n2,n3=sunbeam.nbeigs(G1,neigs(G1),'2D'),sunbeam.nbeigs(G2,neigs(G2),'2D'),sunbeam.nbeigs(G3,neigs(G3),'2D')
n4,n5,n7=sunbeam.nbeigs(G4,neigs(G4),'2D'),sunbeam.nbeigs(G5,neigs(G5),'2D'),sunbeam.nbeigs(G7,neigs(G7),'2D')

cols=[]
eigs_matrix=np.zeros((counter,2))
k=0
while k<counter:
	for i in range(len(n2)):
		elem=n2[i]
		eigs_matrix[k][0],eigs_matrix[k][1]=elem[0],elem[1]
		cols.append(colores['Descent'])
		k+=1
	for i in range(len(n5)):
		elem=n5[i]
		eigs_matrix[k][0],eigs_matrix[k][1]=elem[0],elem[1]
		cols.append(colores['Friendly'])
		k+=1
	for i in range(len(n7)):
		elem=n7[i]
		eigs_matrix[k][0],eigs_matrix[k][1]=elem[0],elem[1]
		cols.append(colores['Unfriendly'])
		k+=1
'''
	for i in range(len(n3)):
		elem=n3[i]
		eigs_matrix[k][0],eigs_matrix[k][1]=elem[0],elem[1]
		cols.append(colores['Gisli_hostile'])
		k+=1
	for i in range(len(n4)):
		elem=n4[i]
		eigs_matrix[k][0],eigs_matrix[k][1]=elem[0],elem[1]
		cols.append(colores['Sparse'])
		k+=1'''

'''
	for i in range(len(n1)):
		elem=n1[i]
		eigs_matrix[k][0],eigs_matrix[k][1]=elem[0],elem[1]
		cols.append(colores['Complete'])
		k+=1 '''


embedding = umap.UMAP(n_neighbors=75, metric='canberra',
                n_epochs=1000, min_dist=0.01, repulsion_strength=10,
                negative_sample_rate=50, transform_queue_size=10)

H = embedding.fit_transform(eigs_matrix)
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

