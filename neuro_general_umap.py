import networkx as nx
import os 
import sunbeam
import numpy as np
import importlib
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

importlib.import_module("sunbeam")

"""INSTRUCTIONS
---Introduce directory of neural networs and the classes of the networks in ALPHABETICAL ORDER
---You can also create neural networks such as Erdos-Renyi or Watts-Strogatz for comparison
		If you will create new networks, do that in the code around line 100 where it starts with: for j in range(20)
---Choose the colors to distinguish the classes
"""

#dir_base="spectralconnectomes-master/"
dir_nets="spectralconnectomes/Coma/Parcel_90/Thresh_400" 
types="Coma/"
parcel="Parcel_90/"
thresh="Thresh_400/"
dir_name=types+parcel+thresh
category1="Control"
category2="Patient"
#category3="WS"
#category4="ER"

if not os.path.exists('results/'):
    os.system('mkdir results/')
if not os.path.exists('results/'+types):
    os.system('mkdir results/'+types)
if not os.path.exists('results/'+types+parcel):
    os.system('mkdir results/'+types+parcel)
if not os.path.exists('results/'+types+parcel+thresh):
    os.system('mkdir results/'+types+parcel+thresh)
colores={category1:'blue',category2:'red'} #,category3:'green',category4:'yellow'}

Kcolores=list(colores.keys()) # Kcolores =  ['Control', 'Old', 'Patient', 'Young']
counter={category2:0,category1:0}#,category3:0,category4:0} # cuenta cuantos de cada categoria 'Old':0,"Young":0,



dic={} # columnas por categoria
dic[category1]=0
dic[category2]=1
#dic[category3]=2
#dic[category4]=3


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
files = os.listdir(dir_nets)
if types=="HCP/":
	files.sort()

cols = []
n_eigs =[]
for f in files[:]:
	G = nx.read_edgelist(os.path.join(dir_nets,f))
	n_eigs.append(neigs(G))
max_eigs = min(n_eigs)
#Creando los grafos, y calculando los eigenvalores de redes neuronales sacadas de fMRI
for f in files[:]:
	#plt.figure(dirs.index(dir))
	#Se verifica cual de las palabras en el diccionario está en el nombre del archivo
	w=[(key in f) for key in Kcolores].index(True)
	#Se cuenta al counter el grafo correspondiente
	counter[Kcolores[w]]+=1
	#Se agrega a los colores el color correspondiente al tipo de grafo
	#cols.append(colores[Kcolores[w]])
	G = nx.read_edgelist(os.path.join(dir_nets,f))
	#Se usa la funcion neigs() pero tienes que ser la misma cantidad de eigenvalores calculados para todo caso
	eigs = sunbeam.nbeigs(G, max_eigs,fmt="1D") 
	data.append(eigs)
	cols.append(colores[Kcolores[w]])
#	for elem in eigs:
#		#Se agrega a los datos[key] el eigenvalor
#		data[dic[Kcolores[w]]].append(elem)
#		#De nuevo se agrega al counter la cantidad de eigenvalores
#		counter[Kcolores[w]]+=1


#Aquí se crean las gráficas adicionales, tales como Erdos-Renyi, etc. y se calculan sus eigenvalores
"""for j in range(20):
	G = nx.erdos_renyi_graph(90,0.1)
	eigs = sunbeam.nbeigs(G, 80,fmt="1D") 
	data.append(eigs)
	cols.append(colores['ER'])
	G=nx.watts_strogatz_graph(90, 10, 0.5, seed=None) 
	eigs = sunbeam.nbeigs(G, 80,fmt="1D") 
	data.append(eigs)
	cols.append(colores['WS'])
"""
#numero total de eigenvalores
cm=sum([len(data[i]) for i in range(len(data)) ])
D=np.zeros((cm,2))
k=0

"""
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
"""



#splitting data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, cols, test_size=0.3)



embedding=umap.UMAP(n_components=2,n_neighbors=25,spread=2,metric=dist_eigs,verbose=True,n_epochs=500)
#H = embedding.fit_transform(data,y=cols)
H = embedding.fit_transform(data,y=cols)
#adding legend
one = mpatches.Patch(facecolor=colores[category1], label=category1, linewidth = 0.5, edgecolor = 'black')
two = mpatches.Patch(facecolor=colores[category2], label = category2, linewidth = 0.5, edgecolor = 'black')
fig = plt.figure()
plt.scatter(H[:,0],H[:,1],c=cols,s=5)
legend = plt.legend(handles=[one, two], loc = 4, fontsize = 'small', fancybox = True)
#plt.title("Title")
plt.savefig('results/'+dir_name+"/2D.png")
#plt.show()




embedding_3d=umap.UMAP(n_components=3,n_neighbors=25,spread=1,metric=dist_eigs,verbose=True,n_epochs=500)
#H_3d = embedding_3d.fit_transform(data,y=cols)
H_3d = embedding_3d.fit(X_train,y=y_train)
H_3d = embedding_3d.transform(data)

one = mpatches.Patch(facecolor=colores[category1], label=category1, linewidth = 0.5, edgecolor = 'black')
two = mpatches.Patch(facecolor=colores[category2], label = category2, linewidth = 0.5, edgecolor = 'black')
fig = plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(H_3d[:,0],H_3d[:,1],H_3d[:,2],c=cols,s=5)
legend = plt.legend(handles=[one, two], loc = 4, fontsize = 'small', fancybox = True)
#plt.title("Title")
plt.savefig('results/'+dir_name+"/3D.png")
#plt.show()



#Connecting dots in scatter plot for HCP only.
#G = [H[i:i+2] for i in range(0,len(H),2)]
G = [H[i:i+2].transpose() for i in range(0,len(H),2)]
fig = plt.figure()
for i in range(len(G)):
	plt.plot(G[i][0],G[i][1],linewidth=0.2)
	plt.plot(G[i][0][0], G[i][1][0], 'ob',markersize=2)
	plt.plot(G[i][0][1], G[i][1][1], 'or',markersize=2)





#Training with SVC and KNN
embedding=umap.UMAP(n_components=2,n_neighbors=10,spread=2,metric=dist_eigs,verbose=True,n_epochs=500)
#H = embedding.fit_transform(data,y=cols)
trans = embedding.fit(X_train,y=y_train)
#trans = embedding.fit(X_train)

test_embedding = embedding.transform(X_test)


plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=y_train, cmap='Spectral')
plt.title('Embedding of the test set by UMAP', fontsize=24);

plt.scatter(test_embedding[:, 0], test_embedding[:, 1], s= 5, c=y_test, cmap='Spectral')
plt.title('Embedding of the test set by UMAP', fontsize=24);



svc = SVC().fit(trans.embedding_, y_train)
knn = KNeighborsClassifier().fit(trans.embedding_, y_train)


svc.score(trans.transform(X_test), y_test), knn.score(trans.transform(X_test), y_test)





embedding_3d=umap.UMAP(n_components=3,n_neighbors=25,spread=1,metric=dist_eigs,verbose=True,n_epochs=500)
#H_3d = embedding_3d.fit_transform(data,y=cols)
H_3d = embedding_3d.fit(X_train,y=y_train)
H_3d_test = embedding_3d.transform(X_test)

one = mpatches.Patch(facecolor=colores[category1], label=category1, linewidth = 0.5, edgecolor = 'black')
two = mpatches.Patch(facecolor=colores[category2], label = category2, linewidth = 0.5, edgecolor = 'black')
fig = plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(H_3d.embedding_[:,0],H_3d.embedding_[:,1],H_3d.embedding_[:,2],c=y_train,s=5)

ax.scatter(H_3d_test[:,0],H_3d_test[:,1],H_3d_test[:,2],c=y_test,s=5)


"""
nx.draw(G_coma[1], with_labels=True, font_weight='bold')
plt.subplot(224)
nx.draw(G_old[1], with_labels=True, font_weight='bold')
plt.show()
np.savetxt("coma_props.csv",coma_props,delimiter=",",fmt="%s")
nx.degree_centrality(Graphs[0])
nx.eigenvector_centrality(Graphs[0])
"""

