import pandas as pd
import matplotlib.pyplot as plt

def plotWordEmbeddings(w2v):
  
  X=w2v[w2v.wv.vocab]
  df=pd.DataFrame(X)
  df['Name'] = list(w2v.wv.vocab)
  df.set_index('Name',drop=True,inplace=True)

  #Computing the correlation matrix
  X_corr=df.corr()

  #Computing eigen values and eigen vectors
  values,vectors=np.linalg.eig(X_corr)

  #Sorting the eigen vectors coresponding to eigen values in descending order
  args = (-values).argsort()
  values = vectors[args]
  vectors = vectors[:, args]

  #Taking first 2 components which explain maximum variance for projecting
  new_vectors=vectors[:,:2]

  #Projecting it onto new dimesion with 2 axis
  neww_X=np.dot(X,new_vectors)

  plt.figure(figsize=(8,5))
  plt.scatter(neww_X[:,0],neww_X[:,1],linewidths=10,color='green')
  plt.xlabel("PC1",size=10)
  plt.ylabel("PC2",size=10)
  plt.title("Word Embedding Space",size=18)
  vocab=list(w2v.wv.vocab)
  for i, word in enumerate(vocab):
    plt.annotate(word,xy=(neww_X[i,0],neww_X[i,1]))
  

plotWordEmbeddings()