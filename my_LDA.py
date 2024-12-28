import numpy as np
import scipy.linalg as linalg

def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """    
    classLabels = np.unique(Y)  # different class labels on the dataset
    classNum = len(classLabels)
    datanum, dim = X.shape  # dimensions of the dataset
    totalMean = np.mean(X, 0)  # total mean of the data
    totalMean = totalMean.T

    #Moyenne au sein de chaque classe :
    
    means = {classe : np.zeros(dim,) for classe in classLabels}
    counting = {classe : 0 for classe in classLabels}
    
    for k in range(datanum) :
        x = X[k,:]
        classe = Y[k]
        means[classe] += x.T
        counting[classe] += 1
        
    for classe in means :
        means[classe] = means[classe]/counting[classe]
        
    #Calcul de la matrice représentant la variance au sein de chaque classe.
    Sw = np.zeros((dim, dim))
    for k in range(datanum) :
        x = X[k,:]
        classe = Y[k]
        #Sw += np.matmul(x.T - means[classe], (x.T- means[classe]).T)
        diff = x.T - means[classe]
        Sw += np.outer(diff, diff)

        
    #Calcul de la matrice représentant la différence des moyennes.
    
    Sb = np.zeros((dim, dim))
    for classe in means :
        Sb += counting[classe]*np.matmul(means[classe]-totalMean, (means[classe]-totalMean).T)
        
    #Calcul de W
    
    A = np.matmul(linalg.inv(Sw), Sb)
    values, vectors = linalg.eig(A)

    # On trie les valeurs propres en ordre décroissant
    sorted_indices = np.argsort(-values.real)  
    values = values[sorted_indices]
    vectors = vectors[:, sorted_indices]
    W = vectors[:, :len(classLabels)-1]
    
    X_lda = np.matmul(X, W)
    projected_centroid = np.array([np.matmul(W.T, means[classe]) for classe in means])
        
    return W, projected_centroid, X_lda
