import numpy as np

def TGG(X,k):
    '''
    Temporal Graph Generation
    input: N time series from X(|X|=N)
    output: Weighted Matrix W of Temporal Graph.
    '''
    #Weighted Matrix WN×N (initial zero matrix), CMC: Cost Matrix Calculation defined in Eq. (2);
    N = len(X)
    W = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            W[i][j] = CMC(X[i],X[j])
        np.sort(W[:,:k])
        for k_ in range(k):
            W[i][k_] = W[k_][i] = 1
    for i in range(N):
        for j in range(N):
            if i==j:
                W[i][j] = 1
    return W
def DPQ():
    pass
def CMC(Xi,Xj):
    '''
    Cost Matrix Calculation
    Dc(i, j)= Di,j +min(Dc(i, j −1),Dc(i−1,j),Dc(i−1,j −1))
    '''
    pass
