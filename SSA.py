import numpy as np
def SSA(signal, lag, numComp, method = "classic"):
    assert isinstance(signal, np.ndarray) and signal.ndim == 1, "signal must be one dimensional and of type ndarray"
    assert isinstance(lag, int) and 2 <= lag < len(signal) // 2, "Lag length must be less than len(signal)//2 and higher than 2"
    assert isinstance(numComp, int) and 1 <= numComp <= lag, "Number of components to be computed must be less than lag and greater than 0"
    if method == "classic":
        return classic_ssa(signal, lag, numComp)
    elif method == "accelerated":
        return accelerated_ssa(signal, lag, numComp)
    else: 
        raise Exception("Options are either 'classic' or 'accelerated'")
def classic_ssa(signal, L=10, numComp=10):
    '''
    This approach is based on the nice tutorial that can be found here: 
    https://www.kaggle.com/code/jdarcy/introducing-ssa-for-time-series-decomposition
    '''
    N = len(signal) # get the lag length
    K = N - L + 1 # get the length of the embedded signals
    # Embedding
    X = np.array([signal[i:L+i] for i in range(0, K)]).T
    # Decomposition by Singular Value Decomposition
    U, Sigma, VT = np.linalg.svd(X)
    # Calculating the reconstructed components
    RC = np.zeros((N, numComp)) # init for speed
    for i in range(numComp):
        buff = Sigma[i]*np.outer(U[:,i], VT[i,:]) # get the buffer
        buff = buff[::-1] # flip the matrix for easier selection of anti-diagonal
        RC[:,i] = [buff.diagonal(j).mean() for j in range(-buff.shape[0]+1, buff.shape[1])]
    return RC
def accelerated_ssa(signal, L=10, numComp=10):
    # Import needed tools
    from joblib import Parallel, delayed, cpu_count
    import numba as nb
    import threading
    from tqdm import tqdm
    N = len(signal) # get the lag length
    K = N - L + 1 # get the length of the embedded signals
    # Define just in time functions with @jit decorators and eager compilation
    @nb.jit(nb.float64[:,:](nb.float64[:,:], nb.float64[:,:]), nopython=True)
    def fastDot(X, Y): # might be faster to go without jit for this, warning in numba
        return np.dot(X, Y)
    @nb.jit(nb.float64[:,:](nb.float64[:], nb.float64[:]), nopython=True)
    def fastBuf(principleComponentColumn, eigenVectorsRow):
        return np.outer(principleComponentColumn,eigenVectorsRow)[::-1]
    @nb.jit(nb.float64[:,:](nb.float64[:,:]), nopython=True)
    def fastCovariance(X):
        return np.cov(X)
    @nb.jit(nopython=True)
    def fastMean(X):
        return X.mean()      
    # Helper functions for multi-threading
    def getColumn(s,K,m):
        return s[m:K+m]
    def laggingFunction(idx, LaggedMatrix, s, K):
        LaggedMatrix[:,idx]=getColumn(s,K,m)
    # Helper function for the anti-diagonal averaging
    def calcRC(antiDiagonal):
        return fastMean(antiDiagonal)
    # Embedding
    X=np.zeros((K,L))
    threads=[]
    for m in range (0,L): # generate and start threads
        threads.append(threading.Thread(target=laggingFunction,args=(m,X,signal,K)))
        threads[m].start()
    for m in range (0, L): # combine all the threads
        threads[m].join()
    # Eigenvalue decomposition approach
    # alternatively use cupy.eigh if a big GPU card is available
    eigenValues, eigenVectors = np.linalg.eigh(fastCovariance(X.T)/K)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    # NOTE: eigh returns Fortran contiguous matrix, convert to C contiguous for faster calculation
    # Principle components
    eigenVectors = np.ascontiguousarray(eigenVectors)
    X = np.ascontiguousarray(X)    
    PC = fastDot(X, eigenVectors)
    # Reconstructed components    
    RC = np.zeros((N, numComp), dtype=np.float64)
    for i in range(0, numComp):
        # Create Hankelised Components
        buf = fastBuf(PC[:,i], eigenVectors[:,i].T)
        Diag0 = -buf.shape[0] + 1
        Diag1 = +buf.shape[1]
        RC[:,i]= Parallel(n_jobs = cpu_count())\
        (delayed(calcRC)(buf.diagonal(j))\
        for j in range(Diag0, Diag1))
    return RC