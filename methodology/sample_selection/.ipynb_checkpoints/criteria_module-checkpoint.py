# ------------------------------------------------------------------------
#                            criteria functions
# ------------------------------------------------------------------------


import numpy as np
from sklearn.metrics import pairwise_distances


# --- criterion determinant  and eigenvalues

def criterion_determinant(X,selected_rows,r):

    ''' 
    evaluate determinant based on selected rows of X with rank r

    '''

    # --- declare initial variables

    N,K = X.shape  
    mean_Xs = np.zeros((1,K))



    # --- get selected matrix

    Xs = X[selected_rows,:]

    # --- cov(X_s)     


    for kk in range(K):
        mean_Xs[0,kk] = np.mean(Xs[:,kk])

    Xs_c = Xs - mean_Xs
    Xs_c_t = np.ascontiguousarray(Xs_c.transpose())
    cov_Xs = Xs_c_t.dot(Xs_c)/Xs_c.shape[0]

    # --- svd of cov(X_s)

    Us,Ds,Vs = np.linalg.svd(cov_Xs, full_matrices=False)

    # --- apply previous svd to total matrix centering with mean_Xs

    X_cs = X - mean_Xs
    Vs_r = Vs[0:r,:]
    Vs_t = np.ascontiguousarray(Vs_r.transpose())
    T = X_cs.dot(Vs_t)
    T_t = np.ascontiguousarray(T.transpose())
    T_norm = T.dot(np.diag(np.sqrt(1/np.diag(T_t.dot(T)))))

    # --- get optimality criterion
    
    T_norm_t = np.ascontiguousarray(T_norm.transpose())
    A = T_norm_t.dot(T_norm)
    
    
    criterion = np.linalg.det(A)
    
    return criterion

def criterion_singvalues(X,selected_rows,r):

    ''' 
    evaluate eigenvalues diff based on selected rows of X with rank r

    '''

    # --- declare initial variables

    N,K = X.shape  
    mean_Xs = np.zeros((1,K))
    mean_X = np.zeros((1,K))



    # --- get selected matrix

    Xs = X[selected_rows,:]

    # --- cov(X_s)   and cov(X)     


    for kk in range(K):
        mean_Xs[0,kk] = np.mean(Xs[:,kk])
        mean_X[0,kk] = np.mean(X[:,kk])

    Xs_c = Xs - mean_Xs
    Xs_c_t = np.ascontiguousarray(Xs_c.transpose())
    cov_Xs = Xs_c_t.dot(Xs_c)/Xs_c.shape[0]
    
    
    X_c = X - mean_X
    X_c_t = np.ascontiguousarray(X_c.transpose())
    cov_X = X_c_t.dot(X_c)/X_c.shape[0]

    # --- svd of cov(X_s)

    Us,Ds,Vs = np.linalg.svd(cov_Xs, full_matrices=False)
    U,ref_eigenvals,V = np.linalg.svd(cov_X, full_matrices=False)
   
    eigenvals_discrepance = (Ds/ref_eigenvals)[0:r]
    criterion = np.amax(eigenvals_discrepance) - np.amin(eigenvals_discrepance) 

    
    return criterion

# --- criterion distance



def criterion_distance(X,selected_rows,r):
    
    '''
    
    distance between X selected and X unselected
    
    '''
    
    U,D,V = np.linalg.svd(X, full_matrices=False)
    
    Xs = U[selected_rows,0:r]
    Xus = np.delete(U, obj = selected_rows,axis=0)[:,0:r]  
    
    dist_sel_usel = pairwise_distances(Xs,Xus, metric='euclidean')
    #print(dist_sel_usel.shape)
    criterion = np.amax(np.amin(dist_sel_usel,axis=1))
    
    return criterion
