# ------------------------------------------------------------------------

#                               mcw-pls functions with numba
# by: valeria fonseca diaz
# supervisors: Wouter Saeys, Bart De Ketelaere
# ------------------------------------------------------------------------


import numpy as np
import numba


@numba.njit(fastmath=True)
def simpls_loadings(xx, yy, P, ncp):
    
    xx = np.ascontiguousarray(xx)
    yy = np.ascontiguousarray(yy)
    P = np.ascontiguousarray(P)
        
    mu_x = ((P.dot(xx)).sum(axis=0))/ P.sum()
    mu_y = ((P.dot(yy)).sum(axis=0))/ P.sum()
        

    Xc = xx.copy() - mu_x
    Yc = yy.copy() - mu_y

    N = Xc.shape[0]
    K = Xc.shape[1]
    q = Yc.shape[1]

    R = np.zeros((K, ncp))  # Weights to get T components
    V = np.zeros((K, ncp))  # orthonormal base for X loadings
    S = Xc.T.dot(P).dot(Yc)  # cov matrix
    r = np.zeros((K, 1))
    v = np.zeros((K, 1))

    aa = 0

    while aa < ncp:
        
        r[:,0] = S[:,0].flatten()
        
#         if q > 1:
                
#             U0, sval, Qsvd = np.linalg.svd(S)
#             Sval = np.zeros((U0.shape[0], Qsvd.shape[0]))
#             Sval[0:sval.shape[0], 0:sval.shape[0]] = np.diag(sval)

#             r[:,0] = U0.dot(Sval)[:,0]

                
        tt = Xc.dot(r)
        tt = tt - ((P.dot(tt)).sum(axis=0)/ P.sum())
        TT_scale = np.sqrt(tt.T.dot(P).dot(tt))
            # Normalize
        tt = tt / TT_scale            
        r = r / TT_scale
        
        p = Xc.T.dot(P).dot(tt)
        v[:,0] = p.flatten()
       
        
        if aa > 0:
            
            v = v - V.dot(V.T.dot(p))

        v = v / np.sqrt(v.T.dot(v))
        S = S - v.dot(v.T.dot(S))

        R[:, aa] = r.flatten()
        V[:, aa] = v.flatten()
        
        aa += 1

    return R
    

@numba.njit(fastmath=True)
def mcw_pls_fit(xx, yy, ncp, sigma_factor, max_iter=30):
    
    
    X = np.ascontiguousarray(xx)
    Y = np.ascontiguousarray(yy)
    
    N = X.shape[0]
    K = X.shape[1]
    q = Y.shape[1]
    
    mu_x = np.zeros((1,K))
    mu_y = np.zeros((1,q))
    
 
    P = np.identity(N)
        
        
    
    sigma_tol = 0.000001  

        
    mu_x[0,:] = ((P.dot(X)).sum(axis=0)) / P.sum()
    Xc = X - mu_x
    mu_y[0,:] = ((P.dot(Y)).sum(axis=0)) / P.sum()
    Yc = Y - mu_y


    R = simpls_loadings(X, Y, P, ncp)


        # --- Iterative process

    kk = 0
    sigma2 = 100000
        
        

    while kk < max_iter and sigma2 > sigma_tol:

        TT = Xc.dot(R)  # TT are not weighted

        tcal_raw0 = np.concatenate((np.ones((Xc.shape[0], 1)), TT), axis=1)
        wtemp = np.linalg.solve(tcal_raw0.T.dot(P.dot(tcal_raw0)), tcal_raw0.T.dot(P.dot(Y)))

        sigma_vector = np.power(Y - tcal_raw0.dot(wtemp), 2).sum(axis=1)
        sigma2 = sigma_factor * (sigma_vector).mean()
        P = np.multiply(np.exp(-sigma_vector / (2 * sigma2)), np.identity(N))

        mu_x[0,:] = ((P.dot(X)).sum(axis=0)) / P.sum()
        Xc = X - mu_x
        mu_y[0,:] = wtemp[0, :]
        Yc = Y - mu_y

        R = simpls_loadings(X, Y, P, ncp)

        kk += 1
            
    TT = Xc.dot(R)
               
    # --- mcw-pls final regression

        
    tcal_raw0 = np.concatenate((np.ones((X.shape[0], 1)), TT), axis=1)
    wtemp = np.linalg.solve(tcal_raw0.T.dot(P.dot(tcal_raw0)), tcal_raw0.T.dot(P.dot(Y)))
            
    wtemp_bool = np.zeros(ncp+1, dtype=np.int64) == 0  
    wtemp_bool[0] = False
    
    wtemp1 = wtemp[wtemp_bool, :]
    BPLS = np.dot(R,wtemp1)
    
    x_mu = mu_x.copy()
    y_mu = wtemp[0:1, :]
    sample_weights = np.diag(P)
    
    
    
    return (BPLS,x_mu, y_mu,sample_weights)



@numba.njit(fastmath=True)
def mcw_pls_predict(X, BPLS, x_mu, y_mu):
    
    Ypred = y_mu + (X - x_mu).dot(BPLS)

    return Ypred

@numba.njit(fastmath=True)
def rmse(yy, y_pred, sample_weights):
    
    N = yy.shape[0]

    P = np.diag(sample_weights.flatten()) / sample_weights.flatten().sum()

    r = yy.copy() - y_pred.copy()
    r = np.power(r,2)
    msep = ((P.dot(r)).sum(axis=0)) / P.sum()
    rmse = np.sqrt(msep)

    return rmse[0]  

@numba.njit(fastmath=True)
def mcw_pls_cv(xx, yy, ncp_range, sigma2_range, number_splits=10, max_iters_cv = 30):
    
    
    X = xx.copy()
    Y = yy.copy()

    N = X.shape[0]
    K = X.shape[1]
    q = Y.shape[1]
    
    size_split = int(N/number_splits)
    sample_in_group = np.arange(0,number_splits)
    
    for ss in range(size_split):
        sample_in_group = np.concatenate((sample_in_group, np.arange(0,number_splits)), axis = 0)
        
    sample_in_group_shuffled = np.random.permutation(sample_in_group[0:N])
    
    cv_performance_array = np.zeros((ncp_range.shape[0], sigma2_range.shape[0]))

    ii = 0
    
    
    for ncp in ncp_range:  
        
        
        jj = 0
        
        for sigma_factor in sigma2_range: 
            
            train_for_sample_weights = mcw_pls_fit(X, Y, ncp, sigma_factor)
            

            cv_predicted = np.zeros((N,q))
            
            
            for ss in range(number_splits):
                
                
                test_obs = np.zeros(N, dtype=np.int64) == 1  
                cal_obs = np.zeros(N, dtype=np.int64) == 0 
                test_obs[np.where(sample_in_group_shuffled == ss)[0]] = True
                cal_obs[np.where(sample_in_group_shuffled == ss)[0]] = False                
                trained = mcw_pls_fit(X[cal_obs,:], Y[cal_obs,:], ncp, sigma_factor, max_iter = max_iters_cv)
                predicted = mcw_pls_predict(X[test_obs,:], trained[0], trained[1], trained[2])                
                cv_predicted[test_obs,:] = predicted

            sample_weights = train_for_sample_weights[3]
            
            cv_performance_array[ii,jj] = rmse(Y, cv_predicted, sample_weights = sample_weights)
            
       
            jj += 1
            
        ii += 1
        
    return cv_performance_array
    