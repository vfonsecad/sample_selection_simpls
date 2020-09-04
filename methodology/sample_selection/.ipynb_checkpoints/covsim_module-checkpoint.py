# ------------------------------------------------------------------------
#                            covsim for numba
# ------------------------------------------------------------------------


import numpy as np
import numba


# --- f1: determinant criterion
    
    # --- criterion
@numba.njit(fastmath=True)
def get_criterion(X,selected_rows):

    ''' 
    evaluate criterion based on selected rows of X with rank r

    '''

    # --- declare initial variables

    N,K = X.shape  
    mean_X = np.zeros((1,K))
    mean_Xs = np.zeros((1,K))



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
    
    
    criterion = np.sum(np.abs(cov_X-cov_Xs))   
    
    

    return criterion


#fast_get_criterion = numba.njit(fastmath=True)(get_criterion)

# --- f2: swap rows
@numba.njit(fastmath=True)
def swap_candidate(all_obs, selected_obs, except_ob):


    ''' 
    randomly swap one sample currently in for one sample currently out

    '''


    bool_all_unselected = np.zeros(all_obs.shape[0])>0    

    return_selected = selected_obs.copy()
    bool_selected_except_id = (selected_obs!=except_ob)

    candidate_selected = selected_obs[bool_selected_except_id] 

    jj=0
    for ii in all_obs:
        check = np.where(selected_obs==ii)[0].shape[0]
        if check == 0:            
            bool_all_unselected[jj]=True
        jj+=1    


    candidate_unselected = all_obs[bool_all_unselected]


    selected_for_out = np.random.choice(a=candidate_selected, size=1, replace=False)
    selected_for_in = np.random.choice(a=candidate_unselected, size=1, replace=False)

    return_selected[np.where(selected_obs==selected_for_out)[0][0]] = selected_for_in[0]


    return (return_selected,selected_for_in[0])

#fast_swap_candidate = numba.njit(fastmath=True)(swap_candidate)


# --- covmap algorithm

@numba.njit(fastmath=True)
def covmap(xx,n_sel, total_iters=4000, total_starts = 5):

    ''' 
    covmap covariance maximum approximation algorithm performs a sample selection procedure. The selected samples are such that they produce an svd
    of the covariance matrix as close as possible to the one that all samples produce
    '''
        

    X_all = xx.copy()
    n_all = X_all.shape[0]
    all_rows = np.arange(n_all)
    convergence_algorithm = np.zeros(total_iters)
    
    
    final_criterion = 1000000
    
    for jj in range(total_starts):

        # --- initialize random selection, row exchange and criterion

        current_selected_rows = np.random.choice(a=n_all, size=n_sel, replace=False)
        current_criterion = 1000000
        current_keep_row = current_selected_rows[1]

        # --- start search

        for ii in range(total_iters):

                # --- swap

            current_exchanged_rows, current_exchanged_row = swap_candidate(all_obs=all_rows, selected_obs=current_selected_rows, except_ob=current_keep_row)


                # --- function to calculate optimality criterion

            criterion_result = get_criterion(X=X_all,selected_rows=current_exchanged_rows)
            convergence_algorithm[ii] = criterion_result

                # --- evaluation statement

            if criterion_result<=current_criterion:

                current_keep_row = current_exchanged_row
                current_selected_rows = current_exchanged_rows.copy()
                current_criterion = criterion_result
                
        if current_criterion<=final_criterion:            

            
            final_selected_rows = current_selected_rows.copy()
            final_criterion = current_criterion
            final_convergence_algorithm = convergence_algorithm.copy()
            
    sample_selected_bool = np.zeros(n_all)>0
    
    for ii in final_selected_rows:
        sample_selected_bool[ii]=True
        
    sample_selected = sample_selected_bool*1

    return (sample_selected,final_criterion, final_convergence_algorithm)

#fast_covmap = numba.njit(fastmath=True)(covmap)
