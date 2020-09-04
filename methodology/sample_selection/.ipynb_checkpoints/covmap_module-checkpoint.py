# ------------------------------------------------------------------------
#                            covmap for numba
# ------------------------------------------------------------------------


import numpy as np
import numba


# --- f1: determinant criterion
    
    # --- criterion
@numba.njit(fastmath=True)
def get_criterion(X,selected_rows,r,ref_svd):

    ''' 
    evaluate criterion based on selected rows of X with rank r

    '''

    # --- declare initial variables

    N,K = X.shape  
    mean_Xs = np.zeros((1,K))
    mean_X = np.zeros((1,K))



    # --- get selected matrix

    Xs = X[selected_rows,:]


    # --- cov(X_s)     


    for kk in range(K):
        mean_Xs[0,kk] = np.mean(Xs[:,kk])
        mean_X[0,kk] = np.mean(X[:,kk])

    Xs_c = Xs - mean_Xs
    Xs_c_t = np.ascontiguousarray(Xs_c.transpose())
    cov_Xs = Xs_c_t.dot(Xs_c)/Xs_c.shape[0]

    # --- svd of cov(X_s)

    Us,Ds,Vs = np.linalg.svd(cov_Xs, full_matrices=False)  
    
        

    if r<Xs.shape[0]:
        last_r = r
    else:
        last_r = Xs.shape[0]-1
        
    ref_eigen_vects = np.ascontiguousarray(ref_svd[0][:,0:last_r])
    ref_eigen_vects_t = np.ascontiguousarray(ref_eigen_vects.transpose())
    ref_eigen_vals_m = np.ascontiguousarray(np.diag(ref_svd[1][0:last_r])) 
    ref_eigen_t = ref_eigen_vals_m.dot(ref_eigen_vects_t)
    
    sample_eigen_vects = np.ascontiguousarray(Us[:,0:last_r])
#     sample_eigen_vects_t = np.ascontiguousarray(sample_eigen_vects.transpose())
    sample_eigen_vals_inv = np.ascontiguousarray(np.diag(1/Ds[0:last_r]))
    sample_eigen_inv = sample_eigen_vects.dot(sample_eigen_vals_inv)
    
    criterion = np.abs(np.linalg.det(np.eye(last_r)-ref_eigen_t.dot(sample_eigen_inv)))
    
#     eigen_vals_bin = np.prod(Ds[0:last_r]>=ref_svd[1][0:last_r])
    
#     if eigen_vals_bin<1:
#         criterion = -1
#     else:    
#         ref_eigen_vects = ref_svd[0][:,0:last_r]
#         ref_eigen_vects_t = np.ascontiguousarray(ref_eigen_vects.transpose())
#         Us_final = np.ascontiguousarray(Us[:,0:last_r])
#         A = np.abs(ref_eigen_vects_t.dot(Us_final))
        
#         eigen_vals_ratio = np.prod(ref_svd[1][0:last_r]/Ds[0:last_r])
#         criterion = np.linalg.det(A)*eigen_vals_ratio

    
    # --- apply previous svd to total matrix centering with mean_Xs

    #X_cs = X - mean_Xs
    #Vs_r = Vs[0:last_r,:]
    #Vs_t = np.ascontiguousarray(Vs_r.transpose())
    #t_scores_subsample = X_cs.dot(Vs_t).dot(np.diag(1/np.sqrt(Ds[0:last_r])))
    
    #X_c = X - mean_X
    #V_r = ref_svd[2][0:last_r,:]
    #V_t = np.ascontiguousarray(V_r.transpose())
    #t_scores_all = X_c.dot(V_t).dot(np.diag(1/np.sqrt(ref_svd[1][0:last_r])))
    
    
    #for pc in range(r):
   
     #   sign_corr = np.corrcoef(np.concatenate((t_scores_all[:,pc:(pc+1)], t_scores_subsample[:,pc:(pc+1)]), axis=1).T)[0,1]    
     #   t_scores_subsample[:,pc] = t_scores_subsample[:,pc]*np.array([1,-1])[np.array([sign_corr>0,sign_corr<0])][0]
     #   new_sign_corr = np.corrcoef(np.concatenate((t_scores_all[:,pc:(pc+1)], t_scores_subsample[:,pc:(pc+1)]), axis=1).T)[0,1]


    
    #criterion = np.exp(-np.power(t_scores_all-t_scores_subsample,2)).sum(axis=1).mean()  
    #criterion = np.sqrt(np.power(t_scores_all-t_scores_subsample,2).sum(axis=1).mean())  

      
  
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
def covmap(xx,n_sel,rank_covmatrix,initial_selected_rows, total_iters=4000, total_starts = 5):

    ''' 
    covmap covariance maximum approximation algorithm performs a sample selection procedure. The selected samples are such that they produce an svd
    of the covariance matrix as close as possible to the one that all samples produce
    '''
        

    X_all = xx.copy()
    n_all, K = X_all.shape
    all_rows = np.arange(n_all)
    convergence_algorithm = np.zeros(total_iters)
    mean_X = np.zeros((1,K))
    
    # --- svd cov(X)     


    for kk in numba.prange(K):
        mean_X[0,kk] = np.mean(X_all[:,kk])

    X_c = X_all - mean_X
    X_c_t = np.ascontiguousarray(X_c.transpose())
    cov_X = X_c_t.dot(X_c)/n_all

    # --- svd of cov(X_s)

    svd_all = np.linalg.svd(cov_X, full_matrices=False)
    
    #if initial_selected_rows==0:
        
    #    initial_selected_rows = np.random.choice(n_all,n_sel, replace=False)
    
    # --- start iterations
    
    
    final_criterion = get_criterion(X=X_all,selected_rows=initial_selected_rows,r=rank_covmatrix, ref_svd=svd_all)
    
    for jj in range(total_starts):

        # --- initialize random selection, row exchange and criterion

        current_selected_rows = initial_selected_rows.copy()
        current_criterion = get_criterion(X=X_all,selected_rows=initial_selected_rows,r=rank_covmatrix, ref_svd=svd_all)
        current_keep_row = current_selected_rows[1]

        # --- start search

        for ii in range(total_iters):

                # --- swap

            current_exchanged_rows, current_exchanged_row = swap_candidate(all_obs=all_rows, selected_obs=current_selected_rows, except_ob=current_keep_row)


                # --- function to calculate optimality criterion

            criterion_result = get_criterion(X=X_all,selected_rows=current_exchanged_rows,r=rank_covmatrix, ref_svd=svd_all)
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


# --- covmap sample size iteration function

@numba.njit(fastmath=True, parallel=True)
def covmap_iterations_range(xx,n_sel_range,rank_covmatrix_range, total_iters_ss, total_starts_ss):    
    
    criterion_results = np.zeros((n_sel_range.shape[0], rank_covmatrix_range.shape[0]))
    

    for ii in numba.prange(n_sel_range.shape[0]):
        
        n_sel = n_sel_range[ii]   
        
        
        for jj in numba.prange(rank_covmatrix_range.shape[0]):
            
            chosen_rank = rank_covmatrix_range[jj]
            covmap_result = covmap(xx,n_sel=n_sel,rank_covmatrix=chosen_rank, total_iters=total_iters_ss, total_starts = total_starts_ss)
            criterion_results[ii,jj] = covmap_result[1]
            
        
    return (n_sel_range,rank_covmatrix_range, criterion_results)
 