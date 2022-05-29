library("Rcpp")
library("pracma")
library("tidyverse")
library("uwot")
library("irlba")
library("mclust")
library("mcclust")
library("readxl")
library("iterators")
library("glue")

sourceCpp("lamb_code/DL_linear_split_merge_package.cpp") ##This is the souce C++ file


#################

set.seed(1234)

#nrun=5e3, burn=1e3
nburn=2
niter =2
thin= 1

############################################
## RUN ON DIFFERENT DATASETS AND SETTINGS ##
############################################

p_s = c(100, 200)
d_s = c(2, 5)
M_s = c(4, 8)
n_percluster_s = c(50)

# Outer cycle for reading the different datasets and perform the estimation
for (p in p_s){
  for (dtrue in d_s){
    for (M in M_s){
      for (npc in n_percluster_s){
        #######################################
        ### READ DATA AND PRE-PROCESSING ######
        #######################################
        
        # read the dataset
        data = read_csv(file = glue("data/Student_data/stud_p_{p}_d_{dtrue}_M_{M}_npc_{npc}_data.csv") , col_names = FALSE)
          
        # scaling of data
        centering.var=median(colMeans(data))
        scaling.var=median(apply(data,2,sd))
        data_scaled=as.matrix((data-centering.var)/scaling.var)
        
        # Perform a sparse PCA on the pre-processed data.
        pca.results=irlba::irlba(data_scaled, nv=10)
        
        cum_eigs= cumsum(pca.results$d)/sum(pca.results$d)
        d=min(which(cum_eigs>.90)) 
        d=ifelse(d<=15,15,d) # Set d= at least 15
        
        outpath_d = glue("data/Student_data/lamb_p_{p}_d_{dtrue}_M_{M}_npc_{npc}_out")
        if (!(dir.exists(outpath_d))){
          dir.create(outpath_d)
        }
        
        # Initialization eta and lambda
        eta= pca.results$u %*% diag(pca.results$d)
        eta=eta[,1:d] #Left singular values of y are used as to initialize eta
        lambda=pca.results$v[,1:d] #Right singular values of y are used to initialize lambda
        
        # Initialization cluster allocations
        cluster.start = kmeans(data, 80)$cluster - 1
        
        ### Set Prior Parameter of `Lamb`
        as = 1; bs = 0.3 
        a=0.5
        diag_psi_iw=20
        niw_kap=1e-3
        nu=d+50
        a_dir_s=c(.1)
        b_dir_s=c(.1)
       
        for (rho in rho_s){
          #### Fit the `Lamb` Model
          result.lamb <- DL_mixture(a_dir, b_dir, diag_psi_iw=diag_psi_iw, niw_kap=niw_kap, niw_nu=nu, 
                                    as=as, bs=bs, a=a,
                                    nrun=niter, burn=nburn, thin=thin, 
                                    nstep = 5,prob=0.5, #With probability `prob` either the Split-Merge sampler with `nstep` Gibbs scans or Gibbs sampler in performed 
                                    lambda, eta, y,
                                    del = cluster.start, 
                                    dofactor=1 #If dofactor set to 0, clustering will be done on the initial input eta values only; eta will not be updated in MCMC
          )
          
          # Posterior Summary
          burn=2e2 #burn/thin
          post.samples=result.lamb[-(1:burn),]+1
          write.table(post.samples, file = "alloc_matrix/mat2.csv",
                      quote=FALSE, eol="\n", row.names=FALSE, col.names=FALSE,  sep=",")
          
          sim.mat.lamb <- mcclust::comp.psm(post.samples) # Compute posterior similarity matrix
          clust.lamb <-   minbinder(sim.mat.lamb)$cl # Minimizing Binder loss across MCMC estimates
          numclust.lamb <- apply(post.samples,1,function(x) {length(unique(x))})
          numclust_50more.lamb <- apply(post.samples,1, function(x) {sum(table(x)>50)} )
          plot(numclust.lamb,type='l')
        }
          
        
      }
    }
  }
}





# Save results in folder
base_outpath_rho = os.path.join(outpath_d, "rho_{0}_out".format(rho)) + "_{0}"
i = 0
while os.path.exists(base_outpath_rho.format(i)):
  i = i+1
outpath = base_outpath_rho.format(i)
os.makedirs(outpath)

# Save the serialized chain produced by the sampler
sampler.serialize_chains(os.path.join(outpath, "chains.recordio"))


# save the parameters
with open(os.path.join(outpath, "params.asciipb"), 'w') as fp:
  fp.write(text_format.MessageToString(hyperpar))


chain = sampler.chains

# plots
fig = plt.figure()
tau_chain = np.array([x.lamb_block.tau for x in chain])
plt.plot(tau_chain)
plt.title("tau chain")
plt.savefig(os.path.join(outpath, "tau_chain.pdf"))
plt.close()

fig = plt.figure()
first_sbar_chain = np.array([to_numpy(x.sigma_bar)[0] for x in chain])
plt.plot(first_sbar_chain,color='red')
last_sbar_chain = np.array([to_numpy(x.sigma_bar)[-1] for x in chain])
plt.plot(last_sbar_chain,color='blue')
plt.title("sbar_chain")
plt.savefig(os.path.join(outpath, "sbar_chain.pdf"))
plt.close()

# Compute Posterior Summaries
fig = plt.figure()
n_cluster_chain = np.array([x.ma for x in chain])
plt.plot(n_cluster_chain)
plt.title("number of clusters chain")
plt.savefig(os.path.join(outpath, "nclus_chain.pdf"))
plt.close()

fig = plt.figure()
n_nonall_chain = np.array([x.mna for x in chain])
plt.plot(n_nonall_chain)
plt.title("number of non allocated components chain")
plt.savefig(os.path.join(outpath, "non_alloc_chain.pdf"))
plt.close()

post_mode_nclus = mode(n_cluster_chain)[0][0] # store in dataframe
post_avg_nclus = n_cluster_chain.mean() # store in dataframe
post_avg_nonall =  n_nonall_chain.mean() # store in dataframe

clus_alloc_chain = [x.clus_alloc for x in chain]
best_clus = cluster_estimate(np.array(clus_alloc_chain))
np.savetxt(os.path.join(outpath, "best_clus.txt"), best_clus)

n_clus_best_clus = np.size(np.unique(best_clus))
true_clus = np.repeat(range(M),npc)
ari_best_clus = adjusted_rand_score(true_clus, best_clus) # store in dataframe
aris_chain = np.array([adjusted_rand_score(true_clus, x) for x in clus_alloc_chain])
mean_aris, sigma_aris = np.mean(aris_chain), np.std(aris_chain) # store mean_aris in dataframe
CI_aris = norm.interval(0.95, loc=mean_aris, scale=sigma_aris/sqrt(len(aris_chain))) # store in dataframe
list_performance = list()
list_performance.append([p,dtrue,d,M,npc,sampler.means_ar, sampler.lambda_ar, post_mode_nclus,
                         post_avg_nclus, post_avg_nonall, ari_best_clus, CI_aris])
df_performance = pd.DataFrame(list_performance, columns=('p','dtrue','d','M','npc','means_ar','lambda_ar',
                                                         'mode_nclus', 'avg_nclus', 'avg_nonalloc', 'ari_best_clus', 'CI_aris'))
df_performance.to_csv(os.path.join(outpath, "df_performance.csv"))






