library(XBART)




pred = matrix(0,10000,10)

if (get_param == TRUE){
  num_trees = 120
  num_sweeps = 40
  burnin = 15
  
  
  delta = seq(0.1, 2, 0.05)
  concn = 1
  Nmin = 1
  max_depth = 10
  mtry = 10
  num_cutpoints = 20
}
tau = 100 / num_trees
tau_later = 100 / num_trees


t = proc.time()
fit = XBART.multinomial(y=matrix(y), num_class=10, X=X_train, Xtest=X_test, 
                        num_trees=num_trees, num_sweeps=num_sweeps, max_depth=max_depth, 
                        Nmin=Nmin, num_cutpoints=num_cutpoints, alpha=0.95, beta=1.25, tau=50/num_trees, 
                        no_split_penality = 1, burnin = burnin, mtry = mtry, p_categorical = 0L, 
                        kap = 1, s = 1, verbose = TRUE, parallel = FALSE, set_random_seed = TRUE, 
                        random_seed = NULL, sample_weights_flag = TRUE, 
                        delta = delta, concn = concn)
t = proc.time() - t

pred = apply(fit$yhats_test[(burnin+1):(num_sweeps-0),,], c(2,3), mean)
yhat = max.col(pred)-1

spr <- split(pred, row(pred))
logloss <- sum(unlist(mapply(function(x,y) -log(x[y]), spr, ytest, SIMPLIFY =TRUE)))

cat("running time ", t[3], " seconds \n")

cat("XBART error rate ", mean(yhat != ytest), "\n")

cat(paste("xbart logloss : ",round(logloss,3)),"\n")


table(fit$delta)

mean(fit$delta)

for(i in 0:9){
  cat("XBART error rate in ", i, ": ", round(mean(yhat[ytest==i]!=i), 4), 
      " misclassified as ", tail(names(sort(table(yhat[ytest==i]))), 2)[1], "\n " )
}



