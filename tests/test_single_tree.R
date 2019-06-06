#######################################################################
# set parameters of XBART
get_XBART_params <- function(n, d, y) {
  XBART_params = list(num_trees = 2,                 # number of trees 
                      num_sweeps = 20,          # number of sweeps (samples of the forest)
                      n_min = 1,               # minimal node size
                      alpha = 0.95,           # BART prior parameter 
                      beta = 1.25,            # BART prior parameter
                      mtry = 1,               # number of variables sampled in each split
                      burnin = 5,
                      no_split_penality = "Auto"
                      )            # burnin of MCMC sample
  num_tress = XBART_params$num_trees
  XBART_params$max_depth = matrix(250, num_tress, XBART_params$num_sweeps)   # max depth of each tree, should be a num_trees by num_sweeps matrix
  XBART_params$num_cutpoints = 10;                                           # number of adaptive cutpoints
  XBART_params$tau = var(y) / num_tress                                   # prior variance of mu (leaf parameter)
  return(XBART_params)
}

grow_tree <- function(x, y, split_prob, depth, max_depth){
    split = rbinom(1, 1, split_prob)
    if (split & depth < max_depth){
      # print("split")
      split_point = runif(1, -10, 10)
      cat('split point', split_point, "\n")
      y[x<split_point] = grow_tree(x[x<split_point], y[x<split_point], split_prob, depth+1, max_depth)
      y[x>=split_point] = grow_tree(x[x>=split_point], y[x>=split_point], split_prob, depth+1, max_depth)
      return(y)
    }
    else{
      # print("no split")
      mu = runif(1, -10, 10)
      cat('leaf mean' , mu, "\n")
      return(rep(mu, length(y)))
    }
}


#######################################################################
library(XBART)

set.seed(100)
d = 1# number of TOTAL variables
dcat = 0 # number of categorical variables
# must be d >= dcat

# (X_continuous, X_categorical), 10 and 10 for each case, 20 in total



n = 10000 # size of training set
nt = 5000 # size of testing set

new_data = TRUE # generate new data
run_dbarts = FALSE # run dbarts
run_xgboost = FALSE # run xgboost
run_lightgbm = FALSE # run lightgbm
parl = TRUE # parallel computing


#######################################################################
# Data generating process

#######################################################################
# Have to put continuous variables first, then categorical variables  #
# X = (X_continuous, X_cateogrical)                                   #
#######################################################################
if (new_data) {
  x_total = runif(n + nt, -10, 10)
  max_depth = 5
  split_prob = 0.6
  y_total = rep(0, n + nt)
  y_total = grow_tree(x_total, y_total, split_prob, 0, max_depth)
  
  
  #xtest[,1] = round(xtest[,1],2)
  # ftrue = f(x)
  # ftest = f(xtest)
  train_set = sample(1:(n+nt), n, replace = FALSE)
  
  x = x_total[train_set]
  xtest = x_total[-train_set]
  ftrue = y_total[train_set]
  ftest = y_total[-train_set]
  sigma = sd(ftrue)

  #y = ftrue + sigma*(rgamma(n,1,1)-1)/(3+x[,d])
  #y_test = ftest + sigma*(rgamma(nt,1,1)-1)/(3+xtest[,d])

  y = ftrue + sigma * rnorm(n)
  y_test = ftest + sigma * rnorm(nt)
}

#######################################################################
# XBART
categ <- function(z, j) {
  q = as.numeric(quantile(x[, j], seq(0, 1, length.out = 100)))
  output = findInterval(z, c(q, + Inf))
  return(output)
}


params = get_XBART_params(n, d, y)
time = proc.time()
fit = XBART(as.matrix(y), as.matrix(x), as.matrix(xtest), p_categorical = dcat,
            params$num_trees, params$num_sweeps, params$max_depth,
            params$n_min, alpha = params$alpha, beta = params$beta, tau = params$tau, s = 1, kap = 1,
            mtry = params$mtry, draw_mu = TRUE,
            num_cutpoints = params$num_cutpoints, parallel = parl, random_seed = 100,no_split_penality=params$no_split_penality)

################################
# two ways to predict on testing set

# 1. set xtest as input to main fitting function
fhat.1 = apply(fit$yhats_test[, params$burnin:params$num_sweeps], 1, mean)
time = proc.time() - time
print(time[3])

# 2. a separate predict function
pred = predict(fit, xtest)
pred = rowMeans(pred[, params$burnin:params$num_sweeps])

time_XBART = round(time[3], 3)


#######################################################################
# dbarts
if (run_dbarts) {
  library(dbarts)

  time = proc.time()
  fit = bart(x, y, xtest, verbose = FALSE, numcut = 100, ndpost = 1000, nskip = 500)
  time = proc.time() - time
  print(time[3])
  fhat.db = fit$yhat.test.mean
  time_dbarts = round(time[3], 3)
}else{
  fhat.db = fhat.1
  time_dbarts = time_XBART
}


#######################################################################
# XGBoost
if(run_xgboost){
  library(xgboost)
}



#######################################################################
# LightGBM
if(run_lightgbm){
  library(xgboost)
}


#######################################################################
# print
xbart_rmse = sqrt(mean((fhat.1 - ftest) ^ 2))
print(paste("rmse of fit xbart: ", round(xbart_rmse, digits = 4)))
print(paste("rmse of fit dbart: ", round(sqrt(mean((fhat.db - ftest) ^ 2)), digits = 4)))

print(paste("running time, dbarts", time_dbarts))
print(paste("running time, XBART", time_XBART))


plot(ftest, fhat.db, pch = 20, col = 'orange')
points(ftest, fhat.1, pch = 20, col = 'slategray')
legend("topleft", c("dbarts", "XBART"), col = c("orange", "slategray"), pch = c(20, 20))


# For Travis
stopifnot(xbart_rmse < 1)
stopifnot(time_XBART < 5)


