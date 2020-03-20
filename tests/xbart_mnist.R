library(XBART)
library(xgboost)

D <- read.csv('~/Dropbox/MNIST/mnist_train.csv',header=FALSE)
y = D[,1]
D = D[,-1]
# 
Dtest <- read.csv('~/Dropbox/MNIST/mnist_test.csv',header=FALSE)
ytest = Dtest[,1]
Dtest = Dtest[,-1]
# 
# 
#k = 2
#X = matrix(NA,28*28,k*10)
#for (h in 0:9){
#   print(h)
#S = svd(t(D[y==h,]))
#X[,(h*k):(h*k+k-1)+1] = S$u[,1:k]
# }

#XXinv = solve(t(X)%*%X)
#P = XXinv%*%t(X)
#X_train = t(P%*%t(D))
#X_test = t(P%*%t(Dtest))

X_train = D
X_test = Dtest
p = ncol(X_train)

#load("mnist_data.rda")
X.train = X_train
X.test = X_test
#X_train = X_train + 0.0001*rnorm(ncol(X_train)*nrow(X_train))
#X_test = X_test + 0.0001*rnorm(ncol(X_test)*nrow(X_test))
v = 0
for (h in 1:p){
breaks =unique(as.numeric(quantile(c(X_train[,h],X_test[,h]),seq(0,1,length.out=4))))
#breaks = seq(min(c(X_train[,h],X_test[,h])),max(c(X_train[,h],X_test[,h])),length.out = 25)
breaks = c(-Inf,breaks,+Inf)
#print(breaks)
if (length(breaks)>3){
v = v + 1
  X.train[,v] = cut(X_train[,h],breaks = breaks,include.lowest=TRUE,labels=FALSE)
  X.test[,v] = cut(X_test[,h],breaks = breaks,include.lowest=TRUE,labels=FALSE)
}
}

#print(v)
X_train = X.train[,1:v]
X_test = X.test[,1:v]
p = v

X_train[,1] = X_train[,1] + 0.01*rnorm(length(y))

# add a garbage variable
# X_train = cbind(rnorm(nrow(X_train)), X_train)
# X_test = cbind(rnorm(nrow(X_test)), X_test)
# colnames(X_train) = 1:(p+1)
# colnames(X_test) = 1:(p+1)


# X_train = X_train[1:1000,]
# y = y[1:1000]


# load("data.rda")




#concn = 1
# delta = seq(0.5,1.5,length.out = 20)
num_sweeps= 15
num_trees = 10
burnin = 2
Nmin = 5
max_depth = 25
mtry = 50
num_cutpoints=20

tm = proc.time()
fit = XBART.multinomial(y=matrix(y), num_class=10, X=X_train, Xtest=X_test,
                        num_trees=num_trees, num_sweeps=num_sweeps, max_depth=max_depth,
                        Nmin=Nmin, num_cutpoints=num_cutpoints, alpha=0.95, beta=1.25, tau=100/num_trees,
                        no_split_penality = 1,  weight = seq(1, 5, 0.5),burnin = burnin, mtry = mtry, p_categorical = p-1,
                        kap = 1, s = 1, verbose = TRUE, parallel = FALSE, set_random_seed = FALSE,
                        random_seed = NULL, sample_weights_flag = TRUE, sample_var_per_tree = FALSE, separate_trees = TRUE, phi_threshold = 0.2)

# # 
# # fit = XBART(y=y, X=X_train, Xtest=X_test, 
# #                         num_trees=num_trees, num_sweeps=num_sweeps, max_depth=max_depth, 
# #                         Nmin=Nmin, num_cutpoints=num_cutpoints, alpha=0.95, beta=1.25, tau=var(y)/num_trees, 
# #                         no_split_penality = 1, burnin = burnin, mtry = mtry, p_categorical = p-1, 
# #                         kap = 1, s = 1, verbose = TRUE, parallel = FALSE, set_random_seed = FALSE, 
# #                         random_seed = NULL, sample_weights_flag = TRUE)
# 
# # number of sweeps * number of observations * number of classes
# #dim(fit$yhats_test)
# tm = proc.time()-tm
# cat(paste("\n", "xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
# 
# pred = apply(fit$yhats_test[(burnin+1):(num_sweeps-0),,], c(2,3), mean)
# yhat = max.col(pred)-1



 xgb.basic.mod1 <- xgboost(data = as.matrix(X_train),label=y,
                           num_class=10,
                           max_depth = 5,
                           subsample = 0.80,
                           nrounds=50,
                           early_stopping_rounds = 2,
                           eta = 0.9,
                           params=list(objective="multi:softprob"))

 xgb.basic.pred <- predict(xgb.basic.mod1, as.matrix(X_test))
 xgb.basic.pred <- matrix(xgb.basic.pred, ncol=10, byrow=TRUE)
 pred.xgb <- max.col(xgb.basic.pred) - 1

cat("running time ", t[3], " seconds \n")

cat("XBART error rate ", mean(yhat != ytest), "\n")

cat("xgboost error rate ", mean(pred.xgb != ytest), "\n")

fit$importance




