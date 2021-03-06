# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

xbart_predict <- function(X, y_mean, tree_pnt) {
    .Call(`_XBART_xbart_predict`, X, y_mean, tree_pnt)
}

xbart_multinomial_predict <- function(X, y_mean, num_class, tree_pnt, iteration) {
    .Call(`_XBART_xbart_multinomial_predict`, X, y_mean, num_class, tree_pnt, iteration)
}

r_to_json <- function(y_mean, tree_pnt) {
    .Call(`_XBART_r_to_json`, y_mean, tree_pnt)
}

json_to_r <- function(json_string_r) {
    .Call(`_XBART_json_to_r`, json_string_r)
}

sample_int_crank <- function(n, size, prob) {
    .Call(`_XBART_sample_int_crank`, n, size, prob)
}

sample_int_ccrank <- function(n, size, prob) {
    .Call(`_XBART_sample_int_ccrank`, n, size, prob)
}

sample_int_expj <- function(n, size, prob) {
    .Call(`_XBART_sample_int_expj`, n, size, prob)
}

sample_int_expjs <- function(n, size, prob) {
    .Call(`_XBART_sample_int_expjs`, n, size, prob)
}

XBART_cpp <- function(y, X, Xtest, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau, no_split_penality, burnin = 1L, mtry = 0L, p_categorical = 0L, kap = 16, s = 4, verbose = FALSE, parallel = TRUE, set_random_seed = FALSE, random_seed = 0L, sample_weights_flag = TRUE, nthread = 0) {
    .Call(`_XBART_XBART_cpp`, y, X, Xtest, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau, no_split_penality, burnin, mtry, p_categorical, kap, s, verbose, parallel, set_random_seed, random_seed, sample_weights_flag, nthread)
}

XBART_CLT_cpp <- function(y, X, Xtest, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau, no_split_penality, burnin = 1L, mtry = 0L, p_categorical = 0L, kap = 16, s = 4, verbose = FALSE, parallel = TRUE, set_random_seed = FALSE, random_seed = 0L, sample_weights_flag = TRUE, nthread = 0) {
    .Call(`_XBART_XBART_CLT_cpp`, y, X, Xtest, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau, no_split_penality, burnin, mtry, p_categorical, kap, s, verbose, parallel, set_random_seed, random_seed, sample_weights_flag, nthread)
}

XBART_multinomial_cpp <- function(y, num_class, X, Xtest, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau_a, tau_b, no_split_penality, burnin = 1L, mtry = 0L, p_categorical = 0L, kap = 16, s = 4, verbose = FALSE, parallel = TRUE, set_random_seed = FALSE, random_seed = 0L, sample_weights_flag = TRUE, separate_tree = FALSE, stop_threshold = 0, nthread = 0L, weight = 1, hmult = 2, heps = 0.1, update_tau = FALSE) {
    .Call(`_XBART_XBART_multinomial_cpp`, y, num_class, X, Xtest, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau_a, tau_b, no_split_penality, burnin, mtry, p_categorical, kap, s, verbose, parallel, set_random_seed, random_seed, sample_weights_flag, separate_tree, stop_threshold, nthread, weight, hmult, heps, update_tau)
}

XBART_Probit_cpp <- function(y, X, Xtest, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau, no_split_penality, burnin = 1L, mtry = 0L, p_categorical = 0L, kap = 16, s = 4, verbose = FALSE, parallel = TRUE, set_random_seed = FALSE, random_seed = 0L, sample_weights_flag = TRUE, nthread = 0) {
    .Call(`_XBART_XBART_Probit_cpp`, y, X, Xtest, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau, no_split_penality, burnin, mtry, p_categorical, kap, s, verbose, parallel, set_random_seed, random_seed, sample_weights_flag, nthread)
}

XBART_MH_cpp <- function(y, X, Xtest, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau, no_split_penality, burnin = 1L, mtry = 0L, p_categorical = 0L, kap = 16, s = 4, verbose = FALSE, parallel = TRUE, set_random_seed = FALSE, random_seed = 0L, sample_weights_flag = TRUE, nthread) {
    .Call(`_XBART_XBART_MH_cpp`, y, X, Xtest, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau, no_split_penality, burnin, mtry, p_categorical, kap, s, verbose, parallel, set_random_seed, random_seed, sample_weights_flag, nthread)
}

