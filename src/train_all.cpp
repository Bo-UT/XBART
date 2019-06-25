// [[Rcpp::depends(RcppArmadillo)]]

#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "fit_std_main_loop.h"

using namespace Rcpp;
using namespace std;
using namespace chrono;

////////////////////////////////////////////////////////////////////////
//                                                                    //
//                                                                    //
//  Full function, support both continuous and categorical variables  //
//                                                                    //
//                                                                    //
////////////////////////////////////////////////////////////////////////

void rcpp_to_std2(arma::mat y, arma::mat X, arma::mat Xtest, arma::mat max_depth_num, std::vector<double> &y_std, double &y_mean, Rcpp::NumericMatrix &X_std, Rcpp::NumericMatrix &Xtest_std, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std)
{
    // The goal of this function is to convert RCPP object to std objects

    // TODO: Refactor code so for loops are self contained functions
    // TODO: Why RCPP and not std?
    // TODO: inefficient Need Replacement?

    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;

    // Create y_std
    for (size_t i = 0; i < N; i++)
    {
        y_std[i] = y(i, 0);
        y_mean = y_mean + y_std[i];
    }
    y_mean = y_mean / (double)N;

    // X_std
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }

    //X_std_test
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xtest_std(i, j) = Xtest(i, j);
        }
    }

    //max_depth_std_test
    for (size_t i = 0; i < max_depth_num.n_rows; i++)
    {
        for (size_t j = 0; j < max_depth_num.n_cols; j++)
        {
            max_depth_std[j][i] = max_depth_num(i, j);
        }
    }

    // Create Xorder
    // Order
    arma::umat Xorder(X.n_rows, X.n_cols);
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = arma::sort_index(X.col(i));
    }
    // Create
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xorder_std[j][i] = Xorder(i, j);
        }
    }

    return;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART(arma::mat y, arma::mat X, arma::mat Xtest,
                 size_t num_trees, size_t num_sweeps, arma::mat max_depth_num,
                 size_t n_min, size_t num_cutpoints, double alpha, double beta,
                 double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0,
                 double kap = 16, double s = 4, bool verbose = false,
                 bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
{

    bool draw_mu = true;

    auto start = system_clock::now();

    size_t N = X.n_rows;
    // number of total variables
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;

    // number of continuous variables
    size_t p_continuous = p - p_categorical;

    // suppose first p_continuous variables are continuous, then categorical

    assert(mtry <= p);
    assert(burnin <= num_sweeps);

    if (mtry == 0)
    {
        mtry = p;
    }

    if (mtry != p)
    {
        COUT << "Sample " << mtry << " out of " << p << " variables when grow each tree." << endl;
    }

    arma::umat Xorder(X.n_rows, X.n_cols);
    xinfo_sizet Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);

    std::vector<double> y_std(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    xinfo_sizet max_depth_std;
    ini_xinfo_sizet(max_depth_std, max_depth_num.n_rows, max_depth_num.n_cols);

    rcpp_to_std2(y, X, Xtest, max_depth_num, y_std, y_mean, X_std, Xtest_std, Xorder_std, max_depth_std);

    ///////////////////////////////////////////////////////////////////

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    xinfo yhats_std;
    ini_xinfo(yhats_std, N, num_sweeps);
    xinfo yhats_test_std;
    ini_xinfo(yhats_test_std, N_test, num_sweeps);

    xinfo yhats_xinfo;
    ini_xinfo(yhats_xinfo, N, num_sweeps);

    xinfo yhats_test_xinfo;
    ini_xinfo(yhats_test_xinfo, N, num_sweeps);

    xinfo sigma_draw_xinfo;
    ini_xinfo(sigma_draw_xinfo, num_trees, num_sweeps);

    std::vector<double> mtry_weight_current_tree(p);

    // // Create trees
    vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2)[i] = vector<tree>(num_trees);
    }

    // prior settings
    Prior prior = Prior(tau, alpha, beta, kap, s);

    // FitInfo settings
    std::vector<double> initial_theta(1, 0);
    std::unique_ptr<FitInfo> fit_info(new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta, n_min, num_cutpoints, parallel, mtry, Xpointer, draw_mu, num_sweeps, sample_weights_flag));

    // define model
    NormalModel *model = new NormalModel();
    model->setNoSplitPenality(no_split_penality);

    std::vector<double> accept_count;
    std::vector<double> MH_vector;
    std::vector<double> Q_ratio;
    std::vector<double> P_ratio;
    std::vector<double> prior_ratio;

    /////////////////////////////////////////////////////////////////
   fit_std(y_std, y_mean, Xorder_std, max_depth_std, burnin, verbose, yhats_xinfo, sigma_draw_xinfo, mtry_weight_current_tree, *trees2, no_split_penality, prior, fit_info, model, accept_count, MH_vector, P_ratio, Q_ratio, prior_ratio);

    predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2, y_mean);

    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats_test(i, j) = yhats_test_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)mtry_weight_current_tree[i];
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
    return Rcpp::List::create(
        Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("accept_count") = accept_count,
        Rcpp::Named("MH") = MH_vector,
        Rcpp::Named("Q_ratio") = Q_ratio,
        Rcpp::Named("P_ratio") = P_ratio,
        Rcpp::Named("prior_ratio") = prior_ratio,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt,
                                                       Rcpp::Named("y_mean") = y_mean,
                                                       Rcpp::Named("p") = p));
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_CLT(arma::mat y, arma::mat X, arma::mat Xtest,
                     size_t num_trees, size_t num_sweeps, arma::mat max_depth_num,
                     size_t n_min, size_t num_cutpoints, double alpha, double beta,
                     double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0,
                     double kap = 16, double s = 4, bool verbose = false,
                     bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
{
    bool draw_mu = true;

    auto start = system_clock::now();

    size_t N = X.n_rows;
    // number of total variables
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;

    // number of continuous variables
    size_t p_continuous = p - p_categorical;

    // suppose first p_continuous variables are continuous, then categorical

    assert(mtry <= p);
    assert(burnin <= num_sweeps);

    if (mtry == 0)
    {
        mtry = p;
    }

    if (mtry != p)
    {
        COUT << "Sample " << mtry << " out of " << p << " variables when grow each tree." << endl;
    }

    arma::umat Xorder(X.n_rows, X.n_cols);
    xinfo_sizet Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);

    std::vector<double> y_std(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    xinfo_sizet max_depth_std;
    ini_xinfo_sizet(max_depth_std, max_depth_num.n_rows, max_depth_num.n_cols);

    rcpp_to_std2(y, X, Xtest, max_depth_num, y_std, y_mean, X_std, Xtest_std, Xorder_std, max_depth_std);

    ///////////////////////////////////////////////////////////////////

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    xinfo yhats_std;
    ini_xinfo(yhats_std, N, num_sweeps);
    xinfo yhats_test_std;
    ini_xinfo(yhats_test_std, N_test, num_sweeps);

    xinfo yhats_xinfo;
    ini_xinfo(yhats_xinfo, N, num_sweeps);

    xinfo yhats_test_xinfo;
    ini_xinfo(yhats_test_xinfo, N, num_sweeps);

    xinfo sigma_draw_xinfo;
    ini_xinfo(sigma_draw_xinfo, num_trees, num_sweeps);

    std::vector<double> mtry_weight_current_tree(p);

    // // Create trees
    vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2)[i] = vector<tree>(num_trees);
    }

    // prior settings
    Prior prior = Prior(tau, alpha, beta, kap, s);

    // FitInfo settings
    std::vector<double> initial_theta(1, 0);
    std::unique_ptr<FitInfo> fit_info(new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta, n_min, num_cutpoints, parallel, mtry, Xpointer, draw_mu, num_sweeps, sample_weights_flag));

    // define model
    CLTClass *model = new CLTClass();
    model->setNoSplitPenality(no_split_penality);

    /////////////////////////////////////////////////////////////////

    fit_std_clt(y_std, y_mean, Xorder_std, max_depth_std, burnin, verbose, yhats_xinfo, sigma_draw_xinfo, mtry_weight_current_tree, *trees2, no_split_penality, prior, fit_info, model);

    predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2, y_mean);

    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats_test(i, j) = yhats_test_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }

    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)mtry_weight_current_tree[i];
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
    return Rcpp::List::create(
        Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt,
                                                       Rcpp::Named("y_mean") = y_mean,
                                                       Rcpp::Named("p") = p));
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_multinomial(arma::mat y, arma::mat X, arma::mat Xtest,
                             size_t num_trees, size_t num_sweeps, arma::mat max_depth_num,
                             size_t n_min, size_t num_cutpoints, double alpha, double beta,
                             double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0,
                             double kap = 16, double s = 4, bool verbose = false,
                             bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
{
    bool draw_mu = true;

    auto start = system_clock::now();

    size_t N = X.n_rows;
    // number of total variables
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;

    // number of continuous variables
    size_t p_continuous = p - p_categorical;

    // suppose first p_continuous variables are continuous, then categorical

    assert(mtry <= p);
    assert(burnin <= num_sweeps);

    if (mtry == 0)
    {
        mtry = p;
    }

    if (mtry != p)
    {
        COUT << "Sample " << mtry << " out of " << p << " variables when grow each tree." << endl;
    }

    arma::umat Xorder(X.n_rows, X.n_cols);
    xinfo_sizet Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);

    std::vector<double> y_std(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    xinfo_sizet max_depth_std;
    ini_xinfo_sizet(max_depth_std, max_depth_num.n_rows, max_depth_num.n_cols);

    rcpp_to_std2(y, X, Xtest, max_depth_num, y_std, y_mean, X_std, Xtest_std, Xorder_std, max_depth_std);

    ///////////////////////////////////////////////////////////////////

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    xinfo yhats_std;
    ini_xinfo(yhats_std, N, num_sweeps);
    xinfo yhats_test_std;
    ini_xinfo(yhats_test_std, N_test, num_sweeps);

    xinfo yhats_xinfo;
    ini_xinfo(yhats_xinfo, N, num_sweeps);

    xinfo yhats_test_xinfo;
    ini_xinfo(yhats_test_xinfo, N, num_sweeps);

    xinfo sigma_draw_xinfo;
    ini_xinfo(sigma_draw_xinfo, num_trees, num_sweeps);

    std::vector<double> mtry_weight_current_tree(p);

    // // Create trees
    vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2)[i] = vector<tree>(num_trees);
    }

    /////////////////////////////////////////////////////////////////
    //
    //
    //      Need to define n_class
    //
    //
    /////////////////////////////////////////////////////////////////

    size_t n_class;

    // prior settings
    Prior prior = Prior(tau, alpha, beta, kap, s);

    // FitInfo settings
    std::vector<double> initial_theta(1, 0);
    std::unique_ptr<FitInfo> fit_info(new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta, n_min, num_cutpoints, parallel, mtry, Xpointer, draw_mu, num_sweeps, sample_weights_flag));

    // define model
    LogitClass *model = new LogitClass();
    model->setNoSplitPenality(no_split_penality);

    /////////////////////////////////////////////////////////////////
    fit_std_multinomial(y_std, y_mean, Xorder_std, max_depth_std, burnin, verbose, yhats_xinfo, sigma_draw_xinfo, mtry_weight_current_tree, *trees2, no_split_penality, prior, fit_info, model);

    predict_std_multinomial(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2, y_mean);

    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats_test(i, j) = yhats_test_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }

    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)mtry_weight_current_tree[i];
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
    return Rcpp::List::create(
        Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt,
                                                       Rcpp::Named("y_mean") = y_mean,
                                                       Rcpp::Named("p") = p));
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_Probit(arma::mat y, arma::mat X, arma::mat Xtest,
                        size_t num_trees, size_t num_sweeps, arma::mat max_depth_num,
                        size_t n_min, size_t num_cutpoints, double alpha, double beta,
                        double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0,
                        double kap = 16, double s = 4, bool verbose = false,
                        bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
{
    bool draw_mu = true;

    auto start = system_clock::now();

    size_t N = X.n_rows;
    // number of total variables
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;

    // number of continuous variables
    size_t p_continuous = p - p_categorical;

    // suppose first p_continuous variables are continuous, then categorical

    assert(mtry <= p);
    assert(burnin <= num_sweeps);

    if (mtry == 0)
    {
        mtry = p;
    }

    if (mtry != p)
    {
        COUT << "Sample " << mtry << " out of " << p << " variables when grow each tree." << endl;
    }

    arma::umat Xorder(X.n_rows, X.n_cols);
    xinfo_sizet Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);

    std::vector<double> y_std(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    xinfo_sizet max_depth_std;
    ini_xinfo_sizet(max_depth_std, max_depth_num.n_rows, max_depth_num.n_cols);

    rcpp_to_std2(y, X, Xtest, max_depth_num, y_std, y_mean, X_std, Xtest_std, Xorder_std, max_depth_std);

    ///////////////////////////////////////////////////////////////////

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    xinfo yhats_std;
    ini_xinfo(yhats_std, N, num_sweeps);
    xinfo yhats_test_std;
    ini_xinfo(yhats_test_std, N_test, num_sweeps);

    xinfo yhats_xinfo;
    ini_xinfo(yhats_xinfo, N, num_sweeps);

    xinfo yhats_test_xinfo;
    ini_xinfo(yhats_test_xinfo, N, num_sweeps);

    xinfo sigma_draw_xinfo;
    ini_xinfo(sigma_draw_xinfo, num_trees, num_sweeps);

    vec_d mtry_weight_current_tree(p);

    // // Create trees
    vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2)[i] = vector<tree>(num_trees);
    }

    // prior settings
    Prior prior = Prior(tau, alpha, beta, kap, s);

    // FitInfo settings
    std::vector<double> initial_theta(1, 0);
    std::unique_ptr<FitInfo> fit_info(new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta, n_min, num_cutpoints, parallel, mtry, Xpointer, draw_mu, num_sweeps, sample_weights_flag));

    // define model
    NormalModel *model = new NormalModel();
    model->setNoSplitPenality(no_split_penality);

    /////////////////////////////////////////////////////////////////

    fit_std_probit(y_std, y_mean, Xorder_std, max_depth_std, burnin, verbose, yhats_xinfo, sigma_draw_xinfo, mtry_weight_current_tree, *trees2, no_split_penality, prior, fit_info, model);

    predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2, y_mean);

    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats_test(i, j) = yhats_test_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)mtry_weight_current_tree[i];
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
    return Rcpp::List::create(
        Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt,
                                                       Rcpp::Named("y_mean") = y_mean,
                                                       Rcpp::Named("p") = p));
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_MH(arma::mat y, arma::mat X, arma::mat Xtest,
                    size_t num_trees, size_t num_sweeps, arma::mat max_depth_num,
                    size_t n_min, size_t num_cutpoints, double alpha, double beta,
                    double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0,
                    double kap = 16, double s = 4, bool verbose = false,
                    bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
{

    cout << "MHMHMH" << endl;

    bool draw_mu = true;

    auto start = system_clock::now();

    size_t N = X.n_rows;
    // number of total variables
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;

    // number of continuous variables
    size_t p_continuous = p - p_categorical;

    // suppose first p_continuous variables are continuous, then categorical

    assert(mtry <= p);
    assert(burnin <= num_sweeps);

    if (mtry == 0)
    {
        mtry = p;
    }

    if (mtry != p)
    {
        COUT << "Sample " << mtry << " out of " << p << " variables when grow each tree." << endl;
    }

    arma::umat Xorder(X.n_rows, X.n_cols);
    xinfo_sizet Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);

    std::vector<double> y_std(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    xinfo_sizet max_depth_std;
    ini_xinfo_sizet(max_depth_std, max_depth_num.n_rows, max_depth_num.n_cols);

    rcpp_to_std2(y, X, Xtest, max_depth_num, y_std, y_mean, X_std, Xtest_std, Xorder_std, max_depth_std);

    ///////////////////////////////////////////////////////////////////

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    xinfo yhats_std;
    ini_xinfo(yhats_std, N, num_sweeps);
    xinfo yhats_test_std;
    ini_xinfo(yhats_test_std, N_test, num_sweeps);

    xinfo yhats_xinfo;
    ini_xinfo(yhats_xinfo, N, num_sweeps);

    xinfo yhats_test_xinfo;
    ini_xinfo(yhats_test_xinfo, N, num_sweeps);

    xinfo sigma_draw_xinfo;
    ini_xinfo(sigma_draw_xinfo, num_trees, num_sweeps);

    std::vector<double> mtry_weight_current_tree(p);

    // // Create trees
    vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2)[i] = vector<tree>(num_trees);
    }

    // prior settings
    Prior prior = Prior(tau, alpha, beta, kap, s);

    // FitInfo settings
    std::vector<double> initial_theta(1, 0);
    std::unique_ptr<FitInfo> fit_info(new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta, n_min, num_cutpoints, parallel, mtry, Xpointer, draw_mu, num_sweeps, sample_weights_flag));

    // define model
    NormalModel *model = new NormalModel();

    /////////////////////////////////////////////////////////////////
    std::vector<double> accept_count;
    std::vector<double> MH_vector;
    std::vector<double> Q_ratio;
    std::vector<double> P_ratio;
    std::vector<double> prior_ratio;

    fit_std_MH(y_std, y_mean, Xorder_std, max_depth_std, burnin, verbose, yhats_xinfo, sigma_draw_xinfo, mtry_weight_current_tree, *trees2, no_split_penality, prior, fit_info, model, accept_count, MH_vector, P_ratio, Q_ratio, prior_ratio);

    predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2, y_mean);

    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats_test(i, j) = yhats_test_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)mtry_weight_current_tree[i];
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
    return Rcpp::List::create(
        Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("accept_count") = accept_count,
        Rcpp::Named("MH") = MH_vector,
        Rcpp::Named("Q_ratio") = Q_ratio,
        Rcpp::Named("P_ratio") = P_ratio,
        Rcpp::Named("prior_ratio") = prior_ratio,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt,
                                                       Rcpp::Named("y_mean") = y_mean,
                                                       Rcpp::Named("p") = p));
}
