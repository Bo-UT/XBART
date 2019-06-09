#include "fit_std_main_loop.h"

void fit_std(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
             size_t N, size_t p,
             size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
             size_t n_min, size_t Ncutpoints, double alpha, double beta,
             double tau, size_t burnin, size_t mtry,
             double kap, double s,
             bool verbose,
             bool draw_mu, bool parallel,
             xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree,
             size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, 
             bool set_random_seed, size_t random_seed, double no_split_penality,bool sample_weights_flag)
{

    std::vector<double> initial_theta(1,0);
    std::unique_ptr<FitInfo> fit_info (new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta));
    
    // xinfo accept_prob;
    // xinfo drawn_accept;
    // xinfo prior_ratio;
    // xinfo proposal_ratio;
    // xinfo likelihood_ratio;
    // ini_xinfo(accept_prob, num_trees, num_sweeps-burnin);
    // ini_xinfo(drawn_accept, num_trees, num_sweeps-burnin);
    // ini_xinfo(prior_ratio, num_trees, num_sweeps-burnin);
    // ini_xinfo(proposal_ratio, num_trees, num_sweeps-burnin);
    // ini_xinfo(likelihood_ratio, num_trees, num_sweeps-burnin);
    std::vector<double> accept_vec;
    std::vector<double> MH_ratio;
    std::vector<double> proposal_ratio;
    std::vector<double> likelihood_ratio;
    std::vector<double> prior_ratio;
    std::vector<double> tree_ratio;


    if (parallel)
        thread_pool.start();

    //std::unique_ptr<NormalModel> model (new NormalModel);
    NormalModel *model = new NormalModel();
    model->setNoSplitPenality(no_split_penality);

    // initialize predcitions
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Set yhat_std to mean
    row_sum(fit_info->predictions_std, fit_info->yhat_std);

    // Residual for 0th tree
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    double sigma = 1.0;


    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {

            // Draw Sigma
            fit_info->residual_std_full = fit_info->residual_std - fit_info->predictions_std[tree_ind];
            std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(fit_info->residual_std_full) + s));
            sigma = 1.0 / sqrt(gamma_samp(fit_info->gen));
            sigma_draw_xinfo[sweeps][tree_ind] = sigma;

            // add prediction of current tree back to residual
            // then it's m - 1 trees residual
            fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

            if (fit_info->use_all && (sweeps > burnin) && (mtry != p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if(sample_weights_flag){
                mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
            }

            // trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, fit_info->residual_std, Xorder_std, Xpointer, mtry, fit_info->use_all, fit_info->split_count_all_tree, mtry_weight_current_tree, fit_info->split_count_current_tree, fit_info->categorical_variables, p_categorical, p_continuous, fit_info->X_values, fit_info->X_counts, fit_info->variable_ind, fit_info->X_num_unique, model, fit_info->data_pointers, tree_ind, fit_info->gen,sample_weights_flag);



            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // X_counts and X_num_unique should not be in fit_info because they depend on node
            // but they are initialized in fit_info object
            // so I'll pass fit_info->X_counts to root node, then create X_counts_left, X_counts_right for other nodes
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            trees[sweeps][tree_ind].grow_from_root(fit_info, sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, Xorder_std, Xpointer, mtry, mtry_weight_current_tree, p_categorical, p_continuous, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, sample_weights_flag);            // metropolis adjustment

            // Add split counts
            mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;
            fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

            if (sweeps >= 80)
            { 
                trees[sweeps-1][tree_ind].recalculate_prob(fit_info, sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, Xorder_std, Xpointer, mtry, mtry_weight_current_tree, p_categorical, p_continuous, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, sample_weights_flag);
                // trees[sweeps-1][tree_ind].update_split_prob(fit_info, sum_vec(fit_info->residual_std) / (double)N, 0,  max_depth_std[sweeps][tree_ind],  n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, Xorder_std, Xpointer, mtry, mtry_weight_current_tree, p_categorical, p_continuous, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, sample_weights_flag);
                metropolis_adjustment(fit_info, Xpointer, model, trees[sweeps-1][tree_ind], trees[sweeps][tree_ind], N, sigma, tree_ind, tau, alpha, beta, accept_vec, MH_ratio, proposal_ratio, likelihood_ratio, prior_ratio, tree_ratio);
            }

            // Update Predict
            predict_from_datapointers(Xpointer, N, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers ,model);
            model->updateResidual(fit_info->predictions_std, tree_ind, num_trees, fit_info->residual_std);
            

            fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];
        }

        // COUT << "average likelihood ratio " << accumulate(likelihood_ratio.begin(), likelihood_ratio.end(), 0.0) / likelihood_ratio.size() << endl;
        COUT << "average MH ratio " << accumulate(MH_ratio.begin(), MH_ratio.end(), 0.0) / MH_ratio.size() << endl;
        COUT << "average acceptance " << accumulate(accept_vec.begin(), accept_vec.end(), 0.0) / accept_vec.size() << endl;
        double accept_tree_ratio = 0.0;
        double reject_tree_ratio = 0.0;
        for (size_t i = 0; i < accept_vec.size(); i++){
            if (accept_vec[i] == 1.0){
                accept_tree_ratio += tree_ratio[i];
            }
            else{
                reject_tree_ratio += tree_ratio[i];
            }
        }
        accept_tree_ratio = accept_tree_ratio / accumulate(accept_vec.begin(), accept_vec.end(), 0.0);
        reject_tree_ratio = reject_tree_ratio / (accept_vec.size() - accumulate(accept_vec.begin(), accept_vec.end(), 0.0));
        COUT << "accepted tree ratio " << accept_tree_ratio << endl;
        COUT << "rejected tree ratio " << reject_tree_ratio << endl;



        fit_info->data_pointers_cp = fit_info->data_pointers;

        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit_info->yhat_std;
    }

    thread_pool.stop();

    delete model;
}

void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees,
                 size_t num_sweeps, xinfo &yhats_test_xinfo,
                 vector<vector<tree>> &trees, double y_mean)
{

    NormalModel *model = new NormalModel();
    xinfo predictions_test_std;
    ini_xinfo(predictions_test_std, N_test, num_trees);

    std::vector<double> yhat_test_std(N_test);
    row_sum(predictions_test_std, yhat_test_std);

    // initialize predcitions and predictions_test
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(predictions_test_std[ii].begin(), predictions_test_std[ii].end(), y_mean / (double)num_trees);
    }
    row_sum(predictions_test_std, yhat_test_std);

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {

            yhat_test_std = yhat_test_std - predictions_test_std[tree_ind];
            predict_from_tree(trees[sweeps][tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind],model);
            yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];
        }
        yhats_test_xinfo[sweeps] = yhat_test_std;
    }

    delete model;
    return;
}

void predict_std_multinomial(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees,
                 size_t num_sweeps, xinfo &yhats_test_xinfo,
                 vector<vector<tree>> &trees, double y_mean)
{

    NormalModel *model = new NormalModel();
    xinfo predictions_test_std;
    ini_xinfo(predictions_test_std, N_test, num_trees);

    std::vector<double> yhat_test_std(N_test);
    row_sum(predictions_test_std, yhat_test_std);

    // initialize predcitions and predictions_test
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(predictions_test_std[ii].begin(), predictions_test_std[ii].end(), y_mean / (double)num_trees);
    }
    row_sum(predictions_test_std, yhat_test_std);

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {

            yhat_test_std = yhat_test_std - predictions_test_std[tree_ind];
            predict_from_tree(trees[sweeps][tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind],model);
            yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];
        }
        yhats_test_xinfo[sweeps] = yhat_test_std;
    }

    delete model;
    return;
}

void fit_std_clt(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
                 size_t N, size_t p,
                 size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
                 size_t n_min, size_t Ncutpoints, double alpha, double beta,
                 double tau, size_t burnin, size_t mtry,
                 double kap, double s,
                 bool verbose,
                 bool draw_mu, bool parallel,
                 xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree,
                 size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed, double no_split_penality, bool sample_weights_flag)
{

    std::vector<double> initial_theta(1,0);
    std::unique_ptr<FitInfo> fit_info (new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta));

    if (parallel)
        thread_pool.start();

    CLTClass *model = new CLTClass();
    model->setNoSplitPenality(no_split_penality);

    // initialize predcitions and predictions_test
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Residual for 0th tree
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    double sigma = 0.0;

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {
            std::cout << "Tree " << tree_ind << std::endl;
            fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

            model->total_fit = fit_info->yhat_std;

            if ((sweeps > burnin) && (mtry < p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

            //COUT << fit_info->split_count_current_tree << endl;

            // subtract old tree for sampling case
            if(sample_weights_flag){
                mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
            }


            trees[sweeps][tree_ind].grow_from_root(fit_info, sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, Xorder_std, Xpointer, mtry, mtry_weight_current_tree, p_categorical, p_continuous, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, sample_weights_flag);


            mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;

            fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

            // fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
            predict_from_datapointers(Xpointer, N, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers,model);

            // update residual, now it's residual of m trees
            model->updateResidual(fit_info->predictions_std, tree_ind, num_trees, fit_info->residual_std);

            fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];

            std::cout << "stuff stat" << model->suff_stat_total << std::endl;
        }
        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit_info->yhat_std;
    }
    thread_pool.stop();
    delete model;
}

// void fit_std_multinomial(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
//                  size_t N, size_t p,
//                  size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
//                  size_t n_min, size_t Ncutpoints, double alpha, double beta,
//                  double tau, size_t burnin, size_t mtry,
//                  double kap, double s,
//                  bool verbose,
//                  size_t = n_class,
//                  bool draw_mu, bool parallel,
//                  xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree,
//                  size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed, double no_split_penality, bool sample_weights_flag)
// {

//     std::vector<double> initial_theta(1,0);
//     std::unique_ptr<FitInfo> fit_info (new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta));
	

//     if (parallel)
//         thread_pool.start();

//     LogitClass *model = new LogitClass(n_class);
    
//     model->setNoSplitPenality(no_split_penality);
//     model->setNumClasses(n_class)
    
//     // initialize Phi
//     std::vector<double> Phi(N,1.0);
    
//       // initialize partialFits
//  	std::vector<std::vector<double>> partialFits(N, std::vector<double>(n_class, 1.0));
    

// 	model->slop = &partialFits
// 	model->phi = &Phi



//     // initialize predcitions and predictions_test
//     for (size_t ii = 0; ii < num_trees; ii++)
//     {
//         std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)num_trees);
//     }

//     // Residual for 0th tree
//     fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

//     double sigma = 0.0;

//     for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
//     {

//         if (verbose == true)
//         {
//             COUT << "--------------------------------" << endl;
//             COUT << "number of sweeps " << sweeps << endl;
//             COUT << "--------------------------------" << endl;
//         }

//         for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
//         {
//             std::cout << "Tree " << tree_ind << std::endl;
//             fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

//             model->total_fit = fit_info->yhat_std;

//             if ((sweeps > burnin) && (mtry < p))
//             {
//                 fit_info->use_all = false;
//             }

//             // clear counts of splits for one tree
//             std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

//             //COUT << fit_info->split_count_current_tree << endl;

//             // subtract old tree for sampling case
//             if(sample_weights_flag){
//                 mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
//             }


//             trees[sweeps][tree_ind].grow_from_root(fit_info, sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, Xorder_std, Xpointer, mtry, mtry_weight_current_tree, p_categorical, p_continuous, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, sample_weights_flag);


//             mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;

//             fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

//             // fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
//             predict_from_datapointers(Xpointer, N, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers,model);

           
//            //updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std)
//             // update residual, now it's residual of m trees
//             model->updateResidual(fit_info->predictions_std, tree_ind, num_trees, slop);

//             fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];

//             std::cout << "stuff stat" << model->suff_stat_total << std::endl;
//         }
//         // save predictions to output matrix
//         yhats_xinfo[sweeps] = fit_info->yhat_std;
//     }
//     thread_pool.stop();
//     delete model;
// }

void fit_std_probit(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
                    size_t N, size_t p,
                    size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
                    size_t n_min, size_t Ncutpoints, double alpha, double beta,
                    double tau, size_t burnin, size_t mtry,
                    double kap, double s,
                    bool verbose,
                    bool draw_mu, bool parallel,
                    xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree,
                    size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed, double no_split_penality, bool sample_weights_flag)
{

    std::vector<double> initial_theta(1,0);
    std::unique_ptr<FitInfo> fit_info (new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta));


    if (parallel)
        thread_pool.start();

    NormalModel *model = new NormalModel();
    model->setNoSplitPenality(no_split_penality);

    // initialize predcitions
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Set yhat_std to mean
    row_sum(fit_info->predictions_std, fit_info->yhat_std);

    // Residual for 0th tree
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    double sigma = 1.0;

    // Probit
    std::vector<double> z = y_std;
    std::vector<double> z_prev(N);

    double a = 0;
    double b = 1;
    double mu_temp;
    double u;

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {

            // Update Z
            if (verbose){
                cout << "Tree "<< tree_ind << endl;
                cout << "Updating Z" << endl;
            }
            z_prev = z;
            for (size_t i = 0; i < N; i++)
            {
                a = 0;
                b = 1;

                mu_temp = normCDF(z_prev[i]);

                // Draw from truncated normal via inverse CDF methods
                if (y_std[i] > 0)
                {
                    a = std::min(mu_temp, 0.999);
                }
                else
                {
                    b = std::max(mu_temp, 0.001);
                }

                std::uniform_real_distribution<double> unif(a, b);
                u = unif(fit_info->gen);
                z[i] = normCDFInv(u) + mu_temp;
            }

            // add prediction of current tree back to residual
            // then it's m - 1 trees residual
            fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

            if (fit_info->use_all && (sweeps > burnin) && (mtry != p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

            if (verbose){
                cout << "Grow from root" << endl;
            }

            if(sample_weights_flag){
                mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
            }


            trees[sweeps][tree_ind].grow_from_root(fit_info, sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, Xorder_std, Xpointer, mtry, mtry_weight_current_tree, p_categorical, p_continuous, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, sample_weights_flag);

            // Add split counts
            //            fit_info->mtry_weight_current_tree = fit_info->mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];

            mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;
            fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

            //	COUT << "outer loop split_count" << fit_info->split_count_current_tree << endl;
            //	COUT << "outer loop weights" << fit_info->mtry_weight_current_tree << endl;

            // Update Predict
            predict_from_datapointers(Xpointer, N, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers,model);

            // update residual, now it's residual of m trees
            model->updateResidual(fit_info->predictions_std, tree_ind, num_trees, fit_info->residual_std);
            for (size_t i = 0; i < N; i++)
            {
                fit_info->residual_std[i] = fit_info->residual_std[i] - z_prev[i] + z[i];
            }

            fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];
        }
        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit_info->yhat_std;
    }

    thread_pool.stop();
    delete model;
}



void fit_std_MH(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
             size_t N, size_t p,
             size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
             size_t n_min, size_t Ncutpoints, double alpha, double beta,
             double tau, size_t burnin, size_t mtry,
             double kap, double s,
             bool verbose,
             bool draw_mu, bool parallel,
             xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree,
             size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, 
             bool set_random_seed, size_t random_seed, double no_split_penality,bool sample_weights_flag)
{

    std::vector<double> initial_theta(1,0);
    std::unique_ptr<FitInfo> fit_info (new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta));

    if (parallel)
        thread_pool.start();

    //std::unique_ptr<NormalModel> model (new NormalModel);
    NormalModel *model = new NormalModel();
    model->setNoSplitPenality(no_split_penality);

    // initialize predcitions
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Set yhat_std to mean
    row_sum(fit_info->predictions_std, fit_info->yhat_std);

    // Residual for 0th tree
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    double sigma = 1.0;

    std::vector<tree> temp_tree = trees[0];

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {

            // Draw Sigma
            fit_info->residual_std_full = fit_info->residual_std - fit_info->predictions_std[tree_ind];
            std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(fit_info->residual_std_full) + s));
            sigma = 1.0 / sqrt(gamma_samp(fit_info->gen));
            sigma_draw_xinfo[sweeps][tree_ind] = sigma;

            // add prediction of current tree back to residual
            // then it's m - 1 trees residual
            fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

            if (fit_info->use_all && (sweeps > burnin) && (mtry != p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if(sample_weights_flag){
                mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
            }

            // trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, fit_info->residual_std, Xorder_std, Xpointer, mtry, fit_info->use_all, fit_info->split_count_all_tree, mtry_weight_current_tree, fit_info->split_count_current_tree, fit_info->categorical_variables, p_categorical, p_continuous, fit_info->X_values, fit_info->X_counts, fit_info->variable_ind, fit_info->X_num_unique, model, fit_info->data_pointers, tree_ind, fit_info->gen,sample_weights_flag);



            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // X_counts and X_num_unique should not be in fit_info because they depend on node
            // but they are initialized in fit_info object
            // so I'll pass fit_info->X_counts to root node, then create X_counts_left, X_counts_right for other nodes
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


            if(sweeps > 1){
                cout << "start " << endl;
                cout << temp_tree[tree_ind].getprob_split() << endl;
                cout << temp_tree[tree_ind].getv() << " " << temp_tree[tree_ind].getc() << endl;

                temp_tree[tree_ind].update_split_prob(fit_info, sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, Xorder_std, Xpointer, mtry, mtry_weight_current_tree, p_categorical, p_continuous, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, sample_weights_flag);

                cout << temp_tree[tree_ind].getprob_split() << endl;
                cout << "-------------------" << endl;
            }

            temp_tree[tree_ind].transition_prob();

            temp_tree[tree_ind].log_like_tree(pow(sigma,2 ), tau);

            trees[sweeps][tree_ind].grow_from_root(fit_info, sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, Xorder_std, Xpointer, mtry, mtry_weight_current_tree, p_categorical, p_continuous, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, sample_weights_flag);

            temp_tree[tree_ind].tonull();
            temp_tree[tree_ind] = trees[sweeps][tree_ind];

            // Add split counts
            mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;
            fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

            // Update Predict
            predict_from_datapointers(Xpointer, N, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers,model);

            // update residual, now it's residual of m trees
            model->updateResidual(fit_info->predictions_std, tree_ind, num_trees, fit_info->residual_std);

            fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];
        }
        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit_info->yhat_std;
    }
    thread_pool.stop();

    delete model;
}
