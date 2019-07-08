#include "fit_std_main_loop.h"

void fit_std(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, 
            size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, 
            vector<vector<tree>> &trees, double no_split_penality, Prior &prior, std::unique_ptr<FitInfo> &fit_info,
            Model *model, std::vector<double>& accept_count,
            std::vector<double>& MH_vector, std::vector<double>& P_ratio, std::vector<double>& Q_ratio, std::vector<double>& prior_ratio)
{

    std::vector<double> tree_ratio;

    if (fit_info->parallel)
        thread_pool.start();

    // initialize predcitions
    for (size_t ii = 0; ii < fit_info->num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double) fit_info->num_trees);
    }

    // Set yhat_std to mean
    row_sum(fit_info->predictions_std, fit_info->yhat_std);

    // Residual for 0th tree
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    double sigma = 1.0;

    NodeData root_data;

    for (size_t sweeps = 0; sweeps < fit_info->num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < fit_info->num_trees; tree_ind++)
        {

            // Draw Sigma
            fit_info->residual_std_full = fit_info->residual_std - fit_info->predictions_std[tree_ind];
            std::gamma_distribution<double> gamma_samp((fit_info->n_y + prior.kap) / 2.0, 2.0 / (sum_squared(fit_info->residual_std_full) + prior.s));
            sigma = 1.0 / sqrt(gamma_samp(fit_info->gen));
            sigma_draw_xinfo[sweeps][tree_ind] = sigma;

            // add prediction of current tree back to residual
            // then it's m - 1 trees residual
            fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

            if (fit_info->use_all && (sweeps > burnin) && (fit_info->mtry != fit_info->p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (fit_info->sample_weights_flag)
            {
                mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
            }

            // set sufficient statistics at root node first
            trees[sweeps][tree_ind].suff_stat[0] = sum_vec(fit_info->residual_std) / (double) fit_info->n_y;
            trees[sweeps][tree_ind].suff_stat[1] = sum_squared(fit_info->residual_std);

            // initialize node data for the root node
            root_data.update_value(sigma, fit_info->n_y);

            // set sufficient statistics at root node first 
            trees[sweeps][tree_ind].suff_stat[0] = sum_vec(fit_info->residual_std) / (double) fit_info->n_y;
            trees[sweeps][tree_ind].suff_stat[1] = sum_squared(fit_info->residual_std);

            trees[sweeps][tree_ind].grow_from_root(fit_info, max_depth_std[sweeps][tree_ind], Xorder_std, mtry_weight_current_tree, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, prior, root_data, true, false, true);
            // Add split counts
            mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;
            fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

            if (sweeps > burnin)
            {
                trees[sweeps-1][tree_ind].recalculate_prob(fit_info, max_depth_std[sweeps][tree_ind], Xorder_std, mtry_weight_current_tree, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, prior, root_data, true, false, true);
                // trees[sweeps-1][tree_ind].update_split_prob(fit_info, sum_vec(fit_info->residual_std) / (double)N, 0,  max_depth_std[sweeps][tree_ind],  n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, Xorder_std, Xpointer, mtry, mtry_weight_current_tree, p_categorical, p_continuous, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, sample_weights_flag);
                metropolis_adjustment(fit_info, fit_info->X_std, model, trees[sweeps-1][tree_ind], trees[sweeps][tree_ind], fit_info->n_y, sigma, tree_ind, prior,  accept_count, MH_vector, Q_ratio, P_ratio, prior_ratio, tree_ratio);
            }
            
            // Update Predict
            predict_from_datapointers(fit_info->X_std, fit_info->n_y, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers, model);

            // update residual, now it's residual of m trees
            model->updateResidual(fit_info->predictions_std, tree_ind, fit_info->num_trees, fit_info->residual_std);

            fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];
        }

        // COUT << "average likelihood ratio " << accumulate(likelihood_ratio.begin(), likelihood_ratio.end(), 0.0) / likelihood_ratio.size() << endl;
        if (sweeps > burnin)
        {
            COUT << "average MH ratio " << accumulate(MH_vector.end() - fit_info->num_trees, MH_vector.end(), 0.0) / fit_info->num_trees << endl;
            COUT << "average acceptance " << accumulate(accept_count.end() - fit_info->num_trees, accept_count.end(), 0.0) / fit_info->num_trees << endl;
            
            // std::vector<double> accept_tree;
            // std::vector<double> reject_tree;
            // for (size_t i = accept_count.size() - fit_info->num_trees; i < accept_count.size(); i++){
            //     if (accept_count[i] == 1.0){
            //         accept_tree.push_back(tree_ratio[i]);
            //     }
            //     else{
            //         reject_tree.push_back(tree_ratio[i]);
            //     }
            // }
            // COUT << "accepted tree ratio " << accumulate(accept_tree.begin(), accept_tree.end(), 0.0) / accept_tree.size()<< endl;
            // COUT << "rejected tree ratio " << accumulate(reject_tree.begin(), reject_tree.end(), 0.0) / reject_tree.size() << endl;
            
        }

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
            predict_from_tree(trees[sweeps][tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind], model);
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
            predict_from_tree(trees[sweeps][tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind], model);
            yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];
        }
        yhats_test_xinfo[sweeps] = yhat_test_std;
    }

    delete model;
    return;
}

void fit_std_clt(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, 
                size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, 
                vector<vector<tree>> &trees, double no_split_penality, Prior &prior, std::unique_ptr<FitInfo> &fit_info, CLTClass *model)
{

    if (fit_info->parallel)
        thread_pool.start();

    // initialize predcitions and predictions_test
    for (size_t ii = 0; ii < fit_info->num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)fit_info->num_trees);
    }

    // Residual for 0th tree
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    double sigma = 0.0;

    NodeData root_data;

    for (size_t sweeps = 0; sweeps < fit_info->num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < fit_info->num_trees; tree_ind++)
        {
            std::cout << "Tree " << tree_ind << std::endl;
            fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

            model->total_fit = fit_info->yhat_std;

            if ((sweeps > burnin) && (fit_info->mtry < fit_info->p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

            //COUT << fit_info->split_count_current_tree << endl;

            // subtract old tree for sampling case
            if (fit_info->sample_weights_flag)
            {
                mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
            }
        
            // set sufficient statistics at root node first
            trees[sweeps][tree_ind].suff_stat[0] = sum_vec(fit_info->residual_std) / (double) fit_info->n_y;
            trees[sweeps][tree_ind].suff_stat[1] = sum_squared(fit_info->residual_std);
        
            root_data.update_value(sigma, fit_info->n_y);

            trees[sweeps][tree_ind].grow_from_root(fit_info, max_depth_std[sweeps][tree_ind], Xorder_std, mtry_weight_current_tree, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, prior, root_data, true, false, true);

            mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;

            fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

            // fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
            predict_from_datapointers(fit_info->X_std, fit_info->n_y, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers, model);

            // update residual, now it's residual of m trees
            model->updateResidual(fit_info->predictions_std, tree_ind, fit_info->num_trees, fit_info->residual_std);

            fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];

            // std::cout << "stuff stat" << model->suff_stat_total << std::endl;
        }
        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit_info->yhat_std;
    }
    thread_pool.stop();
    delete model;
}

void fit_std_multinomial(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, Prior &prior, std::unique_ptr<FitInfo> &fit_info, LogitClass *model)
{

    // if (parallel)
    //     thread_pool.start();

    // // initialize Phi
    // std::vector<double> Phi(N,1.0);

    //   // initialize partialFits
    // std::vector<std::vector<double>> partialFits(N, std::vector<double>(n_class, 1.0));

    // model->slop = &partialFits
    // model->phi = &Phi

    // // initialize predcitions and predictions_test
    // for (size_t ii = 0; ii < num_trees; ii++)
    // {
    //     std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)num_trees);
    // }

    // // Residual for 0th tree
    // fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    // double sigma = 0.0;

    // for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    // {

    //     if (verbose == true)
    //     {
    //         COUT << "--------------------------------" << endl;
    //         COUT << "number of sweeps " << sweeps << endl;
    //         COUT << "--------------------------------" << endl;
    //     }

    //     for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
    //     {
    //         std::cout << "Tree " << tree_ind << std::endl;
    //         fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

    //         model->total_fit = fit_info->yhat_std;

    //         if ((sweeps > burnin) && (mtry < p))
    //         {
    //             fit_info->use_all = false;
    //         }

    //         // clear counts of splits for one tree
    //         std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

    //         //COUT << fit_info->split_count_current_tree << endl;

    //         // subtract old tree for sampling case
    //         if(sample_weights_flag){
    //             mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
    //         }

            // // set sufficient statistics at root node first
            // trees[sweeps][tree_ind].suff_stat[0] = sum_vec(fit_info->residual_std) / (double)N;
            // trees[sweeps][tree_ind].suff_stat[1] = sum_squared(fit_info->residual_std);

    //         trees[sweeps][tree_ind].grow_from_root(fit_info, sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], tau, sigma, alpha, beta, draw_mu, Xorder_std, Xpointer, mtry, mtry_weight_current_tree, p_categorical, p_continuous, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, sample_weights_flag);

    //         mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;

    //         fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

    //         // fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
    //         predict_from_datapointers(Xpointer, N, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers,model);

    //        //updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std)
    //         // update residual, now it's residual of m trees
    //         model->updateResidual(fit_info->predictions_std, tree_ind, num_trees, slop);

    //         fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];

    //         std::cout << "stuff stat" << model->suff_stat_total << std::endl;
    //     }
    //     // save predictions to output matrix
    //     yhats_xinfo[sweeps] = fit_info->yhat_std;
    // }
    // thread_pool.stop();
    // delete model;
}

void fit_std_probit(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, Prior &prior, std::unique_ptr<FitInfo> &fit_info, Model *model)
{

    if (fit_info->parallel)
        thread_pool.start();

    // initialize predcitions
    for (size_t ii = 0; ii < fit_info->num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double) fit_info->num_trees);
    }

    // Set yhat_std to mean
    row_sum(fit_info->predictions_std, fit_info->yhat_std);

    // Residual for 0th tree
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    double sigma = 1.0;

    // Probit
    std::vector<double> z = y_std;
    std::vector<double> z_prev(fit_info->n_y);

    double a = 0;
    double b = 1;
    double mu_temp;
    double u;

    NodeData root_data;
    for (size_t sweeps = 0; sweeps < fit_info->num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < fit_info->num_trees; tree_ind++)
        {

            // Update Z
            if (verbose)
            {
                COUT << "Tree " << tree_ind << endl;
                COUT << "Updating Z" << endl;
            }
            z_prev = z;
            for (size_t i = 0; i < fit_info->n_y; i++)
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

            if (fit_info->use_all && (sweeps > burnin) && (fit_info->mtry != fit_info->p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

            if (verbose)
            {
                COUT << "Grow from root" << endl;
            }

            if (fit_info->sample_weights_flag)
            {
                mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
            }

            // set sufficient statistics at root node first
            trees[sweeps][tree_ind].suff_stat[0] = sum_vec(fit_info->residual_std) / (double) fit_info->n_y;
            trees[sweeps][tree_ind].suff_stat[1] = sum_squared(fit_info->residual_std);

            root_data.update_value(sigma, fit_info->n_y);

            trees[sweeps][tree_ind].grow_from_root(fit_info, max_depth_std[sweeps][tree_ind], Xorder_std, mtry_weight_current_tree, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, prior, root_data, true, false, true);

            // Add split counts
            //            fit_info->mtry_weight_current_tree = fit_info->mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];

            mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;
            fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

            //	COUT << "outer loop split_count" << fit_info->split_count_current_tree << endl;
            //	COUT << "outer loop weights" << fit_info->mtry_weight_current_tree << endl;

            // Update Predict
            predict_from_datapointers(fit_info->X_std, fit_info->n_y, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers, model);

            // update residual, now it's residual of m trees
            model->updateResidual(fit_info->predictions_std, tree_ind, fit_info->num_trees, fit_info->residual_std);
            for (size_t i = 0; i < fit_info->n_y; i++)
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

void fit_std_MH(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, Prior &prior, std::unique_ptr<FitInfo> &fit_info, Model *model, std::vector<double>& accept_count, std::vector<double>& MH_vector, std::vector<double>& P_ratio, std::vector<double>& Q_ratio, std::vector<double>& prior_ratio)
{

    if (fit_info->parallel)
        thread_pool.start();

    // initialize predcitions
    for (size_t ii = 0; ii < fit_info->num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double) fit_info->num_trees);
    }

    // Set yhat_std to mean
    row_sum(fit_info->predictions_std, fit_info->yhat_std);
    // std::fill(fit_info->yhat_std.begin(), fit_info->yhat_std.end(), y_mean);

    // Residual for 0th tree
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];
    // std::fill(fit_info->residual_std.begin(), fit_info->residual_std.end(), y_mean / (double) num_trees * ((double) num_trees - 1.0));

    double sigma = 1.0;

    // std::vector<tree> temp_tree = trees[0];

    double MH_ratio = 0.0;

    double P_new;
    double P_old;
    double Q_new;
    double Q_old;
    double prior_new;
    double prior_old;

    std::uniform_real_distribution<> unif_dist(0, 1);

    tree temp_treetree = tree();

    std::vector<double> temp_vec_proposal(fit_info->n_y);
    std::vector<double> temp_vec(fit_info->n_y);
    std::vector<double> temp_vec2(fit_info->n_y);
    std::vector<double> temp_vec3(fit_info->n_y);
    std::vector<double> temp_vec4(fit_info->n_y);

    bool accept_flag = true;

    NodeData root_data;

    for (size_t sweeps = 0; sweeps < fit_info->num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < fit_info->num_trees; tree_ind++)
        {
            // Draw Sigma
            fit_info->residual_std_full = fit_info->residual_std - fit_info->predictions_std[tree_ind];
            std::gamma_distribution<double> gamma_samp((fit_info->n_y + prior.kap) / 2.0, 2.0 / (sum_squared(fit_info->residual_std_full) + prior.s));
            sigma = 1.0 / sqrt(gamma_samp(fit_info->gen));
            sigma_draw_xinfo[sweeps][tree_ind] = sigma;

            // add prediction of current tree back to residual
            // then it's m - 1 trees residual
            fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

            if (fit_info->use_all && (sweeps > burnin) && (fit_info->mtry != fit_info->p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (fit_info->sample_weights_flag)
            {
                mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
            }

            root_data.update_value(sigma, fit_info->n_y);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // X_counts and X_num_unique should not be in fit_info because they depend on node
            // but they are initialized in fit_info object
            // so I'll pass fit_info->X_counts to root node, then create X_counts_left, X_counts_right for other nodes
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            if (sweeps < 10)
            {

                // The first several sweeps are used as initialization
                // fit_info->data_pointers is calculated in this function
                // trees[sweeps][tree_ind].tonull();
                // cout << "aaa" << endl;
                
                // set sufficient statistics at root node first
                trees[sweeps][tree_ind].suff_stat[0] = sum_vec(fit_info->residual_std) / (double) fit_info->n_y;
                trees[sweeps][tree_ind].suff_stat[1] = sum_squared(fit_info->residual_std);


                trees[sweeps][tree_ind].grow_from_root(fit_info, max_depth_std[sweeps][tree_ind], Xorder_std, mtry_weight_current_tree, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, prior, root_data, true, false, true);
                accept_count.push_back(0);
                MH_vector.push_back(0);
                // cout << "bbb" << endl;
            }
            else
            {
                //     // fit a proposal

                /*

                    BE CAREFUL! Growing proposal update data_pointers in fit_info object implictly
                    need to creat a backup, copy from the backup if the proposal is rejected

                */


                // set sufficient statistics at root node first
                trees[sweeps][tree_ind].suff_stat[0] = sum_vec(fit_info->residual_std) / (double) fit_info->n_y;
                trees[sweeps][tree_ind].suff_stat[1] = sum_squared(fit_info->residual_std);

                trees[sweeps][tree_ind].grow_from_root(fit_info, max_depth_std[sweeps][tree_ind], Xorder_std, mtry_weight_current_tree, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, prior, root_data, true, false, true);

                predict_from_tree(trees[sweeps][tree_ind], fit_info->X_std, fit_info->n_y, fit_info->p, temp_vec_proposal, model);


                // evaluate old tree on new residual, thus need to update sufficient statistics on new data first 
                // update_theta = false and update_split_prob = true
                trees[sweeps - 1][tree_ind].suff_stat[0] = sum_vec(fit_info->residual_std) / (double) fit_info->n_y;
                trees[sweeps - 1][tree_ind].grow_from_root(fit_info, max_depth_std[sweeps][tree_ind], Xorder_std, mtry_weight_current_tree, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, prior, root_data, false, true, false);

                Q_old = trees[sweeps - 1][tree_ind].transition_prob();
                P_old = trees[sweeps - 1][tree_ind].tree_likelihood(fit_info->n_y, sigma, fit_info->residual_std);
                // P_old = trees[sweeps-1][tree_ind].tree_likelihood(N, sigma, tree_ind, model, fit_info, Xpointer, fit_info->residual_std, false);



                prior_old = trees[sweeps - 1][tree_ind].prior_prob(prior);

                // // proposal
                Q_new = trees[sweeps][tree_ind].transition_prob();
                P_new = trees[sweeps][tree_ind].tree_likelihood(fit_info->n_y, sigma, fit_info->residual_std);
                // P_new = trees[sweeps][tree_ind].tree_likelihood(N, sigma, tree_ind, model, fit_info, Xpointer, fit_info->residual_std, true);

                prior_new = trees[sweeps][tree_ind].prior_prob(prior);

                // cout << "tree size comparison " << trees[sweeps - 1][tree_ind].treesize() << "   " << trees[sweeps][tree_ind].treesize() << endl;

                MH_ratio = P_new + prior_new + Q_old - P_old - prior_old - Q_new;

                if (MH_ratio > 0)
                {
                    MH_ratio = 1;
                }
                else
                {
                    MH_ratio = exp(MH_ratio);
                }
                MH_vector.push_back(MH_ratio);

                Q_ratio.push_back(Q_old - Q_new);
                P_ratio.push_back(P_new - P_old);
                prior_ratio.push_back(prior_new - prior_old);

                // cout << "ratio is fine " << endl;

                if (unif_dist(fit_info->gen) <= MH_ratio)
                {
                    // accept
                    // do nothing
                    // cout << "accept " << endl;
                    accept_flag = true;
                    accept_count.push_back(1);
                }
                else
                {
                    // reject
                    // cout << "reject " << endl;
                    accept_flag = false;
                    accept_count.push_back(0);

                    // // // keep the old tree

                    // predict_from_tree(trees[sweeps - 1][tree_ind], Xpointer, N, p, temp_vec2, model);

                    trees[sweeps][tree_ind].copy_only_root(&trees[sweeps - 1][tree_ind]);

                    // predict_from_tree(trees[sweeps][tree_ind], Xpointer, N, p, temp_vec3, model);

                    // // update theta
                    /*
                    
                        update_theta() not only update leaf parameters, but also fit_info->data_pointers
                    
                    */

                    // update_theta = true, update_split_prob = true
                    // resample leaf parameters
                    trees[sweeps][tree_ind].grow_from_root(fit_info, max_depth_std[sweeps][tree_ind], Xorder_std, mtry_weight_current_tree, fit_info->X_counts, fit_info->X_num_unique, model, tree_ind, prior, root_data, true, true, false);

                    // predict_from_tree(trees[sweeps][tree_ind], Xpointer, N, p, temp_vec4, model);

                    // // keep the old tree, need to update fit_info object properly
                    // fit_info->data_pointers[tree_ind] = fit_info->data_pointers_copy[tree_ind];
                    fit_info->restore_data_pointers(tree_ind);

                }

                // cout << "copy is ok" << endl;
            }


            if(accept_flag){    
                // Add split counts
                mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;
                fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;
            }

            // Update Predict
            // I think this line can update corresponding column of predictions_std if the proposal is rejected. Not necessary to restore manually 
            // predict_from_datapointers(Xpointer, N, tree_ind, temp_vec, fit_info->data_pointers, model);
// cout << "before datapointers " << endl;
// cout << "tree size " << trees[sweeps][tree_ind].treesize() << endl;
            predict_from_datapointers(fit_info->X_std, fit_info->n_y, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers, model);
            // cout << "after datapointers " << endl;

            // predict_from_tree(trees[sweeps][tree_ind], Xpointer, N, p, fit_info->predictions_std[tree_ind], model);

            // if(!accept_flag){
            //     cout << "tree index " << tree_ind << endl;

            //     cout << "diff of proposal and vec2 " << sq_vec_diff(temp_vec_proposal, temp_vec2) << endl;

            //     cout << "diff of proposal and vec3 " << sq_vec_diff(temp_vec_proposal, temp_vec3) << endl;
                
            //     cout << "diff of vec2 and vec3 " << sq_vec_diff(temp_vec2, temp_vec3) << endl;

            //     cout << "diff of vec3 and vec4 " << sq_vec_diff(temp_vec2, temp_vec4) << endl;

            //     cout << "diff of vec and vec3 " << sq_vec_diff(temp_vec, temp_vec3) << endl;

            //     cout << "diff of vec and vec4 " << sq_vec_diff(temp_vec, temp_vec4) << endl;

            //     cout << "diff of prediction and vec4 " << sq_vec_diff(fit_info->predictions_std[tree_ind], temp_vec4) << endl;

            //     cout << "diff of prediction and vec " << sq_vec_diff(temp_vec, fit_info->predictions_std[tree_ind]) << endl;

            //     cout << "------------" << endl;
            // }
            

            // update residual
            model->updateResidual(fit_info->predictions_std, tree_ind, fit_info->num_trees, fit_info->residual_std);

            fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];



        }

        // after loop over all trees, backup the data_pointers matrix
        // data_pointers_copy save result of previous sweep
        fit_info->data_pointers_copy = fit_info->data_pointers;
        // fit_info->create_backup_data_pointers();
                
        double average = accumulate(accept_count.end() - fit_info->num_trees, accept_count.end(), 0.0) / fit_info->num_trees;
        double MH_average = accumulate(MH_vector.end() - fit_info->num_trees, MH_vector.end(), 0.0) / fit_info->num_trees;
        // cout << "size of MH " << accept_count.size() << "  " << MH_vector.size() << endl;

        COUT << "percentage of proposal acceptance " << average << endl;
        COUT << "average MH ratio " << MH_average << endl;

        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit_info->yhat_std;
    }
    thread_pool.stop();

    delete model;
}

