#include "tree.h"
#include "model.h"
#include <cfenv>
#include <functional>

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Normal Model
//
//
//////////////////////////////////////////////////////////////////////////////////////

void NormalModel::incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats)
{
    // I have to pass matrix<double> &residual_std, size_t index_next_obs
    // which allows more flexibility for multidimensional residual_std

    suffstats[0] += residual_std[0][index_next_obs];
    return;
}

void NormalModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    // test result should be theta
    theta_vector[0] = suff_stat[0] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)) + sqrt(1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2))) * normal_samp(state->gen); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));

    // also update probability of leaf parameters
    // prob_leaf = normal_density(theta_vector[0], suff_stat[0] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), 1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), true);

    return;
}

void NormalModel::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
{
    // Draw Sigma
    // state->residual_std_full = state->residual_std - state->predictions_std[tree_ind];

    // residual_std is only 1 dimensional for regression model

    std::vector<double> full_residual(state->n_y);

    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        full_residual[i] = state->residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0];
    }

    std::gamma_distribution<double> gamma_samp((state->n_y + kap) / 2.0, 2.0 / (sum_squared(full_residual) + s));
    state->update_sigma(1.0 / sqrt(gamma_samp(state->gen)));
    return;
}

void NormalModel::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{
    // sum of y
    suff_stat[0] = sum_vec(state->residual_std[0]);
    // sum of y squared
    suff_stat[1] = sum_squared(state->residual_std[0]);
    // number of observations in the node
    suff_stat[2] = state->n_y;
    return;
}

void NormalModel::updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    suff_stat[0] += residual_std[0][Xorder_std[split_var][row_ind]];
    suff_stat[1] += pow(residual_std[0][Xorder_std[split_var][row_ind]], 2);
    suff_stat[2] += 1;
    return;
}

void NormalModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
{

    // in function split_xorder_std_categorical, for efficiency, the function only calculates suff stat of ONE child
    // this function calculate the other side based on parent and the other child

    if (compute_left_side)
    {
        rchild_suff_stat = parent_suff_stat - lchild_suff_stat;
    }
    else
    {
        lchild_suff_stat = parent_suff_stat - rchild_suff_stat;
    }
    return;
}

void NormalModel::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    ////////////////////////////////////////////////////////
    // Be care of line 151 in train_all.cpp, initial_theta
    ////////////////////////////////////////////////////////

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        residual_std[0][i] = residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0] + (*(x_struct->data_pointers[next_index][i]))[0];
    }
    return;
}

double NormalModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    // double y_sum = (double)suff_stat_all[2] * suff_stat_all[0];
    // double y_sum = suff_stat_all[0];
    double sigma2 = state->sigma2;
    double ntau;
    // double suff_one_side;

    /////////////////////////////////////////////////////////////////////////
    //
    //  I know combining likelihood and likelihood_no_split looks nicer
    //  but this is a very fundamental function, executed many times
    //  the extra if(no_split) statement and value assignment make the code about 5% slower!!
    //
    /////////////////////////////////////////////////////////////////////////

    size_t nb;
    double nbtau;
    double y_sum;
    double y_squared_sum;

    if (no_split)
    {
        // ntau = suff_stat_all[2] * tau;
        // suff_one_side = y_sum;

        nb = suff_stat_all[2];
        nbtau = nb * tau;
        y_sum = suff_stat_all[0];
        y_squared_sum = suff_stat_all[1];
    }
    else
    {
        if (left_side)
        {
            nb = N_left + 1;
            nbtau = nb * tau;
            // ntau = (N_left + 1) * tau;
            y_sum = temp_suff_stat[0];
            y_squared_sum = temp_suff_stat[1];
            // suff_one_side = temp_suff_stat[0];
        }
        else
        {
            nb = suff_stat_all[2] - N_left - 1;
            nbtau = nb * tau;
            y_sum = suff_stat_all[0] - temp_suff_stat[0];
            y_squared_sum = suff_stat_all[1] - temp_suff_stat[1];

            // ntau = (suff_stat_all[2] - N_left - 1) * tau;
            // suff_one_side = y_sum - temp_suff_stat[0];
        }
    }

    // return 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));

    return -0.5 * nb * log(2 * 3.141592653) - 0.5 * nb * log(sigma2) + 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) - 0.5 * y_squared_sum / sigma2 + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));
}

// double NormalModel::likelihood_no_split(std::vector<double> &suff_stat, std::unique_ptr<State> &state) const
// {
//     // the likelihood of no-split option is a bit different from others
//     // because the sufficient statistics is y_sum here
//     // write a separate function, more flexibility
//     double ntau = suff_stat[2] * tau;
//     // double sigma2 = pow(state->sigma, 2);
//     double sigma2 = state->sigma2;
//     double value = suff_stat[2] * suff_stat[0]; // sum of y

//     return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2));
// }

void NormalModel::ini_residual_std(std::unique_ptr<State> &state)
{
    // initialize partial residual at (num_tree - 1) / num_tree * yhat
    double value = state->ini_var_yhat * ((double)state->num_trees - 1.0) / (double)state->num_trees;
    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        state->residual_std[0][i] = (*state->y_std)[i] - value;
    }
    return;
}

void NormalModel::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees)
{

    matrix<double> output;

    // row : dimension of theta, column : number of trees
    ini_matrix(output, this->dim_theta, trees[0].size());

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {
            getThetaForObs_Outsample(output, trees[sweeps], data_ind, Xtestpointer, N_test, p);

            // take sum of predictions of each tree, as final prediction
            for (size_t i = 0; i < trees[0].size(); i++)
            {
                yhats_test_xinfo[sweeps][data_ind] += output[i][0];
            }
        }
    }
    return;
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Logit Model
//
//
//
//////////////////////////////////////////////////////////////////////////////////////

//incSuffStat should take a state as its first argument
void LogitModel::incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats)
{
    // I have to pass matrix<double> &residual_std, size_t index_next_obs
    // which allows more flexibility for multidimensional residual_std

    // suffstats[0] += residual_std[0][index_next_obs];

    // sufficient statistics have 2 * num_classes

    suffstats[(*y_size_t)[index_next_obs]] += 1; //weight;


    for (size_t j = 0; j < dim_theta; ++j)
    {
        // count number of observations, y_{ij}
        // if ((*y_size_t)[index_next_obs] == j)
            // suffstats[j] += 1;

        // psi * f
        suffstats[dim_theta + j] += (*phi)[index_next_obs] * residual_std[j][index_next_obs]; // residual already get powered by weight
        // if (isnan(suffstats[dim_theta + j])) {cout << "phi = " << (*phi)[index_next_obs] << "; resid = " << residual_std[j][index_next_obs] << "; j = " << j << endl; }
    }

    return;
}

void LogitModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{

    //redefine these to use prior pars from Model class
    // int c = dim_theta; //suffstats.size() / 2;

    // double r;
    // double s;
    double temp;

    for (size_t j = 0; j < dim_theta; j++)
    {
        // std::gamma_distribution<double> gammadist(tau_a + suff_stat[j], 1.0);
        std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] +1 / weight[j], 1.0);

        theta_vector[j] =  gammadist(state->gen) / (tau_b + suff_stat[dim_theta + j]); // store theta_vector as lambda^weight
        // theta_vector[j] = pow( gammadist(state->gen) / (tau_b + suff_stat[dim_theta + j]), 1 / weight[j]) ;
    }
    // cout << "weight " << weight  << " theta_vector" << theta_vector << endl;

    return;
}

void LogitModel::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
{

    std::vector<double> loglike_weight(weight_candidate.size(), 0.0);
   
    for (size_t j = 0; j < weight_candidate.size(); j++)
    {
        #pragma omp task shared(loglike_weight, j, state, x_struct, tree_ind, weight_candidate, dim_residual)
        {
            for (size_t i = 0; i < state->residual_std[0].size(); i++)
            {
                loglike_weight[j] += loglike_x(state, x_struct, tree_ind, i, weight_candidate[j]);
            }

        }
    }
    #pragma omp taskwait

    // Draw weight for each class
    for (size_t l = 0; l < dim_theta; l++)
    {
        #pragma omp task shared(loglike_weight, state, x_struct, tree_ind, weight_candidate)
        {
            double max_loglike_weight = -INFINITY;
            std::vector<double> log_lambda_prior(weight_candidate.size(), 0.0);
            std::vector<double> weight_likelihood(weight_candidate.size(), 0.0);

            for (size_t i = 0; i < state->residual_std[0].size(); i++)
            {
                for (size_t j = 0; j < weight_candidate.size(); j++)
                {
                    for (size_t k = 0; k < state->num_trees; k++)
                    {
                        log_lambda_prior[j] += log_dlambda((*(x_struct->data_pointers[k][i]))[l], weight_candidate[j], l);
                    }
                }

            }

            for (size_t i = 0; i < weight_candidate.size(); i++)
            {
                // loglike_weight[i] = weight_candidate[i] * loglike_pi + lgamma(weight_candidate[i] * n + 1) - lgamma(n + 1) - lgamma((weight_candidate[i] - 1) * n + 1);
                // loglike_weight[i] = weight_candidate[i] * loglike_pi - loglike_weight[i];
                if (loglike_weight[i] + log_lambda_prior[i] > max_loglike_weight){max_loglike_weight = loglike_weight[i] + log_lambda_prior[i];}
            }

            for (size_t i = 0; i < weight_candidate.size(); i++)
            {
                weight_likelihood[i] = exp(loglike_weight[i]  + log_lambda_prior[i] - max_loglike_weight);
            }
            // cout << "loglike_weight " << loglike_weight <<endl;
            
            std::discrete_distribution<> d(weight_likelihood.begin(), weight_likelihood.end());
            weight[l] = weight_candidate[d(state->gen)];
            // cout << "weight " << weight << endl;
        }
    }
    #pragma omp taskwait


    // Draw phi
    std::gamma_distribution<double> gammadist(1.0, 1.0);
    std::vector<double> fits_w(dim_theta);
    
    for (size_t i = 0; i < state->residual_std[0].size(); i++){
        for (size_t j = 0; j < dim_theta; ++j)
        {
            fits_w[j] = state->residual_std[j][i] * (*(x_struct->data_pointers[tree_ind][i]))[j]; // residuals already powered by weight
            // if (fits_w[j] == 0) {cout << "resid = "<< state->residual_std[j][i] << "; data_pointer = "  << (*(x_struct->data_pointers[tree_ind][i]))[j] << "; weight = " << weight << endl;}
        }
        // (*phi)[i] = gammadist(state->gen) / (1.0*sum_fits_v[i]); 
        (*phi)[i] = gammadist(state->gen) / (1.0 * accumulate(fits_w.begin(), fits_w.end(), 0.0) );
        if (isinf((*phi)[i])) {
            cout << "current weight " << weight << endl;
            terminate();
        }
    }


    return;
}

void LogitModel::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{

    /*
    // sum of y
    suff_stat[0] = sum_vec(state->residual_std[0]);
    // sum of y squared
    suff_stat[1] = sum_squared(state->residual_std[0]);
    // number of observations in the node
    suff_stat[2] = state->n_y;
    */

    // JINGYU check -- should i always plan to resize this vector?
    // reply: use it for now. Not sure how to call constructor of tree when initialize vector<vector<tree>>, see definition of trees2 in XBART_multinomial, train_all.cpp

    // remove resizing it does not work, strange

    suff_stat.resize(2 * dim_theta);
    std::fill(suff_stat.begin(), suff_stat.end(), 0.0);
    for (size_t i = 0; i < state->n_y; i++)
    {
        // from 0
        incSuffStat(state->residual_std, i, suff_stat);
    }

    return;
}

void LogitModel::updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    /*
    suff_stat[0] += residual_std[0][Xorder_std[split_var][row_ind]];
    suff_stat[1] += pow(residual_std[0][Xorder_std[split_var][row_ind]], 2);
    suff_stat[2] += 1;
    */

    incSuffStat(residual_std, Xorder_std[split_var][row_ind], suff_stat);

    return;
}

void LogitModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
{

    // in function split_xorder_std_categorical, for efficiency, the function only calculates suff stat of ONE child
    // this function calculate the other side based on parent and the other child

    if (compute_left_side)
    {
        rchild_suff_stat = parent_suff_stat - lchild_suff_stat;
        // this is a wired bug. Should look into it in the future.
        for (size_t j = 0; j < rchild_suff_stat.size() ; j++)
        {
            rchild_suff_stat[j] = rchild_suff_stat[j] > 0 ? rchild_suff_stat[j] : 0;
            // if (isnan(rchild_suff_stat[j])) {cout <<"j = " << j <<  "; parent_suff_stat = " << parent_suff_stat[j] << "; lchild_suff_stat = " << lchild_suff_stat[j]<< "; rchild_suff_stat = " << rchild_suff_stat[j]  << endl;}
            
        }
    }
    else
    {
        lchild_suff_stat = parent_suff_stat - rchild_suff_stat;
        for (size_t j = 0; j < rchild_suff_stat.size() ; j++)
        {
            lchild_suff_stat[j] = lchild_suff_stat[j] > 0 ? lchild_suff_stat[j] : 0;
            // if (isnan(lchild_suff_stat[j])) { cout <<"j = " << j <<  "; parent_suff_stat = " << parent_suff_stat[j] << "; rchild_suff_stat = " << rchild_suff_stat[j]<< "; lchild_suff_stat = " << lchild_suff_stat[j]  << endl;}
            
        }
    }
    return;
}

void LogitModel::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{

    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    ////////////////////////////////////////////////////////
    // Be care of line 151 in train_all.cpp, initial_theta
    ////////////////////////////////////////////////////////

    // cumulative product of trees, multiply current one, divide by next one

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        for (size_t j = 0; j < dim_theta; ++j)
        {
            residual_std[j][i] = residual_std[j][i] * (*(x_struct->data_pointers[tree_ind][i]))[j] / (*(x_struct->data_pointers[next_index][i]))[j];
        }
    }

    return;
}

double LogitModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    // double y_sum = (double)suff_stat_all[2] * suff_stat_all[0];
    // double y_sum = suff_stat_all[0];
    // double suff_one_side;

    /////////////////////////////////////////////////////////////////////////
    //
    //  I know combining likelihood and likelihood_no_split looks nicer
    //  but this is a very fundamental function, executed many times
    //  the extra if(no_split) statement and value assignment make the code about 5% slower!!
    //
    /////////////////////////////////////////////////////////////////////////

    //could rewrite without all these local assigments if that helps...
    std::vector<double> local_suff_stat = suff_stat_all; // no split

    //COUT << "LIK" << endl;

    //COUT << "all suff stat dim " << suff_stat_all.size();

    if (!no_split)
    {
        if (left_side)
        {
            //COUT << "LEFTWARD HO" << endl;
            //COUT << "local suff stat dim " << local_suff_stat.size() << endl;
            //COUT << "temp suff stat dim " << temp_suff_stat.size() << endl;
            local_suff_stat = temp_suff_stat;
        }
        else
        {
            //COUT << "RIGHT HO" << endl;
            //COUT << "local suff stat dim " << local_suff_stat.size() << endl;
            //COUT << "temp suff stat dim " << temp_suff_stat.size() << endl;
            local_suff_stat = suff_stat_all - temp_suff_stat;

            // ntau = (suff_stat_all[2] - N_left - 1) * tau;
            // suff_one_side = y_sum - temp_suff_stat[0];
        }
    }

    // return 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));

    //return - 0.5 * nb * log(2 * 3.141592653) -  0.5 * nb * log(sigma2) + 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) - 0.5 * y_squared_sum / sigma2 + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));

    return (LogitLIL(local_suff_stat));
}

void LogitModel::ini_residual_std(std::unique_ptr<State> &state)
{
    //double value = state->ini_var_yhat * ((double)state->num_trees - 1.0) / (double)state->num_trees;
    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        // init leaf pars are all 1, partial fits are all 1
        for (size_t j = 0; j < dim_theta; ++j)
        {
            state->residual_std[j][i] = 1.0; // (*state->y_std)[i] - value;
        }
    }
    return;
}

void LogitModel::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees, std::vector<double> &output_vec)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    tree::tree_p bn;

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            for (size_t i = 0; i < trees[0].size(); i++)
            {
                // search leaf
                bn = trees[sweeps][i].search_bottom_std(Xtestpointer, data_ind, p, N_test);

                for (size_t k = 0; k < dim_residual; k++)
                {
                    // add all trees

                    // product of trees, thus sum of logs

                    output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] += log(bn->theta_vector[k]);
                }
            }
        }
    }

    // normalizing probability

    double denom = 0.0;
    double max_log_prob = -INFINITY;

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            max_log_prob = -INFINITY;
            // take exp, subtract max to avoid overflow

            // this line does not work for some reason, havd to write loops manually
            // output.tube(sweeps, data_ind) = exp(output.tube(sweeps, data_ind) - output.tube(sweeps, data_ind).max());

            // find max of probability for all classes
            for (size_t k = 0; k < dim_residual; k++)
            {
                if(output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] > max_log_prob){
                    max_log_prob = output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
                }
            }

            // take exp after subtracting max to avoid overflow
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] = exp(output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] - max_log_prob);
            }

            // calculate normalizing constant
            denom = 0.0;
            for (size_t k = 0; k < dim_residual; k++)
            {
                // denom += output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
                denom += output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
            }

            // normalizing
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] = output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] / denom;
            }
        }
    }
    return;
}

// this function is for a standalone prediction function for classification case.
// with extra input iteration, which specifies which iteration (sweep / forest) to use
void LogitModel::predict_std_standalone(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees, std::vector<double> &output_vec, std::vector<size_t>& iteration)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    size_t num_iterations = iteration.size();

    tree::tree_p bn;

    cout << "number of iterations " << num_iterations << " " << num_sweeps << endl;

    size_t sweeps;

    for (size_t iter = 0; iter < num_iterations; iter++)
    {
        sweeps = iteration[iter];

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            for (size_t i = 0; i < trees[0].size(); i++)
            {
                // search leaf
                bn = trees[sweeps][i].search_bottom_std(Xtestpointer, data_ind, p, N_test);

                for (size_t k = 0; k < dim_residual; k++)
                {
                    // add all trees

                    // product of trees, thus sum of logs

                    output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] += log(bn->theta_vector[k]);
                }
            }
        }
    }
    // normalizing probability

    double denom = 0.0;
    double max_log_prob = -INFINITY;

    for (size_t iter = 0; iter < num_iterations; iter++)
    {

        sweeps = iteration[iter];

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            max_log_prob = -INFINITY;
            // take exp, subtract max to avoid overflow

            // this line does not work for some reason, havd to write loops manually
            // output.tube(sweeps, data_ind) = exp(output.tube(sweeps, data_ind) - output.tube(sweeps, data_ind).max());

            // find max of probability for all classes
            for (size_t k = 0; k < dim_residual; k++)
            {
                if(output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] > max_log_prob){
                    max_log_prob = output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test];
                }
            }

            // take exp after subtracting max to avoid overflow
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] = exp(output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] - max_log_prob);
            }

            // calculate normalizing constant
            denom = 0.0;
            for (size_t k = 0; k < dim_residual; k++)
            {
                // denom += output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test];
                denom +=  output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] ;
            }

            // normalizing
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] = output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] / denom;
            }
        }
    }
    return;
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Probit Model
//
//
//////////////////////////////////////////////////////////////////////////////////////
void ProbitClass::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
{
    // Draw Sigma
    // state->residual_std_full = state->residual_std - state->predictions_std[tree_ind];

    // residual_std is only 1 dimensional for regression model

    // std::vector<double> full_residual(state->n_y);

    // for (size_t i = 0; i < state->residual_std[0].size(); i++)
    // {
    //     full_residual[i] = state->residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0];
    // }

    // For probit model, do not need to sample gamma
    // std::gamma_distribution<double> gamma_samp((state->n_y + kap) / 2.0, 2.0 / (sum_squared(full_residual) + s));
    // state->update_sigma(1.0 / sqrt(gamma_samp(state->gen)));

    //update latent variable Z

    z_prev = z;

    double mu_temp;
    double u;

    for (size_t i = 0; i < state->n_y; i++)
    {
        a = 0;
        b = 1;

        mu_temp = normCDF(z_prev[i]);

        // Draw from truncated normal via inverse CDF methods
        if ((*state->y_std)[i] > 0)
        {
            a = std::min(mu_temp, 0.999);
        }
        else
        {
            b = std::max(mu_temp, 0.001);
        }

        std::uniform_real_distribution<double> unif(a, b);
        u = unif(state->gen);
        z[i] = normCDFInv(u) + mu_temp;
    }
    return;

    //NormalModel::update_state(state, tree_ind, x_struct);
}

void ProbitClass::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{

    NormalModel::state_sweep(tree_ind, M, residual_std, x_struct);
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    ////////////////////////////////////////////////////////
    // Be care of line 151 in train_all.cpp, initial_theta
    ////////////////////////////////////////////////////////

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        residual_std[0][i] = residual_std[0][i] - z_prev[i] + z[i];
    }

    return;
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  CLT Model
//
//
//////////////////////////////////////////////////////////////////////////////////////
