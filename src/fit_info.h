#ifndef GUARD_fit_info_h
#define GUARD_fit_info_h

#include <ctime>
#include "common.h"
#include "utility.h"
#include <chrono>

struct FitInfo
{
  public:
    // Categorical
    bool categorical_variables = false;
    std::vector<double> X_values;
    std::vector<size_t> X_counts;
    std::vector<size_t> variable_ind;
    size_t total_points;
    std::vector<size_t> X_num_unique;

    // Result containers
    xinfo predictions_std;
    std::vector<double> yhat_std;
    std::vector<double> residual_std;
    std::vector<double> residual_std_full;

    // Random
    std::vector<double> prob;
    std::random_device rd;
    std::mt19937 gen;
    std::discrete_distribution<> d;

    // Splits
    xinfo split_count_all_tree;
    std::vector<double> split_count_current_tree;
    std::vector<double> mtry_weight_current_tree;

    // mtry
    bool use_all = true;


    // fitinfo
    size_t n_min;
    size_t n_cutpoints;
    bool parallel;
    size_t p_categorical;
    size_t p_continuous;
    size_t p; // total number of variables = p_categorical + p_continuous
    size_t mtry;
    size_t n_y; // number of total data points in root node
    const double *X_std; // pointer to original data
    bool draw_mu;
    size_t num_trees;
    size_t num_sweeps;
    bool sample_weights_flag;

    // Vector pointers
    matrix<std::vector<double>*> data_pointers;
    matrix<std::vector<double>*> data_pointers_cp;
    matrix<std::vector<double>*> data_pointers_copy;
    
    void create_backup_data_pointers(){
        // create a backup copy of data_pointers
        // used in MH adjustment
        data_pointers_copy = data_pointers;
        return;
    }

    void restore_data_pointers(size_t tree_ind){
        // restore pointers of one tree from data_pointers_copy
        // used in MH adjustment
        data_pointers[tree_ind] = data_pointers_copy[tree_ind];
        return;
    }

    void init_tree_pointers(std::vector<double>* initial_theta, size_t N, size_t num_trees)
    {
        ini_matrix(data_pointers, N, num_trees);
        for (size_t i = 0; i < num_trees; i++)
        {
            std::vector<std::vector<double>*> &pointer_vec = data_pointers[i];
            for (size_t j = 0; j < N; j++)
            {
                pointer_vec[j] = initial_theta;
            }
        }
    }

    FitInfo(const double *Xpointer, xinfo_sizet &Xorder_std, size_t N, size_t p, size_t num_trees, size_t p_categorical, size_t p_continuous, bool set_random_seed, size_t random_seed, std::vector<double>* initial_theta, size_t n_min, size_t n_cutpoints, bool parallel, size_t mtry, const double *X_std, bool draw_mu, size_t num_sweeps, bool sample_weights_flag)
    {

        // Handle Categorical
        if (p_categorical > 0)
        {
            this->categorical_variables = true;
        }
        this->variable_ind = std::vector<size_t>(p_categorical + 1); // cummulated number of unique values in categorical variables
        this->X_num_unique = std::vector<size_t>(p_categorical);
        unique_value_count2(Xpointer, Xorder_std, this->X_values, this->X_counts,
                            this->variable_ind, this->total_points, this->X_num_unique, p_categorical, p_continuous);

        // // Init containers
        ini_xinfo(this->predictions_std, N, num_trees);

        yhat_std = std::vector<double>(N);
        row_sum(this->predictions_std, this->yhat_std);
        this->residual_std = std::vector<double>(N);

        // Random
        this->prob = std::vector<double>(2, 0.5);
        this->gen = std::mt19937(rd());
        if (set_random_seed)
        {
            gen.seed(random_seed);
        }
        this->d = std::discrete_distribution<>(prob.begin(), prob.end());

        // Splits
        ini_xinfo(this->split_count_all_tree, p, num_trees);
        this->split_count_current_tree = std::vector<double>(p, 0);
        this->mtry_weight_current_tree = std::vector<double>(p, 0.1);

        init_tree_pointers(initial_theta, N, num_trees);

        this->n_min = n_min;
        this->n_cutpoints = n_cutpoints;
        this->parallel = parallel;
        this->p_categorical = p_categorical;
        this->p_continuous = p_continuous;
        this->mtry = mtry;
        this->X_std = X_std;
        this->p = p_categorical + p_continuous;
        this->n_y = N;
        this->draw_mu = draw_mu;
        this->num_trees = num_trees;
        this->num_sweeps = num_sweeps;
        this->sample_weights_flag = sample_weights_flag;
    }
};

#endif