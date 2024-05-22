/*
 


 */

/* 
 * File:   DiversifiedFCFitness.cpp
 * Author: Sijben
 *
 * Created on Aug 4 2022
 */


#include "GPGOMEA/Fitness/DiversifiedFCFitness.h"
void DiversifiedFCFitness::SetFitnessCases(const arma::mat& X, FitnessCasesType fct) {

    arma::vec Y = X.col(X.n_cols - 1);
    arma::mat Xx = X;
    Xx.shed_col(X.n_cols-1);
    Xx.shed_col(0);

    arma::vec ID = X.col(0);
    std::vector<size_t> group_len;
    size_t current_id = ID[0];
    size_t count = 1;
    for (size_t i = 1; i<ID.size(); i++){
        if (ID[i] != current_id ){
            group_len.push_back(count);
            count = 1;
            current_id = ID[i];
            if (i == ID.size()-1 and count == 1){
                group_len.push_back(1);
            }
        }
        else if (i == ID.size()-1){
            group_len.push_back(count + 1);
        }else{
            count ++;
        }
    }

    if (fct == FitnessCasesTRAIN) {
        TrainY = Y;
        TrainX = Xx;
        group_len_train = group_len;
        trainY_mean = arma::mean(TrainY);
        trainY_std = arma::stddev(TrainY, 1);

        var_comp_trainY = Y - trainY_mean;
    } else if (fct == FitnessCasesTEST) {
        group_len_test = group_len;
        TestY = Y;
        TestX = Xx;
    } else if (fct == FitnessCasesVALIDATION) {
        group_len_val = group_len;
        ValidationY = Y;
        ValidationX = Xx;
    } else {
        throw std::runtime_error("Fitness::SetFitnessCases invalid fitness cases type provided.");
    }
}

double_t DiversifiedFCFitness ::ComputeFitness(Node* n, bool use_caching) {

    size_t trees = ((Multitree *) n)->nodes.size();
    arma::mat error_total_per_set = arma::mat(((Multitree *) n)->nodes[0]->cached_train_error_per_set.size(),trees,arma::fill::none);
    for (size_t i =0 ; i < trees; i++) {
        error_total_per_set.col(i) = ((Multitree *) n)->nodes[i]->cached_train_error_per_set;
    }
    double_t fit = arma::mean(((arma::min(error_total_per_set, 1))));
    if (std::isnan(fit)) {
        fit = arma::datum::inf;
    }
    n->cached_fitness = fit;
    return fit;

}

double_t DiversifiedFCFitness::GetValidationFit(Node* n) {
    size_t trees = ((Multitree *) n)->nodes.size();
    arma::mat error_total_per_set = arma::mat(((Multitree *) n)->nodes[0]->cached_val_error_per_set.size(),trees,arma::fill::none);
    for (size_t i =0 ; i < trees; i++) {
        error_total_per_set.col(i)  = ((Multitree *) n)->nodes[i]->cached_val_error_per_set;
    }
    double_t fit = arma::mean(((arma::min(error_total_per_set, 1))));
    if (std::isnan(fit)) {
        fit = arma::datum::inf;
    }

    return fit;

}


double_t DiversifiedFCFitness::GetTestFit(Node* n) {
    size_t trees = ((Multitree *) n)->nodes.size();
    arma::mat error_total_per_set = arma::mat(((Multitree *) n)->nodes[0]->cached_test_error_per_set.size(),trees,arma::fill::none);
    for (size_t i =0 ; i < trees; i++) {
        error_total_per_set.col(i)  =((Multitree *) n)->nodes[i]->cached_test_error_per_set ;
    }
    double_t fit = arma::mean(((arma::min(error_total_per_set, 1))));
    if (std::isnan(fit)) {
        fit = arma::datum::inf;
    }

    return fit;

}

double_t DiversifiedFCFitness::ComputeMSE(const arma::vec& P, const arma::vec& Y) {
    arma::vec res = Y - P;
    double_t mse = arma::mean( arma::square(res) );
    return mse;
}



