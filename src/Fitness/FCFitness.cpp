/*
 


 */

/* 
 * File:   FCFitness.cpp
 * Author: Sijben
 *
 * Created on march 28 2022
 */


#include "GPGOMEA/Fitness/FCFitness.h"


void FCFitness::SetFitnessCases(const arma::mat& X, FitnessCasesType fct) {

    arma::vec Y = X.col(X.n_cols - 1);
    arma::mat Xx = X;
    Xx.shed_col(X.n_cols-1);
    Xx.shed_col(0);

    std::vector<size_t> ID;
    for (double id:X.col(0) ){
        ID.push_back((size_t)id);
    }
    std::vector<size_t> group_len;
    size_t non_zero_count = 0;
    size_t current_id = ID[0];
    size_t count = 1;
    for (size_t i = 1; i<ID.size(); i++){
        //add a check if subsuquent ID
        if (ID[i] != current_id ){
            group_len.push_back(count);
            non_zero_count = non_zero_count +1;
            count = 1;
            while ((size_t)ID[i] != (current_id +1)){
                group_len.push_back(0);
                current_id ++;
            }
            if (i == ID.size()-1 and count == 1){
                group_len.push_back(1);
                non_zero_count = non_zero_count +1;
            }
            current_id = ID[i];
        }
        else if (i == ID.size()-1){
            group_len.push_back(count + 1);
            non_zero_count = non_zero_count +1;
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
        test_size = non_zero_count;
    } else if (fct == FitnessCasesVALIDATION) {
        group_len_val = group_len;
        ValidationY = Y;
        ValidationX = Xx;
    } else {
        throw std::runtime_error("Fitness::SetFitnessCases invalid fitness cases type provided.");
    }

}
void FCFitness::SetBatchingIndex(size_t number_of_individuals) {
    size_t total_len = group_len_train.size() ;
    samples.clear();
    arma::uvec idxs = arma::randperm(total_len);
    size_t count = 0;
    batching = true;
    for (size_t ind : idxs){
        samples.push_back(ind);
        count++;
        if (count == number_of_individuals){
            break;
        }
    }

}

double_t FCFitness ::ComputeFitness(Node* n, bool use_caching) {
    evaluations++;
    n->constraints_violated = 0;
    double_t fit;
    auto &hash_table = GetHashTable();
    //check whether we need to deal with a multi-tree
    if (n->type == NodeType::Multi) {
        fit = 0;
        arma::vec output;
        //for each tree in the multi-tree...
        for (SingleNode *sn: ((Multitree *) n)->nodes) {
            sn->constraints_violated = 0;
            sn->MakeShadowTree(arma::max(arma::max(arma::abs(this->TrainX))));
            //if the tree is already in the hashtable, get the cached values
            if (hash_table.SolutionInTable(sn) ){
                CachedFCFitness cash = hash_table.GetSolution(sn);
                output = cash.first;
                sn->constraints_violated = cash.second;
            }else {
                output = GetErrorVector(sn, use_caching,FitnessCasesTRAIN,false);
                //check if we can add the calculated output to the table
                hash_table.AddSolutionToTable(sn,CreateCachedFitness(output, sn->constraints_violated));
            }
            //add the mean over the group members to the fitness
            fit += arma::mean(output);
            //cach the error per group member for calculating the diversified fitness
            sn->cached_train_error_per_set.reset();
            sn->cached_train_error_per_set = output;
            //add the violated constraints to the multi-tree
            n->constraints_violated += sn->constraints_violated;
        }
    }
    else{
        arma::vec output = arma::vec(group_len_train.size(), arma::fill::none);
        ((SingleNode * )n)->MakeShadowTree(arma::max(arma::max(arma::abs(this->TrainX))));
        if ( hash_table.SolutionInTable(n)){
            CachedFCFitness cash = hash_table.GetSolution(n);
            output = cash.first;
            n->constraints_violated = cash.second;
        }else {
            output = GetErrorVector(n,  use_caching,FitnessCasesTRAIN,false);
            hash_table.AddSolutionToTable(n, CreateCachedFitness(output, n->constraints_violated));
        }
        fit = arma::mean(output);
    }
    if (std::isnan(fit)) {
        fit = arma::datum::inf;
    }
    n->cached_fitness = fit;

    return fit;

}

double_t FCFitness::GetValidationFit(Node* n) {
    double_t fit;

    if (n->type == NodeType::Multi) {
        fit = 0;
        arma::vec output;
        for (SingleNode *sn: ((Multitree *) n)->nodes) {
            (sn)->MakeShadowTree(arma::max(arma::max(arma::abs(this->TrainX))));
            output = GetErrorVector(sn,  false,FitnessCasesVALIDATION,false);
            //add the mean over the groupmembers to the fitness
            fit += arma::mean(output);
            //cach the error per group member for calculating the differsified fitness
            sn->cached_val_error_per_set.reset();
            sn->cached_val_error_per_set = output;
        }
    }
    else{
        ((SingleNode *)n)->MakeShadowTree(arma::max(arma::max(arma::abs(this->TrainX))));

        fit = arma::mean(GetErrorVector(n,false,FitnessCasesVALIDATION,false));
    }
    if (std::isnan(fit)) {
        fit = arma::datum::inf;
    }
    return fit;
}

arma::vec FCFitness::GetPopulationTestFitnessFC(const std::vector<Node*>& population, const std::string& path) {

    arma::vec fitnesses(population.size());
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < population.size(); i++) {
        fitnesses[i] = GetTestFitWithTree(population[i],path, i) ;
    }
    return fitnesses;
}

double_t FCFitness::GetTestFitWithTree(Node* n, const std::string& path, size_t tree_index) {
    double_t fit;
    if (n->type == NodeType::Multi) {
        fit = 0;
        arma::vec output;
        for (size_t i =0; i< ((Multitree *) n)->nodes.size(); i++){
            (((Multitree *) n)->nodes[i])->MakeShadowTree(arma::max(arma::max(arma::abs(this->TrainX))));
            output = GetErrorVector(((Multitree *) n)->nodes[i], true,FitnessCasesTEST,true, path,tree_index, i);
            //add the mean over the groupmembers to the fitness
            fit += arma::mean(output);
            //cach the error per group member for calculating the differsified fitness
            ((Multitree *) n)->nodes[i]->cached_test_error_per_set.reset();
            ((Multitree *) n)->nodes[i]->cached_test_error_per_set = output;
        }
    }
    else{
        ((SingleNode *)n)->MakeShadowTree(arma::max(arma::max(arma::abs(this->TrainX))));
        arma::vec output = (GetErrorVector(n,  false,FitnessCasesTEST,true, path, tree_index,1));
        ((SingleNode *)n)->cached_test_error_per_set.reset();
        ((SingleNode *)n)->cached_test_error_per_set = output;
        fit = arma::mean(output);
    }

    if (std::isnan(fit)) {
        fit = arma::datum::inf;
    }
    return fit;

}



double_t FCFitness::GetTestFit(Node* n) {
    double_t fit;
    if (n->type == NodeType::Multi) {
        fit = 0;
        for (size_t i =0; i< ((Multitree *) n)->nodes.size(); i++){
            if (((Multitree *) n)->nodes[i]->cached_test_error_per_set.is_empty()){
                (((Multitree *) n)->nodes[i])->MakeShadowTree(arma::max(arma::max(arma::abs(this->TrainX))));
                arma::vec output = GetErrorVector(((Multitree *) n)->nodes[i], true,FitnessCasesTEST,false);
                //add the mean over the groupmembers to the fitness
                fit += arma::mean(output);
                //cach the error per group member for calculating the differsified fitness
                ((Multitree *) n)->nodes[i]->cached_test_error_per_set.reset();
                ((Multitree *) n)->nodes[i]->cached_test_error_per_set = output;
            }else{
                arma::vec output = (((Multitree *) n)->nodes[i]->cached_test_error_per_set);
                fit += arma::mean(output);
            }
        }
    }
    else{
        if (((SingleNode *) n)->cached_test_error_per_set.is_empty()) {
            ((SingleNode *) n)->MakeShadowTree(arma::max(arma::max(arma::abs(this->TrainX))));
            fit = arma::mean(GetErrorVector(n, false, FitnessCasesTEST, false));
        }else{
            arma::vec output = ((SingleNode *) n)->cached_test_error_per_set;
            fit = arma::mean(output);
        }
    }
    if (std::isnan(fit)) {
        fit = arma::datum::inf;
    }
    return fit;

}


arma::vec FCFitness::GetErrorVector(Node * n,  bool use_caching,FitnessCasesType f, bool get_solutions, const std::string& path, size_t tree_index, size_t tree){
    arma::Mat<double> *x;
    arma::Mat<double> *y;
    arma::Mat<double> *x_t;
    arma::Mat<double> *y_t;
    std::vector<size_t> *group_len_vec;
    size_t nr_of_sets;
    std::vector<size_t> *group_len_vec_t;
    bool get_constraints = false;
    switch (f) {
        case FitnessCasesTRAIN:
            x = &TrainX;
            y = &TrainY;
            x_t = &ValidationX;
            y_t = &ValidationY;
            group_len_vec = &group_len_train;
            nr_of_sets = group_len_train.size();
            group_len_vec_t = &group_len_val;
            get_constraints = calc_constraints;
            break;
        case FitnessCasesVALIDATION:
            x = &TrainX;
            y = &TrainY;
            x_t = &ValidationX;
            y_t = &ValidationY;
            group_len_vec = &group_len_train;
            nr_of_sets = group_len_train.size();
            group_len_vec_t = &group_len_val;
            use_caching = false;
            get_constraints = false;
            break;
        case FitnessCasesTEST:
            x = &TrainX;
            y = &TrainY;
            x_t = &TestX;
            y_t = &TestY;
            group_len_vec = &group_len_train;
            nr_of_sets = test_size;
            group_len_vec_t = &group_len_test;
            use_caching = false;
            get_constraints = false;
            break;
    }
    arma::vec output;
    if (batching){
        output = arma::vec(samples.size(), arma::fill::zeros);
    }else{
        output = arma::vec(nr_of_sets, arma::fill::zeros);
    }
    if (n->GetNrFCConstants().empty()) {
        size_t idx_train = 0;
        size_t idx_val = 0;
        size_t count = 0;
        arma::mat P = n->GetOutput(*x_t, use_caching);
        for (size_t i=0; i<(*group_len_vec).size(); i++) {

            if ((*group_len_vec_t)[i] > 0 && ((!batching  ) || find(samples.begin(), samples.end(), i) != samples.end())){
                //compute error per group member
                output(count) = ComputeMSE(P.rows(idx_val, idx_val + (*group_len_vec_t)[i] - 1),
                                             y->rows(idx_train, idx_train + (*group_len_vec)[i] - 1),
                                             y_t->rows(idx_val, idx_val + (*group_len_vec_t)[i] - 1));
                count = count +1;
            }
            idx_train = idx_train + (*group_len_vec)[i];
            idx_val = idx_val + (*group_len_vec_t)[i];
        }
        if (get_constraints){
            ComputeConstraints(n, false);
        }
    } else {
            size_t idx_train = 0;
            size_t idx_val = 0;
            size_t count = 0;
            for (size_t i=0; i<(*group_len_vec).size(); i++) {
                if ((*group_len_vec_t)[i] > 0 && ((!batching  ) || find(samples.begin(), samples.end(), i) != samples.end()) ){
                    n->ClearCachedOutput(true, true);
                    //optimize FCC's per group member, save error per group member and compute constraints per group member
                    RV rv(n, n->GetNrFCConstants(), x->rows(idx_train, idx_train + (*group_len_vec)[i] - 1),
                          y->rows(idx_train, idx_train + (*group_len_vec)[i] - 1), get_solutions, (*group_len_vec)[i], rv_evals, path, tree_index, tree, i+1, calc_constraints);
                    rv.run();
                    if (get_solutions){
                        boost::unique_lock<boost::shared_mutex> write_guard(write_solution_lock);
                        rv.saveSolutions();
                    }

                    output.at(count) = CalculateRVScore(n, x_t->rows(idx_val, idx_val + (*group_len_vec_t)[i]- 1), y_t->rows(idx_val, idx_val +(*group_len_vec_t)[i] - 1), y->rows(idx_train, idx_train + (*group_len_vec)[i] - 1) );
                    count = count +1;
                    if (get_constraints){
                        ComputeConstraints(n, true);
                    }
                }else{
                    if (get_solutions){
                        n->ClearCachedOutput(true, true);
                        //optimize FCC's per group member, save error per group member and compute constraints per group member
                        RV rv(n, n->GetNrFCConstants(), x->rows(idx_train, idx_train + (*group_len_vec)[i] - 1),
                              y->rows(idx_train, idx_train + (*group_len_vec)[i] - 1), get_solutions, (*group_len_vec)[i], rv_evals, path, tree_index, tree, i+1, calc_constraints);
                        rv.run();
                        boost::unique_lock<boost::shared_mutex> write_guard(write_solution_lock);
                        rv.saveSolutions();
                    }
                }
                idx_train = idx_train + (*group_len_vec)[i];
                idx_val = idx_val + (*group_len_vec_t)[i];

            }

    }

    return output;

}

double_t FCFitness::ComputeMSE(const arma::vec& P,const arma::vec& y, const arma::vec& y_t) {

    return arma::mean(arma::square((y_t - P) ));
}


double FCFitness::CalculateRVScore(Node * n, const arma::mat&  x_t, const arma::mat& y_t, const arma::mat& y){
    arma::vec preds = ((SingleNode*)n)->GetOutput(x_t,false).col(0);
    arma::vec trues = y_t;
    arma::vec diffs = preds - trues;

    return arma::mean(arma::square(diffs));

}


void FCFitness ::ComputeConstraints(Node* n, bool hasFCs) const {
    arma::mat m = arma::mat(100, 1, arma::fill::none);
    double j = 0;
    size_t violation = 0;
    for(size_t idx = 0; idx<100; idx++) {
        m.at(idx,0) = j;
        j = j+1;
    }
    arma::vec output = (n->GetOutput(m , false)).col(0);
    for(size_t idx = 0; idx<99; idx++) {
        if (output[idx]>output[idx+1] || (!isnan(output[idx]) && isnan(output[idx+1])) ){
            violation ++;
            break;
        }
    }
    if (std::isnan(output[99]) || output[99]>1500 ){
        violation ++;
    }
    if (std::isnan(output[0])  || output[0] > 0.01 || output[0] < 0.0){
        violation++;
    }
    if (!hasFCs){
        violation *= (samples.size());
    }
    n->constraints_violated += violation;
}