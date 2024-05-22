/*
 


 */

/* 
 * File:   FCFitness.cpp
 * Author: Sijben
 *
 * Created on march 28
 */

#ifndef FCFITNESS_H
#define FCFITNESS_H

#include "GPGOMEA/Fitness/Fitness.h"
#include "GPGOMEA/Genotype/Multitree.h"
#include "GPGOMEA/RV/RV.h"

#include <armadillo>

class FCFitness : public Fitness {

public:
    explicit FCFitness(size_t max_num_runs) {
    }
    using CachedFCFitness = std::pair<arma::vec, size_t>;

    CachedFCFitness CreateCachedFitness(arma::vec output, size_t constraints) {
        return std::make_pair<arma::vec, size_t>(std::move(output), std::move(constraints));
    }

    double_t ComputeFitness(Node * n, bool use_caching) override;
    
    double_t GetTestFit(Node * n) override;
    
    double_t GetValidationFit(Node * n) override;

    void SaveFit(Node* n, const std::string& path, size_t index);

    void SetFitnessCases(const arma::mat & X, FitnessCasesType fct) override;

    void SetBatchingIndex(size_t number_of_individuals);

    arma::vec GetPopulationTestFitnessFC (const std::vector<Node*>& population, const std::string& path) ;

    double_t GetTestFitWithTree(Node* n, const std::string& path, size_t tree_index);
    boost::shared_mutex write_solution_lock;

    std::vector<size_t> group_len_train;
    std::vector<size_t> group_len_val;
    std::vector<size_t> group_len_test;

    std::vector<size_t> samples;

    bool calc_constraints = false;

    bool batching = false;

    HashTable<CachedFCFitness> &GetHashTable() {
        assert(rv_evals != INVALID_EVALS);
        auto it = hash_tables.find(rv_evals);
        assert(it != hash_tables.end());
        return it->second;
    }

    void SetRVevals(size_t new_rv_evals) {
        rv_evals = new_rv_evals;
        auto it = hash_tables.find(rv_evals);
        if(it == hash_tables.end()){
            hash_tables[rv_evals];
        }
    }

    void ClearHashTable() {
        for (auto it = hash_tables.begin(); it != hash_tables.end();) {
                auto &ht = it->second;
                ht.ClearTable();
                it = hash_tables.erase(it);
        }
    }

    void CleanHashTable() {
        for (auto it = hash_tables.begin(); it != hash_tables.end(); it++) {
            auto &ht = it->second;
            ht.ClearTable();
        }
    }

    void ClearHashTableOf(size_t key) {
        hash_tables[key].ClearTable();
        hash_tables.erase(key);

    }


private:
    size_t test_size;
    static constexpr size_t INVALID_EVALS = (size_t)-1;
    size_t rv_evals = INVALID_EVALS;


    std::unordered_map<size_t, HashTable<CachedFCFitness>> hash_tables;


    double_t ComputeMSE(const arma::vec& P, const arma::vec &y, const arma::vec &y_t);

    void ComputeConstraints(Node* n, bool hasFCs) const;
    arma::vec GetErrorVector(Node * sn,  bool use_caching,FitnessCasesType f, bool get_solutions, const std::string& path = "", size_t tree_index =0, size_t tree =0);
    static double CalculateRVScore(Node * n, const arma::mat&  x_t, const arma::mat& y_t, const arma::mat& y);

};

#endif /* FCFITNESS_H */

