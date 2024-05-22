/*
 


 */

/* 
 * File:   DiversifiedFCFitness.cpp
 * Author: Sijben
 *
 * Created on march 28
 */

#ifndef DIVERSIFIEDFCFITNESS_H
#define DIVERSIFIEDFCFITNESS_H

#include "GPGOMEA/Fitness/Fitness.h"
#include "GPGOMEA/Genotype/Multitree.h"

#include <armadillo>

class  DiversifiedFCFitness : public Fitness {
public:

    double_t ComputeFitness(Node * n, bool use_caching) override;
    
    double_t GetTestFit(Node * n) override;
    
    double_t GetValidationFit(Node * n) override;

    void SetFitnessCases(const arma::mat & X, FitnessCasesType fct) override;

    std::vector<size_t> group_len_train;
    std::vector<size_t> group_len_val;
    std::vector<size_t> group_len_test;



private:

    double_t ComputeMSE(const arma::vec & P, const arma::vec & Y);

};

#endif /* DIVERSIFIEDFCFITNESS_H */

