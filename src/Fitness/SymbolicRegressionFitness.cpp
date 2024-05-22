/*
 


 */

/*
 * File:   SymbolicRegressionFitness.cpp
 * Author: virgolin
 *
 * Created on June 27, 2018, 6:23 PM
 */


#include "GPGOMEA/Fitness/SymbolicRegressionFitness.h"
#include "GPGOMEA/Genotype/SingleNode.h"
#include "GPGOMEA/Genotype/Multitree.h"

double_t SymbolicRegressionFitness::ComputeFitness(Node* n, bool use_caching) {

    evaluations++;


    double fit;
    if (n->type == NodeType::Multi){
        fit = 0;
        for (SingleNode * sn : ((Multitree *) n)->nodes){
            fit += ComputeFitness(sn,use_caching);
        }
    }
    else{
        arma::mat P = n->GetOutput(TrainX, use_caching);
        fit = ComputeMSE(P, TrainY);
    }
    if (std::isnan(fit)) {
        fit = arma::datum::inf;
    }
    n->cached_fitness = fit;
    return fit;

}

double_t SymbolicRegressionFitness::GetValidationFit(Node* n) {

    double fit;
    if (n->type == NodeType::Multi){
        fit = 0;
        for (SingleNode * sn : ((Multitree *) n)->nodes){
            fit += GetValidationFit(sn);
        }
    }
    else{
        arma::mat P = n->GetOutput(ValidationX, false);
        fit = ComputeMSE(P, ValidationY);
    }
    return fit;
}


double_t SymbolicRegressionFitness::GetTestFit(Node* n) {
    double fit;
    if (n->type == NodeType::Multi){
        fit = 0;
        for (SingleNode * sn : ((Multitree *) n)->nodes){
            fit += GetTestFit(sn);
        }
    }
    else{
        arma::mat P = n->GetOutput(TestX, false);
        fit = ComputeMSE(P, TestY);
    }
    return fit;

}

double_t SymbolicRegressionFitness::ComputeMSE(const arma::vec& P, const arma::vec& Y) {
    arma::vec res = Y - P;
    double_t mse = arma::mean( arma::square(res) );
    return mse;
}

