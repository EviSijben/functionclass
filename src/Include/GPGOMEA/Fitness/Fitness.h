/*
 


 */

/* 
 * File:   Fitness.h
 * Author: virgolin
 *
 * Created on June 27, 2018, 6:14 PM
 */

#ifndef FITNESS_H
#define FITNESS_H

#include "GPGOMEA/Utils/Exceptions.h"
#include "GPGOMEA/Genotype/Node.h"
#include "GPGOMEA/Genotype/SingleNode.h"
#include "GPGOMEA/Genotype/Multitree.h"
#include "GPGOMEA/Fitness/HashTable.h"

#include <armadillo>
#include <string>

enum FitnessCasesType {
    FitnessCasesTRAIN, FitnessCasesTEST, FitnessCasesVALIDATION
};

class Fitness {
    
public:
    
    Fitness();
    Fitness(const Fitness& orig);
    virtual ~Fitness();

    arma::mat ReadFitnessCases(std::string filepath);

    virtual void SetFitnessCases(const arma::mat & X, FitnessCasesType fct);

    virtual double_t ComputeFitness(Node * n, bool use_caching);
    
    arma::vec GetPopulationFitness(const std::vector<Node *> & population, bool compute, bool use_caching);

    virtual arma::vec GetPopulationTestFitness(const std::vector<Node*>& population);
    
    Node * GetBest(const std::vector<Node*> & population, bool compute, bool use_caching);
    
    virtual double_t GetTestFit(Node * n);
    
    virtual double_t GetValidationFit(Node * n);

    Node * GetBestWithConstraints(const std::vector<Node*>& population, bool compute, bool use_caching);
    
    arma::mat TrainX;
    arma::vec TrainY;
    arma::mat ValidationX;
    arma::vec ValidationY;
    arma::mat TestX;
    arma::mat TestY;
    
    double_t trainY_mean = arma::datum::nan;
    double_t trainY_std = arma::datum::nan;
    
    arma::vec var_comp_trainY;

    size_t evaluations = 0;
    size_t macro_generation;

private:

};

#endif /* FITNESS_H */

