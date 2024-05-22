/*
 


 */

/* 

 */

#ifndef DIVERSIFIEDSYMBOLICREGRESSIONFITNESS_H
#define DIVERSIFIEDSYMBOLICREGRESSIONFITNESS_H

#include "GPGOMEA/Fitness/Fitness.h"

#include <armadillo>

class DiversifiedSymbolicRegressionFitness : public Fitness {
public:

    double_t ComputeFitness(Node * n, bool use_caching) override;
    
    double_t GetTestFit(Node * n) override;
    
    double_t GetValidationFit(Node * n) override;



private:

    double_t ComputeMSE(const arma::vec & res);

    double_t ComputeMSE(const arma::vec& P, const arma::vec& Y);




};

#endif /* DIVERSIFIEDSYMBOLICREGRESSIONFITNESS_H */

