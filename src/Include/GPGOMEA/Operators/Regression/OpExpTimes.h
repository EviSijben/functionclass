/*
 


 */

/* 
 * File:   OpExpTimes.h
 * Author: Sijben

 */

#ifndef OPEXPTIMES_H
#define OPEXPTIMES_H

#include "GPGOMEA/Operators/Operator.h"

class OpExpTimes : public Operator {
public:

    OpExpTimes() {
        arity = 2;
        name = "exp*";
        type = OperatorType::opFunction;
        is_arithmetic = false;
    }

    Operator * Clone() const override {
        return new OpExpTimes(*this);
    }

    arma::vec ComputeOutput(const arma::mat& x) override {
//        arma::vec res = arma::vec(x.col(0).n_elem, arma::fill::none);
//        for (std::size_t i = 0; i < res.n_elem; ++i) {
//            if (x.col(0)[i]* x.col(1)[i] > 709){
//                res[i] = arma::datum::inf;
//            } else if (x.col(0)[i]* x.col(1)[i] < -709){
//                res[i] = 0;
//            } else {
//
//                res[i] = arma::exp(arma::vex.col(0)[i] * x.col(1)[i]));
//            }
//        }
        return arma::exp(x.col(0)%x.col(1));
//        return arma::exp(x.col(0) * x.col(1) );

//        return arma::exp(x.col(0));
    }

    arma::vec Invert(const arma::vec& desired_elem, const arma::vec& output_siblings, size_t idx) override {
        throw std::runtime_error("OpExpTimes::Invert - Not implemented");
    }
    
    std::string GetHumanReadableExpression(const std::vector<std::string>& args) override {
        return "( "+args[0] + name + args[1]  +" )";
    }

    std::string GetPythonReadableExpression(const std::vector<std::string> & args) override {
        return "(" + args[0] + name + args[1] + ")";
    }
    

private:

};

#endif /* OPEXPTIMES_H */

