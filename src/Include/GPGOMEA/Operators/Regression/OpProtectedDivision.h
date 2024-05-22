/*
 


 */

/* 
 * File:   OpNewProtectedDivision.h
 * Author: Sijben
 */

#ifndef OPPROTECTEDDIVISION_H
#define OPPROTECTEDDIVISION_H

#include "GPGOMEA/Operators/Operator.h"

class OpProtectedDivision : public Operator {
public:

    OpProtectedDivision() {
        arity = 2;
        name = "/";
        type = OperatorType::opFunction;
        is_arithmetic = true;
    }

    Operator* Clone() const override {
        return new OpProtectedDivision(*this);
    }

    arma::vec ComputeOutput(const arma::mat& x) override {
        arma::vec result = x.col(0) / x.col(1);
        for(size_t i=0; i < x.col(1).n_elem; i++){
            if (x.col(1)[i] == 0) {
                result[i] = arma::datum::nan;
            }
        }

     
        return result;
    }

    std::string GetHumanReadableExpression(const std::vector<std::string>& args) override {
        return "(" + args[0] + name + args[1] + ")";
    }

    std::string GetPythonReadableExpression(const std::vector<std::string> & args) override {
        return "(" + args[0] + "/" + args[1] + ")";
    }
    
private:

};

#endif /* OPPROTECTEDDIVISION_H */

