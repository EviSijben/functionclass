/*
 


 */

/* 
 * File:   OpNewProtectedDivisionS.h
 * Author: Sijben
 */

#ifndef OPPROTECTEDDIVISIONS_H
#define OPPROTECTEDDIVISIONS_H

#include "GPGOMEA/Operators/Operator.h"

class OpProtectedDivisionS : public Operator {
public:

    OpProtectedDivisionS() {
        arity = 2;
        name = "/s";
        type = OperatorType::opFunction;
        is_arithmetic = true;
    }

    Operator* Clone() const override {
        return new OpProtectedDivisionS(*this);
    }

    arma::vec ComputeOutput(const arma::mat& x) override {
        arma::vec result = x.col(1) / x.col(0);
        for(size_t i=0; i < x.col(0).n_elem; i++){
            if (x.col(0)[i] == 0) {
                result[i] = arma::datum::nan;
            }
        }

     
        return result;
    }

    std::string GetHumanReadableExpression(const std::vector<std::string>& args) override {
        return "(" + args[0] + name + args[1] + ")";
    }

    std::string GetPythonReadableExpression(const std::vector<std::string> & args) override {
        return "(" + args[0] + "/s" + args[1] + ")";
    }
    
private:

};

#endif /* OPPROTECTEDDIVISIONS_H */

