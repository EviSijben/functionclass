/*
 


 */

/* 
 * File:   OpPow.h
 * Author: Sijben
 *
 *
 */

#ifndef OPPOW_H
#define OPPOW_H

#include "GPGOMEA/Operators/Operator.h"

class OpPow : public Operator {
public:

    OpPow() {
        arity = 2;
        name = "pow";
        type = OperatorType::opFunction;
        is_arithmetic = false;
    }

    Operator * Clone() const override {
        return new OpPow(*this);
    }

    arma::vec ComputeOutput(const arma::mat& x) override {
        return arma::pow(x.col(0),x.col(1));
    }


    std::string GetHumanReadableExpression(const std::vector<std::string>& args) override {
        return "("+args[0]+"^"+ args[1]+ ")";
    }

    std::string GetPythonReadableExpression(const std::vector<std::string>& args) override {
        return "("+args[0]+"^"+ args[1]+ ")";
    }



private:

};

#endif /* OPPOW_H */

