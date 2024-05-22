/*
 


 */

/* 
 * File:   OpPowS.h
 * Author: Sijben
 *
 *
 */

#ifndef OPPOWS_H
#define OPPOWS_H

#include "GPGOMEA/Operators/Operator.h"

class OpPowS : public Operator {
public:

    OpPowS() {
        arity = 2;
        name = "pows";
        type = OperatorType::opFunction;
        is_arithmetic = false;
    }

    Operator * Clone() const override {
        return new OpPowS(*this);
    }

    arma::vec ComputeOutput(const arma::mat& x) override {
        return arma::pow(x.col(1),x.col(0));
    }


    std::string GetHumanReadableExpression(const std::vector<std::string>& args) override {
        return "("+args[0]+"^s"+ args[1]+ ")";
    }

    std::string GetPythonReadableExpression(const std::vector<std::string>& args) override {
        return "("+args[0]+"^s"+ args[1]+ ")";
    }



private:

};

#endif /* OPPOWS_H */

