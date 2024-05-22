/*
 


 */

/* 
 * File:   OpFunctionClassConst.h
 * Author: sijben
 *
 */

#ifndef OPFUNCTIONCLASSCONST_H
#define OPFUNCTIONCLASSCONST_H


#include "GPGOMEA/Operators/Operator.h"

#include <armadillo>
#include <math.h>

class OpFunctionClassConst : public Operator {
public:

    OpFunctionClassConst(double_t lower_b, double_t upper_b) {
        arity = 0;
        constant = arma::datum::nan;
        this->lower_b = lower_b;
        this->upper_b = upper_b;
        name = "FC CONST";
        type = OperatorType::opFCConstant;
    }
    OpFunctionClassConst() {
        throw std::runtime_error("OpFunctionClassConst::OpFunctionClassConst cannot be initialized");
    }


    Operator * Clone() const override {
        return new OpFunctionClassConst(*this);
    }

    arma::vec ComputeOutput(const arma::mat& x) override {
        return arma::ones(x.n_rows)* constant;
    }

    std::string GetHumanReadableExpression(const std::vector<std::string>& args) override {
        return name;

    }

    double_t GetConstant() {
        return constant;
    }

    void SetConstant(double_t new_value) {
        constant = new_value;
    }

    std::string GetPythonReadableExpression(const std::vector<std::string> & args) override {
        return name;
    }

    double_t lower_b, upper_b;

protected:
    double_t constant;

private:



};

#endif /* OPFUNCTIONCLASSCONST_H */

