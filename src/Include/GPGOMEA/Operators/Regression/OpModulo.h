/*
 


 */

/* 
 * File:   OpModulo.h
 * Author: Sijben
 *
 * Created on Augustus 15, 2022
 */

#ifndef OPMODULO_H
#define OPMODULO_H

#include "GPGOMEA/Operators/Operator.h"


class OpModulo : public Operator {
public:

    OpModulo() {
        arity = 2;
        name = "%";
        type = OperatorType::opFunction;
        is_arithmetic = true;
    }

    Operator* Clone() const override {
        return new OpModulo(*this);
    }

    arma::vec ComputeOutput(const arma::mat& x) override {
        arma::vec res = arma::vec(x.col(0).n_elem, arma::fill::none);
        for (std::size_t i = 0; i < res.n_elem; ++i) {
            if (x.col(0)[i]<0 || x.col(1)[i]<0){
                res[i] = arma::datum::nan;
            } else {
                double times = std::floor(x.col(0)[i] / x.col(1)[i]);
                res[i] = x.col(0)[i] - (times * x.col(1)[i]);
            }
        }
        return res;
    }

    std::string GetHumanReadableExpression(const std::vector<std::string>& args) override {
        return "(" + args[0] + name + args[1] + ")";
    }
    std::string GetPythonReadableExpression(const std::vector<std::string> & args) override {
        return "(" + args[0] + "%" + args[1] + ")";
    }
private:

};

#endif /* OPTIMES_H */

