/*
 


 */

/* 
 * File:   IMSHandler.h
 * Author: virgolin
 *
 * Created on September 6, 2018, 1:38 PM
 */

#ifndef IMSHANDLER_H
#define IMSHANDLER_H

#include <iostream>
#include <Python.h>

#include "GPGOMEA/Evolution/EvolutionState.h"
#include "GPGOMEA/Evolution/EvolutionRun.h"

#include "GPGOMEA/Utils/Logger.h"
#include "GPGOMEA/Utils/Utils.h"

class IMSHandler {
public:

    IMSHandler(EvolutionState * st) {
        this->st = st;
        max_num_runs = st->config->max_num_runs;
        subgenerations_performed = std::vector<size_t>(max_num_runs, 0);
        runs = std::vector<EvolutionRun *>(max_num_runs, NULL);
        terminated_runs = std::vector<bool>(max_num_runs, false);
        initialized_runs = std::vector<bool>(max_num_runs, false);
        number_of_times_run_was_worse_than_later_run = std::vector<size_t>(max_num_runs, 0);
        elitist_per_run = std::vector<Node*>(max_num_runs, NULL);
        elitist_per_run_fit = std::vector<double_t>(max_num_runs, arma::datum::inf);

        archive_per_run.resize(max_num_runs);
    }

    IMSHandler(const IMSHandler& orig) {
    }

    virtual ~IMSHandler() {
        for(Node * el : elitist_per_run) {
            if (el)
                el->ClearSubtree();
        }
        
        for (EvolutionRun * r : runs) {
            if (r)
                delete r;
        }
    }
    std::string PadWithZeros(size_t number);
    void UpdateFitness( bool rv, bool init_i, size_t i, size_t rv_evals =0);
    void ClearSpecificHashTable(size_t i);

    void Start();

    void Terminate();
    
    Node * GetFinalElitist();

    std::vector<Node*> GetAllActivePopulations(bool copy_solutions);

    EvolutionState * st;

    std::vector<size_t> subgenerations_performed;
    std::vector<EvolutionRun *> runs;
    
    std::vector<Node *> elitist_per_run;
    std::vector<double_t> elitist_per_run_fit;

    std::vector<std::shared_ptr<MOArchive>>  archive_per_run;
    
    int min_run_idx = 0;
    int max_run_idx = -1;
    std::vector<bool> terminated_runs;
    std::vector<bool> initialized_runs;
    std::vector<size_t> number_of_times_run_was_worse_than_later_run;
    size_t max_num_runs;

    size_t macro_generation;
    
    double_t elitist_fit = arma::datum::inf;
    size_t elitist_size;

private:

};

#endif /* IMSHANDLER_H */

