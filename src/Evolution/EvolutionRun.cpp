/*
 * File:   EvolutionRun.cpp
 * Author: virgolin
 *
 * Created on June 28, 2018, 11:28 AM
 */

#include "GPGOMEA/Evolution/EvolutionRun.h"

using namespace std;
using namespace arma;

std::string EvolutionRun::PadWithZeros(size_t number){
    if (std::to_string(number).size() == 1)
        return "00" + std::to_string(number);
    if (std::to_string(number).size() == 2)
        return "0" + std::to_string(number);
    else{
        return std::to_string(number);
    }
}

void EvolutionRun::Initialize() {
    elitist = NULL;

    // Initialize population
    population = PopulationInitializer::InitializeTreePopulation(*config, *tree_initializer, *fitness);
    // Compute fitness of the population if not batching
    if (!(generation_handler->conf->batch_size > 0)){
        pop_fitnesses = fitness->GetPopulationFitness(population, true, config->caching);
        if (is_gomea_and_multiobj) {
            mo_archive->InitSOArchive(population,false );
            mo_archive->InitMOArchive(population, false );
            ((GOMEAMOGenerationHandler *) generation_handler)->InitLinkageMatrix(population);
        }
    }else{
        // if batching and multiobj gomea, only init the linkagematrix
        if (is_gomea_and_multiobj) {
            ((GOMEAMOGenerationHandler *) generation_handler)->InitLinkageMatrix(population);
        }

    }
    // create semantic library if needed
    if (config->semantic_variation && config->semback_library_type == SemanticLibraryType::SemLibRandomStatic)
        semantic_library->GenerateRandomLibrary(config->semback_library_max_height, config->semback_library_max_size,
                                                *fitness, config->functions, config->terminals, *tree_initializer,
                                                config->caching);
}

void EvolutionRun::DoGeneration() {

    // create semantic library if needed
    if (config->semantic_variation) {
        if (config->semback_library_type == SemanticLibraryType::SemLibRandomDynamic)
            semantic_library->GenerateRandomLibrary(config->semback_library_max_height,
                                                    config->semback_library_max_size, *fitness, config->functions,
                                                    config->terminals, *tree_initializer, config->caching);
        else if (config->semback_library_type == SemanticLibraryType::SemLibPopulation)
            semantic_library->GeneratePopulationLibrary(config->semback_library_max_height,
                                                        config->semback_library_max_size, population, *fitness,
                                                        config->caching);
    }
    // if batching, temporarily set the batch
    arma::mat backupX;
    arma::vec backupY;
    if (generation_handler->conf->batch_size > 0) {
        //clear old results from batching archive
        if (is_gomea_and_multiobj){
            mo_archive->ClearArchive(true);
        }
        if (generation_handler->conf->functionclass){

            //pick only sample of data set
            if (dynamic_cast<MOFitness*>(fitness)) {
                dynamic_cast<FCFitness *>(dynamic_cast<MOFitness*>(fitness)->sub_fitness_functions[0])->CleanHashTable();
                dynamic_cast<FCFitness *>(dynamic_cast<MOFitness*>(fitness)->sub_fitness_functions[0])->SetBatchingIndex(generation_handler->conf->batch_size);
            }else{
                dynamic_cast<FCFitness *>(fitness)->CleanHashTable();
                dynamic_cast<FCFitness *>(fitness)->SetBatchingIndex(generation_handler->conf->batch_size);
            }

        }else{
            backupX = arma::mat(fitness->TrainX);
            backupY = arma::vec(fitness->TrainY);
            // sample a batch
            arma::uvec samples = arma::randperm(generation_handler->conf->batch_size);
            arma::uvec unique_samples = arma::unique(samples).as_col();
            arma::mat batchX = fitness->TrainX.rows(samples);
            arma::vec batchY = fitness->TrainY.elem(samples);
            batchX.insert_cols(batchX.n_cols, batchY);
            // set as training fitness cases
            fitness->SetFitnessCases(batchX, FitnessCasesType::FitnessCasesTRAIN);
            //calculate fitness
            pop_fitnesses = fitness->GetPopulationFitness(population, true, config->caching);
        }

        if (is_gomea_and_multiobj){
            pop_fitnesses = fitness->GetPopulationFitness(population, true, config->caching);
            mo_archive->InitMOArchive(population, true);
            mo_archive->InitSOArchive(population, true);
        }


    }

    // perform generation
    generation_handler->PerformGeneration(population);



    if (generation_handler->conf->batch_size > 0) {
        if (!generation_handler->conf->functionclass){
            backupX.insert_cols(backupX.n_cols, backupY);
            fitness->SetFitnessCases(backupX, FitnessCasesType::FitnessCasesTRAIN);
        }else{
            // turn of batching
            if (dynamic_cast<MOFitness*>(fitness)) {
                dynamic_cast<FCFitness *>(dynamic_cast<MOFitness*>(fitness)->sub_fitness_functions[0])->CleanHashTable();
                dynamic_cast<FCFitness *>(dynamic_cast<MOFitness*>(fitness)->sub_fitness_functions[0])->batching = false;
            }else{
                dynamic_cast<FCFitness *>(fitness)->CleanHashTable();
                dynamic_cast<FCFitness *>(fitness)->batching = false;
            }
        }
        if (is_gomea_and_multiobj){
            // check if we need to add new members to full batch archive
            pop_fitnesses = fitness->GetPopulationFitness(mo_archive->mo_archive_batch, true, config->caching);
            mo_archive->InitMOArchive(mo_archive->mo_archive_batch, false, config->add_copied_tree);
            mo_archive->InitSOArchive(mo_archive->mo_archive_batch, false, config->add_copied_tree);
        }
    }

    gen = gen + 1;

    if (is_gomea_and_multiobj) {
        if (config->functionclass) {
            std::string plotpath = config->results_path + "/mo_archive_popsize" + to_string(population.size())+ "_rvevals"+ to_string(rvevals) + "_plot" + PadWithZeros(gen) + ".csv";
            mo_archive->SaveResultsFC ( plotpath, config->linear_scaling, config->caching);
        }else{
            std::string plotpath = config->results_path + "/mo_archive_plot" + PadWithZeros(gen) + ".csv";
            mo_archive->SaveResults( plotpath,  config->linear_scaling, config->caching);
        }
        elitist_fit = mo_archive->so_archive_full[0]->cached_fitness;
        elitist_last_changed_gen = mo_archive->LastGenChanged();
        if (elitist)
            elitist->ClearSubtree();
        elitist = mo_archive->so_archive_full[0]->CloneSubtree();
        elitist_size = mo_archive->so_archive_full[0]->GetSubtreeNodes(true).size();
    }else {
        //update stats
        Node *best = fitness->GetBestWithConstraints(population, false, config->caching);
        size_t best_size = best->GetSubtreeNodes(true).size();
        double_t best_fit = best->cached_fitness;
        size_t best_constraints = best->constraints_violated;

        if (best_constraints < elitist_contraints ||
            (best_fit < elitist_fit && best_constraints == elitist_contraints) ||
            generation_handler->conf->batch_size > 0) {
            elitist_fit = best_fit;
            if (elitist)
                elitist->ClearSubtree();
            elitist = best->CloneSubtree();
            fitness->ComputeFitness(elitist, false);
            elitist_size = best_size;
            elitist_contraints = best_constraints;
        }
        if (config->functionclass){
            std::string plotpath = config->results_path + "/solution_popsize" + to_string(population.size())+ "_rvevals"+ to_string(rvevals) + "_plot" + PadWithZeros(gen) + ".csv";
            std::ofstream myfile(plotpath, std::ios::trunc);
            myfile << "val"  << "|" << "exp" <<"|" << "exp_infix"<< "|" << "c" <<std::endl;
            myfile << best_fit  << "|" << elitist->GetSubtreeHumanExpression( true) << "|" << elitist->GetSubtreeExpression(true, true, true) << "|" <<elitist_contraints <<std::endl;

        }
    }
    if (is_nsga_and_multiobj) {
        // For each solution in the population with best rank, try to fit it in the archive
        for (Node *solution: population) {
            if (solution->rank != 0 )
                continue;

            // check if worth inserting in the archive_size
            bool solution_is_dominated = false;
            bool identical_objectives_already_exist = false;
            for (size_t i = 0; i < mo_archive_n.size(); i++) {
                // check domination
                Node *n = mo_archive_n[i];
                solution_is_dominated = n->Dominates(solution);
                if (solution_is_dominated)
                    break;

                identical_objectives_already_exist = true;
                for (size_t j = 0; j < solution->cached_objectives.n_elem; j++) {
                    if (solution->cached_objectives[j] != n->cached_objectives[j]) {
                        identical_objectives_already_exist = false;
                        break;
                    }
                }
                if (identical_objectives_already_exist)
                    break;

                bool n_is_dominated = solution->Dominates(n);
                if (n_is_dominated) {
                    n->ClearSubtree();
                    mo_archive_n[i] = NULL;  // keep this guy
                }
            }

            if (!solution_is_dominated && !identical_objectives_already_exist) {
                Node * add = solution->CloneSubtree();
                add->cached_objectives = solution->cached_objectives;
                mo_archive_n.push_back(add);    // clone it
            }

            vector<Node *> updated_archive;
            updated_archive.reserve(mo_archive_n.size());
            for (size_t i = 0; i < mo_archive_n.size(); i++)
                if (mo_archive_n[i])
                    updated_archive.push_back(mo_archive_n[i]);
            mo_archive_n = updated_archive;
        }
    }
}
