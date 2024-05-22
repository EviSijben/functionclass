/*
 


 */

/* 
 * File:   IMSHandler.cpp
 * Author: virgolin
 * 
 * Created on September 6, 2018, 1:38 PM
 */

#include "GPGOMEA/RunHandling/IMSHandler.h"

using namespace std;

std::string IMSHandler::PadWithZeros(size_t number){
    if (std::to_string(number).size() == 1)
        return "00" + std::to_string(number);
    if (std::to_string(number).size() == 2)
        return "0" + std::to_string(number);
    else{
        return std::to_string(number);
    }


}


void IMSHandler::UpdateFitness(bool rv, bool init_i, size_t i, size_t rv_evals ) {
    if (dynamic_cast<MOFitness*>(st->fitness)){
        for (Fitness * f : dynamic_cast<MOFitness*>(st->fitness)->sub_fitness_functions){
            if (dynamic_cast<FCFitness*>(f)) {
                if (rv)
                    dynamic_cast<FCFitness*>(f)->SetRVevals(rv_evals);
            }
        }
    }else{
        if (rv)
            dynamic_cast<FCFitness*>(st->fitness)->SetRVevals(rv_evals);

    }


}

void IMSHandler::ClearSpecificHashTable(size_t i) {
    if (dynamic_cast<MOFitness*>(st->fitness)){
        for (Fitness * f : dynamic_cast<MOFitness*>(st->fitness)->sub_fitness_functions){
            if (dynamic_cast<FCFitness*>(f)) {
                (dynamic_cast<FCFitness *>(f))->ClearHashTableOf(runs[i]->rvevals);
            }
        }
    }else{
        (dynamic_cast<FCFitness *>(st->fitness))->ClearHashTableOf(runs[i]->rvevals);
    }
}

void IMSHandler::Start() {

    string stats_file = "stats_generations.txt";
    string heading = "gen\ttime\tevals\tbest_fit\tbest_size\tpop_size";
    if (!st->config->running_from_python) {
        Logger::GetInstance()->Log(heading, st->config->results_path + "/" + stats_file);
    }

    macro_generation = 0;

    size_t maximum_arity_functions = 0;
    for (Operator *op: st->config->functions) {
        if (op->arity > maximum_arity_functions) {
            maximum_arity_functions = op->arity;
        }
    }

    // begin timer
    st->timer.tic();

    size_t current_pop_size = st->config->population_size;
    size_t biggest_pop_size_reached = 0;
    size_t current_rvevals = st->config->rv_evals;

    while (true) {

        // Check if termination criteria is met
        if ((st->fitness->evaluations > 0 && st->fitness->evaluations >= st->config->max_evaluations) ||
            (st->config->max_generations > 0 && macro_generation >= st->config->max_generations) ||
            (st->config->max_time > 0 && st->timer.toc() > st->config->max_time)) {
            break;
        } else if (terminated_runs[max_num_runs - 1]) {
            break;
        } else if (st->config->running_from_python) {
            // check if Ctrl+c from python
            if (PyErr_CheckSignals() == -1) {
                exit(1);
            }
        }

        for (int i = min_run_idx; i <= min(max_run_idx + 1, (int) max_num_runs - 1); i++) {

            // determine previous active run
            int prev_run_idx = -1;
            for (int j = i - 1; j >= min_run_idx; j--) {
                if (!terminated_runs[j]) {
                    prev_run_idx = j;
                    break;
                }
            }

            // check if run should start
            if (!terminated_runs[i] && // must not be terminated, AND
                (i == min_run_idx // it is the smallest run, OR
                 || (subgenerations_performed[prev_run_idx] == pow((int) st->config->num_sugen_IMS, i -
                                                                                                    prev_run_idx)) // previous active run did enough subgenerations
                )) {

                if (i != min_run_idx) { // reset counter for previous run
                    subgenerations_performed[prev_run_idx] = 0;
                }


                // check if run needs to be initialized
                if (!initialized_runs[i]) {
                    if (st->config->functionclass ){
                        UpdateFitness(false, true, i);
                        st->config->rv_evals = current_rvevals;
                        st->config->population_size = current_pop_size;
                        runs[i] = new EvolutionRun(*st,i);
                        runs[i]->Initialize();
                        initialized_runs[i] = true;
                        if (((i+1) % st->config->mod_IMS) == 0 ){
                            current_rvevals *= 10;

                        }else{
                            biggest_pop_size_reached = current_pop_size;
                            current_pop_size *= 2;
                        }

                    }else{
                        st->config->population_size = current_pop_size;
                        runs[i] = new EvolutionRun(*st,i);
                        runs[i]->Initialize();
                        initialized_runs[i] = true;
                        biggest_pop_size_reached = current_pop_size;
                        current_pop_size *= 2;
                    }

                    if (max_num_runs > 1)
                        cout << " ! IMS: ";
                    cout << " Initialized run " << i << " with population size " << runs[i]->population.size()
                         << " and initial tree height " << st->config->initial_maximum_tree_height;
                    if (st->config->functionclass){
                        cout<< " and number of RV evaluations " << runs[i]->rvevals << endl;
                    }else{
                        cout << endl;
                    }
                    max_run_idx++;
                }

                // set RV evals
                if (st->config->functionclass){
                    UpdateFitness(true, true, i , runs[i]->rvevals);
                }


                // Perform generation
                runs[i]->DoGeneration();

                if (st->config->functionclass){
                    UpdateFitness(false, false, i );
                }


                if (runs[i]->elitist_fit < elitist_per_run_fit[i]) {
                    elitist_per_run_fit[i] = runs[i]->elitist_fit;
                    if (elitist_per_run[i]) {
                        elitist_per_run[i]->ClearSubtree();
                    }
                    elitist_per_run[i] = runs[i]->elitist->CloneSubtree();
                }

                if (dynamic_cast<MOFitness*>(st->fitness)) {
                    if (!archive_per_run[i]){
                        archive_per_run[i] = runs[i]->mo_archive;
                    }
                    for (int j = i + 1; j < max_run_idx; j++) {
                        if (initialized_runs[j] && !terminated_runs[j]) {
                            if (archive_per_run[j]->DominatesArchive(archive_per_run[i].get())) {
                                number_of_times_run_was_worse_than_later_run[i] =
                                        number_of_times_run_was_worse_than_later_run[i] + 1;
                                break;
                            }
                        }
                    }
                    // just for output purposes
                    if (runs[i]->elitist_fit < elitist_fit) {
                        elitist_fit = runs[i]->elitist_fit;
                        elitist_size = runs[i]->elitist->GetSubtreeNodes(true).size();

                    }

                }
                else {
                    if (runs[i]->elitist_fit < elitist_fit) {
                        elitist_fit = runs[i]->elitist_fit;
                        elitist_size = runs[i]->elitist->GetSubtreeNodes(true).size();

                        // reset the count on the number of times this run performed poorly
                        number_of_times_run_was_worse_than_later_run[i] = 0;
                    } else {
                        // check if this run is performing poorly:
                        // determine if subsequent run exists with better fitness
                        for (int j = i + 1; j < max_run_idx; j++) {
                            if (initialized_runs[j] && !terminated_runs[j]) {
                                if (elitist_per_run_fit[j] < elitist_per_run_fit[i]) {
                                    number_of_times_run_was_worse_than_later_run[i] =
                                            number_of_times_run_was_worse_than_later_run[i] + 1;
                                    break;
                                }
                            }
                        }
                    }
                }
                // increment subgeneration performed by this run
                subgenerations_performed[i] = subgenerations_performed[i] + 1;
            }

            // check if run should terminate
            if (max_num_runs > 1 && initialized_runs[i] && !terminated_runs[i]) {

                bool should_terminate = false;
                // check if it should terminate
                if (initialized_runs[i] && st->config->gomea){
                    if (dynamic_cast<GOMEAGenerationHandler *> (runs[i]->generation_handler)){
                        if(((GOMEAGenerationHandler *) runs[i]->generation_handler)->gomea_converged) {
                            should_terminate = true;
                        }
                    }else{
                        if(((GOMEAMOGenerationHandler *) runs[i]->generation_handler)->gomea_converged) {
                            should_terminate = true;
                        }
                        if (runs[i]->elitist_last_changed_gen > 3){
                            should_terminate = true;
                        }
                    }
                }


                if (initialized_runs[i] &&
                    number_of_times_run_was_worse_than_later_run[i] >= st->config->early_stopping_IMS) {
                    should_terminate = true;
                }

                if (should_terminate) {
                    //check whether it is a functionclass run, and we should thus save the parameters.
                    if(st->config->functionclass){
                        //check if it is last run with this number of params, and if so, clear the hashtable
                        bool clear_table = true;
                        for (size_t relevant_run = i-(i%st->config->mod_IMS); relevant_run < (i+(st->config->mod_IMS-(i%st->config->mod_IMS))); relevant_run++ ) {
                            if ((!initialized_runs[relevant_run] || !terminated_runs[relevant_run]) && relevant_run != i) {
                                clear_table = false;
                            }

                        }

                        if (clear_table){
                            ClearSpecificHashTable(i);
                        }

                    }

                    terminated_runs[i] = true;
                    delete runs[i];
                    runs[i] = NULL;

                    if (i == min_run_idx) {
                        for (size_t j = min_run_idx + 1; j <= max_run_idx + 1; j++) {
                            if (!terminated_runs[j]) {
                                min_run_idx = j;
                                break;
                            }
                        }
                        if (max_run_idx + 1 < min_run_idx) {
                            max_run_idx++;
                        }
                    }
                    cout << " ! IMS: terminated run " << i << endl;
                }
            }
        }


        macro_generation++;
        st->fitness->macro_generation = macro_generation;

        string generation_stats = to_string(macro_generation) + "\t" + to_string(st->timer.toc()) + "\t" +
                                  to_string(st->fitness->evaluations) + "\t" + to_string(elitist_fit) + "\t" +
                                  to_string(elitist_size) + "\t" + to_string(biggest_pop_size_reached);

        if (!st->config->running_from_python)
            Logger::GetInstance()->Log(generation_stats,  st->config->results_path  + "/" + stats_file);
        cout << " > generation " << macro_generation << " - best fit: " << elitist_fit << endl;

    }


    Terminate();

}

Node *IMSHandler::GetFinalElitist() {
    Node *final_elitist;
    size_t final_elitist_idx = 0;
    if (st->fitness->ValidationY.empty() || st->config->functionclass) {
        for (size_t i = 0; i < elitist_per_run_fit.size(); i++) {
            if (elitist_per_run_fit[i] == elitist_fit) {
                final_elitist_idx = i;
                break;
            }
        }
    } else {
        // compute the best solution according to the validation set
        double_t best_validation_fitness = arma::datum::inf;
        for (size_t i = 0; i < elitist_per_run.size(); i++) {
            if (!elitist_per_run[i])
                continue;
            double_t val_fit = st->fitness->GetValidationFit(elitist_per_run[i]);
            if (val_fit < best_validation_fitness) {
                final_elitist_idx = i;
                best_validation_fitness = val_fit;
            }
        }
    }
    final_elitist = elitist_per_run[final_elitist_idx];
    return final_elitist;
}

std::vector<Node *> IMSHandler::GetAllActivePopulations(bool copy_solutions) {
    vector<Node *> result;
    result.reserve(10000);
    for (int i = 0; i < runs.size(); i++) {
        if (runs[i]) {
            for (Node *n: runs[i]->population) {
                if (copy_solutions)
                    result.push_back(n->CloneSubtree());
                else
                    result.push_back(n);
            }
        }
    }
    return result;
}

void IMSHandler::Terminate() {
    // Terminate
    cout << " -=-=-=-=-=-= TERMINATED =-=-=-=-=-=- " << endl;
    if(st->config->functionclass){
        return;
    }

    string msg = "";
    string out = "";

    // find best solution
    Node * final_elitist = GetFinalElitist();

    // best solution found
    out = "Best solution found:\t" + final_elitist->GetSubtreeHumanExpression(true) + "\n";
    cout << out;
    msg += out;


    if (st->config->linear_scaling) {
        if (final_elitist->type == NodeType::Multi){
            size_t count = 1;
            for (SingleNode * sn : ((Multitree *) final_elitist)->nodes){
                pair<double_t, double_t> ab = Utils::ComputeLinearScalingTerms(sn->GetOutput(st->fitness->TrainX, st->config->caching), st->fitness->TrainY, &st->fitness->trainY_mean, &st->fitness->var_comp_trainY);
                out = "Linear scaling coefficients:\ta=" + to_string(ab.first) + "\tb=" + to_string(ab.second) + "(Expression)" + to_string(count) + "\n";
                cout << out;
                msg += out;
                count += 1;
            }
        }
        else{
            pair<double_t, double_t> ab = Utils::ComputeLinearScalingTerms(final_elitist->GetOutput(st->fitness->TrainX, st->config->caching), st->fitness->TrainY, &st->fitness->trainY_mean, &st->fitness->var_comp_trainY);
            out = "Linear scaling coefficients:\ta=" + to_string(ab.first) + "\tb=" + to_string(ab.second) + "\n";
            cout << out;
            msg += out;
        }

    }

    out = "Number of nodes:\t" + to_string(final_elitist->GetSubtreeNodes(true).size()) + "\n";
    cout << out;
    msg += out;

    // Training fitness
    double_t training_fit = st->fitness->ComputeFitness(final_elitist, false);
    out = "Train fit:\t" + to_string(training_fit) + "\n";
    cout << out;
    msg += out;

    // Validation fitness
    if (!st->fitness->ValidationY.empty()) {
        double_t validation_fit = st->fitness->GetValidationFit(final_elitist);
        out = "Validation fit:\t" + to_string(validation_fit) + "\n";
        cout << out;
        msg += out;
    }

    if (!st->fitness->TestY.empty() && !st->config->running_from_python) {
        double_t test_fit = st->fitness->GetTestFit(final_elitist);
        out = "Test fit:\t" + to_string(test_fit) + "\n";
        cout << out;
        msg += out;
    }

    // Running time
    out = "Running time:\t" + to_string(std::round(st->timer.toc()*100) / 100) + "\n";
    cout << out;
    msg += out;

    // Evaluations
    out = "Evaluations:\t" + to_string(st->fitness->evaluations) + "\n";
    cout << out;
    msg += out;


    if (!st->config->running_from_python)
        Logger::GetInstance()->Log(msg, "result.txt");
}




