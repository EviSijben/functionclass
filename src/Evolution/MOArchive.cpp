#include "GPGOMEA/Evolution/MOArchive.h"


size_t MOArchive::LastGenChanged(){
    if(!changed_this_gen){
        last_changed_gen += 1;
    }else{
        last_changed_gen = 0;
    }
    changed_this_gen = false;
    return last_changed_gen;
}

void MOArchive::UpdateSOArchive(Node *offspring, bool batching) {
    boost::unique_lock<boost::shared_mutex> write_guard((batching ? so_lock_batch : so_lock_full));
    for (size_t i = 0; i < offspring->cached_objectives.size(); i++) {
        if ((batching ? so_archive_batch : so_archive_full)[i] == nullptr) {
            Node *new_node = offspring->CloneSubtree();
            fitness->ComputeFitness(new_node,false);
            (batching ? so_archive_batch : so_archive_full)[i] = new_node;    // clone it
        } else if ((batching ? so_archive_batch : so_archive_full)[i]->cached_objectives[i] > offspring->cached_objectives[i] || offspring->constraints_violated < (batching ? so_archive_batch : so_archive_full)[i]->constraints_violated) {
            (batching ? so_archive_batch : so_archive_full)[i]->ClearSubtree();
            (batching ? so_archive_batch : so_archive_full)[i] = nullptr;
            Node *new_node = offspring->CloneSubtree();
            fitness->ComputeFitness(new_node,false);
            (batching ? so_archive_batch : so_archive_full)[i] = new_node;    // clone it
        }
    }

}

void MOArchive::UpdateMOArchive(Node *offspring, bool batching) {
    boost::unique_lock<boost::shared_mutex> write_guard((batching ? mo_lock_batch : mo_lock_full));
    bool solution_is_dominated = false;
    bool diversity_added = false;
    bool identical_objectives_already_exist;
    for (size_t i = 0; i < (batching ? mo_archive_batch : mo_archive_full).size(); i++) {
        // check domination
        solution_is_dominated = (batching ? mo_archive_batch : mo_archive_full)[i]->Dominates(offspring);
        if (solution_is_dominated)
            break;

        identical_objectives_already_exist = true;
        for (size_t j = 0; j < offspring->cached_objectives.n_elem; j++) {
            if (offspring->cached_objectives[j] != (batching ? mo_archive_batch : mo_archive_full)[i]->cached_objectives[j]) {
                identical_objectives_already_exist = false;
                break;
            }
        }
        identical_objectives_already_exist = identical_objectives_already_exist &&  offspring->constraints_violated == (batching ? mo_archive_batch : mo_archive_full)[i]->constraints_violated;
        if (identical_objectives_already_exist) {
            if (DiversityAdded(offspring, i, batching)) {
                diversity_added = true;
                (batching ? mo_archive_batch : mo_archive_full)[i]->ClearSubtree();
                (batching ? mo_archive_batch : mo_archive_full)[i] = nullptr;
                break;
            }else{
                break;
            }
        }

        if (offspring->Dominates((batching ? mo_archive_batch : mo_archive_full)[i])) {
            (batching ? mo_archive_batch : mo_archive_full)[i]->ClearSubtree();
            (batching ? mo_archive_batch : mo_archive_full)[i] = nullptr;
        }
    }
    (batching ? mo_archive_batch : mo_archive_full).erase(
            std::remove_if((batching ? mo_archive_batch : mo_archive_full).begin(), (batching ? mo_archive_batch : mo_archive_full).end(), [](Node *node) { return node == nullptr; }),
            (batching ? mo_archive_batch : mo_archive_full).end());

    if ((!solution_is_dominated && !identical_objectives_already_exist) || (diversity_added)) {
        changed_this_gen = true;
        Node *new_node = offspring->CloneSubtree();
        fitness->ComputeFitness(new_node,false);
        (batching ? mo_archive_batch : mo_archive_full).push_back(new_node);    // clone it
    }

}


Node * MOArchive::ReturnCopyRandomMOMember(bool batching) {
    boost::shared_lock<boost::shared_mutex> read_guard((batching ? mo_lock_batch : mo_lock_full));
    size_t index = arma::randu() * (batching ? mo_archive_batch : mo_archive_full).size();
    Node *copy = (batching ? mo_archive_batch : mo_archive_full)[index]->CloneSubtree();
    fitness->ComputeFitness(copy,false);
    return copy;
}

Node* MOArchive::ReturnCopySOMember(size_t idx, bool batching) {
    boost::shared_lock<boost::shared_mutex> read_guard((batching ? so_lock_batch : so_lock_full));
    Node *copy = (batching ? so_archive_batch : so_archive_full)[idx]->CloneSubtree();
    fitness->ComputeFitness(copy,false);
    return copy;
}


bool MOArchive::NonDominated(Node *offspring, bool batching) {
    boost::shared_lock<boost::shared_mutex> read_guard((batching ? mo_lock_batch : mo_lock_full));
    bool solution_is_dominated = false;
    bool identical_objectives_already_exist;
    bool diversity_added = false;
    if ((batching ? mo_archive_batch : mo_archive_full).empty())
        return true;
    for (size_t i = 0; i < (batching ? mo_archive_batch : mo_archive_full).size(); i++) {
        // check domination
        solution_is_dominated = (batching ? mo_archive_batch : mo_archive_full)[i]->Dominates(offspring);
        if (solution_is_dominated)
            break;

        identical_objectives_already_exist = true;
        for (size_t j = 0; j < offspring->cached_objectives.n_elem; j++) {
            if (offspring->cached_objectives[j] != (batching ? mo_archive_batch : mo_archive_full)[i]->cached_objectives[j]) {
                identical_objectives_already_exist = false;
                break;
            }
        }
        identical_objectives_already_exist = identical_objectives_already_exist &&  offspring->constraints_violated == (batching ? mo_archive_batch : mo_archive_full)[i]->constraints_violated;
        if (identical_objectives_already_exist) {
            if (DiversityAdded(offspring, i, batching)) {
                diversity_added = true;
                break;
            }else{
                break;
            }
        }
    }

    return (!solution_is_dominated && !identical_objectives_already_exist) || (diversity_added);
}

bool MOArchive::DiversityAdded(Node *offspring, size_t idx, bool batching) {
    if ( dynamic_cast<FCFitness *>((dynamic_cast<MOFitness *>(fitness))->sub_fitness_functions[0])) {
        return false;
    }
    arma::mat diff = offspring->GetOutput(fitness->TrainX, false) - (batching ? mo_archive_batch : mo_archive_full)[idx]->GetOutput(fitness->TrainX, false);
    if ( arma::mean(arma::vectorise(abs(diff))) == 0 ){
        return false;
    }else return true;
}


void MOArchive::InitSOArchive(const std::vector<Node *>& population, bool batching,  bool add_copied_tree) {
    for (Node *n: population) {
        UpdateSOArchive(n, batching);
        if (add_copied_tree){
            if(n->type != NodeType::Multi){
                throw std::runtime_error("MOArchive:: can't copy trees in singletree");
            }else{
                for (size_t i =0; i<((Multitree*)n)->nodes.size(); i++){
                    Node *copy = ((Multitree*)n)->CloneSingleSubtree(i);
                    fitness->ComputeFitness(copy,false);
                    UpdateSOArchive(copy, batching);
                    copy->ClearSubtree();
                }

            }
        }
    }

}

void MOArchive::InitMOArchive(const std::vector<Node *>& population, bool batching, bool add_copied_tree) {
    for (Node *n: population) {
        UpdateMOArchive(n, batching);
        if (add_copied_tree){
            if(n->type != NodeType::Multi){
                throw std::runtime_error("MOArchive:: can't copy trees in singletree");
            }else{

                for (size_t i =0; i<((Multitree*)n)->nodes.size(); i++){
                    Node *copy = ((Multitree*)n)->CloneSingleSubtree(i);
                    fitness->ComputeFitness(copy,false);
                    UpdateMOArchive(copy, batching);
                    copy->ClearSubtree();
                }

            }
        }
    }

}


void MOArchive::SaveResults(std::string plotpath, bool scaling, bool caching) {
    boost::unique_lock<boost::shared_mutex> read_guard(mo_lock_full);
    fitness->GetPopulationTestFitness(mo_archive_full);
    std::ofstream myfile(plotpath, std::ios::trunc);
    myfile << "nr" << "|" << "obj1_train" << "|" << "obj2_train" << "|" << "obj1_test" << "|" << "obj2_test" << "|" << "exp1" << "|" << "exp2" << std::endl;
    for (size_t p = 0; p < mo_archive_full.size(); p++) {

        if ((mo_archive_full[0])->type == NodeType::Multi) {
            myfile << p << "| " << std::setprecision(8) << std::scientific << mo_archive_full[p]->cached_objectives[0] << "| " << std::setprecision(8) << std::scientific << mo_archive_full[p]->cached_objectives[1]<< "| " << std::setprecision(8) << std::scientific << mo_archive_full[p]->cached_objectives_test[0]<< "| " << std::setprecision(8) << std::scientific <<  mo_archive_full[p]->cached_objectives_test[1];
            for (Node *k: (((Multitree *) mo_archive_full[p])->nodes)) {
                if (scaling) {
                    std::pair<double_t,double_t> ab = Utils::ComputeLinearScalingTerms(k->GetOutput(fitness->TrainX, caching), fitness->TrainY, &fitness->trainY_mean, &fitness->var_comp_trainY);
                    myfile << "|" << std::to_string(ab.first)+ "+" + std::to_string(ab.second) + "*" + mo_archive_full[p]->GetPythonExpression(true);
                }else {
                    myfile << "|" << k->GetSubtreeHumanExpression(true);
                }
                myfile << "|" << k->constraints_violated;
            }
            myfile << std::endl;

        } else {
            myfile << p << "| " << mo_archive_full[p]->cached_objectives[0] << "| " << mo_archive_full[p]->cached_objectives[1]<< "| " << mo_archive_full[p]->cached_objectives_test[0]<< "| " << mo_archive_full[p]->cached_objectives_test[1];
            if (scaling) {
                std::pair<double_t,double_t> ab = Utils::ComputeLinearScalingTerms(mo_archive_full[p]->GetOutput(fitness->TrainX, caching), fitness->TrainY, &fitness->trainY_mean, &fitness->var_comp_trainY);
                myfile << "|" << std::to_string(ab.first)+ "+" + std::to_string(ab.second) + "*" + mo_archive_full[p]->GetPythonExpression(true);
            }else {
                myfile << "|" << mo_archive_full[p]->GetSubtreeHumanExpression(true);
            }

        }


    }


}


void MOArchive::SaveResultsFC(std::string plotpath, bool scaling, bool caching) {
    boost::unique_lock<boost::shared_mutex> read_guard(mo_lock_full);
    std::ofstream myfile(plotpath, std::ios::trunc);
    myfile << "nr" << "|" << "obj1_train" << "|" << "obj2_train" << "|" << "exp1" << "|" << "exp1_infix" << "|" << "constraints1" << "|" << "exp2" << "|" << "exp2_infix" << "|" << "constraints2" << std::endl;
    for (size_t p = 0; p < mo_archive_full.size(); p++) {

        if ((mo_archive_full[0])->type == NodeType::Multi) {
            myfile << p << "| " << std::setprecision(8) << std::scientific << mo_archive_full[p]->cached_objectives[0] << "| " << std::setprecision(8) << std::scientific << mo_archive_full[p]->cached_objectives[1];
            for (Node *k: (((Multitree *) mo_archive_full[p])->nodes)) {
                myfile << "|" << k->GetSubtreeHumanExpression(true);
                myfile << "|" << k->GetSubtreeExpression(true, true, true);
                myfile << "|" << k->constraints_violated;
            }
            myfile << std::endl;

        } else {
            myfile << p << "| " << mo_archive_full[p]->cached_objectives[0] << "| " << mo_archive_full[p]->cached_objectives[1];
            myfile << "|" << mo_archive_full[p]->GetSubtreeHumanExpression(true);
            myfile << "|" << mo_archive_full[p]->GetSubtreeExpression(true, true, true);
            myfile << "|" << mo_archive_full[p]->constraints_violated;

        }


    }


}
bool MOArchive::DominatesArchive(MOArchive* new_mo_archive){
    for (auto other_node : new_mo_archive->mo_archive_full) {
        bool dominated = false;
        for (auto this_node : mo_archive_full) {
            if (this_node->DominatesTernary(other_node) >= 0) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            return false;
        }
    }
    return true;


}

void MOArchive::ClearArchive(bool batching){
    for (auto & i : (batching ? mo_archive_batch : mo_archive_full)){
        if (i != nullptr){
            i->ClearSubtree();
        }
        i = nullptr;
    }
    for (auto & i : (batching ? so_archive_batch : so_archive_full)){
        if (i != nullptr){
            i->ClearSubtree();
        }
        i = nullptr;
    }
    (batching ? mo_archive_batch : mo_archive_full).clear();
}

bool MOArchive::IsNotEmpty(bool batching){
    return !(batching ? mo_archive_batch : mo_archive_full).empty();
}

void MOArchive::SetAllSOarchives(size_t size, bool batching){
    so_archive_full = std::vector<Node*>(size,nullptr);
    if (batching){
        so_archive_batch = std::vector<Node*>(size,nullptr);
    }
}




