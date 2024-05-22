/*

 */

/*
 * File:   main.cpp
 * Author: Sijben
 *
 * Created December 2023
 */


#include <iostream>
#include <armadillo>
#include "GPGOMEA/Evolution/EvolutionState.h"
#include "GPGOMEA/Utils/ConfigurationOptions.h"


using namespace std;
using namespace arma;

std::vector<Operator *> all_operators = {new OpPlus(), new OpMinus(), new OpTimes(), new OpAnalyticQuotient(),
                                         new OpAnalyticQuotient01(), new OpNewProtectedDivision(),new OpProtectedDivision(), new OpAnalyticLog01(),
                                         new OpExp(), new OpExpTimes, new OpLog(), new OpSin(), new OpCos(), new OpSquare(), new OpSquareRoot(),
                                         new OpAnd(), new OpOr(), new OpNand(), new OpNor(), new OpNot(), new OpXor(), new OpPow(), new OpModulo(),
                                         new OpPowS(), new OpProtectedDivisionS()
};
bool mo;
double_t min_erc;
double_t max_erc;

std::pair<Node *,std::vector<string>> GenerateSingleTree(std::vector<string> tree){
    Node * singletree;

    if (tree[0] == "FC CONST"){
        singletree = new SingleNode((new OpFunctionClassConst(min_erc,max_erc)));
        tree.erase(tree.begin());
        return make_pair(singletree,tree);

    }else{
        if (tree[0][0] == 'x'){
            string variable_name = tree[0];
            variable_name.erase(variable_name.begin());
            std::stringstream sstream(variable_name);
            size_t id;
            sstream >> id;
            singletree = new SingleNode((new OpVariable(id)));
            tree.erase(tree.begin());
            return make_pair(singletree,tree);
        }else{
            for(auto & all_operator : all_operators) {
                if (all_operator->name == tree[0]) {
                    singletree = new SingleNode(all_operator->Clone());
                    tree.erase(tree.begin());
                    break;
                }
            }
            for (size_t j = 0; j < singletree->GetArity(); j++) {
                std::pair<Node *,std::vector<string> > child = GenerateSingleTree(tree);
                tree = child.second;
                singletree->AppendChild(child.first);
            }

        }
    }

    return std::make_pair(singletree, tree);
}
Node* String2Tree(std::vector<string> trees){
    Node * multitree = new Multitree();
    for (size_t i = 0; i<trees.size();i++){
        std::vector<string> string_vec;
        stringstream value_stream(trees[i]);
        string op;
        while (std::getline(value_stream, op, ' ')) {
            if (op != "CONST"){
                if (op == "FC"){
                    string_vec.emplace_back("FC CONST");
                }else{
                    string_vec.push_back(op);
                }

            }
        }
        std::pair<Node *,std::vector<string> > result = GenerateSingleTree(string_vec);
        if (trees.size() == 1)
            return result.first;
        else multitree->AppendSingleNode(result.first);
    }
    return multitree;


}

void SaveTestResultsFC(std::vector<Node * > mo_archive, const std::string& filename) {
    std::ofstream myfile(filename + "/test_results.csv", std::ios::trunc);
    myfile << "nr" << "|" << "obj1_test" << "|" << "obj2_test" << std::endl;
    for (size_t p = 0; p < mo_archive.size(); p++) {

        if ((mo_archive[0])->type == NodeType::Multi) {
            myfile << p << "| " << std::setprecision(8) << std::scientific << mo_archive[p]->cached_objectives_test[0] << "| " << std::setprecision(8) << std::scientific << mo_archive[p]->cached_objectives_test[1];
            myfile << std::endl;

        } else {
            myfile << p << "| " << mo_archive[p]->cached_objectives[0] << "| " << mo_archive[p]->cached_objectives[1];

        }


    }



}
int main(int argc, char** argv) {
    auto * st = new EvolutionState();
    st->SetOptions(argc, argv);
    mo = dynamic_cast<MOFitness *>(st->fitness);
    double_t biggest_val = arma::max(arma::max(arma::abs(st->fitness->TrainX)));
    min_erc = -5 * biggest_val;
    max_erc = 5 * biggest_val;
    string mo_path = st->config->mopath;

    std::ifstream fin(mo_path);

    // Make sure the file is open
    if(!fin.is_open()) throw std::runtime_error("Could not open file");
    string line;
    string value;
    std::vector<Node*> mo_archive;
    while(fin.good()){
        getline(fin,line );
        stringstream line_stream(line);
        size_t count = 0;
        std::vector<string> trees;
        while(std::getline(line_stream,value , '|')) {
            if (value[0] == 'n' or value[0] == 'v'){
                break;
            }
            if ((count-1)%3 == 0 && count >1 && mo ) {
                trees.push_back(value);
            }
            if (count ==2  && (!mo)) {
                trees.push_back(value);
            }
            count++;
        }
        if (!trees.empty()){
            mo_archive.push_back(String2Tree(trees));
        }

    }
    if (mo){
        (dynamic_cast<FCFitness *>(dynamic_cast<MOFitness *>(st->fitness)->sub_fitness_functions[0]))->GetPopulationTestFitnessFC(mo_archive, st->config->results_path + "/constants_fc.csv");
    }else{
        (dynamic_cast<FCFitness *>(st->fitness))->GetPopulationTestFitnessFC(mo_archive, st->config->results_path + "/constants_fc.csv");
    }
    if (mo){
        for (Node * n : mo_archive){
            st->fitness->GetTestFit(n);
        }
        SaveTestResultsFC(mo_archive, st->config->results_path);
    }else{
        std::ofstream myfile(st->config->results_path + "/test_results.csv", std::ios::trunc);
        myfile<<"test"<<std::endl;
        myfile<<st->fitness->GetTestFit(mo_archive[0])<<std::endl;
    }
    return 0;
}

