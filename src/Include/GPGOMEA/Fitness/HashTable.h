//
// Created by evi on 9/8/22.
//

#ifndef INC_3M_FC_GOMEA_HASHTABLE_H
#define INC_3M_FC_GOMEA_HASHTABLE_H

#include <boost/thread.hpp>
#include <unordered_map>


template<class FITNESS>
struct HTValue{
    FITNESS fitness;
};

template<class FITNESS>
class HashTable {
public:

    std::unordered_map<std::string, HTValue<FITNESS>> hash_table;

private:
    boost::shared_mutex hash_lock;


public:

    bool SolutionInTable(Node *n) {
        auto tree_str = n->GetSubtreeExpression(true, true);
        boost::unique_lock<boost::shared_mutex> read_guard(hash_lock);
        return hash_table.find(tree_str) != hash_table.end();
    }

    void AddSolutionToTable(Node *n,  FITNESS f) {
        auto tree_str = n->GetSubtreeExpression(true, true);
        boost::unique_lock<boost::shared_mutex> write_guard(hash_lock);
        if (hash_table.find(tree_str) == hash_table.end()) {
            hash_table[tree_str] = {f };
        }
    }

    void ClearTable() {
        boost::unique_lock<boost::shared_mutex> write_guard(hash_lock);
        std::unordered_map<std::string, HTValue<FITNESS>> empty_hash_table;
        std::swap(hash_table, empty_hash_table);
    }

    FITNESS GetSolution(Node *n) {
        auto &result = hash_table[n->GetSubtreeExpression(true, true)];
        return result.fitness;
    }


};

#endif //INC_3M_FC_GOMEA_HASHTABLE_H
