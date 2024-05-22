/*
 * File:   Node.cpp
 * Author: virgolin
 *
 * Created on June 27, 2018, 11:57 AM
 *
 * ##########################################
 *
 * New filename: SingleNode.cpp
 * Adjusted by: Sijben
 *
 * Adjusted on: April 7
 */

#include "GPGOMEA/Genotype/SingleNode.h"

using namespace std;
using namespace arma;

using child_iter = std::vector<Node *>::iterator;

SingleNode::SingleNode(Operator * op) : Node(NodeType::Single)  {
    this->op = op;
    this->shadow_operator = nullptr;
    this->parent = NULL;
    this->shadow_operator = nullptr;
    this->cached_fitness = arma::datum::inf;
    this->cached_objectives = arma::datum::inf;
}

SingleNode::SingleNode(const SingleNode& orig) : Node(NodeType::Single)  {
    op = orig.op->Clone();
    parent = NULL;
    if (orig.shadow_operator == nullptr){
        shadow_operator = nullptr;
    }else{
        shadow_operator = orig.shadow_operator->Clone();
    }
    children.clear();
    cached_fitness = orig.cached_fitness;
    cached_objectives = orig.cached_objectives;
}

SingleNode::~SingleNode() {
    delete this->shadow_operator;
    delete this->op;
}

arma::mat SingleNode::GetOutput(const arma::mat & x, bool use_caching) {
    vec out;
    if (nr_fcc == INVALID_NRFCC){
        if (use_caching && !cached_output.empty())
            return cached_output;
        if (op->type == OperatorType::opFunction) {
            size_t arity = this->GetArity();
            mat xx(x.n_rows, arity);
            for (size_t i = 0; i < arity; i++)
                xx.col(i) = children[i]->GetOutput(x, use_caching);
            out = op->ComputeOutput(xx);
        } else
            out = op->ComputeOutput(x);

        if (use_caching)
            cached_output = out;
    }else{
        if (use_caching && !cached_output.empty() && nr_fcc == 0){
            return cached_output;
        }
        Operator * right_op;
        if (shadow_operator) {
            right_op = shadow_operator;
        } else {
            right_op = op;
        }
        if (right_op->type == OperatorType::opFunction) {
            size_t arity = this->GetArity();
            mat xx(x.n_rows, arity);
            for (size_t i = 0; i < arity; i++)
                xx.col(i) = ((SingleNode*)children[i])->GetOutput(x, use_caching);
            out = right_op->ComputeOutput(xx);
        } else
            out = right_op->ComputeOutput(x);

        if (use_caching && nr_fcc == 0) {
            cached_output = out;
        }
    }


    return out;
}

size_t SingleNode::GetArity() const {
    return op->arity;
}

std::string SingleNode::GetValue() const {
    return op->name;
}

void SingleNode::ClearCachedOutput(bool also_ancestors, bool also_children) {
    cached_output.clear();
    if (also_ancestors && parent)
        parent->ClearCachedOutput(also_ancestors);
    if (also_children){
        for (auto child: children){
            child->ClearCachedOutput(false, also_children);
        }
    }
}

void SingleNode::AppendChild(Node * c) {
    auto * sn = (SingleNode *)c;
    children.push_back(sn);
    sn->parent = this;
}

child_iter SingleNode::DetachChild(Node* c) {
    auto * sn = (SingleNode *)c;
    auto it = children.begin();
    for (it; it < children.end(); it++) {
        if ((*it) == sn) {
            break;
        }
    }
    assert(it != children.end());
    children.erase(it);
    sn->parent = NULL;
    return it;
}

Node* SingleNode::DetachChild(size_t index) {
    auto it = children.begin() + index;
    Node * c = children[index];
    auto * sn = (SingleNode *)c;
    children.erase(it);
    sn->parent = NULL;
    return c;
}

void SingleNode::InsertChild(Node* c, child_iter it) {
    auto * sn = (SingleNode *)c;
    children.insert(it, sn);
    sn->parent = this;
}

size_t SingleNode::GetDepth() const {
    Node * p = parent;
    auto * sn = (SingleNode *)p;
    size_t depth = 0;
    while (p != NULL) {
        depth++;
        p = sn->parent;
        sn = (SingleNode *)p;
    }
    return depth;
}

size_t SingleNode::GetHeight(bool only_active) const {
    size_t h = GetDepth();
    size_t max_h = h;
    ComputeHeightRecursive(max_h, only_active);

    return max_h - h;
}

Operator* SingleNode::ChangeOperator(const Operator & other) {
    Operator * old = op;
    op = other.Clone();
    return old;
}

void SingleNode::ComputeHeightRecursive(size_t & h, bool only_active) const {
    size_t arity;
    if (only_active)
        arity = this->GetArity();
    else
        arity = children.size();

    if (arity == 0) {
        size_t d = this->GetDepth();
        if (d > h)
            h = d;
    }

    for (size_t i = 0; i < arity; i++) {
        auto sn = (SingleNode *) children[i];
        sn->ComputeHeightRecursive(h, only_active);
    }
}

std::vector<Node*> SingleNode::GetSubtreeNodes(bool only_active) {
    SubtreeParseType pt = SubtreeParseType::SubtreeParsePREORDER;
    std::vector<Node *> subtree_nodes;
    if (pt == SubtreeParseType::SubtreeParsePREORDER)
        GetSubtreeNodesPreorderRecursive(subtree_nodes, only_active);
    else if (pt == SubtreeParseType::SubtreeParsePOSTORDER)
        GetSubtreeNodesPostorderRecursive(subtree_nodes, only_active);
    else if (pt == SubtreeParseType::SubtreeParseLEVELORDER) {
        GetSubtreeNodesLevelOrder(subtree_nodes, only_active);
    } else
        throw std::runtime_error("Node::GetSubtreeNodes unrecognized subtree parse type");
    return subtree_nodes;
}

void SingleNode::GetSubtreeNodesPreorderRecursive(std::vector<Node*>& v, bool only_active) {
    v.push_back(this);
    if (only_active) {
        size_t arity = GetArity();
        for (size_t i = 0; i < arity; i++) {
            auto sn = (SingleNode *) children[i];
            sn->GetSubtreeNodesPreorderRecursive(v, only_active);
        }
    } else {
        for (size_t i = 0; i < children.size(); i++) {
            auto sn = (SingleNode *) children[i];
            sn->GetSubtreeNodesPreorderRecursive(v, only_active);
        }
    }
}

void SingleNode::GetSubtreeNodesPostorderRecursive(std::vector<Node*>& v, bool only_active) {
    if (only_active) {
        size_t arity = GetArity();
        for (size_t i = 0; i < arity; i++) {
            auto sn = (SingleNode *) children[i];
            sn->GetSubtreeNodesPreorderRecursive(v, only_active);
        }
    } else {
        for (size_t i = 0; i < children.size(); i++) {
            auto sn = (SingleNode *) children[i];
            sn->GetSubtreeNodesPreorderRecursive(v, only_active);
        }
    }
    v.push_back(this);
}

void SingleNode::GetSubtreeNodesLevelOrder(std::vector<Node*> & v,bool only_active) {

    queue<SingleNode*> q;
    SingleNode * curr;

    q.push(this);
    q.push(NULL);

    while(q.size() > 1) {
        curr = q.front();
        q.pop();
        if (!curr)
            q.push(NULL);
        else {
            v.push_back(curr);
            size_t to_consider = only_active ? curr->GetArity() : curr->children.size();
            for(size_t i = 0; i < to_consider; i++) {
                q.push((SingleNode *)curr->children[i]);
            }
        }
    }
}

Node* SingleNode::CloneSubtree() const {
    SingleNode * new_node = new SingleNode(*this);
    for (Node * c : children) {
        Node * new_child = c->CloneSubtree();
        new_node->AppendChild(new_child);
    }
    return new_node;
}


std::string SingleNode::GetSubtreeExpression(bool only_active_nodes, bool shadow, bool spaces) {
    string expr = "";
    GetSubtreeExpressionRecursive(expr, only_active_nodes, shadow, spaces);
    return expr;

}




std::string SingleNode::GetSubtreeHumanExpression(bool shadow) {

    string result = "";

    GetSubtreeHumanExpressionRecursive(result, shadow);


    return result;
}


std::string SingleNode::GetPythonExpression(bool shadow) {
    string result = "";

    GetPythonExpR(result, shadow);

    return result;
}

void SingleNode::GetSubtreeExpressionRecursive(string &expr, bool only_active_nodes, bool shadow, bool spaces) {
    if (shadow && shadow_operator) {
        expr += shadow_operator->name;
        if (spaces){
            expr += " ";
        }
        return;
    }
    expr += GetValue();
    if (spaces){
        expr += " ";
    }

    if (only_active_nodes) {
        for (size_t i = 0; i < GetArity(); i++) {
            auto sn = (SingleNode *) children[i];
            sn->GetSubtreeExpressionRecursive(expr, only_active_nodes, shadow,spaces);
        }
    } else {
        for (Node *c: children) {
            auto sn = (SingleNode *) c;
            sn->GetSubtreeExpressionRecursive(expr, only_active_nodes, shadow, spaces);
        }
    }
}

void SingleNode::GetSubtreeHumanExpressionRecursive(std::string& expr, bool shadow) {
    size_t arity = GetArity();
    vector<string> args;
    args.reserve(arity);

    for (size_t i = 0; i < arity; i++) {
        auto sn = (SingleNode *) children[i];
        sn->GetSubtreeHumanExpressionRecursive(expr, shadow);
        args.push_back(expr);
    }
    if(shadow && shadow_operator){
        expr = shadow_operator->GetHumanReadableExpression(args);
    }else{
        expr = op->GetHumanReadableExpression(args);
    }

}

void SingleNode::GetPythonExpR(std::string& expr, bool shadow) {
    size_t arity = GetArity();

    vector<string> args;
    args.reserve(arity);

    for (size_t i = 0; i < arity; i++) {
        auto sn = (SingleNode *) children[i];
        sn->GetPythonExpR(expr,shadow);
        args.push_back(expr);
    }


    if(shadow && shadow_operator){
        expr = shadow_operator->GetPythonReadableExpression(args);
    }else{
        expr = op->GetPythonReadableExpression(args);
    }
}


void SingleNode::ClearSubtree() {
    std::vector<Node *> subtree_nodes = GetSubtreeNodes(false);
    for (size_t i = 1; i < subtree_nodes.size(); i++) {
        delete subtree_nodes[i];
    }
    delete this;
}

size_t SingleNode::GetRelativeIndex() {
    if (parent == NULL)
        return 0;
    size_t i = 0;
    auto sn = (SingleNode *) parent;
    for (Node * c : sn->children) {
        if (c == this)
            return i;
        i++;
    }
    throw std::runtime_error("Node::GetRelativeIndex unreachable code reached.");
}

bool SingleNode::IsActive() {
    auto p = (SingleNode *) parent;
    if (p == NULL)
        return true;
    SingleNode * n = this;
    while (p != NULL) {
        if (n->GetRelativeIndex() >= p->GetArity())
            return false;
        n = p;
        p = (SingleNode *) n->parent;
    }
    return true;
}

std::vector<arma::vec> SingleNode::GetDesiredOutput(const std::vector<arma::vec> & Y, const arma::mat & X, bool use_caching) {

    // if it is the root, return
    if (!parent) {
        return Y;
    }

    size_t idx = GetRelativeIndex();
    auto sn = (SingleNode *) parent;
    size_t num_siblings = sn->GetArity() - 1;
    arma::mat output_siblings(X.n_rows, num_siblings);

    size_t j = 0;
    for (size_t i = 0; i < num_siblings + 1; i++) {
        Node * sibling = sn->children[j];
        if (i == idx) {
            continue;
        }
        vec sibling_output = sibling->GetOutput(X, use_caching);
        output_siblings.col(j) = sibling_output;
        j++;
    }

    vector<vec> inversion;
    inversion.reserve(Y.size());

    for (size_t i = 0; i < Y.size(); i++) {
        vec y = Y[i];

        // propagate impossibility or dont' care
        if (std::isinf(y[0]) || std::isnan(y[0])) {
            inversion.push_back(y);
            continue;
        }

        vec r;
        vec out_sib;
        if (num_siblings > 0) {
            out_sib = output_siblings.row(i);
            if (out_sib.has_nan()) { // can happen, e.g. the sibling is sin(exp(exp(100)));
                inversion.clear();
                r = vec(1);
                r[0] = arma::datum::inf;
                inversion.push_back(r);
                return inversion;
            }
        }

        r = sn->op->InvertAndPostproc(y, out_sib, idx);

        if (std::isinf(r[0])) {
            inversion.clear();
            inversion.push_back(r);
            return inversion;
        }

        inversion.push_back(r);
    }

    for (vec x : inversion) {
        assert(x.n_elem > 0);
    }

    return inversion;

}


Node * SingleNode::GetParent(){
    return this->parent;
}

std::vector<Node *> SingleNode::GetChildren() {
    return this->children;
}

Operator * SingleNode::GetOperator() {
    return this->op;
}


void SingleNode::AppendSingleNode(Node * n){
    throw NotImplementedException("SingleNode::AppendSingleNode not implemented.");
}


int SingleNode::Count_N_NaComp(int count) {
    if (count == -1)
        count = 0;
    if (!this->GetOperator()->is_arithmetic) {
        count += 1;
        vector<int> count_args; count_args.reserve(10);
        for(Node * c : this->GetChildren()) {
            count_args.push_back(c->Count_N_NaComp(count));
        }
        int max_count = 0;
        for(int arg : count_args) {
            if (arg > max_count)
                max_count = arg;
        }
        count = max_count;
    }
    else {
        if (this->GetOperator()->type == OperatorType::opFunction) {
            vector<int> count_args; count_args.reserve(10);
            for(Node * c : this->GetChildren()) {
                count_args.push_back(c->Count_N_NaComp(count));
            }
            int max_count = 0;
            for(int arg : count_args) {
                if (arg > max_count)
                    max_count = arg;
            }
            count = max_count;
        }
    }
    return count;
}

std::vector<Operator *> SingleNode::GetNrFCConstants() {
    std::vector<Operator *> FC_consts;
    Operator * right_op = shadow_operator ? shadow_operator : op;
    if (right_op->type == OperatorType::opFunction) {
        size_t arity = this->GetArity();
        for (size_t i = 0; i < arity; i++) {
            std::vector<Operator  *> output = children[i]->GetNrFCConstants();
            for (Operator * o: output) {
                FC_consts.push_back(o);
            }
        }
    } else {
        if (right_op->type == OperatorType::opFCConstant) {
            FC_consts.push_back(right_op);
        }
    }
    nr_fcc = FC_consts.size();
    return FC_consts;
}


size_t SingleNode::GetNrSingleNodes() {
    throw NotImplementedException("SingleNode::GetNrSingleNodes not implemented.");
}

void SingleNode::MakeShadowTree(double_t biggest_val) {
    this->CalculateReplaceSubtree();
    this->MakeShadowOperators(biggest_val);
}

std::pair<bool,bool> SingleNode::CalculateReplaceSubtree() {
    switch(op->type) {
        case OperatorType::opFunction: {
            bool FCCs = false;
            bool variables = false;
            for (size_t i = 0; i < GetArity(); i++) {
                auto bools = ((SingleNode *) children[i])->CalculateReplaceSubtree();
                FCCs = FCCs || bools.first;
                variables = variables || bools.second;
            }
            replace_with_fcc = FCCs && !variables;
            return make_pair(FCCs,variables);
        }
        case OperatorType::opFCConstant:
            replace_with_fcc = false;
            return make_pair(true, false);
        case OperatorType::opTermVariable:
            replace_with_fcc = false;
            return make_pair(false, true);
        default:
            replace_with_fcc = false;
            return make_pair(false, false);
    }
}

void SingleNode::MakeShadowOperators(double_t biggest_val){
    if (this->replace_with_fcc){
        if (shadow_operator) {
            delete shadow_operator;
        }
        shadow_operator =  new OpFunctionClassConst(-5*biggest_val, 5*biggest_val);
    } else {
        if (shadow_operator) {
            delete shadow_operator;
            shadow_operator = nullptr;
        }
        for (size_t i = 0; i < GetArity(); i++) {
            ((SingleNode *)children[i])->MakeShadowOperators(biggest_val);
        }
    }
}




