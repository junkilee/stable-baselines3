import hydra
import logging
import os
import math
from pysat.formula import CNF # , WCNF
from pysat.card import *
from sympy import symbols
from sympy.logic.boolalg import to_cnf
from sympy.logic import simplify_logic
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import And, Or, Not, Equivalent, Implies
from sympy.parsing.sympy_parser import parse_expr
from experiment.decision_tree.utils.ansi_colors import *
from pysat.solvers import Solver # , Glucose4
from experiment.decision_tree.utils.tree_visualizer import check_true


def output_cnf(cnf_str):
    print(BRIGHT_RED, cnf_str, RESET)


def implies_to_cnf(left_id, right_cnfs):
    result_cnfs = []
    for cnf in right_cnfs:
        result_cnfs += [[-left_id] + cnf]
    return result_cnfs


def clauses_to_names(clauses, var_list):
    clauses_in_str = []
    for clause in clauses:
        clause_in_str = []
        for var in clause:
            if var < 0:
                addition = '-'
            else:
                addition = ''
            clause_in_str += [addition + var_list[abs(var) - 1]]
        clauses_in_str += [clause_in_str]
    return clauses_in_str
        

def update_clause(left_clause, right_clause, operation):
    if left_clause is None:
        return right_clause
    else:
        return operation(left_clause, right_clause)


def check_cnf_clause(cnf_clause, truth_table):
    count = 0
    for truth_val in cnf_clause:
        var_id = abs(truth_val)
        val = truth_table[var_id - 1]
        if truth_val == val:
            count += 1
    return count >= 1


def _check_list_of_cnfs(list_of_cnfs, truth_table, debug=True):
    count = 0
    for cnf_clause in list_of_cnfs:
        if debug:
            print(BRIGHT_YELLOW + str(cnf_clause) + RESET, end='')
        if check_cnf_clause(cnf_clause, truth_table):
            if debug:
                print(BLUE + " - OK" + RESET)
            count += 1        
        else:
            if debug:
                print(RED + " - NOT OK" + RESET)

    return count == len(list_of_cnfs)


class CNFGenerator(object):
    def __init__(self, tree_size, feature_size, example_size, debug=True):
        self._n = tree_size
        self._k = feature_size
        self._m = example_size
        self._counter = 0
        self._var_list = []
        self._var_map = dict()
        self._tree_related_cnfs = None
        self._decision_tree_related_cnfs = None
        self._example_constraints_cnfs = None

        if debug:
            print(BRIGHT_CYAN)
            print("Tree size        : {}".format(self._n))
            print("Feature size     : {}".format(self._k))
            print("Example size     : {}".format(self._m))
            print(RESET)

    @property
    def var_list(self):
        return self._var_list

    @property
    def var_map(self):
        return self._var_map

    def register(self, var_name):        
        assert var_name not in self._var_map
        self._counter += 1
        self._var_map[var_name] = self._counter
        self._var_list += [var_name]        
        assert self._var_list[self._var_map[var_name] - 1] == var_name
        return self._counter

    def register_dummy_variables(self, orig_vars, cnfs, name):
        dummy_vars = []
        new_dummy_ids = {}
        new_ids = []
        for cnf in cnfs:
            for var in cnf:
                real_var = var if var > 0 else -var
                if real_var not in orig_vars and real_var not in dummy_vars:
                    dummy_vars += [real_var]
        change = False
        for var in dummy_vars:
            change = True
            new_id = self.register("dummy_{}_{}".format(name, var))
            #print("NEW VAR", var, new_id)
            new_dummy_ids[var] = new_id
        new_cnfs = []
        for cnf in cnfs:
            new_cnf = []
            for var in cnf:
                real_var = var if var > 0 else -var
                sin = 1 if var > 0 else -1
                if real_var not in orig_vars:
                    new_cnf += [sin * new_dummy_ids[real_var]]
                else:
                    new_cnf += [var]
            new_cnfs += [new_cnf]
        return new_cnfs, change

    def get_var_id(self, var_name):        
        #print(var_name)
        assert var_name in self._var_map
        return self._var_map[var_name]

    def get_var_name(self, var_id):
        #print(var_id)
        assert var_id >= 1 and var_id <= self._counter
        return self._var_list[var_id - 1]
    
    def get_var_names(self, var_ids):
        assert type(var_ids) == list
        return [self.get_var_name(var_id) for var_id in var_ids]

    def LR(self, i):
        l = []
        fr = i + 1
        to = min(2 * i, self._n - 1)
        if fr % 2 != 0:
            fr = fr + 1
        for j in range(fr, to + 1, 2): # even
            l += [j]
        return l

    def RR(self, i):
        l = []
        fr = i + 2
        to = min(2 * i + 1, self._n)
        if fr % 2 == 0:
            fr = fr + 1
        for j in range(fr, to + 1, 2): # odd
            l += [j]
        return l

    def cnf_to_clauses(self, cnf):
        clauses = []
        if type(cnf) == str:
            output_cnf(cnf)
            clauses = self.cnf_to_clauses(to_cnf(cnf))
        if type(cnf) == Symbol:
            #print(cnf.name)
            clauses += [self.get_var_id(cnf.name)]
        if type(cnf) == Not:
            clauses += [-self.get_var_id(cnf.args[0].name)]
        elif type(cnf) == And:
            for arg in cnf.args:
                clauses += self.cnf_to_clauses(arg)
        elif type(cnf) == Or:
            for arg in cnf.args:
                clauses += self.cnf_to_clauses(arg)
            clauses = [clauses]
        return clauses

    def add_to_clauses(self, left_clauses, right_clauses, tag = None):
        processed_clauses = []
        if tag is not None:
            print(BRIGHT_BLUE, tag, RESET)
        print(GREEN, right_clauses, RESET)
        print(GREEN, clauses_to_names(right_clauses, self.var_list), RESET)
        
        if any(isinstance(el, list) for el in right_clauses):
            processed_clauses += right_clauses
        else:
            processed_clauses = right_clauses
        return left_clauses + processed_clauses

    def add_tree_related_cnfs(self, debug=True):
        # Based on 3.1 Encoding Valid Binary Trees [Naro18]
        # N : # of nodes
        # v_i      : 1 iff node i is a leaf node
        # l_i,j    : 1 iff node i has node j as the left child with j \in LR(i), where LR(i) = even([i+1, min(2i, N-1)]), i= 1, ..., N
        # r_i,j    : 1 iff node i has node j as the right child with j \in LR(i), where RR(i) = odd([i+2, min(2i+1, N-1)]), i= 1, ..., N
        # p_j,i    : 1 iff the parent node of node j is node i, j = 2, ..., N, i = 1, ..., N-1
        # --------------------------------------------------------------------------------------------------------------------------------
        # ~ v_1    : root node is not a leaf      as (1)
        # v_i -> ~ l_i,j  and j \in LR(i)         as (2) what about r_i,j?
        #          : If a node is a leaf node, then it has no children
        # l_i,j <-> r_i,j+1                       as (3)
        #          : The left child and the right child of the ith node are numbered 
        # ~v_i -> ( \sum_{j \in LR(i)} l_i,j = 1) as (4)
        #          : A non-leaf node must have a child
        # p_j,i <-> l_i,j, j \in LR(i)            as (5)
        # p_j,i <-> r_i,j, j \in RR(i)            
        #          : If the ith node is a parent then it must have a child
        # ( \sum_{i = |_ j/2 _|}^{min(j-1, N)} p_j,i = 1) with j= 2, ..., N as (6)
        clauses = []
        for i in range(1, self._n+1):
            self.register("v_{}".format(i))
        if debug:
            output_cnf("~v_1")
        clauses = self.add_to_clauses(clauses, [[-self.get_var_id("v_1")]], tag="(1)")                                        # as (1)
        for i in range(1, self._n + 1):
            lr_ids = []
            print(BLUE + "i: " + str(i) + RESET)
            print(BLUE + "LR: " + str(self.LR(i)) + RESET)
            print(BLUE + "RR: " + str(self.RR(i)) + RESET)
            for j in self.LR(i):                
                lr_ids.append(self.register("l_{}_{}".format(i,j)))
                #print(to_cnf("v_{} >> ~l_{}_{}".format(i, i, j)))
                clauses = self.add_to_clauses(clauses, self.cnf_to_clauses("v_{} >> ~l_{}_{}".format(i, i, j)), tag="(2)")   # as (2)
            # rr_ids = []
            # for j in self.RR(i):                
            #     rr_ids.append(self.register("r_{}_{}".format(i,j)))
            #     #print(to_cnf("v_{} >> ~l_{}_{}".format(i, i, j)))
            #     clauses = self.add_to_clauses(clauses, self.cnf_to_clauses("v_{} >> ~r_{}_{}".format(i, i, j)))   # as (2)

            as4_clauses = CardEnc.equals(lits=lr_ids, bound=1, encoding=EncType.seqcounter).clauses            
            new_clauses, change = self.register_dummy_variables(lr_ids, as4_clauses, "v_{}".format(i))
            if change:
                #print(BRIGHT_RED + "new variables added." + RESET)
                as4_clauses = new_clauses
            if as4_clauses == []:
                as4_clauses = [[self.get_var_id("v_{}".format(i))]]
            else:
                as4_clauses = implies_to_cnf(-self.get_var_id("v_{}".format(i)), as4_clauses)                    # as (4)
            clauses = self.add_to_clauses(clauses, as4_clauses, tag="(4)")
            if debug:
                print("lr_ids : i={}, {} {}".format(i, lr_ids, self.get_var_names(lr_ids)))
                print("as 4 : ", as4_clauses)
            for j in self.RR(i):
                self.register("r_{}_{}".format(i,j))                
            for j in self.LR(i):
                clauses = self.add_to_clauses(clauses, self.cnf_to_clauses("Equivalent(l_{}_{}, r_{}_{})".format(i, j, i, j+1)), tag="(3)") # as (3)
        for j in range(2, self._n+1):
            p_jis = []
            for i in range(max(1, j // 2), min(j, self._n+1)):
                p_jis += [self.register("p_{}_{}".format(j,i))]
                lr_i = self.LR(i)
                rr_i = self.RR(i)
                if j in lr_i:
                    clauses = self.add_to_clauses(clauses, self.cnf_to_clauses("Equivalent(p_{}_{}, l_{}_{})".format(j, i, i, j)), tag="(5)") # as (5)
                elif j in rr_i:
                    clauses = self.add_to_clauses(clauses, self.cnf_to_clauses("Equivalent(p_{}_{}, r_{}_{})".format(j, i, i, j)), tag="(5)") # as (5)
            as6_clauses = CardEnc.equals(lits=p_jis, bound=1, encoding=EncType.seqcounter).clauses
            new_clauses, change = self.register_dummy_variables(p_jis, as6_clauses, "p_{}_is".format(j))
            print(BRIGHT_RED, " + ".join(self.get_var_names(p_jis)) + " = 1", RESET)
            if change:
                #print(BRIGHT_RED +  " new variables added." + RESET)
                as6_clauses = new_clauses
            clauses = self.add_to_clauses(clauses, as6_clauses, tag="(6)")  # as (6)
                
        if debug:
            print(RED + "Tree Related." + RESET)
            print(self._var_map)
            print(clauses)
        return clauses

    def add_decision_tree_related_cnfs(self, debug=True):
        # Based on 3.2 Computing Decision Trees with SAT [Naro18]
        # K : # of features, N : # of nodes, M : # of data

        # a_r,j    : 1 iff feature f_r is assigned to node j, r = 1~K, j = 1~N
        # u_r,j    : 1 iff feature f_r is being discriminated against by node j, r = 1~K, j = 1~N
        # d^0_r,j  : 1 iff feature f_r is discriminated for value 0 by node j, or by one of its ancestors, r = 1~K, j = 1~N
        # d^1_r,j  : 1 iff feature f_r is discriminated for value 1 by node j, or by one of its ancestors, r = 1~K, j = 1~N
        # c_j      : 1 iff class of leaf node j is 1, j = 1~N
        # --------------------------------------------------------------------------------------------------------------------------------
        # j = 2~N
        # d^0_r,j <-> ( OR^{j-1}_{i = |_j/2_|} ((p_ji and d^0_ri) or (a_ri and r_ij))); d^0_{r,1} = 0. as (7)
        # d^1_r,j <-> ( OR^{j-1}_{i = |_j/2_|} ((p_ji and d^1_ri) or (a_ri and l_ij))); d^1_{r,1} = 0. as (8)
        # r=1~K, j=1~N
        # and^{j-1}_{i=|_j/2_|} (u_ri and p_ji -> ~a_rj), u_rj <-> (a_rj or or^{j-1}_{i=|_j/2_|} (u_ri and p_ji)) as (9) 
        # j = 1~N
        # ~v_j -> (sum^K_{r=1} a_rj = 1) as (10)
        # v_j -> (sum^K_{r=1} a_rj = 0) as (11)
        clauses = []

        for r in range(1, self._k + 1):
            for j in range(1, self._n + 1):
                self.register("a_{}_{}".format(r, j))
                self.register("u_{}_{}".format(r, j))
                self.register("d0_{}_{}".format(r, j))
                self.register("d1_{}_{}".format(r, j))

        for j in range(1, self._n + 1):
            self.register("c_{}".format(j))

        for r in range(1, self._k + 1):
            if debug:
                output_cnf("d0_{}_1".format(r))
                output_cnf("d1_{}_1".format(r))
                print(BLUE + "===> r, j: {}, {}".format(r, 1) + RESET)
            clauses = self.add_to_clauses(clauses, [[-self.get_var_id("d0_{}_1".format(r))]], tag = "(7)")       # as (7)
            clauses = self.add_to_clauses(clauses, [[-self.get_var_id("d1_{}_1".format(r))]], tag = "(8)")       # as (8)
            clauses = self.add_to_clauses(clauses, self.cnf_to_clauses(to_cnf(Equivalent(symbols("u_{}_1".format(r)), symbols("a_{}_1".format(r))))), tag="(9-1)") # as (9-1)
            for j in range(2, self._n + 1):                
                exprs_0 = None
                exprs_1 = None
                exprs_9_0 = None
                exprs_9_1 = None
                if debug: 
                    print(BLUE + "===> r, j: {}, {}".format(r, j) + RESET)
                for i in range(j // 2, j):
                    if debug:
                        print(BLUE + "i: {}".format(i) + " LR: " + str(self.LR(i)) + RESET)
                        print(BLUE + "i: {}".format(i) + " RR: " + str(self.RR(i)) + RESET)
                    
                    if j in self.LR(i):
                        expr_0 = parse_expr("(p_{}_{} & d0_{}_{}) | (a_{}_{} & l_{}_{})".format(j, i, r, i, r, i, i, j))                        
                    else:
                        expr_0 = parse_expr("(p_{}_{} & d0_{}_{})".format(j, i, r, i))
                    exprs_0 = update_clause(exprs_0, expr_0, Or)
                    if j in self.RR(i):
                        expr_1 = parse_expr("(p_{}_{} & d1_{}_{}) | (a_{}_{} & r_{}_{})".format(j, i, r, i, r, i, i, j))
                    else:
                        expr_1 = parse_expr("(p_{}_{} & d1_{}_{})".format(j, i, r, i))
                    exprs_1 = update_clause(exprs_1, expr_1, Or)

                    expr_9_0 = parse_expr("(u_{}_{} & p_{}_{}) >> ~a_{}_{}".format(r, i, j, i, r, j))
                    expr_9_1 = parse_expr("(u_{}_{} & p_{}_{})".format(r, i, j, i))
                    
                    exprs_9_0 = update_clause(exprs_9_0, expr_9_0, And)
                    exprs_9_1 = update_clause(exprs_9_1, expr_9_1, Or)
                if exprs_0 is not None:
                    print(">", Equivalent(symbols("d0_{}_{}".format(r, j)), exprs_0))
                    clauses = self.add_to_clauses(clauses, self.cnf_to_clauses(to_cnf(Equivalent(symbols("d0_{}_{}".format(r, j)), exprs_0))), tag="(7)") # as (7)
                else:
                    self.add_to_clauses(clauses, [[-self.get_var_id("d0_{}_{}".format(r, j))]], tag = "(7)")
                    #print(RED + "> [][][][][] (7)" + RESET)
                if exprs_1 is not None:
                    print(">>", Equivalent(symbols("d1_{}_{}".format(r, j)), exprs_1))
                    clauses = self.add_to_clauses(clauses, self.cnf_to_clauses(to_cnf(Equivalent(symbols("d1_{}_{}".format(r, j)), exprs_1))), tag="(8)") # as (8)
                else:
                    self.add_to_clauses(clauses, [[-self.get_var_id("d1_{}_{}".format(r, j))]], tag = "(8)")
                    #print(RED + "> [][][][][] (8)" + RESET)
                if exprs_9_0 is not None:
                    print(">>>", exprs_9_0)
                    clauses = self.add_to_clauses(clauses, self.cnf_to_clauses(to_cnf(exprs_9_0)), tag="(9-0)") # as (9-0)
                else:
                    print(RED + "> [][][][][] (9-0)" + RESET)
                exprs_9_1 = update_clause(exprs_9_1, symbols("a_{}_{}".format(r,j)), Or)
                if exprs_9_1 is not None:                    
                    clauses = self.add_to_clauses(clauses, self.cnf_to_clauses(to_cnf(Equivalent(symbols("u_{}_{}".format(r,j)), exprs_9_1))), tag="(9-1)") # as (9-1)
                else:
                    print(RED + "> [][][][][] (9-1)" + RESET)
                    
        
        for j in range(1, self._n + 1):
            arjs = []
            for r in range(1, self._k + 1):
                arjs.append(self.get_var_id("a_{}_{}".format(r,j)))
            as10_clauses = CardEnc.equals(lits=arjs, bound=1, encoding=EncType.seqcounter).clauses
            new_clauses, change = self.register_dummy_variables(arjs, as10_clauses, "va1_{}".format(j))
            if change:
                #print(BRIGHT_RED + "new variables added." + RESET)
                as10_clauses = new_clauses
            as10_clauses = implies_to_cnf(-self.get_var_id("v_{}".format(j)), as10_clauses)                    # as (10)
            clauses = self.add_to_clauses(clauses, as10_clauses, tag="(10)")

            as11_clauses = CardEnc.equals(lits=arjs, bound=0, encoding=EncType.seqcounter).clauses
            new_clauses, change = self.register_dummy_variables(arjs, as11_clauses, "va0_{}".format(j))
            if change:
                #print(BRIGHT_RED + "new variables added." + RESET)
                as11_clauses = new_clauses
            as11_clauses = implies_to_cnf(self.get_var_id("v_{}".format(j)), as11_clauses)                    # as (11)
            clauses = self.add_to_clauses(clauses, as11_clauses, tag="(11)")
        
        if debug:
            print(RED + "Decision Tree Related." + RESET)
            print(self._var_map)
            print(clauses)
        return clauses


    def add_example_constraints(self, examples, debug=True):
        # Based on 3.2 Computing Decision Trees with SAT [Naro18]
        # K : # of features, N : # of nodes, M : # of data

        # a_r,j    : 1 iff feature f_r is assigned to node j, r = 1~K, j = 1~N
        # u_r,j    : 1 iff feature f_r is being discriminated against by node j, r = 1~K, j = 1~N
        # d^0_r,j  : 1 iff feature f_r is discriminated for value 0 by node j, or by one of its ancestors, r = 1~K, j = 1~N
        # d^1_r,j  : 1 iff feature f_r is discriminated for value 1 by node j, or by one of its ancestors, r = 1~K, j = 1~N
        # c_j      : 1 iff class of leaf node j is 1, j = 1~N
        # -------------------------------------------------------------------------------------------------------------------
        # v_j and ~c_j -> OR^K_{r=1} d^{\sigma(r,q)}_{r,j}     as (12)
        # i.e. any positive example must be discriminated if the leaf node is associated with the negative class
        # v_j and c_j  -> OR^K_{r=1} d^{\sigma(r,q)}_{r,j}     as (13)
        # i.e. any negative example must be discriminated if the leaf node is associated with the positive class
        clauses = []


        for i, example in enumerate(examples):
            #positive and negative
            L_q, c_q = example
            for j in range(1, self._n + 1):
                exprs = None
                for _r, f_r in enumerate(L_q):
                    r = _r + 1
                    if f_r:
                        sigma_r_q  = 1
                    else:
                        sigma_r_q  = 0
                    exprs = update_clause(exprs, symbols("d{}_{}_{}".format(sigma_r_q, r, j)), Or)
                if c_q:     # Positive Example (1)
                    clauses = self.add_to_clauses(clauses, self.cnf_to_clauses(
                        to_cnf(Implies(parse_expr("v_{} & ~c_{}".format(j, j)), exprs))), tag="(12)-{}-{}".format(i, j))                    # as (12)
                else:       # Negative Example (0)
                    clauses = self.add_to_clauses(clauses, self.cnf_to_clauses(
                        to_cnf(Implies(parse_expr("v_{} & c_{}".format(j, j)), exprs))), tag="(13)-{}-{}".format(i, j))                     # as (13)
        if debug:
            print(RED + "Example Related." + RESET)
            print(self._var_map)
            print(clauses)
        return clauses
        

    def add_inference_constraints(self, debug=True):
        # Based on 3.3 Additional Inference Constraints [Naro18]
        # K : # of features, N : # of nodes, M : # of data
        # lda_t,i denote # of leaf nodes until node i
        # 1. lda_0, i = 1, for 1 <= i <= N
        # 2. lda_t, i <-> (lda_t, i-1 | lda_t-1,i-1 and v_i),   i = 1..N, t = 1..|_i/2_|
        # tau_t,i denote # of non-leaf nodes until node i
        # 1. tau_0, i = 1, for 1 <= i <= N
        # 2. tau_t, i <-> (tau, i-1 | tau_t-1,i-1 and ~v_i),   i = 1..N, t = 1..i
        # Pro. 2 (Refine upper bound on descendants' numbers).
        # If lda_t,i = 1 with 0 < t <= |_i/2_|, then l_i,2(i-t+1) = 0, r _i,2(i-t+1)+1 = 0
        # Pro. 3 (Refine lower bound on descendants' numbers).
        # If tau_t,i = 1 with |-i/2-| < t <= i, then l_i,2(t-1) = 0, r _i,2(i-t+1)+1 = 0
        clauses = []
        for t in range(0, self._n+1):
            lda_id = self.register("lda_{}_{}".format(t, 0))
            clauses = add_to_clauses(clauses, [[-lda_id]])
            tau_id = self.register("tau_{}_{}".format(t, 0))
            clauses = add_to_clauses(clauses, [[-tau_id]])
        for i in range(1, self._n+1):
            lda_id = self.register("lda_{}_{}".format(0, i))
            tau_id = self.register("tau_{}_{}".format(0, i))
            for t in range(1, i // 2 + 1):
                self.register("lda_{}_{}".format(t, i))
            for t in range(1, i + 1):
                self.register("tau_{}_{}".format(t, i))
            clauses = add_to_clauses(clauses, [[lda_id]])
            clauses = add_to_clauses(clauses, [[tau_id]])
            for t in range(1, i // 2 + 1):
                clauses = add_to_clauses(clauses, self.cnf_to_clauses(
                    "Equivalent(lda_{}_{}, lda_{}_{} | (lda_{}_{} & ~v_{}))".format(t, i, t, i-1, t-1, i-1, i)), tag="pro2")
            for t in range(1, i + 1):
                clauses = add_to_clauses(clauses, self.cnf_to_clauses(
                    "Equivalent(tau_{}_{}, tau_{}_{} | (tau_{}_{} & ~v_{}))".format(t, i, t, i-1, t-1, i-1, i)), tag="pro2")
            for t in range(1, i // 2 + 1):
                if (2*(i-t+1)) in self.LR(i) and (2*(i-t+1)+1) in self.RR(i):
                    clauses = add_to_clauses(clauses, self.cnf_to_clauses(
                        "lda_{}_{} >> (~l_{}_{} and ~r_{}_{})".format(t, i, i, 2*(i-t+1), i, 2*(i-t+1)+1)), tag="pro3")
            for t in range(math.ceil(i / 2), i + 1):
                if (2*(t-1)) in self.LR(i) and (2*t-1) in self.RR(i):
                    clauses = add_to_clauses(clauses, self.cnf_to_clauses(
                        "tau_{}_{} >> (~l_{}_{} and ~r_{}_{})".format(t, i, i, 2*(t-1), i, 2*t-1)), tag="pro3")
        return clauses

    
    def check_tree_related_cnfs(self, truth_table):
        if self._tree_related_cnfs is not None:
            _check_list_of_cnfs(self._tree_related_cnfs, truth_table)
        else:
            raise Exception("Tree Related CNFs not initialized.")

    def check_decision_tree_related_cnfs(self, truth_table):
        if self._decision_tree_related_cnfs is not None:
            _check_list_of_cnfs(self._decision_tree_related_cnfs, truth_table)
        else:
            raise Exception("Decision Tree Related CNFs not initialized.")

    def check_example_constraints_cnfs(self, truth_table, debug=True):
        if self._example_constraints_cnfs is not None:
            _check_list_of_cnfs(self._example_constraints_cnfs, truth_table, debug)
        else:
            raise Exception("Tree Related CNFs not initialized.")
 
    def generate(self, examples, debug=True):
        self.cnf = CNF() # And of ORs (A or B) and (C or D)
        if debug:
            print(CYAN + "generating cnfs ===== for {} size".format(self._n) + RESET)
        self._tree_related_cnfs = self.add_tree_related_cnfs()
        self._decision_tree_related_cnfs = self.add_decision_tree_related_cnfs()
        self._example_constraints_cnfs = self.add_example_constraints(examples)
        self.cnf.extend(self._tree_related_cnfs)
        self.cnf.extend(self._decision_tree_related_cnfs)
        self.cnf.extend(self._example_constraints_cnfs) 
        #self.cnf.extend(self.add_inference_constraints())

        return self.cnf.clauses

@hydra.main(config_path="config", config_name="cnf_generator_test")
def main(cfg):
    logging.info("Working directory : {}".format(os.getcwd()))
    formulas = CardEnc.atmost(lits=[1,2,3], bound=1, encoding=EncType.seqcounter).clauses
    print(formulas)
    s = Solver(name='g4')
    s.append_formula(formulas)
    result = s.solve()
    print(result)
    for m in s.enum_models():
        print(m)

if __name__ == "__main__":
    main()