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
from experiment.decision_trees.utils.ansi_colors import BRIGHT_RED, RESET, GREEN


def output_cnf(cnf_str):
    print(BRIGHT_RED, cnf_str, RESET)


def implies_to_cnf(left_id, right_cnfs):
    result_cnfs = []
    for cnf in right_cnfs:
        result_cnfs += [[-left_id] + cnf]
    return result_cnfs


def add_to_clauses(left_clauses, right_clauses):
    processed_clauses = []
    print(GREEN, right_clauses, RESET)
    if any(isinstance(el, list) for el in right_clauses):
        processed_clauses += right_clauses
    else:
        processed_clauses = right_clauses
    return left_clauses + processed_clauses


def update_clause(left_clause, right_clause, operation):
    if left_clause is None:
        return right_clause
    else:
        return operation(left_clause, right_clause)


class CNFGenerator(object):
    def __init__(self, tree_size, feature_size, example_size):
        self._n = tree_size
        self._k = feature_size
        self._m = example_size
        self._counter = 0
        self._var_list = []
        self._var_map = dict()

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
        assert var_name in self._var_map
        return self._var_map[var_name]

    def get_var_name(self, var_id):
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
        clauses = add_to_clauses(clauses, [[-self.get_var_id("v_1")]])                                        # as (1)
        for i in range(1, self._n + 1):
            lr_ids = []
            for j in self.LR(i):                
                lr_ids.append(self.register("l_{}_{}".format(i,j)))
                #print(to_cnf("v_{} >> ~l_{}_{}".format(i, i, j)))
                clauses = add_to_clauses(clauses, self.cnf_to_clauses("v_{} >> ~l_{}_{}".format(i, i, j)))   # as (2)                
            # rr_ids = []
            # for j in self.RR(i):                
            #     rr_ids.append(self.register("r_{}_{}".format(i,j)))
            #     #print(to_cnf("v_{} >> ~l_{}_{}".format(i, i, j)))
            #     clauses = add_to_clauses(clauses, self.cnf_to_clauses("v_{} >> ~r_{}_{}".format(i, i, j)))   # as (2)
            as4_clauses = CardEnc.equals(lits=lr_ids, bound=1, encoding=EncType.seqcounter).clauses
            new_clauses, change = self.register_dummy_variables(lr_ids, as4_clauses, "v_{}".format(i))
            if change:
                #print(BRIGHT_RED + "new variables added." + RESET)
                as4_clauses = new_clauses
            as4_clauses = implies_to_cnf(-self.get_var_id("v_{}".format(i)), as4_clauses)                    # as (4)
            clauses = add_to_clauses(clauses, as4_clauses)
            if debug:
                print("lr_ids : i={}, {} {}".format(i, lr_ids, self.get_var_names(lr_ids)))
                print("as 4 : ", as4_clauses)
            for j in self.RR(i):
                self.register("r_{}_{}".format(i,j))                
            for j in self.LR(i):
                clauses = add_to_clauses(clauses, self.cnf_to_clauses("Equivalent(l_{}_{}, r_{}_{})".format(i, j, i, j+1))) # as (3)
        for j in range(2, self._n+1):
            p_jis = []
            for i in range(j // 2, min(j, self._n+1)):
                p_jis += [self.register("p_{}_{}".format(j,i))]
                lr_i = self.LR(i)
                rr_i = self.RR(i)
                if j in lr_i:
                    clauses = add_to_clauses(clauses, self.cnf_to_clauses("Equivalent(p_{}_{}, l_{}_{})".format(j, i, i, j))) # as (5)
                elif j in rr_i:
                    clauses = add_to_clauses(clauses, self.cnf_to_clauses("Equivalent(p_{}_{}, r_{}_{})".format(j, i, i, j))) # as (5)
            as6_clauses = CardEnc.equals(lits=p_jis, bound=1, encoding=EncType.seqcounter).clauses
            new_clauses, change = self.register_dummy_variables(p_jis, as6_clauses, "p_{}_is".format(j))
            print(BRIGHT_RED, " + ".join(self.get_var_names(p_jis)) + " = 1", RESET)
            if change:
                #print(BRIGHT_RED +  " new variables added." + RESET)
                as6_clauses = new_clauses
            clauses = add_to_clauses(clauses, as6_clauses)  # as (6)
                
        if debug:
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
            clauses = add_to_clauses(clauses, [[-self.get_var_id("d0_{}_1".format(r))]])       # as (7)
            clauses = add_to_clauses(clauses, [[-self.get_var_id("d1_{}_1".format(r))]])       # as (8)
            for j in range(1, self._n + 1):
                d0_rj = symbols('d0_{}_{}'.format(r, j))
                d1_rj = symbols('d1_{}_{}'.format(r, j))
                exprs_0 = None
                exprs_1 = None
                exprs_9_0 = None
                exprs_9_1 = None
                for i in range(j // 2, j - 1 + 1):
                    expr_0 = parse_expr("(p_{}_{} & d0_{}_{}) | (a_{}_{} & r_{}_{})".format(j, i, r, i, r, i, i, j))
                    expr_1 = parse_expr("(p_{}_{} & d1_{}_{}) | (a_{}_{} & l_{}_{})".format(j, i, r, i, r, i, i, j))
                    expr_9_0 = parse_expr("(u_{}_{} & p_{}_{} >> ~a_{}_{})".format(r, i, j, i, r, j))
                    expr_9_1 = parse_expr("(u_{}_{} & p_{}_{})".format(r, i, j, i))
                    exprs_0 = update_clause(exprs_0, expr_0, Or)
                    exprs_1 = update_clause(exprs_1, expr_1, Or)
                    exprs_9_0 = update_clause(exprs_9_0, expr_9_0, And)
                    exprs_9_1 = update_clause(exprs_9_1, expr_9_1, Or)
                if exprs_0 is not None:
                    clauses = add_to_clauses(clauses, self.cnf_to_clauses(Equivalent(symbols("d0_{}_{}".format(r, j)), exprs_0))) # as (7)
                if exprs_1 is not None:
                    clauses = add_to_clauses(clauses, self.cnf_to_clauses(Equivalent(symbols("d1_{}_{}".format(r, j)), exprs_1))) # as (8)
                if exprs_9_0 is not None:
                    clauses = add_to_clauses(clauses, self.cnf_to_clauses(exprs_9_0)) # as (9-0)
                exprs_9_1 = update_clause(exprs_9_1, symbols("a_{}_{}".format(r,j)), Or)
                if exprs_9_1 is not None:                    
                    clauses = add_to_clauses(clauses, self.cnf_to_clauses(Equivalent(symbols("u_{}_{}".format(r,j)), exprs_9_1))) # as (9-1)
        
        for j in range(1, self._n + 1):
            arjs = []
            for r in range(1, self._k + 1):
                arjs.append(self.get_var_name("a_{}_{}".format(r,j)))
            as10_clauses = CardEnc.equals(lits=arjs, bound=1, encoding=EncType.seqcounter).clauses
            new_clauses, change = self.register_dummy_variables(arjs, as10_clauses, "va1_{}".format(j))
            if change:
                #print(BRIGHT_RED + "new variables added." + RESET)
                as10_clauses = new_clauses
            as10_clauses = implies_to_cnf(-self.get_var_id("v_{}".format(i)), as10_clauses)                    # as (10)
            clauses = add_to_clauses(clauses, as10_clauses)

            as11_clauses = CardEnc.equals(lits=arjs, bound=0, encoding=EncType.seqcounter).clauses
            new_clauses, change = self.register_dummy_variables(arjs, as11_clauses, "va0_{}".format(j))
            if change:
                #print(BRIGHT_RED + "new variables added." + RESET)
                as11_clauses = new_clauses
            as11_clauses = implies_to_cnf(self.get_var_id("v_{}".format(i)), as11_clauses)                    # as (11)
            clauses = add_to_clauses(clauses, as11_clauses)


    def add_example_constraints(self, examples):
        # Based on 3.2 Computing Decision Trees with SAT [Naro18]
        # K : # of features, N : # of nodes, M : # of data

        # a_r,j    : 1 iff feature f_r is assigned to node j, r = 1~K, j = 1~N
        # u_r,j    : 1 iff feature f_r is being discriminated against by node j, r = 1~K, j = 1~N
        # d^0_r,j  : 1 iff feature f_r is discriminated for value 0 by node j, or by one of its ancestors, r = 1~K, j = 1~N
        # d^1_r,j  : 1 iff feature f_r is discriminated for value 1 by node j, or by one of its ancestors, r = 1~K, j = 1~N
        # c_j      : 1 iff class of leaf node j is 1, j = 1~N
        # --------------------------------------------------------------------------------------------------------------------------------
        # 
        clauses = []

        for example in examples:
            #positive and negative
            L_q, c_q = example
            for j in range(1, self._n + 1):
                exprs = None
                for r, f_r in enumerate(L_q):                
                    if f_r:
                        sigma_r_q  = 1
                    else:
                        sigma_r_q  = 0
                    exprs = update_clause(exprs, symbols("d{}_{}_{}".format(sigma_r_q, r, j)), Or)
                if c_q: # Positive Example (1)
                    clauses = add_to_clauses(clauses, self.cnf_to_clauses(
                        Implies(parse_expr("v_{} & ~c_{}"), exprs))) # as (12)
                else: # Negative Example (0)
                    clauses = add_to_clauses(clauses, self.cnf_to_clauses(
                        Implies(parse_expr("v_{} & c_{}"), exprs))) # as (13)
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
        for i in range(1, self._n+1):
            lda_id = self.register("lda_{}_{}".format(0, i))
            tau_id = self.register("tau_{}_{}".format(0, i))
            for t in range(1, i // 2 + 1):
                self.register("lda_{}_{}".format(0, i))
            for t in range(1, i + 1):
                self.register("tau_{}_{}".format(0, i))
            clauses = add_to_clauses(clauses, [[lda_id]])
            clauses = add_to_clauses(clauses, [[tau_id]])
            for t in range(1, i // 2 + 1):
                clauses = add_to_clauses(clauses, self.cnf_to_clauses(
                    "Equivalent(lda_{}_{}, lda_{}_{} | lda_{}_{} & ~v_{})".format(t, i, t, i-1, t-1, i-1, i)))
            for t in range(1, i + 1):
                clauses = add_to_clauses(clauses, self.cnf_to_clauses(
                    "Equivalent(tau_{}_{}, tau_{}_{} | tau_{}_{} & ~v_{})".format(t, i, t, i-1, t-1, i-1, i)))
            for t in range(1, i // 2 + 1):
                clauses = add_to_clauses(clauses, self.cnf_to_clauses(
                    "lda_{}_{} >> ~l_{}_{} and ~r_{}_{}".format(t, i, i, 2*(i-t+1), i, 2*(i-t+1)+1)))
            for t in range(math.ceil(i / 2), i + 1):
                clauses = add_to_clauses(clauses, self.cnf_to_clauses(
                    "tau_{}_{} >> ~l_{}_{} and ~r_{}_{}".format(t, i, i, 2*(t-1), i, 2*t-1)))
        return clauses
    
    def generate(self, examples):
        self.cnf = CNF() # And of ORs (A or B) and (C or D)
        self.cnf.extend(self.add_tree_related_cnfs())
        self.cnf.extend(self.add_decision_tree_related_cnfs())
        return self.cnf.clauses

@hydra.main(config_path="config", config_name="cnf_generator_test")
def main(cfg):
    logging.info("Working directory : {}".format(os.getcwd()))
    gen = CNFGenerator(5, 4*4, 10)
    #print(gen.generate())    

if __name__ == "__main__":
    main()