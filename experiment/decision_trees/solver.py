import hydra
import logging
import os
from pysat.solvers import Solver # , Glucose4
from pysat.formula import CNF # , WCNF
from .cnf_generator import CNFGenerator 
from experiment.decision_trees.utils.tree_visualizer import plot_tree, display_truth_table

class DecisionTreeSATSolver(object):
    def __init__(self, cfg):
        pass

    def solve(self):
        # Output decision trees from the smallest to the larger ones
        # Save them in a file
        n = 11 # has to be an odd number
        n_features = 4
        n_sub_features = 4
        example_size = 10
        s = Solver(name='g4')

        cnf_gen = CNFGenerator(n, n_features * n_sub_features, example_size)
        s.append_formula(cnf_gen.generate())

        result = s.solve()
        print(result)
        #print(s.get_model())
        counter = 0
        prev_edge_tables = []
        for m in s.enum_models():
            print(m)
            display_truth_table(cnf_gen.var_map, m)
            edge_table = plot_tree("tree_" + str(counter), n, cnf_gen.var_map, cnf_gen.var_list, m, prev_edge_tables)
            if edge_table is not None:
                prev_edge_tables += [edge_table]
            counter += 1
        
        s.delete()

@hydra.main(config_path="config", config_name="solver")
def main(cfg):
    logging.info("Working directory : {}".format(os.getcwd()))
    solver = DecisionTreeSATSolver(cfg)    
    solver.solve()

if __name__ == "__main__":
    main()