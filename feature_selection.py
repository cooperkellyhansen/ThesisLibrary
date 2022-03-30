""" @author: Sam Parry u1008557 """

import numpy as np
import pandas as pd
from time import time


from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.explicit_regression import ExplicitTrainingData
from bingo.symbolic_regression.explicit_regression import ExplicitRegression
from bingo.stats.pareto_front import ParetoFront

# Global parameters
POP_SIZE = 300
STACK_SIZE = 20
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.4
MAX_GENS = 1_000
FIT_TOL = 1e-3
use_simplification = True
regression_metric = 'mse'
clo_algorithm = 'lm'
# If I want a scalar fitness -> BFGS
# If I want a vectorized fitness -> lm (faster)

def agraph_similarity(ag_1, ag_2):
    """a similarity metric between agraphs"""
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

def get_generators(x_data, stack_size: int, use_simplification: bool):
    """
    Create and return the agraph, crossover, and mutation generators.
    :param x_data: numpy array of training data.
    :param stack_size: Maximum stack size for AGraph.
    :param use_simplification: Use simplification in AGraphGenerator.
    :return: agraph_gen, crossover, mutation
    """
    component_generator = ComponentGenerator(x_data.shape[1], constant_probability=0.1, num_initial_load_statements=1)
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("/")
    component_generator.add_operator("cos")
    component_generator.add_operator("sin")
    component_generator.add_operator("**")
    agraph_gen = AGraphGenerator(stack_size, component_generator=component_generator,
                                 use_simplification=use_simplification)
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)
    return agraph_gen, crossover, mutation


def print_pareto_front(pareto_front):
    """
    Print the members of the pareto front in sympy format.
    """
    print(" FITNESS   COMPLEXITY    EQUATION")
    for member in pareto_front:
        eq = member.get_formatted_string("sympy")
        print("%.3e     " % member.fitness, member.get_complexity(), "     f(X_0) =", eq)

def main(X,y):
    # Build x/y data
    x_data, y_data = get_training_data('data.xlsx')
    X_0, X_1, X_2, X_3, X_4 = x_data.T

    #scale y


    # add super feature(s) here


    # Agraph generation/variation
    agraph_gen, crossover, mutation = get_generators(x_data, STACK_SIZE, use_simplification)

    # Explicit evaluation & CLO
    training_data = ExplicitTrainingData(X, y)
    fitness = ExplicitRegression(training_data=training_data, metric=regression_metric)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm=clo_algorithm)
    # local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm=clo_algorithm,
    #                                                   param_init_bounds=(0, 0), tol=100)
    evaluator = Evaluation(local_opt_fitness)

    # Evolution
    ea = AgeFitnessEA(evaluator, agraph_gen, crossover, mutation,
                      CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, POP_SIZE)
    pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                               similarity_function=agraph_similarity)
    t = time()
    island = Island(ea, agraph_gen, POP_SIZE, hall_of_fame=pareto_front)
    island.evolve_until_convergence(max_generations=MAX_GENS, fitness_threshold=FIT_TOL)
    print(f'Elapsed Time: {round((time() - t) / 60, 2)}min')
    print_pareto_front(pareto_front)

if __name__ == '__main__':
    main()