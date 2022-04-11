from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file

import sys

chkpt = sys.argv[-1]

# read bingo pickle file and retrieve best model
archipelago = load_parallel_archipelago_from_file(chkpt)
best_ind = archipelago.get_best_individual()
print('%.3e    ' % best_ind.fitness, best_ind.get_complexity(),
                '   f(X_0) =', best_ind)

hof = archipelago.hall_of_fame
print("  FITNESS    COMPLEXITY    EQUATION")
for member in hof:
    a = str(member)
    a = a.split(")(")
    a = ")*(".join(a)
    print('%.3e    ' % member.fitness, member.get_complexity(),
            '   f(X_0) =', a)

print()
print()
print(hof[5])

