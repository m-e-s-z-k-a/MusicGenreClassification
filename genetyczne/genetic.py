from random import choices, randint, randrange, random, sample
from typing import List, Optional, Callable, Tuple

Genome = List[int]
Population = List[Genome]
PopulateFunction = Callable[[], Population]
FitnessFunction = Callable[[Genome], int]
SelectionFunction = Callable[[Population, FitnessFunction], Tuple[Genome, Genome]]
CrossoverFunction = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunction = Callable[[Genome], Genome]
PrintFunction = Callable[[Population, int, FitnessFunction], None]

def genome_generator(length) -> Genome:
    return choices([0, 1], k = length)

def population_generator(size, genome_lenght) -> Population:
    return [genome_generator(genome_lenght) for _ in range(size)]

def mutation(genome, num, probability) -> Genome:
    for i in range(0, num, 1):
        index = randrange(len(genome))
        if random() <= probability:
            genome[index] = abs(genome[index] - 1)
    return genome

def population_fitness(population, fitness_function):
    retval = 0
    for i in range(0, len(population), 1):
        retval += fitness_function(population[i])
    return retval

def sp_crossover(a, b) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("genomes not of the same lenght!")
    else:
        length = len(a)
        if length < 2:
            return a, b
        else:
            r = randint(1, length - 1)
            return a[0:r] + b[r:], b[0:r] + a[r:]

def weighted_distribution_generator(population, fitness_function) -> Population:
    result = []
    for i in range(0, len(population), 1):
        result.append([population[i]] * int(fitness_function(population[i])+1))
    return result

def pair_selection(population, fitness_function) -> Population:
    return sample(population = weighted_distribution_generator(population, fitness_function), k = 2)

def sort_population(population, fitness_function) -> Population:
    return sorted(population, key = fitness_function, reverse = True)

def genome_to_string(genome):
    return "".join(map(str, genome))

def print_stats(population, id, fitness_function):
    print("GENERATION", id)
    print("--------------------------------")
    #TODO: co to ma robic ? poprawic
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_function) / len(population)))
    sorted_population = sort_population(population, fitness_function)
    print(
        "Best: %s (%f)" % (genome_to_string(sorted_population[0]), fitness_function(sorted_population[0])))
    print("Worst: %s (%f)" % (genome_to_string(sorted_population[-1]),
                              fitness_function(sorted_population[-1])))
    print("")

    return sorted_population[0]


def evolve(populate_function, fitness_function, fitness_limit, selection_function = pair_selection,
           crossover_function = sp_crossover, mutation_function = mutation, gen_lim = 100,
           printer: Optional[PrintFunction] = None) -> Tuple[Population, int]:
    population = populate_function()
    i = 0
    for i in range(0, gen_lim, 1):
        population = sorted(population, key = lambda genome: fitness_function(genome), reverse=True)
        if printer:
            printer(population, i, fitness_function)

        if fitness_function(population[0]) >= fitness_limit:
            break

        next_gen = population[0:2]
        for j in range(int(len(population) / 2) - 1):
            parents = selection_function(population, fitness_function)
            child1, child2 = crossover_function(parents[0], parents[1])
            child1 = mutation_function(child1)
            child2 = mutation_function(child2)
            next_gen += [child1, child2]

        population = next_gen

    return population, i
