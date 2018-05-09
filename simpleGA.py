import random
import string

TARGET = 'Hello World!'
SAMPLE_SPACE = list(string.printable)  # Create list of lower and upper case letters
#  SAMPLE_SPACE = list(string.ascii_letters + ' ')


def generate_chromosome(sample_space, size):
    """
    Generate a random individual, each gene is drawn from sample_space with replacement.

    :param sample_space: list of all characters to search from
    :param size: integer determining the length of the chromosome or individual
    :return: string representing a random chromosome or individual
    """
    return ''.join(random.choices(sample_space, k=size))


def get_candidates(population, candidate_pos):
    """
    Return list of individuals at index candidate_pos in population.

    :param population: list of individuals
    :param candidate_pos: list of index or position in population
    :return: list of candidates from population at position candidate_pos
    """
    candidates = []
    for gene in candidate_pos:
        candidates.append(population[gene])
    return candidates


def get_population(pop_size):
    """
    Calculate an initial set of candidates and their fitness.
    Note element index in each list have to be preserved,
    so that:
    'population[0]=first individual',
    'fitness[0]=fitness first individual'.

    :param pop_size: Number of candidates or individuals in population
    :return: list of population and fitness
    """
    candidate_size = len(TARGET)
    population, fitness = [], []
    for individual in range(pop_size):
        candidate = generate_chromosome(sample_space=SAMPLE_SPACE, size=candidate_size)
        population.append(candidate)
        fitness.append(get_fitness(candidate))
    return population, fitness


def get_fitness(candidate):
    """
    :param candidate: A string of the chromosome or a list of genes in chromosome.
    :return: Integer of single candidate fitness.
    """
    fitness = 0
    for index, gene in enumerate(list(candidate)):  # enumerate returns current index and element in list
        if gene == list(TARGET)[index]:
            fitness += 1
    return fitness


def tournament_selection(fitness, n_tournaments=2, candidate_sample=4):
    """
    Perform n_tour tournaments selecting the best and worst candidate from each tournament.
    Each tournament has a candidate_sample number of individuals, drawn with replacement

    :param fitness: List of all fitness values (integers) for the current population
     and with each index corresponding to candidate index
    :param n_tournaments: Integer number of tournaments to be had
    :param candidate_sample: Integer # of individuals participating in each tournament
    :return Integer list containing indexes of selected candidates
    """
    if n_tournaments % 2 is not 0:  # send out a value error exception with a custom message
        raise ValueError("Number of tournaments have to be even")
    best_candidate_index, worst_candidate_index = [], []
    for tournament_i in range(n_tournaments):
        tmp_candidate_id, tmp_fitness = [], []
        for candidate in range(candidate_sample):
            tmp_candidate_id.append(random.randrange(len(fitness)))
            tmp_fitness.append(fitness[tmp_candidate_id[-1]])  # some_list[-1] returns last element in list
        best_candidate_index.append(tmp_candidate_id[tmp_fitness.index(max(tmp_fitness))])  # Add best candidate id
        # from tournament
        worst_candidate_index.append(tmp_candidate_id[tmp_fitness.index(min(tmp_fitness))])  # Add worst candidate id
        # from tournament
    return best_candidate_index, worst_candidate_index


def crossover(parents_to_breed):
    """
    Perform uniform crossover on parents.
    :param parents_to_breed: List containing two parent strings
    :return: list of tuple with children chromosome as strings
    :raises: IndexError: If the number of parents to breed is not equal to 2, raise error
    """
    if len(parents_to_breed) == 2:
        child_1, child_2, parent_1, parent_2 = [], [], [], []
        [parent_1.append(gene) for gene in chromosome(parents_to_breed[0])]
        [parent_2.append(gene) for gene in chromosome(parents_to_breed[1])]
        for index_gene, gene in enumerate(parent_1):
            if random.random() < 0.5:
                child_1.append(gene)
                child_2.append(parent_2[index_gene])
            else:
                child_1.append(parent_2[index_gene])
                child_2.append(gene)
        return [''.join(child_1), ''.join(child_2)]
    else:
        raise IndexError("Number of parents to breed is not equal to 2")


def mutate(child):
    """
    Mutates chromosome of child with probability prob_mutation
    :param child: string of chromosome
    :return: mutated child as a string
    """
    prob_mutation = 0.01
    mutated_child = []
    for gene in chromosome(child):
        if random.random() > prob_mutation:
            mutated_child.append(gene)
        else:
            mutated_child.append(generate_chromosome(SAMPLE_SPACE, size=1))
    return ''.join(mutated_child)


def chromosome(candidate):
    """
    :param candidate: Candidate solution
    :return: Returns a generator for candidate. A list of the strings or candidate genes
    """
    for gene in candidate:
        for letter in gene:
            yield letter


def breed(population, fitness, n_candidates):
    """
    Main function that create a new generation
    :param population: list of candidate solutions
    :param fitness: list of fitness for the candidate solutions
    :param n_candidates: integer of number of new individuals
    :return: updates the population and the fitness
    """
    try:
        # Best and worst candidate positions from tournaments
        best_cand_id, worst_cand_id = tournament_selection(fitness, n_tournaments=n_candidates,
                                                           candidate_sample=2*n_candidates)
    except ValueError as tournament_selection_error:
        raise tournament_selection_error
    parent_list = get_candidates(population, best_cand_id)
    children, mutated_children = [], []
    for pos, parent in enumerate(parent_list):
        if pos % 2 == 0 and pos < len(parent_list)-1:  # len(list)-2 is maximum pos value - 1
            parents_to_breed = get_candidates(parent_list, [pos, pos+1])
            try:
                children.extend(crossover(parents_to_breed))
                mutated_children.extend(mutate(child) for child in children[-2:])  # [-2:] return the two last children
            except IndexError as message:
                raise message
    " Update population and fitness "
    'pos - position in list'
    'i - current list index/for loop iteration'
    for i, pos in enumerate(worst_cand_id):
        if 'mutated_children' in locals():  # Check so mutated_children exists
            population[pos] = mutated_children[i]  # replace worst candidate with new child
            fitness[pos] = get_fitness(chromosome(mutated_children[i]))  # replace worst candidate fitness with child's
            # fitness


def run():
    random.seed()  # Seed random generator
    population, fitness = get_population(pop_size=200)
    #  print('Population: ', population)
    #  print('Fitness: ', fitness)
    generation = 0
    print('Generation ', generation, '-', population[fitness.index(max(fitness))] + '.', ' Fitness: ', max(fitness))
    while get_fitness(population[fitness.index(max(fitness))]) < len(TARGET):
        generation += 1
        breed(population, fitness, n_candidates=40)
        print('Generation ', generation, '-', population[fitness.index(max(fitness))]+'.', ' Fitness: ', max(fitness))


if __name__ == '__main__':
    run()
