from random import shuffle
import random
import copy


def perturbation_mechanism(current_solution, action):
    # Distruction phase
    num_jobs_remove = action
    new_solution = current_solution

    removed_jobs = random.sample(current_solution.sequence, num_jobs_remove)
    new_solution.sequence = [job for job in current_solution.sequence if job not in removed_jobs]

    # Construction phase
    for job in removed_jobs:
        new_solution.insert_in_best_position(job, True)

    return new_solution


def local_search(solution, best_solution, ref_best=False, until_no_improvement=True, tie_breaking=False, method='insertion_neighborhood'):

    if method == 'insertion_neighborhood':
        insertion_neighborhood(solution, best_solution, ref_best, until_no_improvement, tie_breaking)
    else:
        pass


def insertion_neighborhood(solution, best_solution, ref_best=False, until_no_improvement=True, tie_breaking=False):
    """ removes jobs one by one randomly and insert them in their best possible position """

    current_makespan = solution.makespan
    improve = True
    best_sequence = copy.deepcopy(solution.sequence)
    best_makespan = current_makespan

    while improve:
        improve = False

        # If jobs are removed based on a reference sequence (e.g., best found solution so far)
        if ref_best:
            not_tested = best_solution.sequence.copy()
        else:
            not_tested = solution.sequence.copy()
            shuffle(not_tested)

        for removed_job in not_tested:
            solution.sequence.remove(removed_job)

            solution.insert_in_best_position(removed_job, tie_breaking)

            if solution.makespan < current_makespan:
                improve = True
                current_makespan = solution.makespan
                best_sequence = copy.deepcopy(solution.sequence)
                best_makespan = current_makespan

        if until_no_improvement == False:
            break
    solution.sequence = best_sequence.copy()
    solution.makespan = best_makespan

