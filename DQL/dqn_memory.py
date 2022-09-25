from initialize import make_sol, NEH_heuristic, extract_taillard, extract_vrf
from search import perturbation_mechanism, local_search


def initial_solution(dataset: object, inst_number: object) -> object:
    # Determine which instance to execute
    if dataset == 'taillard':
        processing_times = extract_taillard(inst_number)
    elif dataset == 'vrf_small':
        processing_times = extract_vrf(inst_number, 'small')
    elif dataset == 'vrf_large':
        processing_times = extract_vrf(inst_number, 'large')

    current_solution = make_sol(processing_times)
    current_solution = NEH_heuristic(current_solution, True, True, True, 'insertion_neighborhood')
    current_makespan = current_solution.makespan

    return current_solution, current_makespan


def permute_solution(action, solution):
    action += 1
    reward_before = solution.makespan
    new_solution = perturbation_mechanism(solution, action)

    # local search on the solution
    local_search(new_solution, new_solution, ref_best=False, until_no_improvement=True, tie_breaking=False, method='insertion_neighborhood')
    # next state and reward:
    if new_solution.makespan <= reward_before:
        reward = (reward_before - new_solution.makespan)
        state_next = [0, 1]
    else:
        reward = 0
        state_next = [1, 0]
    return state_next, reward, new_solution, new_solution.makespan
