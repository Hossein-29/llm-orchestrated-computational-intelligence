import lkh
import pulp

def compute_tour_length(tour, dist_matrix):
  """Calculates the total distance of a given tour."""
  length = sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1))
  length += dist_matrix[tour[-1], tour[0]]
  return length


def solve_tsp_lkh(dist_matrix):
  """
  Uses the Lin-Kernighan-Helsgaun heuristic via the LKH-3 solver.
  Suitable for N > 500.
  """
  matrix_as_list = dist_matrix.tolist()
  
  solver_res = lkh.solve(matrix_as_list, dimension=len(dist_matrix), runs=10)
  
  best_tour = solver_res[0]
  best_length = compute_tour_length(best_tour, dist_matrix)
  
  return best_length, best_tour


def solve_tsp_ilp(dist_matrix):
    n = len(dist_matrix)
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)
    
    # Binary variable: x[i][j] is 1 if we go from city i to city j
    x = pulp.LpVariable.dicts("x", (range(n), range(n)), cat='Binary')
    
    # Objective: Minimize total distance
    prob += pulp.lpSum(dist_matrix[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)
    
    # Constraints: Enter and leave each city exactly once
    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(n) if i != j) == 1
        prob += pulp.lpSum(x[j][i] for j in range(n) if i != j) == 1
        
    # Subtour elimination (MTZ constraints) - This is the "magic" part
    u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n-1, cat='Continuous')
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i][j] <= n - 1
                
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.value(prob.objective)
