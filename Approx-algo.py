import gurobipy as grb
from gurobipy import GRB
import numpy as np
import pandas as pd

#Reading the CSV
def read_orders_from_csv(file_path):
    orders_df = pd.read_csv(file_path)
    print(f"Initial number of orders: {len(orders_df)}")
    
    orders_df = orders_df.dropna(subset=['Weight (lbs)', 'Volume (in3)', 'Pallets'], how='any')
    orders_df['Weight (lbs)'] = pd.to_numeric(orders_df['Weight (lbs)'], errors='coerce')
    orders_df['Volume (in3)'] = pd.to_numeric(orders_df['Volume (in3)'], errors='coerce')
    orders_df['Pallets'] = pd.to_numeric(orders_df['Pallets'], errors='coerce')
    orders_df = orders_df.dropna(subset=['Weight (lbs)', 'Volume (in3)', 'Pallets'], how='any')
    
    weights = orders_df['Weight (lbs)'].values
    volumes = orders_df['Volume (in3)'].values
    pallets = orders_df['Pallets'].values
    num_orders = len(orders_df)
    
    return weights, volumes, pallets, num_orders
#BFD get the return matrix
def best_fit_decreasing(weights, volumes, pallets, num_orders):
    MAX_WEIGHT = 45000
    MAX_VOLUME = 3600
    MAX_PALLETS = 60
    
    orders = [(w, v, p, i) for i, (w, v, p) in enumerate(zip(weights, volumes, pallets))]
    orders.sort(key=lambda x: (x[0]/MAX_WEIGHT + x[1]/MAX_VOLUME + x[2]/MAX_PALLETS), reverse=True)
    
    containers = []
    
    for order in orders:
        weight, volume, pallets, orig_idx = order
        best_fit = -1
        min_waste = float('inf')
        
        for i in range(len(containers)):
            rem_weight, rem_volume, rem_pallets, _ = containers[i]
            if (rem_weight >= weight and rem_volume >= volume and rem_pallets >= pallets):
                waste = (rem_weight/MAX_WEIGHT + rem_volume/MAX_VOLUME + rem_pallets/MAX_PALLETS)
                if waste < min_waste:
                    min_waste = waste
                    best_fit = i
        
        if best_fit >= 0:
            containers[best_fit][0] -= weight
            containers[best_fit][1] -= volume
            containers[best_fit][2] -= pallets
            containers[best_fit][3][orig_idx] = 1
        else:
            binary_array = np.zeros(num_orders, dtype=int)
            binary_array[orig_idx] = 1
            containers.append([MAX_WEIGHT-weight, MAX_VOLUME-volume, MAX_PALLETS-pallets, binary_array])
    
    pattern_matrix = np.zeros((num_orders, len(containers)))
    for j, container in enumerate(containers):
        pattern_matrix[:, j] = container[3]
    
    return pattern_matrix

def find_exact_solution(weights, volumes, pallets, target_containers):
    num_orders = len(weights)
    m = grb.Model()
    
    # Make Gurobi search harder
    m.setParam('MIPGap', 0)        # Accept 2% gap
    m.setParam('MIPFocus', 1)         # Focus on finding solutions
    m.setParam('Cuts', 2)             # Moderate cuts
    m.setParam('Heuristics', 1)       # Standard heuristics
    m.setParam('TimeLimit', 600)     # 15min limit
    
    # Binary variables: x[i,j] = 1 if order i is in container j
    x = m.addMVar((num_orders, target_containers), vtype=GRB.BINARY)
    
    # Each order must be in exactly one container
    for i in range(num_orders):
        m.addConstr(x[i,:].sum() == 1)
    
    # Capacity constraints for each container
    for j in range(target_containers):
        m.addConstr(weights @ x[:,j] <= 45000)  # Weight capacity
        m.addConstr(volumes @ x[:,j] <= 3600)   # Volume capacity
        m.addConstr(pallets @ x[:,j] <= 60)     # Pallet capacity
    
    # Objective: minimize sum of used containers (though it will always be target_containers)
    m.setObjective(1, GRB.MINIMIZE)  # Dummy objective since we just want feasibility
    
    m.optimize()
    
    if m.Status == GRB.OPTIMAL:
        solution_matrix = np.zeros((num_orders, target_containers))
        for i in range(num_orders):
            for j in range(target_containers):
                solution_matrix[i,j] = x[i,j].X
        return True, solution_matrix
    else:
        return False, None

def save_solution_to_csv(solution_matrix, weights, volumes, pallets, orders_df, filename="solution_details.csv"):
    WEIGHT_CAPACITY = 45000
    VOLUME_CAPACITY = 3600
    PALLET_CAPACITY = 60

    # Create lists to store data
    container_data = []

    for j in range(solution_matrix.shape[1]):
        orders_in_container = np.where(solution_matrix[:, j] == 1)[0]
        order_numbers = orders_df.iloc[orders_in_container]['Order Number'].tolist()
        
        # Calculate weights, volumes, and pallets used
        weight_used = sum(weights[orders_in_container])
        volume_used = sum(volumes[orders_in_container])
        pallets_used = sum(pallets[orders_in_container])

        weight_percent = (weight_used / WEIGHT_CAPACITY) * 100
        volume_percent = (volume_used / VOLUME_CAPACITY) * 100
        pallet_percent = (pallets_used / PALLET_CAPACITY) * 100

        # Append container data
        container_data.append({
            'Container_Number': j + 1,
            'Orders': str(order_numbers),  # Saving order numbers as a string
            'Weight_Used': weight_used,
            'Volume_Used': volume_used,
            'Pallets_Used': pallets_used,
            'Weight_Utilization_%': weight_percent,
            'Volume_Utilization_%': volume_percent,
            'Pallet_Utilization_%': pallet_percent,
            'Remaining_Weight': WEIGHT_CAPACITY - weight_used,
            'Remaining_Volume': VOLUME_CAPACITY - volume_used,
            'Remaining_Pallets': PALLET_CAPACITY - pallets_used
        })
    df = pd.DataFrame(container_data)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    file_path = "filtered_orders_bad_round3.csv"
    weights, volumes, pallets, num_orders = read_orders_from_csv(file_path)
    orders_df = pd.read_csv(file_path)
    CHUNK_SIZE = 1000


    initial_solutions = {}  # Dictionary to store solutions dynamically
    final_solutions = [] 
    chunk_dfs = []
    
    if num_orders > CHUNK_SIZE:
        # Split dataset into chunks of size 250
        order_indices = np.arange(num_orders)
        chunks = [order_indices[i:i + CHUNK_SIZE] for i in range(0, num_orders, CHUNK_SIZE)]
        
        total_containers = 0
        total_containers_approx = 0
        for idx, chunk in enumerate(chunks, start=1):
            chunk_weights = weights[chunk]
            chunk_volumes = volumes[chunk]
            chunk_pallets = pallets[chunk]
            chunk_orders_df = orders_df.iloc[chunk]
            
            # Apply BFD to the current chunk
            initial_solution = best_fit_decreasing(chunk_weights, chunk_volumes, chunk_pallets, len(chunk))
            
            # Dynamically name the solution
            key_name = f"initial_solution_chunk{idx}"
            initial_solutions[key_name] = initial_solution
            
            print(f"Chunk {idx} ({chunk[0]}-{chunk[-1]}) uses {initial_solution.shape[1]} containers")
            total_containers += initial_solution.shape[1]

            # Attempt to find an exact solution
            target_containers = initial_solution.shape[1] - 1
            solution_found = False  # Track if any feasible solution is found
            
            while True:
                print(f"\nTrying to find solution with {target_containers} containers...")
                feasible, solution = find_exact_solution(chunk_weights, chunk_volumes, chunk_pallets, target_containers)

                if feasible:
                    print(f"Found a feasible solution with {target_containers} containers.")
                    solution_old = solution  # Save the last feasible solution
                    target_containers -= 1
                    solution_found = True
                else:
                    break

            # Save the best solution found for the chunk
            if solution_found:
                save_solution_to_csv(
                    solution_old, chunk_weights, chunk_volumes, chunk_pallets, chunk_orders_df,
                    f'solution_details_Approx_chunk{idx}_optimal_round4.csv'
                )
                final_solutions.append(solution_old)  # Append the last feasible solution
                chunk_dfs.append(chunk_orders_df)      # Keep track of the chunk's dataframe
                print(f"\nSolution details for chunk {idx} saved and used containers are {target_containers + 1}")
                total_containers_approx += target_containers + 1
            else:
                # If no feasible solution was found, save the initial BFD solution
                print(f"\nNo exact solution found for chunk {idx}. Saving initial BFD solution.")
                save_solution_to_csv(
                    initial_solution, chunk_weights, chunk_volumes, chunk_pallets, chunk_orders_df,
                    f'solution_details_Approx_chunk{idx}_initial_round4.csv'
                )
                final_solutions.append(initial_solution)
                chunk_dfs.append(chunk_orders_df)
                total_containers_approx += initial_solution.shape[1]

    else:
        
        initial_solution = best_fit_decreasing(weights, volumes, pallets, num_orders)
        print(f"BFD uses {initial_solution.shape[1]} containers")
        target_containers = initial_solution.shape[1] - 1
        while True:
            print(f"\nTrying to find solution with {target_containers} containers...")
            feasible, solution = find_exact_solution(weights, volumes, pallets, target_containers)

            if feasible:
                print(f"\nFound optimal solution!")
                save_solution_to_csv(solution, weights, volumes, pallets, orders_df,
                                 f'solution_details_filtered_Finalround_{target_containers}_new.csv')
                print(f"\nSolution details saved to solution_details_{target_containers}.csv")
                target_containers -= 1
            
            else:
                print("\nNo feasible solution found")
                save_solution_to_csv(solution, weights, volumes, pallets, orders_df,
                                 f'solution_details_filtered_Finalround_{target_containers}_new.csv')
                print(f"\nSolution details saved to solution_details_{target_containers}.csv")
                break
