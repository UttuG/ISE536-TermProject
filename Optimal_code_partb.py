import pandas as pd

# Load the CSV data
file_path = 'solution_details_268_new.csv'
data = pd.read_csv(file_path)

# Filter containers where all utilization percentages are below 99%
filtered_containers = data[(data['Weight_Utilization_%'] < 99) & 
                           (data['Volume_Utilization_%'] < 99) & 
                           (data['Pallet_Utilization_%'] < 99)]

print(len(filtered_containers['Orders']))

goal=len(filtered_containers['Orders'])

# Initialize an empty list to store all order numbers
all_order_numbers = []

# Iterate through each row in the filtered containers
for orders in filtered_containers['Orders']:
    # Convert the string representation of list to an actual list
    order_list = eval(orders)
    # Extend the main list with this container's order numbers
    all_order_numbers.extend(order_list)

# Load the CSV data
file_path = 'Term project data 1a - Copy.csv'
data = pd.read_csv(file_path)

# Drop any columns with "Unnamed" in their name
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Filter orders based on the order numbers
filtered_orders = data[data['Order Number'].isin(all_order_numbers)]

# Save filtered orders to a new CSV for use in testing.py
filtered_orders.to_csv('filtered_orders.csv', index=False)


import gurobipy as grb
from gurobipy import GRB
import numpy as np
import pandas as pd

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

def find_exact_solution(weights, volumes, pallets, target_containers):
    num_orders = len(weights)
    m = grb.Model()
    
    # Make Gurobi search harder
    m.setParam('MIPGap', 0)        # Accept 2% gap
    m.setParam('MIPFocus', 1)         # Focus on finding solutions
    m.setParam('Cuts', 1)             # Moderate cuts
    m.setParam('Heuristics', 1)       # Standard heuristics
    m.setParam('TimeLimit', 7200)     # 2-hour limit
    
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

def print_solution_stats(solution_matrix, weights, volumes, pallets):
    WEIGHT_CAPACITY = 45000
    VOLUME_CAPACITY = 3600
    PALLET_CAPACITY = 60
    
    num_containers = solution_matrix.shape[1]
    
    print("\nContainer Utilization Statistics:")
    print("-" * 80)
    print(f"{'Container #':<10} {'Weight %':<12} {'Volume %':<12} {'Pallet %':<12}")
    print("-" * 80)
    
    total_weight_util = 0
    total_volume_util = 0
    total_pallet_util = 0
    
    for j in range(num_containers):
        orders_in_container = np.where(solution_matrix[:,j] == 1)[0]
        
        weight_used = sum(weights[orders_in_container])
        volume_used = sum(volumes[orders_in_container])
        pallets_used = sum(pallets[orders_in_container])
        
        weight_percent = (weight_used / WEIGHT_CAPACITY) * 100
        volume_percent = (volume_used / VOLUME_CAPACITY) * 100
        pallet_percent = (pallets_used / PALLET_CAPACITY) * 100
        
        total_weight_util += weight_percent
        total_volume_util += volume_percent
        total_pallet_util += pallet_percent
        
        print(f"{j+1:<10} {weight_percent:,.1f}%{' ':>4} {volume_percent:,.1f}%{' ':>4} {pallet_percent:,.1f}%")
        print(f"Orders in container: {orders_in_container.tolist()}")
        print(f"Container details:")
        print(f"  Total Weight: {weight_used:,.0f} lbs")
        print(f"  Total Volume: {volume_used:,.0f} in³")
        print(f"  Total Pallets: {pallets_used:,.0f}")
        print("-" * 80)
    
    print(f"\nAverage utilization across all containers:")
    print(f"Weight: {total_weight_util/num_containers:,.1f}%")
    print(f"Volume: {total_volume_util/num_containers:,.1f}%")
    print(f"Pallets: {total_pallet_util/num_containers:,.1f}%")

def print_container_stats(solution_matrix, weights, volumes, pallets):
    WEIGHT_CAPACITY = 45000
    VOLUME_CAPACITY = 3600
    PALLET_CAPACITY = 60
    
    num_containers = solution_matrix.shape[1]
    
    print("\nContainer Utilization Statistics:")
    print("-" * 80)
    print(f"{'Container #':<10} {'Weight %':<12} {'Volume %':<12} {'Pallet %':<12}")
    print("-" * 80)
    
    total_weight_util = 0
    total_volume_util = 0
    total_pallet_util = 0
    
    for j in range(num_containers):
        orders_in_container = np.where(solution_matrix[:,j] == 1)[0]
        
        weight_used = sum(weights[orders_in_container])
        volume_used = sum(volumes[orders_in_container])
        pallets_used = sum(pallets[orders_in_container])
        
        weight_percent = (weight_used / WEIGHT_CAPACITY) * 100
        volume_percent = (volume_used / VOLUME_CAPACITY) * 100
        pallet_percent = (pallets_used / PALLET_CAPACITY) * 100
        
        total_weight_util += weight_percent
        total_volume_util += volume_percent
        total_pallet_util += pallet_percent
        
        print(f"{j+1:<10} {weight_percent:,.1f}%{' ':>4} {volume_percent:,.1f}%{' ':>4} {pallet_percent:,.1f}%")
        print(f"Orders in container: {orders_in_container.tolist()}")
        print(f"Container details:")
        print(f"  Total Weight: {weight_used:,.0f} lbs")
        print(f"  Total Volume: {volume_used:,.0f} in³")
        print(f"  Total Pallets: {pallets_used:,.0f}")
        print("-" * 80)
    
    print(f"\nAverage utilization across all containers:")
    print(f"Weight: {total_weight_util/num_containers:,.1f}%")
    print(f"Volume: {total_volume_util/num_containers:,.1f}%")
    print(f"Pallets: {total_pallet_util/num_containers:,.1f}%")

    # And in the main execution:
    if feasible:
        print(f"\nFound optimal solution!")
        # print_container_stats(solution, weights, volumes, pallets)  # Changed this line
    else:
        print("\nNo feasible solution found")
    
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

    # Convert to DataFrame and save
    df = pd.DataFrame(container_data)

    # # Calculate and add summary statistics
    # summary_data = {
    #     'Container_Number': 'AVERAGE',
    #     'Orders': '',
    #     'Weight_Used': df['Weight_Used'].mean(),
    #     'Volume_Used': df['Volume_Used'].mean(),
    #     'Pallets_Used': df['Pallets_Used'].mean(),
    #     'Weight_Utilization_%': df['Weight_Utilization_%'].mean(),
    #     'Volume_Utilization_%': df['Volume_Utilization_%'].mean(),
    #     'Pallet_Utilization_%': df['Pallet_Utilization_%'].mean(),
    #     'Remaining_Weight': df['Remaining_Weight'].mean(),
    #     'Remaining_Volume': df['Remaining_Volume'].mean(),
    #     'Remaining_Pallets': df['Remaining_Pallets'].mean()
    # }

    # df = pd.concat([df, pd.DataFrame([summary_data])], ignore_index=True)
    df.to_csv(filename, index=False)
    
# Main execution
# Main execution
if __name__ == "__main__":
    target_containers = goal - 1
    file_path = "filtered_orders.csv"
    orders_df = pd.read_csv(file_path)
    weights, volumes, pallets, num_orders = read_orders_from_csv(file_path)

    while True:
        print(f"\nTrying to find solution with {target_containers} containers...")
        feasible, solution = find_exact_solution(weights, volumes, pallets, target_containers)

        if feasible:
            print(f"\nFound optimal solution!")
            print_container_stats(solution, weights, volumes, pallets)
            # Save solution to CSV
            save_solution_to_csv(solution, weights, volumes, pallets, orders_df,
                                 f'solution_details_{target_containers}_new.csv')
            print(f"\nSolution details saved to solution_details_{target_containers}.csv")
            target_containers -= 1
        else:
            print("\nNo feasible solution found")
            save_solution_to_csv(solution, weights, volumes, pallets, orders_df,
                                 f'solution_details_{target_containers}_new.csv')
            print(f"\nSolution details saved to solution_details_{target_containers}.csv")
            break

