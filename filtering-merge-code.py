import pandas as pd
import ast

# Load the 268 CSV data
solution_268 = pd.read_csv('solution_details_268_new.csv')

# Filter containers where all utilization percentages are below 99%
filtered_containers_268 = solution_268[(solution_268['Weight_Utilization_%'] >= 99) | 
                                       (solution_268['Volume_Utilization_%'] >= 99) | 
                                       (solution_268['Pallet_Utilization_%'] >= 99)]

print(len(filtered_containers_268))

# Load the 107 CSV data
solution_107 = pd.read_csv('solution_details_107_new.csv')

# Concatenate the filtered 268 containers with the 107 containers
merged_solution = pd.concat([filtered_containers_268, solution_107], ignore_index=True)

# Reset the Container_Number column
merged_solution['Container_Number'] = range(1, len(merged_solution) + 1)

# Save the merged result to a new CSV file
merged_solution.to_csv('merged_solution.csv', index=False)

# Output the number of containers in the merged solution
print(f"Number of containers in merged solution: {len(merged_solution)}")

unique_orders = set()
for orders in merged_solution['Orders']:
    unique_orders.update(ast.literal_eval(orders))
print(f"Number of unique orders: {len(unique_orders)}")