import pandas as pd
import numpy as np

"""
Distribution Center Container Loading Optimization
-----------------------------------------------
Problem Description:
- Daily volume: ~500 orders from various companies (cosmetics to sports memorabilia)
- Container Type: 53' freight container for intermodal transport
- Constraints per container:
    * Weight Capacity: 45,000 lbs
    * Volume Capacity: 3,600 in³
    * Pallet Capacity: 60 (double stacked)
- Objective: Minimize number of containers shipped per day

Input Data Format (CSV):
- Order Number: Unique identifier
- Weight (lbs): Order weight
- Volume (in3): Order volume
- Pallets: Number of pallets needed
"""

def read_orders_from_csv(file_path):
    # Read the CSV file
    orders_df = pd.read_csv(file_path)
    
    # Print initial shape
    print(f"Initial number of orders: {len(orders_df)}")
    
    # Only drop rows where ALL required columns are NaN
    orders_df = orders_df.dropna(subset=['Weight (lbs)', 'Volume (in3)', 'Pallets'], how='any')
    
    # Convert numeric columns - this will help identify any non-numeric values
    orders_df['Weight (lbs)'] = pd.to_numeric(orders_df['Weight (lbs)'], errors='coerce')
    orders_df['Volume (in3)'] = pd.to_numeric(orders_df['Volume (in3)'], errors='coerce')
    orders_df['Pallets'] = pd.to_numeric(orders_df['Pallets'], errors='coerce')
    
    # Drop rows again after conversion
    orders_df = orders_df.dropna(subset=['Weight (lbs)', 'Volume (in3)', 'Pallets'], how='any')
    
    # Store original indices before extraction
    original_indices = orders_df.index.tolist()
    
    # Extract relevant columns and convert to list of tuples
    orders = orders_df[['Weight (lbs)', 'Volume (in3)', 'Pallets']].values.tolist()
    parsed_orders = [(weight, volume, pallets, idx) for (weight, volume, pallets), idx in zip(orders, original_indices)]
    
    print(f"Total valid orders after cleaning: {len(parsed_orders)}")
    
    return parsed_orders, len(orders_df)

def best_fit_decreasing(orders, total_orders):
    MAX_WEIGHT = 45000
    MAX_VOLUME = 3600
    MAX_PALLETS = 60
    
    orders = sorted(orders, key=lambda x: (x[0]/MAX_WEIGHT + x[1]/MAX_VOLUME + x[2]/MAX_PALLETS), reverse=True)
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
            binary_array = np.zeros(total_orders, dtype=int)
            binary_array[orig_idx] = 1
            containers.append([MAX_WEIGHT-weight, MAX_VOLUME-volume, MAX_PALLETS-pallets, binary_array])
    
    pattern_matrix = np.zeros((total_orders, len(containers)))
    for j, container in enumerate(containers):
        pattern_matrix[:, j] = container[3]
    
    print(f"\nTotal number of containers needed: {len(containers)}")
    
    total_weight_util = 0
    total_volume_util = 0
    total_pallet_util = 0
    
    for i, container in enumerate(containers, 1):
        weight_util = ((MAX_WEIGHT - container[0]) / MAX_WEIGHT) * 100
        volume_util = ((MAX_VOLUME - container[1]) / MAX_VOLUME) * 100
        pallet_util = ((MAX_PALLETS - container[2]) / MAX_PALLETS) * 100
        
        total_weight_util += weight_util
        total_volume_util += volume_util
        total_pallet_util += pallet_util
    
    avg_weight_util = total_weight_util / len(containers)
    avg_volume_util = total_volume_util / len(containers)
    avg_pallet_util = total_pallet_util / len(containers)
    
    print(f"\nAverage Utilization Across All Containers:")
    print(f"Weight: {avg_weight_util:.2f}%")
    print(f"Volume: {avg_volume_util:.2f}%")
    print(f"Pallets: {avg_pallet_util:.2f}%")
    
    return len(containers), containers, pattern_matrix
def save_solution_to_csv(solution_matrix, weights, volumes, pallets, orders_df, filename="solution_details.csv"):
    WEIGHT_CAPACITY = 45000
    VOLUME_CAPACITY = 3600
    PALLET_CAPACITY = 60

    container_data = []

    for j in range(solution_matrix.shape[1]):
        orders_in_container = np.where(solution_matrix[:, j] == 1)[0]
        order_numbers = orders_df.iloc[orders_in_container]['Order Number'].tolist()
        
        weight_used = sum(weights[orders_in_container])
        volume_used = sum(volumes[orders_in_container])
        pallets_used = sum(pallets[orders_in_container])

        weight_percent = (weight_used / WEIGHT_CAPACITY) * 100
        volume_percent = (volume_used / VOLUME_CAPACITY) * 100
        pallet_percent = (pallets_used / PALLET_CAPACITY) * 100

        container_data.append({
            'Container_Number': j + 1,
            'Orders': str(order_numbers),
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

def main():
    file_path = "Term project data final.csv"
    
    try:
        orders, total_orders = read_orders_from_csv(file_path)
        
        # Read the original DataFrame to get the 'Order Number' column
        orders_df = pd.read_csv(file_path)
        
        num_containers, container_config, pattern_matrix = best_fit_decreasing(orders, total_orders)
        
        # Extract weights, volumes, and pallets from orders
        weights = np.array([order[0] for order in orders])
        volumes = np.array([order[1] for order in orders])
        pallets = np.array([order[2] for order in orders])
        
        # Save BFD solution to CSV
        save_solution_to_csv(pattern_matrix, weights, volumes, pallets, orders_df, filename="bfd_solution_details.csv")
        
        print(f"BFD solution details saved to bfd_solution_details.csv")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()