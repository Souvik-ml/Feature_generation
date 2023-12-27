import os
import pandas as pd
import numpy as np

directory = '/home/souvik/MXene/files/car_PO'  # Replace with your directory path
output_df = pd.DataFrame(columns=['POSCAR', 'T1-M1', 'M1-X', 'X-M2', 'M2-T2'])

for filename in os.listdir(directory):
    poscar_path = os.path.join(directory, filename)
    
    if os.path.isfile(poscar_path):
        positions = []
        
        with open(poscar_path, 'r') as poscar_file:
            lines = poscar_file.readlines()
            
            for line in lines[8:]:
                atoms = line.strip().split()
                position = [float(atoms[0]), float(atoms[1]), float(atoms[2])]
                positions.append(position)
        
        sorted_positions = sorted(positions, key=lambda x: x[2], reverse=True)
        new_sorted_positions = sorted_positions[1:-1]
        subtracted_data = []
        
        for i in range(len(new_sorted_positions) - 1):
            row1 = new_sorted_positions[i]
            row2 = new_sorted_positions[i + 1]
            subtracted_row = [b - a for a, b in zip(row1, row2)]
            subtracted_data.append(subtracted_row)
        
        squared_data = []
        
        for row in subtracted_data:
            squared_row = [elem ** 2 for elem in row]
            squared_data.append(squared_row)
        
        row_sum = [sum(row) for row in squared_data]
        d = [np.sqrt(i) for i in row_sum]
        
        distance = pd.DataFrame([[filename] + d], columns=['POSCAR', 'T1-M1', 'M1-X', 'X-M2', 'M2-T2'])
        
        output_df = output_df.append(distance, ignore_index=True)

# Print the resulting DataFrame
#print(output_df)

output_df.to_excel('Inter_distance_PO.xlsx')


