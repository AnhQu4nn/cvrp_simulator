"""
Công cụ chuyển đổi dữ liệu CVRP từ CVRPLIB sang định dạng JSON
"""

import os
import json
import argparse


def parse_vrp_file(file_path):
    """Parse a VRP file in the CVRPLIB format"""
    data = {
        'capacity': 0,
        'depot': None,
        'customers': []
    }
    
    # Parse the file
    section = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for section headers
            if line.startswith('CAPACITY'):
                data['capacity'] = int(line.split(':')[1].strip())
                continue
            elif line == 'NODE_COORD_SECTION':
                section = 'coords'
                continue
            elif line == 'DEMAND_SECTION':
                section = 'demands'
                continue
            elif line.startswith('DEPOT_SECTION') or line.startswith('EOF'):
                continue
            
            # Parse coordinates
            if section == 'coords':
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    
                    # Node ID 1 is usually the depot
                    if node_id == 1:
                        data['depot'] = {'x': x, 'y': y}
                    else:
                        data['customers'].append({
                            'id': node_id,
                            'x': x,
                            'y': y,
                            'demand': 0  # Will be filled later
                        })
            
            # Parse demands
            elif section == 'demands':
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    
                    # Skip depot (its demand is 0)
                    if node_id > 1:
                        # Find the customer in the list and update its demand
                        customer_idx = node_id - 2  # -1 for 0-indexing, -1 for depot
                        if 0 <= customer_idx < len(data['customers']):
                            data['customers'][customer_idx]['demand'] = demand
    
    return data


def convert_to_json(vrp_file, json_file):
    """Convert a VRP file to JSON format"""
    print(f"Converting {vrp_file} to {json_file}...")
    
    # Parse VRP file
    vrp_data = parse_vrp_file(vrp_file)
    
    # Check if depot was found
    if not vrp_data['depot']:
        print("Warning: No depot found, using (0,0) as default")
        vrp_data['depot'] = {'x': 0, 'y': 0}
    
    # Save as JSON
    with open(json_file, 'w') as f:
        json.dump(vrp_data, f, indent=2)
    
    print(f"Conversion completed: {len(vrp_data['customers'])} customers, capacity: {vrp_data['capacity']}")


def batch_convert(input_dir, output_dir):
    """Convert all VRP files in a directory"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all VRP files
    vrp_files = [f for f in os.listdir(input_dir) if f.endswith('.vrp')]
    
    for vrp_file in vrp_files:
        input_path = os.path.join(input_dir, vrp_file)
        # Change extension from .vrp to .json
        json_file = os.path.splitext(vrp_file)[0] + '.json'
        output_path = os.path.join(output_dir, json_file)
        
        convert_to_json(input_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert CVRPLIB VRP files to JSON format")
    parser.add_argument("-i", "--input", help="Input VRP file or directory", required=True)
    parser.add_argument("-o", "--output", help="Output JSON file or directory")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        # Batch convert directory
        output_dir = args.output if args.output else "json_output"
        batch_convert(args.input, output_dir)
    else:
        # Convert single file
        if not args.output:
            # Default output filename
            args.output = os.path.splitext(args.input)[0] + ".json"
        convert_to_json(args.input, args.output)


if __name__ == "__main__":
    main() 