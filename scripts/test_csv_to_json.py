#!/usr/bin/env python3
import csv
import json
import os
from csv_to_json import csv_to_json

def create_test_csv():
    """Create a test CSV file with token info"""
    test_csv = "test_messages.csv"
    
    # Sample token info
    token_info = [
        [["The", "A", "In", "France", "Paris"], [[267, 307, 287, 1546, 9458]], [[15.7, 10.2, 8.5, 7.9, 6.3]]],
        [[" capital", " city", " largest", " main", " biggest"], [[4588, 1350, 8273, 1762, 5438]], [[14.2, 12.8, 9.1, 6.7, 5.9]]],
        [[" of", " in", " is", " for", " and"], [[293, 287, 318, 337, 290]], [[18.3, 10.5, 7.2, 4.1, 3.8]]],
        [[" France", " Paris", " the", " Europe", " Italy"], [[1546, 9458, 262, 2325, 3766]], [[19.5, 11.2, 8.4, 5.3, 4.7]]],
        [[" is", " has", " was", " remains", " includes"], [[318, 389, 373, 5438, 8372]], [[17.8, 9.5, 8.3, 6.1, 3.9]]],
        [[" Paris", " Lyon", " located", " situated", " found"], [[9458, 18372, 4829, 8372, 1498]], [[20.1, 8.7, 7.3, 6.5, 5.2]]],
        [[".", ",", "!", "?", " which"], [[29, 30, 0, 93, 1498]], [[16.9, 10.3, 7.8, 5.2, 3.1]]]
    ]
    
    # Write to CSV
    with open(test_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['role', 'content', 'info'])
        
        # Write user message
        writer.writerow(['user', 'What is the capital of France?', ''])
        
        # Write assistant response with token info
        token_info_json = json.dumps(token_info)
        writer.writerow(['assistant', 'The capital of France is Paris.', token_info_json])
    
    return test_csv

def test_conversion():
    """Test the CSV to JSON conversion with token info"""
    # Create test CSV
    test_csv = create_test_csv()
    print(f"Created test CSV: {test_csv}")
    
    # Convert to JSON
    json_file = csv_to_json(test_csv, format_type='messages')
    print(f"Converted to JSON: {json_file}")
    
    # Read and verify JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
        
        print("\nJSON Data Structure:")
        print(f"Number of message pairs: {len(data)}")
        
        # Check first message pair
        if data and len(data) > 0:
            pair = data[0]
            print(f"\nUser message: {pair[0]['role']}, {pair[0]['content']}")
            print(f"Assistant message: {pair[1]['role']}, {pair[1]['content']}")
            
            # Check token info
            if 'info' in pair[1]:
                token_info = pair[1]['info']
                print(f"\nToken info available: {len(token_info)} positions")
                
                # Check first position
                if token_info and len(token_info) > 0:
                    first_pos = token_info[0]
                    print(f"First position tokens: {first_pos[0]}")
                    print(f"First position token IDs: {first_pos[1]}")
                    print(f"First position logits: {first_pos[2]}")
                    
                    # Reconstruct content
                    reconstructed = "".join([pos[0][0] for pos in token_info])
                    print(f"\nReconstructed content: {reconstructed}")
                    print(f"Matches original: {reconstructed == pair[1]['content']}")
    
    # Clean up
    os.remove(test_csv)
    os.remove(json_file)
    print("\nTest completed")

if __name__ == "__main__":
    test_conversion()
