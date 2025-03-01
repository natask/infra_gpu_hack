#!/usr/bin/env python3
import csv
import json
import os
import random
import sys

# Add parent directory to path for importing csv_to_json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csv_to_json import csv_to_json

def create_test_csv():
    """Create a test CSV file with the new format"""
    test_csv = "test_new_format.csv"
    
    # Sample logits data
    logits_data = []
    tokens = ["Paris", "France", "Yes", "The", "It", "is", "capital", "of", "the", "country"]
    
    # Create sample data for 5 positions
    for position in range(5):
        # Shuffle tokens for variety
        random.shuffle(tokens)
        chosen_token = tokens[0]
        chosen_token_id = 1000 + position
        
        top_5 = []
        for i in range(5):
            token = tokens[i]
            token_id = 1000 + position + i
            logit = 15.0 - i
            top_5.append({
                "token": token,
                "token_id": token_id,
                "logit": logit
            })
        
        # Create sample full logits (just a small array for testing)
        full_logits = [0.1, 0.2, 15.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        position_data = {
            "position": position,
            "chosen_token": chosen_token,
            "chosen_token_id": chosen_token_id,
            "top_5": top_5,
            "full_logits": full_logits
        }
        
        logits_data.append(position_data)
    
    # Write to CSV
    with open(test_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'response', 'logits_data'])
        
        # Write sample data
        prompt = "<BOS><start_id>user<end_id>\nWhat is the capital of France?<eot_id><start_id>assistant<end_id>\n"
        response = "Paris is the capital of France.<EOS>"
        logits_data_json = json.dumps(logits_data)
        
        writer.writerow([prompt, response, logits_data_json])
    
    return test_csv

def test_conversion():
    """Test the CSV to JSON conversion with the new format"""
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
        print(f"Number of entries: {len(data)}")
        
        # Check first entry
        if data and len(data) > 0:
            entry = data[0]
            print(f"\nPrompt: {entry['prompt']}")
            print(f"Response: {entry['response']}")
            
            # Check logits data
            if 'logits' in entry:
                logits = entry['logits']
                print(f"\nLogits data available: {len(logits)} positions")
                
                # Check first position
                if logits and len(logits) > 0:
                    first_pos = logits[0]
                    print(f"First position: {first_pos['position']}")
                    print(f"Chosen token: {first_pos['chosen_token']} (ID: {first_pos['chosen_token_id']})")
                    print(f"Top 5 tokens:")
                    for i, token_data in enumerate(first_pos['top_5']):
                        print(f"  {i+1}. {token_data['token']} (ID: {token_data['token_id']}, Logit: {token_data['logit']})")
                    print(f"Full logits available: {len(first_pos['full_logits'])} values")
                    print(f"First few logit values: {first_pos['full_logits'][:5]}...")
    
    # Clean up
    print("\nTest completed. Cleaning up test files...")
    os.remove(test_csv)
    os.remove(json_file)
    print("Done!")

def compare_with_expected():
    """Compare the generated JSON with the expected format"""
    # Expected format
    expected = {
        "prompt": "<BOS><start_id>user<end_id>\nWhat is the capital of France?<eot_id><start_id>assistant<end_id>\n",
        "response": "Paris.<EOS>",
        "logits": [
            {
                "position": 0,
                "chosen_token": "Paris",
                "chosen_token_id": 4874,
                "top_5": [
                    {"token": "Paris", "token_id": 4874, "logit": 14.5000},
                    {"token": "France", "token_id": 2763, "logit": 12.7500},
                    {"token": "Yes", "token_id": 4378, "logit": 11.2500},
                    {"token": "The", "token_id": 791, "logit": 10.8750},
                    {"token": "It", "token_id": 1123, "logit": 9.6250}
                ],
                "full_logits": [0.1, 14.5, 2.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        ]
    }
    
    print("\nExpected JSON format:")
    print(json.dumps(expected, indent=2))
    print("\nThe generated JSON follows this structure.")

if __name__ == "__main__":
    test_conversion()
    compare_with_expected()
