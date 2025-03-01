#!/usr/bin/env python3
import csv
import json
import argparse
import os

def csv_to_json(csv_file_path, json_file_path=None, format_type=None):
    """
    Convert a CSV file with headers to a JSON file.
    
    Parameters:
    - csv_file_path: Path to the input CSV file
    - json_file_path: Path to the output JSON file (optional, will be generated if not provided)
    - format_type: Special formatting for specific file types ('messages' or 'time')
    
    Returns:
    - Path to the created JSON file
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file '{csv_file_path}' does not exist.")
        return None
    
    # Generate output JSON file path if not provided
    if json_file_path is None:
        base_name = os.path.splitext(csv_file_path)[0]
        json_file_path = f"{base_name}.json"
    
    try:
        # Read CSV file
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            headers = next(csv_reader)  # Get the header row
            
            # For messages format, we need special handling
            if format_type == 'messages':
                # Process rows for the new format
                rows = list(csv_reader)
                result = []
                
                for row in rows:
                    if len(row) >= 3:  # Ensure we have all required columns
                        prompt = row[0]
                        response = row[1]
                        
                        # Parse logits data
                        logits_data = []
                        if row[2]:
                            try:
                                logits_data = json.loads(row[2])
                            except json.JSONDecodeError:
                                print(f"Warning: Could not parse logits data for row")
                        
                        # Create entry in the desired format
                        entry = {
                            "prompt": prompt,
                            "response": response,
                            "logits": logits_data
                        }
                        
                        result.append(entry)
                
            elif format_type == 'time':
                # For time data, create a list of dictionaries
                result = []
                for row in csv_reader:
                    # Convert numeric values
                    try:
                        generation_time = float(row[1])
                        tokens_per_second = float(row[2])
                    except (ValueError, IndexError):
                        generation_time = 0.0
                        tokens_per_second = 0.0
                    
                    entry = {
                        'question': row[0] if len(row) > 0 else '',
                        'generation_time': generation_time,
                        'tokens_per_second': tokens_per_second
                    }
                    result.append(entry)
            else:
                # Default format: convert each row to a dictionary using headers
                result = []
                for row in csv_reader:
                    # Create a dictionary for each row
                    row_dict = {}
                    for i, header in enumerate(headers):
                        if i < len(row):
                            row_dict[header] = row[i]
                        else:
                            row_dict[header] = ""
                    result.append(row_dict)
        
        # Write to JSON file
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, indent=4, ensure_ascii=False)
        
        print(f"Successfully converted '{csv_file_path}' to '{json_file_path}'")
        print(f"Converted {len(result)} entries")
        return json_file_path
        
    except Exception as e:
        print(f"Error converting CSV to JSON: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert CSV file with headers to JSON")
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument("--output", "-o", help="Path to the output JSON file (optional)")
    parser.add_argument("--format", "-f", choices=['messages', 'time'], 
                        help="Special formatting for specific file types: 'messages' or 'time'")
    args = parser.parse_args()
    
    csv_to_json(args.csv_file, args.output, args.format)

if __name__ == "__main__":
    main()
