#!/usr/bin/env python3
import json
import argparse
import os

def count_json_elements(file_path):
    """
    Count the number of elements in a JSON file.
    
    The function handles two cases:
    1. JSON file contains a list/array - returns the length of the list
    2. JSON file contains multiple JSON objects, one per line - counts the lines
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return 0
    
    try:
        # First try to load the entire file as a single JSON object
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            count = len(data)
            print(f"File '{file_path}' contains {count} elements in a JSON array.")
            return count
        elif isinstance(data, dict):
            count = len(data.keys()) if data else 1
            print(f"File '{file_path}' contains a JSON object with {count} top-level keys.")
            return count
        else:
            print(f"File '{file_path}' contains a single JSON value.")
            return 1
            
    except json.JSONDecodeError:
        # If that fails, try to count JSON objects line by line
        try:
            count = 0
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            json.loads(line)
                            count += 1
                        except json.JSONDecodeError:
                            # Skip lines that aren't valid JSON
                            pass
            
            print(f"File '{file_path}' contains {count} JSON objects (one per line).")
            return count
            
        except Exception as e:
            print(f"Error processing file '{file_path}': {str(e)}")
            return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count elements in a JSON file")
    parser.add_argument("file_path", help="Path to the JSON file")
    args = parser.parse_args()
    
    count_json_elements(args.file_path)
