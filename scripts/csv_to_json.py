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
                # Group by pairs (user and assistant messages)
                rows = list(csv_reader)
                result = []
                
                # Process rows in pairs (user, assistant)
                i = 0
                while i < len(rows) - 1:
                    if rows[i][0] == 'user' and rows[i+1][0] == 'assistant':
                        # Parse token info if available
                        token_info = None
                        if len(rows[i+1]) > 2 and rows[i+1][2]:
                            try:
                                token_info = json.loads(rows[i+1][2])
                            except json.JSONDecodeError:
                                print(f"Warning: Could not parse token info for row {i+1}")
                        
                        # Create user message
                        user_message = {"role": rows[i][0], "content": rows[i][1]}
                        if len(rows[i]) > 2 and rows[i][2]:
                            user_message["info"] = rows[i][2]
                        
                        # Create assistant message
                        assistant_message = {"role": rows[i+1][0], "content": rows[i+1][1]}
                        if token_info is not None:
                            assistant_message["info"] = token_info
                        
                        message_pair = [user_message, assistant_message]
                        result.append(message_pair)
                        i += 2
                    else:
                        # Handle unpaired messages
                        message = {"role": rows[i][0], "content": rows[i][1]}
                        if len(rows[i]) > 2 and rows[i][2]:
                            try:
                                message["info"] = json.loads(rows[i][2]) if rows[i][2] else ""
                            except json.JSONDecodeError:
                                message["info"] = rows[i][2]
                        result.append([message])
                        i += 1
                
                # Handle any remaining row
                if i < len(rows):
                    message = {"role": rows[i][0], "content": rows[i][1]}
                    if len(rows[i]) > 2 and rows[i][2]:
                        try:
                            message["info"] = json.loads(rows[i][2]) if rows[i][2] else ""
                        except json.JSONDecodeError:
                            message["info"] = rows[i][2]
                    result.append([message])
                
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
