#!/usr/bin/env python3
import json
import os
import argparse

def merge_json_files(output_name_prefix, num_gpus):
    """
    Merge JSON files from multiple GPUs into single consolidated files.
    """
    merged_messages = []
    merged_time = []

    # Iterate over each GPU's output files
    for gpu_id in range(num_gpus):
        messages_file_path = f'{output_name_prefix}__messages_output{gpu_id}.json'
        time_file_path = f'{output_name_prefix}__time_output{gpu_id}.json'

        # Read and merge messages
        if os.path.exists(messages_file_path):
            try:
                with open(messages_file_path, 'r') as messages_file:
                    messages_data = json.load(messages_file)
                    merged_messages.extend(messages_data)
                    print(f"Added {len(messages_data)} message entries from GPU {gpu_id}")
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON from {messages_file_path}")

        # Read and merge time entries
        if os.path.exists(time_file_path):
            try:
                with open(time_file_path, 'r') as time_file:
                    time_data = json.load(time_file)
                    merged_time.extend(time_data)
                    print(f"Added {len(time_data)} time entries from GPU {gpu_id}")
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON from {time_file_path}")

    # Write merged messages to a single file
    with open(f'{output_name_prefix}__merged_messages.json', 'w') as merged_messages_file:
        json.dump(merged_messages, merged_messages_file, indent=4)

    # Write merged time entries to a single file
    with open(f'{output_name_prefix}__merged_time.json', 'w') as merged_time_file:
        json.dump(merged_time, merged_time_file, indent=4)

    print(f"Merged {len(merged_messages)} message entries and {len(merged_time)} time entries.")
    print(f"Output files: {output_name_prefix}__merged_messages.json and {output_name_prefix}__merged_time.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge JSON output files from multiple GPUs')
    parser.add_argument('--name', type=str, required=True, help='Output name prefix used for the files')
    parser.add_argument('--num_gpus', type=int, required=True, help='Number of GPUs used for generation')
    args = parser.parse_args()
    
    merge_json_files(args.name, args.num_gpus)
