#!/usr/bin/env python3
import csv
import os
import argparse

def merge_csv_files(output_name_prefix, num_gpus):
    """
    Merge CSV files from multiple GPUs into single consolidated files.
    """
    # Initialize lists to store merged data
    merged_messages = []
    merged_time = []

    # Iterate over each GPU's output files
    for gpu_id in range(num_gpus):
        messages_file_path = f'{output_name_prefix}__messages_output{gpu_id}.csv'
        time_file_path = f'{output_name_prefix}__time_output{gpu_id}.csv'

        # Read and merge messages
        if os.path.exists(messages_file_path):
            with open(messages_file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                # Skip header row
                header = next(reader, None)
                for row in reader:
                    if row:  # Skip empty rows
                        merged_messages.append(row)

        # Read and merge time entries
        if os.path.exists(time_file_path):
            with open(time_file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                # Skip header row
                header = next(reader, None)
                for row in reader:
                    if row:  # Skip empty rows
                        merged_time.append(row)

    # Write merged messages to a single file
    with open(f'{output_name_prefix}__merged_messages.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['role', 'content'])  # Write header
        writer.writerows(merged_messages)

    # Write merged time entries to a single file
    with open(f'{output_name_prefix}__merged_time.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'generation_time', 'tokens_per_second'])  # Write header
        writer.writerows(merged_time)

    print(f"Merged {len(merged_messages)} message entries and {len(merged_time)} time entries.")
    print(f"Output files: {output_name_prefix}__merged_messages.csv and {output_name_prefix}__merged_time.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge output files from multiple GPUs')
    parser.add_argument('--name', type=str, required=True, help='Output name prefix used for the files')
    parser.add_argument('--num_gpus', type=int, required=True, help='Number of GPUs used for generation')
    args = parser.parse_args()
    
    merge_csv_files(args.name, args.num_gpus)
