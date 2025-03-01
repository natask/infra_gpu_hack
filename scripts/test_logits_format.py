#!/usr/bin/env python3
import json
import csv
import random
import os

def simulate_generation(num_tokens=5, top_k=5):
    """
    Simulate token generation with logits to test our format
    """
    # Simulate token generation
    token_info = []
    actual_tokens = []
    
    # Sample token vocabulary
    vocab = ["The", " is", " a", " capital", " of", " France", " Paris", " city", " located", " in", "."]
    
    for i in range(num_tokens):
        # Shuffle vocabulary to simulate different top-k tokens
        random.shuffle(vocab)
        top_tokens = vocab[:top_k]
        
        # Generate random token IDs
        token_ids = [random.randint(100, 10000) for _ in range(top_k)]
        
        # Generate random logits (higher for first token to simulate selection)
        logits = [random.uniform(5.0, 20.0) for _ in range(top_k)]
        logits.sort(reverse=True)  # Sort in descending order
        
        # Store information for this position
        token_info.append([top_tokens, [token_ids], [logits]])
        
        # Add the "selected" token to our generated sequence
        actual_tokens.append(top_tokens[0])
    
    # Create the actual content by joining the selected tokens
    content = "".join(actual_tokens)
    
    return content, token_info

def test_csv_format():
    """
    Test writing and reading the CSV format with token info
    """
    # Create test files
    test_csv = "test_messages_output.csv"
    
    # Generate sample data
    question = "What is the capital of France?"
    answer_content, token_info = simulate_generation(num_tokens=7)
    
    print(f"Generated content: {answer_content}")
    print(f"Token info (first position): {token_info[0]}")
    
    # Write to CSV
    with open(test_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['role', 'content', 'info'])
        
        # Write user message
        writer.writerow(['user', question, ''])
        
        # Write assistant response with token info
        token_info_json = json.dumps(token_info)
        writer.writerow(['assistant', answer_content, token_info_json])
    
    print(f"\nWrote data to {test_csv}")
    
    # Read back and verify
    with open(test_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        print(f"\nHeaders: {headers}")
        
        # Read user message
        user_row = next(reader)
        print(f"User row: {user_row}")
        
        # Read assistant message
        assistant_row = next(reader)
        print(f"Assistant row: {assistant_row[0]}, {assistant_row[1]}")
        print(f"Info field length: {len(assistant_row[2])}")
        
        # Parse the info field
        parsed_info = json.loads(assistant_row[2])
        print(f"\nParsed info (first position):")
        print(f"Top tokens: {parsed_info[0][0]}")
        print(f"Token IDs: {parsed_info[0][1]}")
        print(f"Logits: {parsed_info[0][2]}")
        
        # Verify we can reconstruct the content
        reconstructed = "".join([pos[0][0] for pos in parsed_info])
        print(f"\nReconstructed content: {reconstructed}")
        print(f"Matches original: {reconstructed == answer_content}")
    
    # Clean up
    os.remove(test_csv)
    print("\nTest completed")

if __name__ == "__main__":
    test_csv_format()
