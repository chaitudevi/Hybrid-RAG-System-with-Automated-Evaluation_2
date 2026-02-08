
import json
import pandas as pd
import glob
import os
import argparse
import sys

def combine_results(output_file='results/combined_results.xlsx'):
    # Search for potential metric files
    detailed_files = glob.glob('results/detailed_*.json')
    eval_files = glob.glob('results/evaluations_*.json')
    metric_files = detailed_files + eval_files
    
    # Search for ablation files (containing answers)
    ablation_files = glob.glob('results/ablation_*.json')
    
    if not metric_files:
        print("Error: No evaluations/detailed results found.")
        return

    # Helper to extract timestamp
    def get_timestamp(filename):
        # Assumes format ..._YYYYMMDD_HHMMSS.json
        parts = filename.replace('.json', '').split('_')
        if len(parts) >= 2:
            return f"{parts[-2]}_{parts[-1]}"
        return ""

    # Try to find a pair with matching timestamps
    best_pair = None
    latest_ts = ""

    for m_file in metric_files:
        m_ts = get_timestamp(m_file)
        # Find matching ablation
        for a_file in ablation_files:
            a_ts = get_timestamp(a_file)
            if m_ts == a_ts:
                # Found a match
                if m_ts > latest_ts:
                    latest_ts = m_ts
                    best_pair = (m_file, a_file)
    
    if best_pair:
        eval_file, ablation_file = best_pair
        print(f"Matched files by timestamp ({latest_ts}):")
        print(f"- Metrics: {eval_file}")
        print(f"- Answers: {ablation_file}")
    else:
        # Fallback: just take the absolute latest of each
        print("Warning: Could not match filenames by timestamp. Using latest files found.")
        eval_file = max(metric_files, key=os.path.getmtime)
        ablation_file = max(ablation_files, key=os.path.getmtime) if ablation_files else None
        print(f"- Metrics: {eval_file}")
        print(f"- Answers: {ablation_file}")

    # Load data
    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
        
    answers_map = {}
    if ablation_file:
        with open(ablation_file, 'r', encoding='utf-8') as f:
            ablation_data = json.load(f)
            
        # Extract answers from ablation data (focus on 'hybrid' method as primary)
        if 'hybrid' in ablation_data:
            for item in ablation_data['hybrid']:
                query = item.get('query', '').strip()
                answer = item.get('answer', '')
                if query:
                    answers_map[query] = answer
    else:
         print("Warning: No ablation file found. Answers will be empty.")

    # Combined list
    combined_rows = []
    
    # Stats for debugging
    matches = 0
    total = 0

    for item in eval_data:
        total += 1
        question = item.get('question', '').strip()
        
        answer = answers_map.get(question, "N/A")
        if answer != "N/A":
            matches += 1

        # Base row with identification
        row = {
            'Question ID': item.get('question_id'),
            'Question': question,
            'Answer': answer,
            'Question Type': item.get('question_type'),
        }
        
        # Add metrics
        # List of potential metrics to include
        metrics = [
            'mrr_url', 
            'precision_at_5', 'precision_at_10', 
            'recall_at_10', 'hit_rate_at_10', 
            'ndcg_at_10', 
            'semantic_similarity', 
            'answer_length_score', 
            'contextual_precision',
            'time_taken', 
            'confidence'
        ]
        
        for metric in metrics:
            row[metric] = item.get(metric)
            
        combined_rows.append(row)

    print(f"Matched {matches}/{total} questions with answers.")

    # Create DataFrame
    df = pd.DataFrame(combined_rows)
    
    # Save to Excel
    try:
        output_path = os.path.abspath(output_file)
        df.to_excel(output_path, index=False)
        print(f"\nSuccess! Combine results saved to:\n{output_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine RAG evaluation results to Excel')
    parser.add_argument('--output', default='results/combined_results.xlsx', help='Output Excel file path')
    args = parser.parse_args()
    
    combine_results(args.output)
