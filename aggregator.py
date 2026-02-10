import os
import re
import glob
import sys
import argparse
import multiprocessing as mp
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from visualizer import IntervalVisualizer
from interval_proc import IntervalMerger

def extract_and_validate_parallel_config(root_dir):
    """
    Extract configuration data and complete PP Groups with their corresponding rank paths.
    Returns:
        tuple: (config_data, complete_pp_groups_with_paths, complete_model_groups_with_paths)
               or (None, None, None) if config file not found.
    """
    # Step 1: Find configuration file
    search_patterns = [os.path.join(root_dir, 'unique_parallel_groups*.txt'),
                      os.path.join(os.path.dirname(root_dir), 'unique_parallel_groups*.txt')]
    config_files = [f for pattern in search_patterns for f in glob.glob(pattern)]
    config_file_path = config_files[0] if config_files else None
    
    if config_file_path is None:
        print(f"⚠ Warning: 'unique_parallel_groups' config file not found in {root_dir}")
        print(f"  Skipping PP Group aggregation analysis.")
        return None, None, None

    # Step 2: Extract all rank IDs from folders
    rank_ids = sorted({int(m.group(1)) for f in os.listdir(root_dir) if (m := re.search(r'rank[_]?(\d+)[._]', f))})
    
    # Step 3: Read config file content
    with open(config_file_path, 'r') as file:
        content = file.read()

    # Step 3.1: Extract scalar values
    def _extract_int(pats, default=None):
        for pat in pats:
            if m := re.search(pat, content): return int(m.group(1))
        return default
        
    config_data = {
        'TP': _extract_int([r"TP \(Tensor Parallel\) Size: (\d+)", r"TP Size: (\d+)"]),
        'PP': _extract_int([r"PP \(Pipeline Parallel\) Size: (\d+)", r"PP Size: (\d+)"]),
        'DP': _extract_int([r"DP \(Data Parallel\) Size: (\d+)", r"DP Size: (\d+)"]),
        'EP': _extract_int([r"EP \(Expert Parallel\) Size: (\d+)", r"EP Size: (\d+)" ]),
        'CP': _extract_int([r"CP \(Context Parallel\) Size: (\d+)", r"CP Size: (\d+)" ]),
        'World Size': _extract_int([r"World Size: (\d+)", r"World Size\s*:\s*(\d+)"]),
        'TP Groups': [], 'PP Groups': [], 'DP Groups': [], 'EP Groups': [], 'Full Model Rank Groups': []
    }

    # Step 3.2: Extract groups
    def _extract_groups(section_name_patterns):
        block = None
        for sec_pat in section_name_patterns:
            patterns = [
                rf"{sec_pat}:?\s*\n-+\s*\n(.*?)(?=\n\n[A-Z][A-Z \(\)]+:?\s*\n-+)",
                rf"{sec_pat}:?\s*\n-+\s*\n(.*?)(?=\nTotal )",
                rf"{sec_pat}:?\s*\n-+\s*\n(.*?)(?=\Z)",
            ]
            for pattern in patterns:
                if m := re.search(pattern, content, flags=re.S | re.MULTILINE):
                    block = m.group(1).strip(); break
            if block: break
        
        if block is None: return []
        filtered_lines = [line for line in block.split('\n') if re.match(r'^\s*(?:TP |PP |DP |EP |Model )?Group\s+\d+:', line)]
        groups = re.findall(r'^(?:TP |PP |DP |EP |Model )?Group\s+\d+:\s*\[([^\]]+)\]', '\n'.join(filtered_lines), re.MULTILINE)
        return [[int(x.strip()) for x in g.split(',') if x.strip()] for g in groups]

    config_data['TP Groups'] = _extract_groups([r"UNIQUE TP GROUPS", r"TENSOR PARALLEL \(TP\) GROUPS"])
    config_data['PP Groups'] = _extract_groups([r"UNIQUE PP GROUPS", r"PIPELINE PARALLEL \(PP\) GROUPS"])
    config_data['DP Groups'] = _extract_groups([r"UNIQUE DP GROUPS", r"DATA PARALLEL \(DP\) GROUPS"])
    config_data['EP Groups'] = _extract_groups([r"UNIQUE EP GROUPS", r"EXPERT PARALLEL \(EP\) GROUPS", r"EXPERT GROUPS"])

    # Step 3.3: Extract Full Model Rank Groups
    full_model_groups = []
    if m_full := re.search(r"FULL MODEL RANK GROUPS.*?\n-+\s*\n.*?\n-+\s*\n(.*?)(?=\n\nTotal\s|$)", content, flags=re.S):
        for gid_str, ranks_str in re.findall(r"^Group\s+(\d+):\s*\[([^\]]+)\]", m_full.group(1).strip(), re.MULTILINE):
            full_model_groups.append({"group_id": int(gid_str), "ranks": [int(x.strip()) for x in ranks_str.split(',') if x.strip()]})
    config_data['Full Model Rank Groups'] = full_model_groups

    # Step 4: Map ranks to folders
    rank_to_folder = {}
    for folder in [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f)) and 'kernel_analysis' in f]:
        if m := re.search(r'rank[_]?(\d+)[._]', folder):
            rank_to_folder[int(m.group(1))] = os.path.join(root_dir, folder)

    # Step 5: Complete PP Groups
    rank_set = set(rank_ids)
    complete_pp_groups_with_paths = [
        {'pp_group': g, 'paths': [rank_to_folder[r] for r in g if r in rank_to_folder]}
        for g in config_data['PP Groups'] if all(r in rank_set for r in g)
        and len([rank_to_folder[r] for r in g if r in rank_to_folder]) == len(g)
    ]

    # Step 6: Complete Full Model Groups
    complete_model_groups_with_paths = [
        {'group_id': g['group_id'], 'ranks': g['ranks'], 'paths': [rank_to_folder[r] for r in g['ranks'] if r in rank_to_folder]}
        for g in config_data['Full Model Rank Groups'] if all(r in rank_set for r in g['ranks'])
        and len([rank_to_folder[r] for r in g['ranks'] if r in rank_to_folder]) == len(g['ranks'])
    ]
    complete_model_groups_with_paths = _split_model_groups_into_pp_groups(complete_model_groups_with_paths, config_data)
    
    return config_data, complete_pp_groups_with_paths, complete_model_groups_with_paths

def _split_model_groups_into_pp_groups(complete_model_groups_with_paths, config_data):
    """Split Full Model Groups into PP Groups organized by model_group_id."""
    pp_groups = config_data['PP Groups']
    if not pp_groups:
        print("⚠ Warning: No PP Groups configuration found")
        return {}
        
    result = {}
    for model_group in complete_model_groups_with_paths:
        model_group_id = model_group['group_id']
        model_ranks = set(model_group['ranks'])
        rank_to_path = dict(zip(model_group['ranks'], model_group['paths']))
        
        result[model_group_id] = []
        for pp_idx, pp_group in enumerate(pp_groups):
            if all(r in model_ranks for r in pp_group):
                pp_paths = [rank_to_path[r] for r in pp_group if r in rank_to_path]
                if len(pp_paths) == len(pp_group):
                    result[model_group_id].append({'pp_group_idx': pp_idx, 'pp_group': pp_group, 'paths': pp_paths})
    return result

def _process_single_rank(rank_id: int, rank_folder: str, files: List[str]) -> Tuple[int, Optional[Dict]]:
    """Helper function to process a single rank in parallel."""
    try:
        parquet_files = [os.path.join(rank_folder, f) for f in files]
        existing_files = [f for f in parquet_files if os.path.exists(f)]

        if not existing_files:
            print(f"⚠ No parquet files found in {os.path.basename(rank_folder)}, skipping...")
            return (rank_id, None)
        
        merger = IntervalMerger()
        merged_dict = {}
        result_df_commcompOver = merger.calculate_comm_comp_overlap(rank_folder)
        result_df_commcompOver.to_csv(os.path.join(rank_folder, f'Rank{rank_id}_commcompOverlap_analysis.csv'), index=False)
    
        for parquet_file in existing_files:
            try:
                merged_df = merger.merge_singleRank_mbs(parquet_file)
                category = os.path.splitext(os.path.basename(parquet_file))[0]
                if not merged_df.empty:
                    merged_dict[category] = merged_df
                else:
                    print(f"  ⚠ Rank {rank_id} - {category}: No intervals")
            except Exception as e:
                print(f"  ✗ Rank {rank_id} - Error processing {os.path.basename(parquet_file)}: {e}")
                continue
        
        if not merged_dict:
            print(f"⚠ No valid data for rank {rank_id}, skipping...")
            return (rank_id, None)
        
        pd.concat(list(merged_dict.values()), ignore_index=True).to_csv(os.path.join(rank_folder, f'Rank{rank_id}_statistics.csv'), index=False)
        
        visualizer = IntervalVisualizer()
        visualizer.plot_combined_timeline({rank_id: merged_dict}, os.path.join(rank_folder, 'overlapping_timeline.png'),
                                          category_order=['optimizer_computation', 'dp_communication', 'pp_communication',
                                                         'fwd_computation', 'bwd_computation', 'bwd_computation_recompute'],
                                         title=f'Rank {rank_id} Overlapping Timeline')
        visualizer.plot_comm_comp_overlap(overlap_df=result_df_commcompOver, output_dir=rank_folder, rank_id=rank_id)
        
        print(f"✓ Rank {rank_id} processed successfully")
        return (rank_id, merged_dict)
        
    except Exception as e:
        print(f"✗ Error processing rank {rank_id}: {e}")
        import traceback; traceback.print_exc()
        return (rank_id, None)

def process_individual_ranks(root_dir: str, files: List[str], max_workers: Optional[int] = None) -> Dict[int, Dict]:
    """Step 1: Process all individual ranks in parallel."""
    print(f"\n{'='*80}\nSTEP 1: PROCESSING ALL INDIVIDUAL RANKS (PARALLEL)\n{'='*80}")
    
    all_rank_folders = {}
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and 'kernel_analysis' in item:
            if match := re.search(r'rank[_-]?(\d+)', item):
                all_rank_folders[int(match.group(1))] = item_path
    
    if not all_rank_folders:
        print("⚠ No rank folders found"); return {}
    
    max_workers = min(mp.cpu_count(), len(all_rank_folders)) if max_workers is None else max_workers
    print(f"Found {len(all_rank_folders)} ranks to process using {max_workers} workers")
    
    all_rank_data = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_rank = {executor.submit(_process_single_rank, rid, f, files): rid for rid, f in all_rank_folders.items()}
        completed = 0
        for future in as_completed(future_to_rank):
            completed += 1
            try:
                rid, mdict = future.result()
                if mdict: all_rank_data[rid] = mdict
                print(f"Progress: {completed}/{len(future_to_rank)} ranks completed")
            except Exception as e:
                print(f"✗ Rank {future_to_rank[future]} failed: {e}")
    
    print(f"\n✓ Successfully processed {len(all_rank_data)}/{len(all_rank_folders)} ranks")
    return dict(sorted(all_rank_data.items()))

def process_pp_groups(root_dir: str, complete_pp_groups_with_paths: List[Dict], all_rank_data: Dict[int, Dict], time_alignment: str = 'global'):
    """Step 2: Generate multi-rank visualizations for PP Groups."""
    print(f"\n{'='*80}\nSTEP 2: GENERATING MULTI-RANK VISUALIZATIONS FOR PP GROUPS\n{'='*80}")
    
    for group_idx, pp_info in enumerate(complete_pp_groups_with_paths):
        pp_group = pp_info['pp_group']
        
        if len(pp_group) <= 1:
            continue
        multirank_data = {rid: all_rank_data[rid] for rid in pp_group if rid in all_rank_data}
        
        if missing := [r for r in pp_group if r not in all_rank_data]:
            print(f"⚠ Warning: Missing data for ranks: {missing}")
        
        if len(multirank_data) >= 2:
            output_dir = os.path.join(root_dir, f'PP_group_{group_idx}')
            os.makedirs(output_dir, exist_ok=True)

            visualizer = IntervalVisualizer()
            visualizer.plot_combined_timeline(multirank_data, 
                os.path.join(output_dir, f'pp_group_{group_idx + 1}_ranks_{"_".join(map(str, pp_group))}_{time_alignment}.png'),
                time_alignment=time_alignment,
                category_order=['optimizer_computation', 'dp_communication', 'fwd_computation', 'bwd_computation'],
                title=f'PP Group {group_idx + 1} Timeline (Ranks: {pp_group}, {time_alignment} alignment)')
        
        elif len(multirank_data) == 1:
            print(f"⚠ PP Group {group_idx}: Only 1 rank with valid data, skipping visualization.")
        else:
            print(f"⚠ PP Group {group_idx}: No valid data, skipping.")


def process_model_groups(root_dir: str, complete_model_groups_with_paths: Dict[int, List[Dict]], all_rank_data: Dict[int, Dict], time_alignment: str = 'global'):
    """Step 3: Generate globally aligned concatenated visualizations for Model Groups."""
    print(f"\n{'='*80}\nSTEP 3: GENERATING GLOBALLY ALIGNED CONCATENATED VISUALIZATIONS FOR MODEL GROUPS\n{'='*80}")
    
    if not complete_model_groups_with_paths:
        print("⚠ No Model Groups found, skipping..."); return
    
    visualizer = IntervalVisualizer()
    for model_group_id, pp_groups_list in complete_model_groups_with_paths.items():
        print(f"\n📊 Processing Model Group {model_group_id} ({len(pp_groups_list)} PP Groups)...")
        output_dir = os.path.join(root_dir, f'Model_Group_{model_group_id}')
        os.makedirs(output_dir, exist_ok=True)
        
        all_model_ranks = sorted({r for p in pp_groups_list for r in p['pp_group']})
        full_multirank_data = {r: all_rank_data[r] for r in all_model_ranks if r in all_rank_data}
        
        if not full_multirank_data:
            print(f"   ⚠ No valid data for Model Group {model_group_id}, skipping..."); continue
            
        pp_group_images = []
        for pp_info in pp_groups_list:
            pp_idx, pp_group = pp_info['pp_group_idx'], pp_info['pp_group']
            pp_data = {r: full_multirank_data[r] for r in pp_group if r in full_multirank_data}
            
            if pp_data:
                out_path = os.path.join(output_dir, f'pp_group_{pp_idx}_ranks_{"_".join(map(str, sorted(pp_data.keys())))}.png')
                visualizer.plot_combined_timeline(pp_data, out_path, time_alignment=time_alignment,
                    category_order=['optimizer_computation', 'dp_communication', 'fwd_computation', 'bwd_computation'],
                    title=f'Model Group {model_group_id} - PP Group {pp_idx} (Ranks: {sorted(pp_data.keys())})',
                    global_time_reference=full_multirank_data)
                pp_group_images.append(out_path)
        
        if pp_group_images:
            visualizer.plot_combined_timeline(full_multirank_data, 
                os.path.join(output_dir, f'Model_Group_{model_group_id}_all_ranks_combined.png'),
                time_alignment=time_alignment,
                category_order=['optimizer_computation', 'dp_communication', 'fwd_computation', 'bwd_computation'],
                title=f'Model Group {model_group_id} - All Ranks Combined ({len(all_model_ranks)} ranks)')

def process_rank_aggregation(input_root: str, files: Optional[List[str]] = None, time_alignment: str = 'global',processes=10) -> int:
    """Process rank aggregation analysis and visualization."""
    if not os.path.isdir(input_root):
        print(f"Error: Root directory not found: {input_root}"); return 1
    if files is None: return 1

    config_data, complete_pp_groups, complete_model_groups = extract_and_validate_parallel_config(input_root)
    if config_data:
        print(f"\n{'='*80}\nPARALLEL CONFIGURATION\n{'='*80}")
        for k in ['TP', 'PP', 'DP', 'EP', 'World Size']: print(f"{k} Size: {config_data.get(k)}")
        print(f"Complete PP Groups found: {len(complete_pp_groups)}")

    all_rank_data = process_individual_ranks(input_root, files, max_workers=processes)

    if config_data and complete_pp_groups:
        process_pp_groups(input_root, complete_pp_groups, all_rank_data, time_alignment)
    else:
        print(f"\n{'='*80}\nSKIPPING STEP 2: PP GROUP ANALYSIS (No config found)\n{'='*80}")
    
    if config_data and complete_model_groups:
        process_model_groups(input_root, complete_model_groups, all_rank_data, time_alignment)
    else:
        print(f"\n{'='*80}\nSKIPPING STEP 3: MODEL GROUP CONCATENATION (No groups found)\n{'='*80}")
    
    print(f"\n{'='*80}\nANALYSIS COMPLETED SUCCESSFULLY\nTotal ranks processed: {len(all_rank_data)}\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n")
    return 0

def main():
    """Single directory processing entry point."""
    parser = argparse.ArgumentParser(description="Merge and visualize kernel time intervals", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', '-i', type=str, required=True, help='Root directory containing rank folders')
    parser.add_argument('--files', nargs='+', type=str, default=['bwd_communication.parquet', 'bwd_computation.parquet', 
                                                                 'fwd_communication.parquet', 'fwd_computation.parquet', 
                                                                 'bwd_communication_recompute.parquet', 'bwd_computation_recompute.parquet', 
                                                                 'optimizer_computation.parquet', 
                                                                 'dp_communication.parquet', 
                                                                 'pp_communication.parquet'], help='Parquet files to process')
    parser.add_argument('--time-alignment', type=str, default='global', choices=['individual', 'global'], help='Time alignment mode')
    parser.add_argument('--processes', '-p', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()
    return process_rank_aggregation(input_root=args.input, files=args.files, time_alignment=args.time_alignment,processes=args.processes)

def main_loop_iterations():
    """Batch processing entry point for multiple iterations."""
    parser = argparse.ArgumentParser(description="Batch merge and visualize kernel time intervals", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-folder', type=str, required=True, help='Base directory containing iteration folders')
    parser.add_argument('--iterations', nargs='+', type=str, required=True, help='Iteration folder names to process')
    parser.add_argument('--files', nargs='+', type=str, default=['bwd_communication.parquet', 'bwd_computation.parquet', 
                                                                 'fwd_communication.parquet', 'fwd_computation.parquet', 
                                                                 'bwd_communication_recompute.parquet', 'bwd_computation_recompute.parquet', 
                                                                 'optimizer_computation.parquet', 
                                                                 'dp_communication.parquet', 
                                                                 'pp_communication.parquet'], help='Parquet files to process')
    parser.add_argument('--time-alignment', type=str, default='global', choices=['individual', 'global'], help='Time alignment mode')
    parser.add_argument('--processes', '-p', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()
    
    print(f"{'='*80}\nBatch Processing: {len(args.iterations)} iterations\nBase folder: {args.base_folder}\n{'='*80}")
    success_count = 0
    for iter_idx, iter_folder in enumerate(args.iterations, 1):
        iter_path = os.path.join(args.base_folder, iter_folder)
        if not os.path.exists(iter_path):
            print(f"[{iter_idx}/{len(args.iterations)}] SKIP: {iter_folder} (not found)"); continue
        
        print(f"\n[{iter_idx}/{len(args.iterations)}] Processing: {iter_folder}\n{'-'*80}")
        if process_rank_aggregation(input_root=iter_path, files=args.files, time_alignment=args.time_alignment,processes=args.processes) == 0:
            success_count += 1; print(f"✓ {iter_folder} completed")
        else:
            print(f"✗ {iter_folder} failed")
            
    print(f"\n{'='*80}\nBatch Summary: {success_count}/{len(args.iterations)} iterations successful\n{'='*80}")
    return 0 if success_count == len(args.iterations) else 1

if __name__ == "__main__":
    sys.exit(main_loop_iterations())