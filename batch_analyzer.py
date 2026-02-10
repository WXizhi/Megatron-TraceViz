"""
Multi-process batch processing for Perfetto trace files.
"""
import os
import sys
import argparse
import logging
import uuid
import fnmatch
import re
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from core_analyzer import SliceCollector, run_complete_analysis

def setup_isolated_logger(output_dir, trace_name):
    """
    Create an isolated logger for each trace to avoid conflicts.
    """
    pid = os.getpid()
    unique_id = f"{trace_name}_{pid}_{uuid.uuid4().hex[:8]}"
    logger_name = f'TraceAnalysis_{unique_id}'
    
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    
    log_file = os.path.join(output_dir, f'{trace_name}_TraceAnalysis.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | PID:%(process)d | %(levelname)-8s | %(message)s'))
    logger.addHandler(file_handler)
    
    return logger

def process_single_trace(trace_path_str, base_output_dir_str):
    """
    Worker function to process a single trace file.
    """
    trace_name = os.path.splitext(os.path.basename(trace_path_str))[0]
    trace_output_dir = os.path.join(base_output_dir_str, f"{trace_name}_kernel_analysis")
    trace_logger = None
    
    try:
        os.makedirs(trace_output_dir, exist_ok=True)
        trace_logger = setup_isolated_logger(trace_output_dir, trace_name)
        
        trace_logger.info(f"Processing trace: {trace_name}")
        trace_logger.info(f"Input file: {trace_path_str}")
        trace_logger.info(f"Output directory: {trace_output_dir}")
        
        start_time = datetime.now()
        
        # Inject isolated logger into collector
        collector = SliceCollector(trace_path_str, logger=trace_logger)
        run_complete_analysis(collector, trace_output_dir)
        
        duration = (datetime.now() - start_time).total_seconds()
        file_size_mb = os.path.getsize(trace_path_str) / (1024 * 1024)
        stats = {'trace_name': trace_name, 'duration_sec': duration, 
                 'file_size_mb': file_size_mb, 'output_dir': trace_output_dir}
        
        trace_logger.info(f"Completed successfully in {duration:.2f}s")
        return (trace_path_str, True, None, stats)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if trace_logger:
            trace_logger.error(f"Processing failed: {error_msg}")
            import traceback
            trace_logger.error(traceback.format_exc())
        return (trace_path_str, False, error_msg, None)
        
    finally:
        # Ensure handlers are closed to prevent file handle leaks
        if trace_logger:
            for handler in trace_logger.handlers[:]:
                handler.close()
                trace_logger.removeHandler(handler)

def find_trace_files(input_dir_str, pattern):
    """Recursively find trace files matching pattern and rank regex."""
    trace_files = []
    for root, _, files in os.walk(input_dir_str):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern) and filename.endswith('.json'):
                if re.search(r'rank[_]?(\d+)[._]', filename):
                    trace_files.append(os.path.join(root, filename))
    return sorted(trace_files)

def process_traces_parallel(trace_files, output_dir_str, num_processes=None):
    """
    Process all trace files in parallel using multiprocessing.
    """
    if num_processes is None: num_processes = cpu_count()
    
    print(f"\n{'='*80}\nBATCH TRACE ANALYSIS\n{'='*80}")
    print(f"Total files: {len(trace_files)} | Workers: {num_processes}")
    print(f"Output dir: {output_dir_str}\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n")
    
    os.makedirs(output_dir_str, exist_ok=True)
    results, failed_files, success_stats = [], [], []
    worker_func = partial(process_single_trace, base_output_dir_str=output_dir_str)
    
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(trace_files), desc="Processing", unit="file", ncols=100, 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for result in pool.imap_unordered(worker_func, trace_files):
                trace_path, success, error_msg, stats = result
                results.append(result)
                trace_filename = os.path.basename(trace_path)
                
                if not success:
                    failed_files.append((trace_path, error_msg))
                    tqdm.write(f"Error {trace_filename}: {error_msg}")
                else:
                    success_stats.append(stats)
                    tqdm.write(f"✓ {trace_filename} ({stats['duration_sec']:.1f}s, {stats['file_size_mb']:.1f}MB)")
                pbar.update(1)
    
    # Summary Report
    print(f"\n{'='*80}\nBATCH ANALYSIS COMPLETED\n{'='*80}")
    print(f"Total: {len(trace_files)} | Success: {len(success_stats)} | Failed: {len(failed_files)}")
    
    if success_stats:
        total_time = sum(s['duration_sec'] for s in success_stats)
        total_size = sum(s['file_size_mb'] for s in success_stats)
        print(f"Time: {total_time:.2f}s (Avg: {total_time/len(success_stats):.2f}s)")
        print(f"Data: {total_size:.2f}MB (Avg: {total_size/len(success_stats):.2f}MB)")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save logs
    if failed_files:
        log_path = os.path.join(output_dir_str, "failed_files.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Failed Files - {datetime.now()}\n{'='*80}\n")
            for i, (path, msg) in enumerate(failed_files, 1):
                f.write(f"{i}. {path}\n   Error: {msg}\n{'-'*80}\n")
        print(f"Failed list saved: {log_path}")
    
    if success_stats:
        stats_path = os.path.join(output_dir_str, "processing_stats.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Processing Statistics - {datetime.now()}\n{'='*80}\n")
            f.write(f"{'Trace Name':<60} {'Time (s)':>10} {'Size (MB)':>12}\n{'-'*80}\n")
            for stat in sorted(success_stats, key=lambda x: x['duration_sec'], reverse=True):
                f.write(f"{stat['trace_name']:<60} {stat['duration_sec']:>10.2f} {stat['file_size_mb']:>12.2f}\n")
            f.write(f"\n{'='*80}\nTotal Time: {total_time:.2f}s | Total Size: {total_size:.2f}MB\n")
        print(f"Stats saved: {stats_path}")
    
    print(f"{'='*80}\n")
    return results, failed_files

def main():
    parser = argparse.ArgumentParser(description="Batch process Perfetto traces with isolated logging")
    parser.add_argument('--input', '-i', type=str, default="./data", help="Base input directory")
    parser.add_argument('--processes', '-p', type=int, default=12, help='Number of worker processes')
    parser.add_argument('--pattern', type=str, default='*.json', help='File pattern to match')
    #Added argument for folder patterns, supports multiple inputs (e.g., -f iter_* run_*)
    parser.add_argument('--folders', '-f', nargs='+', default=['*'], help="Folder patterns to match (default: all)")
    args = parser.parse_args()
    
    base_input_dir = os.path.abspath(args.input)
    if not os.path.isdir(base_input_dir):
        print(f"Error: Directory not found: {base_input_dir}")
        return 1

    # Use arguments from command line
    folder_patterns = args.folders
    
    target_dirs = []
    for pattern in folder_patterns:
        # glob handles the wildcard matching logic
        matched = glob.glob(os.path.join(base_input_dir, pattern))
        target_dirs.extend([d for d in matched if os.path.isdir(d)])
    target_dirs = sorted(set(target_dirs))
    
    if not target_dirs:
        print(f"Error: No matching folders found in: {base_input_dir} with patterns {folder_patterns}")
        return 1
    
    print(f"\n{'='*80}\nMULTI-FOLDER BATCH PROCESSING\n{'='*80}")
    print(f"Base: {base_input_dir}\nPatterns: {folder_patterns}\nTargets ({len(target_dirs)}):")
    for i, d in enumerate(target_dirs, 1):
        print(f"  {i:3d}. {os.path.relpath(d, base_input_dir)}")
    print(f"{'='*80}\n")
    
    total_results, total_failed, folder_stats = [], [], []
    
    for folder_idx, input_dir in enumerate(target_dirs, 1):
        rel_path = os.path.relpath(input_dir, base_input_dir)
        print(f"\n{'#'*80}\n# [{folder_idx}/{len(target_dirs)}] Processing: {rel_path}\n{'#'*80}")
        
        trace_files = find_trace_files(input_dir, args.pattern)
        if not trace_files:
            print(f"No matching trace files found.")
            folder_stats.append({'folder': rel_path, 'status': 'skipped', 'count': 0})
            continue
        
        print(f"Found {len(trace_files)} files.")
        try:
            results, failed = process_traces_parallel(trace_files, input_dir, args.processes)
            total_results.extend(results)
            total_failed.extend(failed)
            folder_stats.append({'folder': rel_path, 'status': 'completed', 'count': len(trace_files)})
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            return 130
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")
            folder_stats.append({'folder': rel_path, 'status': 'error', 'count': len(trace_files)})
    
    print(f"\n{'='*80}\nFINAL SUMMARY\n{'='*80}")
    print(f"Folders: {len(target_dirs)} | Total Files: {len(total_results)} | Failed: {len(total_failed)}")
    for stat in folder_stats:
        sym = '✓' if stat['status'] == 'completed' else ('⊘' if stat['status'] == 'skipped' else '✗')
        print(f"  {sym} {stat['folder']}: {stat.get('count', 0)} files")
    print(f"{'='*80}\n")
    
    return 1 if total_failed else 0

if __name__ == "__main__":
    sys.exit(main())