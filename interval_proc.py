import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

def format_duration(duration_ns):
    """Format nanoseconds into a human-readable string (s, ms, us, ns)."""
    if pd.isna(duration_ns): return "N/A"
    duration_ns = float(duration_ns)
    if duration_ns >= 1_000_000_000: return f"{duration_ns / 1_000_000_000:.2f} s"
    elif duration_ns >= 1_000_000: return f"{duration_ns / 1_000_000:.2f} ms"
    elif duration_ns >= 1_000: return f"{duration_ns / 1_000:.2f} us"
    else: return f"{duration_ns:.2f} ns"

class IntervalMerger:
    """Tool class for merging time intervals."""
    def __init__(self):
        self.step_columns = ['fwdstep_sliceid', 'bwdstep_sliceid', 'pp_sliceid', 'dp_sliceid', 'opt_sliceid']
        self.filename_to_category = {
            'optimizer_computation.parquet': 'optimizer_computation',
            'dp_communication.parquet': 'dp_communication',
            'pp_communication.parquet': 'pp_communication',
            'bwd_communication.parquet': 'bwd_communication',
            'bwd_computation.parquet': 'bwd_computation',
            'fwd_communication.parquet': 'fwd_communication',
            'fwd_computation.parquet': 'fwd_computation',
            'bwd_computation_recompute.parquet': 'bwd_computation_recompute',
            'bwd_communication_recompute.parquet': 'bwd_communication_recompute'
        }
        self.target_files = list(self.filename_to_category.keys())

    def load_parquet(self, filepath):
        if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
        try: return pd.read_parquet(filepath)
        except Exception as e: raise IOError(f"Error loading {filepath}: {e}")

    def detect_step_column(self, df):
        for col in self.step_columns:
            if col in df.columns: return col
        return None

    def validate_dataframe(self, df):
        if missing := [c for c in ['ts', 'dur'] if c not in df.columns]: return False, f"Missing: {missing}"
        if self.detect_step_column(df) is None: return False, f"No step column found in {self.step_columns}"
        return True, None

    def merge_intervals_mbs(self, df, step_col=None):
        """Merge kernel intervals by step_id."""
        if df is None or df.empty: return pd.DataFrame(columns=['step_id', 'start_time', 'end_time', 'duration'])
        valid, msg = self.validate_dataframe(df)
        if not valid: raise ValueError(msg)
        step_col = step_col or self.detect_step_column(df)
        df = df.copy()
        df['end_time'] = df['ts'] + df['dur']
        merged = df.groupby(step_col, as_index=False).agg({'ts': 'min', 'end_time': 'max'})
        merged.rename(columns={step_col: 'step_id', 'ts': 'start_time'}, inplace=True)
        merged['duration'] = merged['end_time'] - merged['start_time']
        return merged.sort_values('start_time').reset_index(drop=True)

    @staticmethod
    def merge_overlapping_intervals_vectorized(df):
        """Vectorized merge of overlapping intervals."""
        if df is None or df.empty: return pd.DataFrame(columns=['start_time', 'end_time', 'duration'])
        if len(df) == 1:
            res = df.copy(); res['duration'] = res['end_time'] - res['start_time']; return res
        df_sorted = df.sort_values('start_time').reset_index(drop=True)
        df_sorted['group_id'] = (df_sorted['start_time'] > df_sorted['end_time'].shift(1).cummax()).cumsum()
        merged = df_sorted.groupby('group_id', as_index=False).agg({'start_time': 'min', 'end_time': 'max'})
        merged['duration'] = merged['end_time'] - merged['start_time']
        return merged.drop(columns=['group_id'])

    def merge_singleRank_mbs(self, input_path):
        """Merge all parquet intervals in a rank by mbs into single blocks."""
        df = self.load_parquet(input_path)
        fname = os.path.basename(input_path)
        cat = self.filename_to_category.get(fname, fname.replace('.parquet', ''))
        step_col = 'slice_id' if fname == 'optimizer_computation.parquet' else self.detect_step_column(df)
        actual_stats = df.groupby(step_col)['dur'].sum().to_dict()
        merged_df = self.merge_intervals_mbs(df, step_col=step_col if fname == 'optimizer_computation.parquet' else None)
        
        if merged_df.empty:
            print("Warning: No intervals to merge")
            return pd.DataFrame(columns=['category', 'mbs_id', 'slice_id', 'start_time', 'end_time', 
                                       'actual_duration', 'merged_duration', 'actual_duration_ns', 'merged_duration_ns'])
        
        merged_df['mbs_id'] = range(len(merged_df))
        merged_df['actual_duration_ns'] = merged_df['step_id'].map(actual_stats).astype('int64')
        merged_df['category'] = cat
        merged_df.rename(columns={'duration': 'merged_duration_ns', 'step_id': 'slice_id'}, inplace=True)
        merged_df['merged_duration_ns'] = merged_df['merged_duration_ns'].astype('int64')
        merged_df['merged_duration'] = merged_df['merged_duration_ns'].apply(format_duration)
        merged_df['actual_duration'] = merged_df['actual_duration_ns'].apply(format_duration)
        return merged_df[['category', 'mbs_id', 'slice_id', 'start_time', 'end_time', 
                         'actual_duration', 'merged_duration', 'actual_duration_ns', 'merged_duration_ns']]

    def calculate_comm_comp_overlap(self, kernel_analysis_dir):
        """Calculate overlap between communication and computation for each MBS."""
        file_map = {'Forward': {'comm': 'fwd_communication.parquet', 'comp': 'fwd_computation.parquet'},
                    'Backward': {'comm': ['bwd_communication.parquet'], 'comp': ['bwd_computation.parquet']}}
        results = []
        for phase, files in file_map.items():
            comm_dfs = [self.load_parquet(os.path.join(kernel_analysis_dir, f)) 
                       for f in (files['comm'] if isinstance(files['comm'], list) else [files['comm']]) 
                       if os.path.exists(os.path.join(kernel_analysis_dir, f))]
            comp_dfs = [self.load_parquet(os.path.join(kernel_analysis_dir, f)) 
                       for f in (files['comp'] if isinstance(files['comp'], list) else [files['comp']]) 
                       if os.path.exists(os.path.join(kernel_analysis_dir, f))]
            
            if not comm_dfs or not comp_dfs:
                print(f"⚠ Warning: Missing files for {phase}"); continue
                
            comm_df, comp_df = pd.concat(comm_dfs, ignore_index=True), pd.concat(comp_dfs, ignore_index=True)
            comm_step, comp_step = self.detect_step_column(comm_df), self.detect_step_column(comp_df)
            if not comm_step or not comp_step: print(f"⚠ Warning: Missing step col for {phase}"); continue
            
            comm_df['end_time'], comp_df['end_time'] = comm_df['ts'] + comm_df['dur'], comp_df['ts'] + comp_df['dur']
            all_slices = sorted(set(comm_df[comm_step].unique()) | set(comp_df[comp_step].unique()))
            
            # Sort slices by start time to assign mbs_id
            slice_starts = {}
            for sid in all_slices:
                starts = [df[df[col] == sid]['ts'].min() for df, col in [(comm_df, comm_step), (comp_df, comp_step)] 
                         if not df[df[col] == sid].empty]
                if starts: slice_starts[sid] = min(starts)
            sorted_slices = sorted(slice_starts, key=slice_starts.get)
            slice_to_mbs = {sid: i for i, sid in enumerate(sorted_slices)}
            
            for sid in all_slices:
                c_mbs, p_mbs = comm_df[comm_df[comm_step] == sid], comp_df[comp_df[comp_step] == sid]
                if c_mbs.empty and p_mbs.empty: continue
                
                starts = ([c_mbs['ts'].min()] if not c_mbs.empty else []) + ([p_mbs['ts'].min()] if not p_mbs.empty else [])
                ends = ([c_mbs['end_time'].max()] if not c_mbs.empty else []) + ([p_mbs['end_time'].max()] if not p_mbs.empty else [])
                wall_ns = int(max(ends) - min(starts))
                
                act_comm = int(self.merge_overlapping_intervals_vectorized(c_mbs[['ts', 'end_time']].rename(columns={'ts': 'start_time'}))['duration'].sum()) if not c_mbs.empty else 0
                act_comp = int(self.merge_overlapping_intervals_vectorized(p_mbs[['ts', 'end_time']].rename(columns={'ts': 'start_time'}))['duration'].sum()) if not p_mbs.empty else 0
                overlap = self._calculate_overlap_time(c_mbs[['ts', 'end_time']].values.tolist(), p_mbs[['ts', 'end_time']].values.tolist())
                
                results.append({'phase': phase, 'mbs_id': slice_to_mbs[sid], 'slice_id': int(sid),
                               'start_time': int(min(starts)), 'wall_clock_ns': wall_ns,
                               'actual_comm_time_ns': act_comm, 'actual_comp_time_ns': act_comp,
                               'comm_comp_overlap_time_ns': overlap, 'wall_clock': format_duration(wall_ns),
                               'actual_comm_time': format_duration(act_comm), 'actual_comp_time': format_duration(act_comp),
                               'comm_comp_overlap_time': format_duration(overlap)})
        
        return pd.DataFrame(results)[['phase', 'mbs_id', 'slice_id', 'start_time', 'wall_clock', 'actual_comm_time', 
                                    'actual_comp_time', 'comm_comp_overlap_time', 'wall_clock_ns', 'actual_comm_time_ns',
                                    'actual_comp_time_ns', 'comm_comp_overlap_time_ns']].sort_values(['phase', 'mbs_id']).reset_index(drop=True) if results else pd.DataFrame()

    def _calculate_overlap_time(self, ints1, ints2):
        if not ints1 or not ints2: return 0
        m1, m2 = self._merge_interval_list(ints1), self._merge_interval_list(ints2)
        return sum(max(0, min(e1, e2) - max(s1, s2)) for s1, e1 in m1 for s2, e2 in m2)

    def _merge_interval_list(self, intervals):
        if not intervals: return []
        merged = self.merge_overlapping_intervals_vectorized(pd.DataFrame(intervals, columns=['start_time', 'end_time']))
        return list(zip(merged['start_time'], merged['end_time']))

class MBS_IntervalLoader:
    def __init__(self):
        self.node_durations = {}
        self.node_intervals = {}
        self.node_comp_ratio = {}
        self.node_overlap_ratio = {}
        self.num_microbatches = 0
        self.pp_ranks = []
        self.merger = IntervalMerger()

    def load_from_pp_group_paths(self, pp_paths: List[str], pp_ranks: List[int]) -> Dict[str, float]:
        """Load data from PP Group ranks and calculate ratios."""
        if len(pp_paths) != len(pp_ranks): raise ValueError(f"Path count ({len(pp_paths)}) != Rank count ({len(pp_ranks)})")
        self.pp_ranks = pp_ranks
        
        first_csv = os.path.join(pp_paths[0], f'Rank{pp_ranks[0]}_statistics.csv')
        if not os.path.exists(first_csv): raise FileNotFoundError(f"CSV not found: {first_csv}")
        self.num_microbatches = self._infer_num_microbatches(pd.read_csv(first_csv))
        
        for idx, (rank, path) in enumerate(zip(pp_ranks, pp_paths)):
            csv_file = os.path.join(path, f'Rank{rank}_statistics.csv')
            if not os.path.exists(csv_file): raise FileNotFoundError(f"CSV not found: {csv_file}")
            df = pd.read_csv(csv_file)
            
            overlap_csv = os.path.join(path, f'Rank{rank}_commcompOverlap_analysis.csv')
            overlap_df = pd.read_csv(overlap_csv) if os.path.exists(overlap_csv) else self.merger.calculate_comm_comp_overlap(path)
            
            for mb in range(self.num_microbatches):
                for phase, p_key in [('fwd', 'Forward'), ('bwd', 'Backward')]:
                    node_id = f"{'F' if phase=='fwd' else 'B'}_{idx}_{mb}"
                    dur, interval = self._extract_phase_duration(df, phase, mb)
                    if dur:
                        self.node_durations[node_id] = dur
                        self.node_intervals[node_id] = interval
                        self.node_comp_ratio[node_id] = self._calculate_comp_ratio(overlap_df, p_key, mb, dur)
                        self.node_overlap_ratio[node_id] = self._calculate_overlap_ratio(overlap_df, p_key, mb, dur)
        return self.node_intervals

    def _calculate_comp_ratio(self, df, phase, mb, dur):
        if df.empty or dur == 0: return 0.0
        row = df[(df['phase'] == phase) & (df['mbs_id'] == mb)]
        return row['actual_comp_time_ns'].iloc[0] / dur if not row.empty else 0.0

    def _calculate_overlap_ratio(self, df, phase, mb, dur):
        if df.empty or dur == 0: return 0.0
        row = df[(df['phase'] == phase) & (df['mbs_id'] == mb)]
        if row.empty: return 0.0
        ratio = row['comm_comp_overlap_time_ns'].iloc[0] / dur
        return max(0.0, min(1.0, ratio))

    def _infer_num_microbatches(self, df):
        if 'mbs_id' not in df.columns or 'category' not in df.columns: raise ValueError("Missing mbs_id/category columns")
        comp_df = df[df['category'].isin(['fwd_computation', 'bwd_computation'])]
        if comp_df.empty: raise ValueError("No computation records found")
        mbs = sorted(comp_df['mbs_id'].dropna().unique())
        if mbs != list(range(len(mbs))): print(f"⚠ Warning: Non-sequential mbs_ids: {mbs}")
        return len(mbs)

    def _extract_phase_duration(self, df, phase, mb):
        rows = df[(df['category'].str.contains(phase, case=False, na=False)) & (df['mbs_id'] == mb)]
        if rows.empty: return None, (None, None)
        ints = rows[['start_time', 'end_time']].values.tolist()
        if not ints: return 0, (0, 0)
        merged = IntervalMerger.merge_overlapping_intervals_vectorized(pd.DataFrame(ints, columns=['start_time', 'end_time']))
        return (int(merged['duration'].sum()), (int(merged['start_time'].min()), int(merged['end_time'].max()))) if not merged.empty else (0, (0, 0))

    def get_node_duration(self, node_id): return self.node_durations.get(node_id, 0.0)
    def get_node_interval(self, node_id): return self.node_intervals.get(node_id, (0, 0))
    def get_node_comp_ratio(self, node_id): return self.node_comp_ratio.get(node_id, 0.0)
    def get_node_overlap_ratio(self, node_id): return self.node_overlap_ratio.get(node_id, 0.0)