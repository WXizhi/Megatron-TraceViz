import sys
import os
import logging
from datetime import datetime
import pandas as pd
from query_utils import TraceAnalyzer, extract_INTlist_from_query
from config import *
import argparse

def setup_logger(output_dir):
    """Configure and return logger."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('TraceAnalysis')

class SliceCollector:
    def __init__(self, trace_file_path: str, logger=None):
        """Initialize trace analyzer."""
        self.trace_analyzer = TraceAnalyzer(trace_file_path, logger)
        self.trace_path = trace_file_path
        self.fwd_utid = self.trace_analyzer.fwd_utid
        self.bwd_utid = self.trace_analyzer.bwd_utid
        self.kernellaunch_cpu_slicename = self.trace_analyzer.kernellaunch_cpu_slicename
        self.logger = logger if logger else logging.getLogger('TraceAnalysis')
    
    def save_df(self, table_df, filepath):
        """Save DataFrame to parquet file."""
        try:
            if table_df is None or (isinstance(table_df, pd.DataFrame) and table_df.empty):
                self.logger.warning(f"Empty DataFrame, skipping save to {filepath}")
                return False
            table_df.to_parquet(filepath, engine="pyarrow", index=False)
            self.logger.info(f"Successfully saved DataFrame to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving DataFrame to {filepath}: {e}")
            return False     
    
    def query_comp_tpep_fwd(self):
        """
        Identify forward steps and associated kernels.
        Returns: Dict with 'computation' and 'communication' DataFrames.
        """
        # Find key function names (forward_step) to identify micro-batches.
        forward_clause = glob_OR('s.name', PATTERNS_GLOB['forward_cpu_slice'])
        sql_fwd_coarse = f""" 
            SELECT s.id, s.name, s.ts, s.dur, s.category, s.track_id
            FROM slice s JOIN thread_track tt ON s.track_id = tt.id
            WHERE tt.utid = {self.fwd_utid} AND {forward_clause} ORDER BY s.ts;
        """
        try:
            # Coarse query, results may contain hierarchy
            fwd_slices_coarse = self.trace_analyzer.execute_SQL(sql_fwd_coarse)
            fwd_ids_coarse = extract_INTlist_from_query(fwd_slices_coarse, 'id')
            if not fwd_ids_coarse:
                self.logger.warning("No forward slices found")
                return None

            # Filter out hierarchy to get root slices
            unique_ids = sorted(set(fwd_ids_coarse))
            id_clause = ', '.join(str(id) for id in unique_ids)
            fwd_slices = self.trace_analyzer.merge_root(f" id IN ({id_clause}) ")
            fwd_ids = extract_INTlist_from_query(fwd_slices, 'slice_id')
            self.logger.info(f"Forward query : found {len(fwd_ids)} slices")
            
            # Query kernel correlations
            kernels_df = self.trace_analyzer.query_kernel_correlations_cpuop_batch(fwd_ids)
            kernel_ids = kernels_df['slice_id'].tolist()
            kernel_ids_cls_df = self.trace_analyzer.classify_kernels(kernel_ids)
            
            kernels_df.rename(columns={'query_id': 'fwdstep_sliceid', 'launch_id': 'launch_slice_id'}, inplace=True)
            
            # Separate computation and communication kernels
            computation_kernel_ids = kernel_ids_cls_df[kernel_ids_cls_df['slice_type'] == 'computation']['id'].tolist()
            communication_kernel_ids = kernel_ids_cls_df[kernel_ids_cls_df['slice_type'] == 'communication']['id'].tolist()

            # Remove loss_func kernels (usually considered DP)
            lossfunc_slices = self.trace_analyzer.execute_SQL("select * from slice where name GLOB '*pretrain*loss_func*'")
            if lossfunc_slices is not None and lossfunc_slices.row_count > 0:
                lossfunc_ids = extract_INTlist_from_query(lossfunc_slices, 'id')
                lossfunc_kernels_df = self.trace_analyzer.query_kernels_with_ancestor_match(communication_kernel_ids, lossfunc_ids)
                lossfunc_kernels = lossfunc_kernels_df['id'].tolist() if lossfunc_kernels_df is not None else []
                self.logger.info(f"Forward pass with {len(lossfunc_kernels)} communication kernels in loss func")
                communication_kernel_ids = list(set(communication_kernel_ids) - set(lossfunc_kernels))

            computation_kernels_df = kernels_df[kernels_df['slice_id'].isin(computation_kernel_ids)]
            communication_kernels_df = kernels_df[kernels_df['slice_id'].isin(communication_kernel_ids)]
            self.logger.info(f"Forward query: {len(computation_kernels_df)} computation, {len(communication_kernels_df)} communication kernels")
            return {'computation': computation_kernels_df, 'communication': communication_kernels_df}
        except Exception as e:
            self.logger.error(f"error in {sys._getframe().f_code.co_name}: {str(e)} of {self.trace_path}")
            return None

    def query_comp_tpep_bwd_wrecompt(self):
        """
        Identify backward steps and associated kernels, including recomputation.
        Strategy: Find bwd_funcs on fwd_utid, then filter slices using time range on bwd thread.
        """
        backward_clause = glob_OR('s.name', PATTERNS_GLOB['backward_cpu_slice'])
        sql_bwd_coarse = f""" 
            SELECT s.id, s.name, s.ts, s.dur, s.category, s.track_id
            FROM slice s JOIN thread_track tt ON s.track_id = tt.id
            WHERE tt.utid = {self.fwd_utid} AND {backward_clause} ORDER BY s.ts;
        """
        try:
            # Coarse query for backward steps on FWD thread
            bwd_slices_coarse = self.trace_analyzer.execute_SQL(sql_bwd_coarse)
            bwd_ids_coarse = extract_INTlist_from_query(bwd_slices_coarse, 'id')
            if not bwd_ids_coarse:
                self.logger.warning("No backward slices found")
                return None

            unique_ids = sorted(set(bwd_ids_coarse))
            id_clause = ', '.join(str(id) for id in unique_ids)
            bwd_slices = self.trace_analyzer.merge_root(f" id IN ({id_clause}) ")
            bwd_ids = extract_INTlist_from_query(bwd_slices, 'slice_id')
            self.logger.info(f"Backward query : found {len(bwd_ids)} slices")
            
            # Use time range to select slices on BWD thread
            bwd_ids4sql = ','.join(map(str, bwd_ids))
            launchkernel_clause = glob_OR('ms.name', self.kernellaunch_cpu_slicename) 
            
            sql_bwd_tmpTable = f"""
            DROP TABLE IF EXISTS temp_launch_kernel_ids;
            CREATE TEMP TABLE temp_launch_kernel_ids AS
            WITH target_slices AS (
                SELECT s.id as target_slice_id, s.ts as start_ts, s.ts + s.dur as end_ts
                FROM slice s WHERE s.id IN ({bwd_ids4sql})
            ),
            matching_slices AS (
                SELECT s.id, s.name, t.target_slice_id
                FROM slice s
                JOIN thread_track tt ON s.track_id = tt.id
                JOIN target_slices t ON (s.ts >= t.start_ts AND s.ts <= t.end_ts)
                WHERE tt.utid = {self.bwd_utid} AND s.dur > 0
                    AND s.id NOT IN (SELECT target_slice_id FROM target_slices)
            )
            SELECT ms.id as launch_slice_id, ms.name as launch_name, ms.target_slice_id
            FROM matching_slices ms WHERE {launchkernel_clause};
            CREATE INDEX IF NOT EXISTS idx_temp_lk_slice ON temp_launch_kernel_ids(launch_slice_id);
            """
            
            sql_bwd_kernels = f"""
            SELECT 
                cm.gpu_slice_id as id, cm.gpu_slice_id as slice_id, cm.gpu_ts as ts, cm.gpu_dur as dur,
                NULL as track_id, cm.gpu_kernel_name as name, 'kernel' as category,
                tlk.target_slice_id, tlk.launch_slice_id, tlk.launch_name, cm.correlation_id,
                cm.gpu_dur / 1000000.0 as duration_ms,
                (cm.gpu_ts - (SELECT MIN(ts) FROM slice WHERE ts > 0)) / 1000000.0 as relative_start_ms
            FROM correlation_mapping cm
            JOIN temp_launch_kernel_ids tlk ON cm.slice_id = tlk.launch_slice_id
            WHERE cm.gpu_slice_id IS NOT NULL
            ORDER BY tlk.target_slice_id, cm.gpu_ts;
            """
            self.trace_analyzer.execute_SQL(sql_bwd_tmpTable)
            bwd_kernels = self.trace_analyzer.execute_SQL(sql_bwd_kernels)   
            self.trace_analyzer.execute_SQL("DROP TABLE IF EXISTS temp_launch_kernel_ids;")
            
            kernel_ids = extract_INTlist_from_query(bwd_kernels, 'slice_id')
            if not kernel_ids:
                self.logger.warning("No backward kernels found")
                return None
            kernel_ids_cls_df = self.trace_analyzer.classify_kernels(kernel_ids)

            bwd_kernels_df = bwd_kernels.as_pandas_dataframe()
            bwd_kernels_df.rename(columns={'target_slice_id': 'bwdstep_sliceid'}, inplace=True)

            computation_kernel_ids = kernel_ids_cls_df[kernel_ids_cls_df['slice_type'] == 'computation']['id'].tolist()
            communication_kernel_ids = kernel_ids_cls_df[kernel_ids_cls_df['slice_type'] == 'communication']['id'].tolist()
            
            computation_kernels_df = bwd_kernels_df[bwd_kernels_df['slice_id'].isin(computation_kernel_ids)]
            communication_kernels_df = bwd_kernels_df[bwd_kernels_df['slice_id'].isin(communication_kernel_ids)]
            self.logger.info(f"Backward query with recompute: {len(computation_kernels_df)} computation, {len(communication_kernels_df)} communication kernels")
            return {'computation': computation_kernels_df, 'communication': communication_kernels_df}
        except Exception as e:
            self.logger.error(f"error in {sys._getframe().f_code.co_name}: {str(e)} of {self.trace_path}")
            return None
    
    def query_comm_pp_bubble(self):
        """Query Pipeline Parallelism (PP) communication kernels (bubbles)."""
        try:
            filter_condition = glob_OR('name', PATTERNS_GLOB['pp_cpu_slice'])
            pp_slices = self.trace_analyzer.merge_root(filter_condition)
            pp_ids = extract_INTlist_from_query(pp_slices, 'slice_id')
            if not pp_ids:
                self.logger.warning("No PP communication slices found")
                return None
            
            kernels_df = self.trace_analyzer.query_kernel_correlations_cpuop_batch(pp_ids)
            kernels_df.rename(columns={'query_id': 'pp_sliceid', 'launch_id': 'launch_slice_id'}, inplace=True)
            
            kernel_ids = kernels_df['slice_id'].tolist()
            kernel_ids_cls_df = self.trace_analyzer.classify_kernels(kernel_ids)
            communication_kernel_ids = kernel_ids_cls_df[kernel_ids_cls_df['slice_type'] == 'communication']['id'].tolist()
            
            ppcomm_kernels_df = kernels_df[kernels_df['slice_id'].isin(communication_kernel_ids)]
            self.logger.info(f"PP query: {len(ppcomm_kernels_df)} communication kernels")
            return ppcomm_kernels_df
        except Exception as e:
            self.logger.error(f"Error in {sys._getframe().f_code.co_name}: {e} of {self.trace_path}")
            return None

    def query_comm_dp(self):
        """Query Data Parallelism (DP) communication kernels."""
        try:
            filter_condition = glob_OR('name', PATTERNS_GLOB['dp_cpu_slice'])
            dpopt_slices = self.trace_analyzer.merge_root(filter_condition)
            dpopt_slices_ids = extract_INTlist_from_query(dpopt_slices, 'slice_id')
            if not dpopt_slices_ids:
                self.logger.warning("No DP communication slices found")
                return None
            
            kernels_df = self.trace_analyzer.query_kernel_correlations_cpuop_batch(dpopt_slices_ids)
            kernels_df.rename(columns={'query_id': 'dp_sliceid', 'launch_id': 'launch_slice_id'}, inplace=True)
            
            kernel_ids = kernels_df['slice_id'].tolist()
            kernel_ids_cls_df = self.trace_analyzer.classify_kernels(kernel_ids)
            communication_kernel_ids = kernel_ids_cls_df[kernel_ids_cls_df['slice_type'] == 'communication']['id'].tolist()
            
            dpcomm_kernels_df = kernels_df[kernels_df['slice_id'].isin(communication_kernel_ids)]
            self.logger.info(f"DP query: {len(dpcomm_kernels_df)} communication kernels")
            return dpcomm_kernels_df
        except Exception as e:
            self.logger.error(f"Error in {sys._getframe().f_code.co_name}: {e} of {self.trace_path}")
            return None
    
    def query_comp_opt(self):
        """Query Optimizer computation kernels."""
        try:
            filter_condition = glob_OR('name', PATTERNS_GLOB['optimizer_cpu_slice'])
            opt_slices = self.trace_analyzer.merge_root(filter_condition)
            opt_slices_ids = extract_INTlist_from_query(opt_slices, 'slice_id')
            if not opt_slices_ids:
                self.logger.warning("No optimizer slices found")
                return None

            kernels_df = self.trace_analyzer.query_kernel_correlations_cpuop_batch(opt_slices_ids)
            kernels_df.rename(columns={'query_id': 'opt_sliceid', 'launch_id': 'launch_slice_id'}, inplace=True)
            
            kernel_ids = kernels_df['slice_id'].tolist()
            kernel_ids_cls_df = self.trace_analyzer.classify_kernels(kernel_ids)
            computation_kernel_ids = kernel_ids_cls_df[kernel_ids_cls_df['slice_type'] == 'computation']['id'].tolist()
            
            optcomp_kernels_df = kernels_df[kernels_df['slice_id'].isin(computation_kernel_ids)]
            self.logger.info(f"Optimizer query: {len(optcomp_kernels_df)} computation kernels")
            return optcomp_kernels_df
        except Exception as e:
            self.logger.error(f"Error in {sys._getframe().f_code.co_name}: {e} of {self.trace_path}")
            return None
    
    def _get_recompute_slices(self):
        """Find recomputation slices on the backward thread."""
        recompute_clause = glob_OR('s.name', PATTERNS_GLOB['recompute_cpu_slice'])
        sql_query = f"""
        WITH RECURSIVE optimize_slices AS (
            SELECT s.id, s.name, s.ts, s.dur, s.track_id, s.parent_id
            FROM slice s JOIN thread_track tt ON s.track_id = tt.id
            WHERE {recompute_clause} AND s.dur > 0 AND tt.utid = {self.bwd_utid}  
        ),
        all_ancestors AS (
            SELECT id as original_id, id, parent_id, 0 as depth FROM optimize_slices
            UNION ALL
            SELECT aa.original_id, parent.id, parent.parent_id, aa.depth + 1
            FROM all_ancestors aa JOIN slice parent ON aa.parent_id = parent.id
            WHERE aa.depth < {self.trace_analyzer.recursive_depth}
        ),
        ancestors_with_flag AS (
            SELECT aa.original_id, s.id, s.name, s.ts, s.dur, s.track_id, s.parent_id, aa.depth,
                CASE WHEN s.id IN (SELECT id FROM optimize_slices) THEN 1 ELSE 0 END as in_optimize_flag
            FROM all_ancestors aa JOIN slice s ON aa.id = s.id
        ),
        final_results AS (
            SELECT original_id,
                CASE WHEN MAX(CASE WHEN in_optimize_flag = 1 AND depth > 0 THEN depth ELSE -1 END) > -1 
                THEN (SELECT id FROM ancestors_with_flag awf2 
                      WHERE awf2.original_id = awf.original_id AND awf2.in_optimize_flag = 1 AND awf2.depth > 0
                      ORDER BY awf2.depth DESC LIMIT 1)
                ELSE original_id END as result_id
            FROM ancestors_with_flag awf GROUP BY original_id
        )
        SELECT DISTINCT os.id, os.ts, os.dur, os.track_id, os.id as slice_id, os.name, os.parent_id
        FROM optimize_slices os JOIN final_results fr ON os.id = fr.result_id ORDER BY os.ts;
        """
        try:
            result = self.trace_analyzer.execute_SQL(sql_query)
            if result.row_count == 0:
                self.logger.warning("No recompute slices found")
                return None
            recompute_ids = extract_INTlist_from_query(result, column_name='id')
            self.logger.info(f"Found {len(recompute_ids)} recompute slices")
            return recompute_ids
        except Exception as e:
            self.logger.error(f"Error in {sys._getframe().f_code.co_name}: {e} of {self.trace_path}")
            return None
   
    def _map_recompute_to_backward_slices(self, recompute_kernels, bwd_target_ids):
        """Map recompute kernels to corresponding backward slices (vectorized)."""
        if recompute_kernels is None or recompute_kernels.empty:
            self.logger.warning("Empty recompute_kernels")
            return None
        if bwd_target_ids is None or len(bwd_target_ids) == 0:
            self.logger.warning("Empty bwd_target_ids")
            return recompute_kernels
        
        try:
            if isinstance(bwd_target_ids, list):
                bwd_ids = list(set(bwd_target_ids))
            elif hasattr(bwd_target_ids, 'tolist'):
                bwd_ids = list(set(bwd_target_ids.tolist()))
            else:
                self.logger.error(f"Unsupported type: {type(bwd_target_ids)}")
                return recompute_kernels
            
            # Query time intervals
            bwd_ids_str = ','.join(map(str, bwd_ids))
            sql_bwd = f"SELECT id as bwd_id, ts as bwd_start_ts, ts + dur as bwd_end_ts FROM slice WHERE id IN ({bwd_ids_str})"
            bwd_time_df = self.trace_analyzer.execute_SQL(sql_bwd).as_pandas_dataframe()
            if bwd_time_df.empty: return recompute_kernels
            
            unique_recomp_ids = recompute_kernels['recomp_sliceid'].unique()
            recomp_ids_str = ','.join(map(str, unique_recomp_ids))
            sql_recomp = f"SELECT id as recomp_id, ts as recomp_ts FROM slice WHERE id IN ({recomp_ids_str})"
            recomp_time_df = self.trace_analyzer.execute_SQL(sql_recomp).as_pandas_dataframe()
            if recomp_time_df.empty: return recompute_kernels
            
            # Vectorized matching
            recomp_time_df['_key'] = 1; bwd_time_df['_key'] = 1
            merged = recomp_time_df.merge(bwd_time_df, on='_key', how='outer')
            matched = merged[(merged['bwd_start_ts'] <= merged['recomp_ts']) & (merged['recomp_ts'] <= merged['bwd_end_ts'])].copy()
            
            if matched.empty:
                self.logger.warning("No matches found between recomp_sliceids and backward slices")
                recompute_kernels = recompute_kernels.copy()
                recompute_kernels['bwdstep_sliceid'] = None
                return recompute_kernels
            
            # Check for 1-to-many anomalies
            dup_counts = matched.groupby('recomp_id').size()
            if (dup_counts > 1).any():
                problematic_ids = dup_counts[dup_counts > 1].index.tolist()
                self.logger.error(f"ABNORMAL: Found {len(problematic_ids)} recomp_sliceids with multiple backward matches!")
                # Fallback: select smallest interval
                matched['interval_size'] = matched['bwd_end_ts'] - matched['bwd_start_ts']
                matched = matched.loc[matched.groupby('recomp_id')['interval_size'].idxmin()]
            
            recomp_to_bwd_map = dict(zip(matched['recomp_id'], matched['bwd_id']))
            recompute_kernels = recompute_kernels.copy()
            recompute_kernels['bwdstep_sliceid'] = recompute_kernels['recomp_sliceid'].map(recomp_to_bwd_map)
            return recompute_kernels
        except Exception as e:
            self.logger.error(f"Mapping error: {e}")
            return recompute_kernels

    def query_comp_comm_recompute(self, bwdslices, bwdstep_sliceid=None):
        """Query recompute kernels and map them to backward slices."""
        recompute_slices = self._get_recompute_slices()
        if recompute_slices is None or not bwdslices:
            return None
        
        recompute_kernels = self.trace_analyzer.query_kernels_with_ancestor_match(bwdslices, recompute_slices)
        if recompute_kernels is None:
            self.logger.info(f"No recompute kernels found")
            return None
        recompute_kernels.rename(columns={'matched_ancestor_id': 'recomp_sliceid'}, inplace=True)
        
        if bwdstep_sliceid is None:
            return recompute_kernels
        else:
            self.logger.debug(f"Using provided bwdstep_sliceid for mapping ({len(bwdstep_sliceid)} IDs)")
            recompute_kernels = self._map_recompute_to_backward_slices(recompute_kernels, bwdstep_sliceid)
            self.logger.info(f"Recompute query: {len(recompute_kernels)} matched kernels")
            return recompute_kernels

def run_complete_analysis(collector, output_dir):
    """Run complete trace analysis and save results."""
    collector.logger.info("=" * 80)
    collector.logger.info("PERFETTO TRACE COMPLETE ANALYSIS")
    collector.logger.info("=" * 80)
    
    analysis_dir = output_dir
    os.makedirs(analysis_dir, exist_ok=True)

    # 1. Forward Analysis
    collector.logger.info("1. Forward Computation/Communication Analysis")
    fwd_results = collector.query_comp_tpep_fwd()
    if fwd_results:
        collector.save_df(fwd_results['computation'], os.path.join(analysis_dir, 'fwd_computation.parquet'))
        collector.save_df(fwd_results['communication'], os.path.join(analysis_dir, 'fwd_communication.parquet'))

    # 2. Backward Analysis
    collector.logger.info("2. Backward Computation/Communication Analysis")
    bwd_results = collector.query_comp_tpep_bwd_wrecompt()
    if bwd_results:
        bwd_comp_df = bwd_results['computation']
        bwd_comm_df = bwd_results['communication']
        collector.save_df(bwd_comp_df, os.path.join(analysis_dir, 'bwd_computation.parquet'))
        collector.save_df(bwd_comm_df, os.path.join(analysis_dir, 'bwd_communication.parquet'))
        
        # 2.1 Recompute Analysis
        collector.logger.info("2.1 Recompute Analysis for Backward Kernels")
        collector.logger.info("Recompute Analysis for Backward Computation Kernels")
        recompute_comp = collector.query_comp_comm_recompute(
            list(bwd_comp_df['slice_id'].unique()), bwdstep_sliceid=bwd_comp_df['bwdstep_sliceid'].unique()
        )
        if recompute_comp is not None:
            collector.save_df(recompute_comp, os.path.join(analysis_dir, 'bwd_computation_recompute.parquet'))
        
        collector.logger.info("Recompute Analysis for Backward Communication Kernels")
        recompute_comm = collector.query_comp_comm_recompute(
            list(bwd_comm_df['slice_id'].unique()), bwdstep_sliceid=bwd_comp_df['bwdstep_sliceid'].unique()
        )
        if recompute_comm is not None:
            collector.save_df(recompute_comm, os.path.join(analysis_dir, 'bwd_communication_recompute.parquet'))
            
    # 3. PP Communication Analysis
    collector.logger.info("3. Pipeline Parallel (PP) Communication Analysis")
    pp_results = collector.query_comm_pp_bubble()
    if pp_results is not None:
        collector.save_df(pp_results, os.path.join(analysis_dir, 'pp_communication.parquet'))
        
    # 4. DP Communication Analysis
    collector.logger.info("4. Data Parallel (DP) Communication Analysis")
    dp_results = collector.query_comm_dp()
    if dp_results is not None:
        collector.save_df(dp_results, os.path.join(analysis_dir, 'dp_communication.parquet'))
      
    # 5. Optimizer Computation Analysis
    collector.logger.info("5. Optimizer Computation Analysis")
    opt_results = collector.query_comp_opt()
    if opt_results is not None:
        collector.save_df(opt_results, os.path.join(analysis_dir, 'optimizer_computation.parquet'))

def main():
    parser = argparse.ArgumentParser(description="Run core analysis on a single Perfetto trace file")
    parser.add_argument('--input', '-i', type=str, required=True, help="Path to input .trace.json file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1

    # Output directory setup (logger will create the directory)
    output_dir = os.path.join(os.path.dirname(args.input), 'kernel_analysis')
    logger = setup_logger(output_dir)

    logger.info(f"Initializing SliceCollector for trace: {args.input}")
    collector = SliceCollector(args.input, logger)
    
    try:
        run_complete_analysis(collector, output_dir)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
    except Exception as e:
        logger.error(f"Fatal error in main analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    import argparse
    sys.exit(main())