import sys
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
from config import *

def extract_INTlist_from_query(query_result, column_name='id') -> Optional[List[int]]:
    logger = logging.getLogger('TraceAnalysis')
    try:
        if query_result is None:
            logger.warning(f"Query result is None for column '{column_name}'")
            return None
        df = query_result.as_pandas_dataframe()
        if column_name not in df.columns:
            logger.error(f"Column '{column_name}' not found. Available: {list(df.columns)}")
            return None
        return df[column_name].dropna().astype(int).tolist()
    except Exception as e:
        logger.error(f"Error in {sys._getframe().f_code.co_name}: {e}")
        return None

class TraceAnalyzer:
    def __init__(self, trace_file_path: str, logger: Optional[logging.Logger] = None):
        """Initialize TraceProcessor and build correlation mapping."""
        self.logger = logger if logger else logging.getLogger('TraceAnalysis')
        self.trace_path = trace_file_path
        try:
            # Use path from config.py
            config = TraceProcessorConfig(bin_path=TRACE_PROCESSOR_BIN_PATH, verbose=False)
            self.tp = TraceProcessor(trace=trace_file_path, config=config)
            self.logger.info("TraceProcessor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize TraceProcessor: {e} for {self.trace_path}")
            raise

        self.recursive_depth = self._get_max_depth() + 1 # Default ~80
        self.logger.info(f"Recursive depth set to: {self.recursive_depth}")
        self.kernellaunch_cpu_slicename = None # Auto-detected later
        
        self._create_correlation_mapping()
        self.fwd_utid = None
        self.bwd_utid = None
        self._assign_fwdbwd_thread()
        self._kernels_cls_df = None
        self._create_kernels_cls() 

    def _get_max_depth(self) -> int:
        try:
            result = self.tp.query('SELECT MAX(depth) AS max_depth FROM slice')
            return result.as_pandas_dataframe()['max_depth'][0]
        except Exception as e:
            self.logger.error(f"Error in _get_max_depth: {e}")
            return 80
    
    def _create_correlation_mapping(self):
        """
        Create correlation mapping table and indexes, avoiding direct flow table usage.
        Populates self.kernellaunch_cpu_slicename.
        """
        create_table_sql = f"""
        -- 1. Extract correlation from args and index it
        DROP TABLE IF EXISTS temp_corr;
        CREATE TEMP TABLE temp_corr AS
        SELECT arg_set_id, int_value as correlation_id
        FROM args WHERE key = 'args.correlation';
        CREATE UNIQUE INDEX idx_tc_argset ON temp_corr(arg_set_id);
        CREATE INDEX idx_tc_corrid ON temp_corr(correlation_id);

        -- 2. Scan slices once, join via index
        DROP TABLE IF EXISTS temp_slices;
        CREATE TEMP TABLE temp_slices AS
        SELECT 
            s.id, s.name, s.category, s.ts, s.dur, s.track_id,
            t.correlation_id, (s.category = 'kernel') as is_gpu
        FROM slice s
        INNER JOIN temp_corr t USING(arg_set_id);
        CREATE INDEX idx_ts_corr ON temp_slices(correlation_id, is_gpu);

        -- 3. Self-join to generate mapping
        DROP TABLE IF EXISTS correlation_mapping;
        CREATE TABLE correlation_mapping AS
        SELECT 
            c.id, c.id as slice_id, c.name as slice_name,
            c.category, c.ts, c.dur, c.track_id, c.correlation_id,
            g.id as gpu_slice_id, g.name as gpu_kernel_name,
            g.ts as gpu_ts, g.dur as gpu_dur
        FROM temp_slices c
        INNER JOIN temp_slices g USING(correlation_id)
        WHERE c.is_gpu = 0 AND g.is_gpu = 1;

        -- Cleanup
        DROP TABLE IF EXISTS temp_corr;
        DROP TABLE IF EXISTS temp_slices;
        """
        
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_corr_map_cid ON correlation_mapping(correlation_id);",
            "CREATE INDEX IF NOT EXISTS idx_corr_map_sid ON correlation_mapping(slice_id);",
            "CREATE INDEX IF NOT EXISTS idx_corr_map_id ON correlation_mapping(id);"
        ]
        
        try:
            self.logger.info("Creating correlation_mapping table...")
            self.tp.query(create_table_sql)
            
            self.logger.info("Creating indexes...")
            for sql in indexes_sql: self.tp.query(sql)

            self.logger.info("Querying distinct slice_names...")
            result = self.tp.query("SELECT DISTINCT slice_name FROM correlation_mapping WHERE slice_name IS NOT NULL")
            
            if result.row_count > 0:
                self.kernellaunch_cpu_slicename = result.as_pandas_dataframe()['slice_name'].tolist()
                self.logger.info(f"Found {len(self.kernellaunch_cpu_slicename)} unique launch names.")
            else:
                self.logger.warning("No slice names found in correlation_mapping.")
                self.kernellaunch_cpu_slicename = []
                
        except Exception as e:
            self.logger.error(f"Error in _create_correlation_mapping: {e}")
            raise
    
    def _assign_fwdbwd_thread(self):
        """Identify Forward and Backward threads based on flow stats."""
        query_sql = f"""
        WITH valid_cpu_slice_filter AS (
            SELECT s.id as slice_id, tt.utid, th.name as thread_name
            FROM slice s
            JOIN thread_track tt ON s.track_id = tt.id
            JOIN thread th ON tt.utid = th.utid
            WHERE s.category NOT IN ('kernel') AND s.category NOT LIKE '%gpu%'
                AND th.name NOT LIKE 'stream %' AND th.name NOT LIKE 'GPU %' 
                AND th.name NOT LIKE 'CUDA %' AND s.dur > 0
        ),
        valid_slice_ids AS (SELECT DISTINCT slice_id FROM valid_cpu_slice_filter),
        outbound_flows AS (
            SELECT vs.utid, vs.thread_name, COUNT(DISTINCT f.id) as flows_out
            FROM valid_cpu_slice_filter vs
            JOIN flow f ON f.slice_out = vs.slice_id
            WHERE EXISTS (SELECT 1 FROM valid_slice_ids v WHERE v.slice_id = f.slice_in)
            GROUP BY vs.utid, vs.thread_name
        ),
        inbound_flows AS (
            SELECT vs.utid, vs.thread_name, COUNT(DISTINCT f.id) as flows_in
            FROM valid_cpu_slice_filter vs
            JOIN flow f ON f.slice_in = vs.slice_id
            WHERE EXISTS (SELECT 1 FROM valid_slice_ids v WHERE v.slice_id = f.slice_out)
            GROUP BY vs.utid, vs.thread_name
        ),
        flow_stats AS (
            SELECT 
                COALESCE(o.utid, i.utid) as utid,
                COALESCE(o.thread_name, i.thread_name) as thread_name,
                COALESCE(o.flows_out, 0) as flows_out,
                COALESCE(i.flows_in, 0) as flows_in,
                COALESCE(o.flows_out, 0) + COALESCE(i.flows_in, 0) as total
            FROM outbound_flows o FULL OUTER JOIN inbound_flows i ON o.utid = i.utid
            WHERE (COALESCE(o.flows_out, 0) + COALESCE(i.flows_in, 0)) > 0
        ),
        top2 AS (
            SELECT *, ROUND(flows_out*100.0/NULLIF(total,0), 2) as out_pct
            FROM flow_stats ORDER BY total DESC LIMIT 2
        )
        SELECT *, 
            CASE WHEN out_pct = MAX(out_pct) OVER () THEN 'FORWARD_THREAD' ELSE 'BACKWARD_THREAD' END as thread_type
        FROM top2 ORDER BY total DESC;
        """
        try:
            df = self.tp.query(query_sql).as_pandas_dataframe()
            if len(df) != 2:
                self.logger.error(f"Expected 2 threads, got {len(df)}")
                return
            
            fwd = df[df['thread_type'] == 'FORWARD_THREAD']
            bwd = df[df['thread_type'] == 'BACKWARD_THREAD']
            
            if len(fwd) == 1 and len(bwd) == 1:
                self.fwd_utid = int(fwd.iloc[0]['utid'])
                self.bwd_utid = int(bwd.iloc[0]['utid'])
            else:
                self.logger.error("Could not uniquely identify FWD/BWD threads.")
        except Exception as e:
            self.logger.error(f"Error in _assign_fwdbwd_thread: {e}")
            
    def _create_kernels_cls(self):
        """Classify slices into communication/computation/memory."""
        comm = glob_OR('name', PATTERNS_GLOB['kernel_comm'])
        mem = glob_OR('name', PATTERNS_GLOB['kernel_mem'])
        query = f"""
        SELECT id, name, category, ts, dur, track_id,
        CASE
            WHEN {comm} THEN 'communication'
            WHEN {mem} THEN 'memory'
            ELSE 'computation'
        END as slice_type
        FROM slice WHERE category IN ('kernel', 'gpu_memcpy', 'gpu_memset') ORDER BY ts;
        """
        try:
            self.logger.info("Building kernel classification cache...")
            self._kernels_cls_df = self.tp.query(query).as_pandas_dataframe().set_index('id')
            self.logger.info(f"Cached {len(self._kernels_cls_df)} slices. "
                             f"Counts: {self._kernels_cls_df['slice_type'].value_counts().to_dict()}")
        except Exception as e:
            self.logger.error(f"Error in _create_kernels_cls: {e}")
            self._kernels_cls_df = pd.DataFrame()
    
    def classify_kernels(self, sliceid_array: List[int]) -> pd.DataFrame:
        """Classify slices into comm/comp/mem/others."""
        if self._kernels_cls_df is None or self._kernels_cls_df.empty:
            return pd.DataFrame({'id': sliceid_array, 'slice_type': ['others']*len(sliceid_array)})
        
        return pd.DataFrame({'id': sliceid_array}).merge(
            self._kernels_cls_df[['slice_type']].reset_index(), on='id', how='left'
        ).fillna('others')[['id', 'slice_type']]

    def is_gpuop(self, sliceid_array: List[int]) -> List[bool]:
        """Check if slices are GPU operations."""
        ids = ','.join(map(str, sliceid_array))
        query = f"""
            SELECT id, CASE WHEN category IN ('kernel', 'gpu_memcpy', 'gpu_memset') THEN 1 ELSE 0 END as is_gpu
            FROM slice WHERE id IN ({ids})
        """
        try:
            res = {int(r.id): bool(r.is_gpu) for r in self.tp.query(query)}
            return [res.get(sid, False) for sid in sliceid_array]
        except Exception as e:
            self.logger.error(f"Error in is_gpuop: {e}")
            return [False] * len(sliceid_array)

    def query_ancestor_cpuop(self, sliceid_array: List[int]) -> Optional[Any]:
        """Return ancestor chain for given slices."""
        ids = ','.join(map(str, sliceid_array))
        sql = f"""
        WITH RECURSIVE caller_chain AS (
            SELECT id, name, ts, dur, parent_id, category, track_id, 0 as level, id as query_id
            FROM slice WHERE id IN ({ids}) 
            UNION ALL
            SELECT s.id, s.name, s.ts, s.dur, s.parent_id, s.category, s.track_id, cc.level + 1, cc.query_id
            FROM slice s JOIN caller_chain cc ON s.id = cc.parent_id 
            WHERE cc.level < {self.recursive_depth} AND cc.parent_id IS NOT NULL AND s.parent_id IS NOT NULL
        )
        SELECT id, ts, dur, track_id, id as slice_id, name, category, level, query_id
        FROM caller_chain WHERE level >= 0 ORDER BY query_id, level DESC, ts;
        """
        return self.execute_SQL(sql)
    
    def query_kernel_correlations_cpuop_batch(self, sliceid_array: List[int], batch_size=300) -> pd.DataFrame:
        """Find GPU kernels triggered by CPU slices (Batch)."""
        unique_ids = sorted(set(sliceid_array))
        
        def _query_batch(batch_ids):
            vals = ', '.join(f"({id})" for id in batch_ids)
            self.tp.query(f"DROP TABLE IF EXISTS t_in; CREATE TEMP TABLE t_in(sid INT); INSERT INTO t_in VALUES {vals};")
            launch = glob_OR('name', self.kernellaunch_cpu_slicename)
            
            sql = f"""
                WITH RECURSIVE tree AS (
                    SELECT sl.id as qid, sl.id, sl.name, sl.ts, sl.dur, sl.parent_id, 0 as depth
                    FROM slice sl JOIN t_in ti ON sl.id = ti.sid
                    UNION ALL
                    SELECT t.qid, s.id, s.name, s.ts, s.dur, s.parent_id, t.depth + 1
                    FROM slice s JOIN tree t ON s.parent_id = t.id WHERE t.depth < {self.recursive_depth}
                ),
                launches AS (SELECT qid, id, name FROM tree WHERE {launch}),
                results AS (
                    SELECT kl.qid as query_id, cm.gpu_slice_id as id, cm.gpu_ts as ts, cm.gpu_dur as dur,
                        cm.track_id, cm.gpu_slice_id as slice_id, cm.gpu_kernel_name as name,
                        cm.correlation_id, kl.id as launch_id, kl.name as launch_name
                    FROM launches kl JOIN correlation_mapping cm ON kl.id = cm.slice_id
                    WHERE cm.gpu_slice_id IS NOT NULL
                )
                SELECT * FROM results ORDER BY query_id, ts;
            """
            try:
                res = self.tp.query(sql)
                self.tp.query("DROP TABLE IF EXISTS t_in")
                return res.as_pandas_dataframe()
            except Exception as e:
                self.logger.error(f"Error in batch query: {e}")
                return None
        
        results = [df for i in range(0, len(unique_ids), batch_size) 
                   if (df := _query_batch(unique_ids[i:i+batch_size])) is not None]
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def query_kernel_correlations_gpuop(self, sliceid_array: List[int]):
        """Find CPU launch info for GPU kernels."""
        ids = ','.join(str(id) for id in sorted(set(sliceid_array)))
        launch = glob_OR('cm.slice_name', self.kernellaunch_cpu_slicename)
        sql = f"""
            SELECT cm.gpu_slice_id as gpu_op_id, cm.gpu_kernel_name as gpu_op_name, cm.correlation_id,
                cm.slice_id as launch_id, cm.slice_name as launch_name, cm.ts as launch_ts, cm.dur as launch_dur
            FROM correlation_mapping cm
            WHERE cm.gpu_slice_id IN ({ids}) AND {launch} AND cm.gpu_slice_id IS NOT NULL
            ORDER BY cm.gpu_slice_id;
        """
        return self.execute_SQL(sql)

    def query_descedents_cpuop(self, sliceid_array: List[int]) -> Dict[int, Any]:
        """Find descendants for given CPU slices. Returns dict {slice_id: result}."""
        results = {}
        for sliceid in sliceid_array:
            sql = f"""
            WITH target AS (
                SELECT s.id, s.ts, s.ts+s.dur as end_ts, s.track_id, s.depth, tt.utid
                FROM slice s LEFT JOIN thread_track tt ON s.track_id = tt.id WHERE s.id = {sliceid}
            )
            SELECT s.id as slice_id, s.id, s.ts, s.dur, s.dur/1e6 as duration_ms, s.track_id,
                s.name as function_name, s.depth as absolute_depth, s.depth - t.depth as relative_depth,
                tt.utid, th.name as thread_name
            FROM slice s
            LEFT JOIN thread_track tt ON s.track_id = tt.id
            LEFT JOIN thread th ON tt.utid = th.utid
            CROSS JOIN target t
            WHERE tt.utid = t.utid AND s.ts >= t.ts AND s.ts+s.dur <= t.end_ts
                AND s.depth BETWEEN t.depth AND t.depth + {self.recursive_depth} AND s.dur > 0
            ORDER BY relative_depth, s.ts;
            """
            try:
                results[sliceid] = self.tp.query(sql)
            except Exception as e:
                self.logger.error(f"Error in descendent query for {sliceid}: {e}")
        return results
    
    def merge_root(self, query_condition: str):
        """Find root-most slices matching condition (e.g. optimizer)."""
        sql = f"""
        WITH RECURSIVE opt_slices AS (
            SELECT id, name, ts, dur, track_id, parent_id FROM slice WHERE {query_condition}
        ),
        ancestors AS (
            SELECT id as orig_id, id, parent_id, 0 as depth FROM opt_slices
            UNION ALL
            SELECT a.orig_id, p.id, p.parent_id, a.depth + 1
            FROM ancestors a JOIN slice p ON a.parent_id = p.id WHERE a.depth < {self.recursive_depth}
        ),
        flagged AS (
            SELECT a.orig_id, s.id, s.name, s.ts, s.dur, s.track_id, s.parent_id, a.depth,
                CASE WHEN s.id IN (SELECT id FROM opt_slices) THEN 1 ELSE 0 END as is_opt
            FROM ancestors a JOIN slice s ON a.id = s.id
        ),
        final AS (
            SELECT orig_id, CASE 
                WHEN MAX(CASE WHEN is_opt=1 AND depth>0 THEN depth ELSE -1 END) > -1 
                THEN (SELECT id FROM flagged f2 WHERE f2.orig_id=f.orig_id AND is_opt=1 AND depth>0 ORDER BY depth DESC LIMIT 1)
                ELSE orig_id END as res_id
            FROM flagged f GROUP BY orig_id
        )
        SELECT DISTINCT os.id, os.ts, os.dur, os.track_id, os.id as slice_id, os.name, os.parent_id
        FROM opt_slices os JOIN final fr ON os.id = fr.res_id ORDER BY os.ts;
        """
        return self.execute_SQL(sql)
    
    def query_kernels_with_ancestor_match(self, kernel_ids: List[int], ancestor_ids: List[int], batch_size=300) -> Optional[pd.DataFrame]:
        """Check if kernel's CPU launch has specific ancestors."""
        if not kernel_ids or not ancestor_ids: return None

        try:
            # Helper to create temp tables
            def _create_temp(name, ids):
                self.execute_SQL(f"DROP TABLE IF EXISTS {name}")
                self.execute_SQL(f"CREATE TEMP TABLE {name}(id INT); CREATE INDEX idx_{name} ON {name}(id);")
                for i in range(0, len(ids), batch_size):
                    vals = ','.join(f"({x})" for x in ids[i:i+batch_size])
                    self.execute_SQL(f"INSERT INTO {name} VALUES {vals}")
            
            _create_temp('t_kernels', kernel_ids)
            _create_temp('t_ancestors', ancestor_ids)

            sql = f"""
            WITH k_to_launch AS (
                SELECT tk.id as kid, cm.gpu_kernel_name as kname, cm.slice_id as lid, cm.slice_name as lname
                FROM t_kernels tk JOIN correlation_mapping cm ON tk.id = cm.gpu_slice_id
                WHERE cm.gpu_slice_id IS NOT NULL
            ),
            ancestors AS (
                SELECT kl.kid, kl.lid, kl.lname, a.id as aid, a.name as aname, a.depth
                FROM k_to_launch kl LEFT JOIN ancestor_slice(kl.lid) a
            ),
            matched AS (
                SELECT kid, lid, lname, MAX(CASE WHEN ta.id IS NOT NULL THEN aid END) as maid,
                    MAX(CASE WHEN ta.id IS NOT NULL THEN aname END) as maname,
                    MAX(CASE WHEN ta.id IS NOT NULL THEN 1 ELSE 0 END) as has_match
                FROM ancestors a LEFT JOIN t_ancestors ta ON a.aid = ta.id
                GROUP BY kid, lid, lname
            )
            SELECT s.id, s.id as slice_id, s.ts, s.dur, s.track_id, s.name as kernel_name,
                m.lid as launch_slice_id, m.lname as launch_name, m.maid as matched_ancestor_id,
                m.maname as matched_ancestor_name,
                CASE WHEN m.has_match=1 THEN 'MATCHED' ELSE 'NO_MATCH' END as match_status
            FROM matched m JOIN slice s ON m.kid = s.id ORDER BY m.has_match DESC, s.ts;
            """
            
            res = self.execute_SQL(sql)
            self.execute_SQL("DROP TABLE IF EXISTS t_kernels; DROP TABLE IF EXISTS t_ancestors;")
            
            df = res.as_pandas_dataframe() if res else pd.DataFrame()
            if df.empty: return None
            
            matched = df[df['match_status'] == 'MATCHED'].reset_index(drop=True)
            self.logger.info(f"Matched {len(matched)} kernels.")
            return matched if not matched.empty else None

        except Exception as e:
            self.logger.error(f"Error in query_kernels_with_ancestor_match: {e}")
            try: self.execute_SQL("DROP TABLE IF EXISTS t_kernels; DROP TABLE IF EXISTS t_ancestors;")
            except: pass
            return None

    def execute_SQL(self, sql_query: str):
        try:
            return self.tp.query(sql_query)
        except Exception as e:
            self.logger.error(f"SQL Execution Error: {e} for {self.trace_path}")
            return None

    def close(self):
        if hasattr(self, 'tp'): del self.tp