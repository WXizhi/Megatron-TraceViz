"""
Time Interval Visualization Tool - Intelligent Automatic Layout
Generates Gantt charts from merged intervals.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
from matplotlib.font_manager import FontProperties
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class IntervalVisualizer:
    """Time Interval Visualization Tool - Intelligent Automatic Layout"""
    # Predefined color scheme
    DEFAULT_COLORS = {
        'fwd':'#496CCE', # Blue tone
        'fwd_computation': '#82AAE7', 
        'fwd_communication': '#A6CCE3',
        'bwd':'#C03F67', # Red tone
        'bwd_computation': '#F7A6BF',
        'bwd_communication': '#F6B293',
        'bwd_computation_recompute':'#5B27F5',
        'bwd_communication_recompute':'#275AF5',
        'pp_communication': '#B28FCE', # Purple
        'dp_communication': '#76E5DF', # Green
        'optimizer_computation': '#095624', # Dark Green
        'default': '#808080'
    }
    
    def __init__(self, plot_width_inches=10.0, row_height_inches=0.6, dpi=300, colors=None, max_label_chars=20):
        """
        Initialize Visualizer.
        Args:
            plot_width_inches: Width of the plotting area (inches).
            row_height_inches: Height of each category row (inches).
            dpi: Image resolution.
            colors: Custom color dictionary.
            max_label_chars: Max characters for y-axis labels before wrapping.
        """
        self.plot_width_inches = plot_width_inches
        self.row_height_inches = row_height_inches
        self.dpi = dpi
        self.max_label_chars = max_label_chars
        self.colors = colors if colors is not None else self.DEFAULT_COLORS.copy()
        # Layout parameters (inches)
        self.ylabel_width = 2.5
        self.right_margin = 0.5
        self.top_margin = 0.8
        self.bottom_base = 0.5
        self.xlabel_height = 0.4
        
    def wrap_labels(self, labels):
        """Intelligently wrap y-axis labels."""
        wrapped_labels = []
        for label in labels:
            if len(label) > self.max_label_chars:
                if '_' in label:
                    parts = label.split('_')
                    wrapped = '\n'.join(textwrap.wrap('_'.join(parts), width=self.max_label_chars, break_long_words=False))
                else:
                    wrapped = '\n'.join(textwrap.wrap(label, width=self.max_label_chars, break_long_words=False))
                wrapped_labels.append(wrapped)
            else:
                wrapped_labels.append(label)
        return wrapped_labels
    
    def calculate_legend_rows(self, num_categories):
        """Calculate legend rows based on category count and plot width."""
        item_width_inches = 1.8
        max_cols = max(1, int(self.plot_width_inches / item_width_inches))
        ncol = min(num_categories, max_cols)
        nrows = int(np.ceil(num_categories / ncol))
        return ncol, nrows
    
    def calculate_figsize(self, num_categories):
        """Calculate total figure size based on category count."""
        ncol, legend_rows = self.calculate_legend_rows(num_categories)
        total_width = self.ylabel_width + self.plot_width_inches + self.right_margin
        legend_height = legend_rows * 0.25
        plot_area_height = num_categories * self.row_height_inches
        total_height = (self.top_margin + plot_area_height + self.xlabel_height + legend_height + self.bottom_base)
        return (total_width, total_height)
    
    def calculate_subplot_params(self, num_categories):
        """Calculate precise subplot parameters."""
        total_width, total_height = self.calculate_figsize(num_categories)
        ncol, legend_rows = self.calculate_legend_rows(num_categories)
        legend_height = legend_rows * 0.25
        bottom_total = self.bottom_base + legend_height
        return {
            'left': self.ylabel_width / total_width,
            'right': (self.ylabel_width + self.plot_width_inches) / total_width,
            'bottom': bottom_total / total_height,
            'top': 1.0 - (self.top_margin / total_height),
            'ncol': ncol,
            'legend_rows': legend_rows
        }
    
    def normalize_time(self, intervals_dict):
        """Normalize time to 0-1 range."""
        all_start_times = []
        all_end_times = []
        for df in intervals_dict.values():
            if not df.empty:
                all_start_times.append(df['start_time'].min())
                all_end_times.append(df['end_time'].max())
        if not all_start_times: return intervals_dict, None
        
        global_min = min(all_start_times)
        global_max = max(all_end_times)
        time_range = global_max - global_min
        if time_range == 0:
            print("Warning: Time range is zero")
            return intervals_dict, None
            
        normalized_dict = {}
        for category, df in intervals_dict.items():
            if df.empty:
                normalized_dict[category] = df; continue
            df_norm = df.copy()
            df_norm['start_time_norm'] = (df['start_time'] - global_min) / time_range
            df_norm['end_time_norm'] = (df['end_time'] - global_min) / time_range
            df_norm['duration_norm'] = df_norm['end_time_norm'] - df_norm['start_time_norm']
            normalized_dict[category] = df_norm
            
        time_info = {'global_min': global_min, 'global_max': global_max, 'range': time_range, 'range_ms': time_range / 1e6}
        return normalized_dict, time_info

    def _compute_all_segments(self, normalized_dict, category_order):
        """Compute all time segments and their active categories."""
        all_intervals = []
        for cat in category_order:
            for _, row in normalized_dict[cat].iterrows():
                all_intervals.append({'start': row['start_time_norm'], 'end': row['end_time_norm'], 'category': cat})
        if not all_intervals: return []
        
        split_points = sorted(set(i['start'] for i in all_intervals) | set(i['end'] for i in all_intervals))
        all_segments = []
        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i + 1]
            mid = (start + end) / 2
            active_cats = [i['category'] for i in all_intervals if i['start'] <= mid < i['end']]
            if active_cats:
                all_segments.append({'start_norm': start, 'end_norm': end,
                    'active_categories': sorted(active_cats, key=lambda c: category_order.index(c)),
                    'is_overlap': len(active_cats) >= 2})
        return all_segments

    def _calculate_adaptive_height_allocation(self, all_segments, category_order):
        """Adaptive height allocation ensuring visibility of all ordered categories."""
        cat_stats = {cat: {'overlap': 0, 'non_overlap': 0} for cat in category_order}
        for seg in all_segments:
            for cat in seg['active_categories']:
                if cat in cat_stats:
                    if seg['is_overlap']: cat_stats[cat]['overlap'] += 1
                    else: cat_stats[cat]['non_overlap'] += 1
                    
        present_cats = [cat for cat in category_order if cat_stats[cat]['overlap'] + cat_stats[cat]['non_overlap'] > 0]
        num_present = len(present_cats)
        if num_present == 0: return {'height_map': {}, 'zorder_map': {}}
        
        if num_present == 1:
            return {'height_map': {present_cats[0]: 0.7}, 'zorder_map': {present_cats[0]: 10}}
            
        base_height = 0.85 if num_present > 5 else (0.80 if num_present > 3 else 0.75)
        min_height = 0.20 if num_present > 5 else (0.25 if num_present > 3 else 0.35)
        height_step = (base_height - min_height) / max(1, num_present - 1)
        
        height_map, zorder_map = {}, {}
        for idx, cat in enumerate(present_cats):
            height_map[cat] = min(base_height, min_height + idx * height_step)
            zorder_map[cat] = 100 - idx
        return {'height_map': height_map, 'zorder_map': zorder_map}

    def _draw_segment_adaptive(self, ax, seg_start, seg_end, y_base, category, height, zorder, MIN_VISIBLE_WIDTH=0.001):
        """Draw segment adaptively."""
        width = seg_end - seg_start
        color = self.colors.get(category, self.colors['default'])
        half_height = height / 2
        if width < MIN_VISIBLE_WIDTH:
            ax.plot([seg_start, seg_start], [y_base - half_height, y_base + half_height],
                color=color, linewidth=1.0, solid_capstyle='butt', clip_on=True, zorder=zorder)
        else:
            rect = mpatches.Rectangle(xy=(seg_start, y_base - half_height), width=width, height=height,
                                    facecolor=color, edgecolor='none', linewidth=0, clip_on=True, zorder=zorder)
            ax.add_patch(rect)

    def plot_combined_timeline(self, pp_group_intervals, output_path, time_alignment='individual',
                            category_order=['optimizer_computation', 'dp_communication', 'pp_communication',
                                            'fwd_computation', 'bwd_computation', 'bwd_computation_recompute',
                                            'bwd_communication_recompute', 'fwd_communication', 'bwd_communication'],
                            title='PP Group Combined Timeline', global_time_reference=None):
        """Plot combined timeline for PP Groups."""
        MIN_VISIBLE_WIDTH = 0.001
        if not pp_group_intervals: print("Warning: No PP group data to plot"); return
        pp_group_ids = sorted(pp_group_intervals.keys())
        num_groups = len(pp_group_ids)
        
        # Step 1: Time Alignment
        if time_alignment == 'global':
            time_ref = global_time_reference if global_time_reference is not None else pp_group_intervals
            global_min = min(min(df['start_time'].min() for df in d.values() if not df.empty) for d in time_ref.values())
            global_max = max(max(df['end_time'].max() for df in d.values() if not df.empty) for d in time_ref.values())
            time_range = global_max - global_min
            if time_range == 0: print("Warning: Global time range is zero"); return
        
        # Step 2: Normalize Data
        normalized_pp_groups, group_time_info = {}, {}
        for group_id in pp_group_ids:
            intervals = pp_group_intervals[group_id]
            if time_alignment == 'individual':
                starts = [df['start_time'].min() for df in intervals.values() if not df.empty]
                ends = [df['end_time'].max() for df in intervals.values() if not df.empty]
                if not starts: continue
                base_time, time_range = min(starts), max(ends) - min(starts)
                if time_range == 0: continue
            else:
                base_time = global_min
            
            norm_dict = {}
            for cat, df in intervals.items():
                if df.empty: norm_dict[cat] = df; continue
                df_norm = df.copy()
                df_norm['start_time_norm'] = (df['start_time'] - base_time) / time_range
                df_norm['end_time_norm'] = (df['end_time'] - base_time) / time_range
                df_norm['duration_norm'] = df_norm['end_time_norm'] - df_norm['start_time_norm']
                norm_dict[cat] = df_norm
            normalized_pp_groups[group_id] = norm_dict
            group_time_info[group_id] = {'range_ms': time_range / 1e6}
            
        # Step 3: Determine Category Order
        all_existing = set().union(*(d.keys() for d in pp_group_intervals.values()))
        if category_order is None: category_order = sorted(all_existing, reverse=True)
        else:
            filtered_order = [cat for cat in category_order if cat in all_existing]
            if not filtered_order: print("Error: No valid categories after filtering!"); return
            category_order = filtered_order

        # Step 4: Layout Calculation
        total_width = self.ylabel_width + self.plot_width_inches + self.right_margin
        ncol, legend_rows = self.calculate_legend_rows(len(category_order))
        total_height = 0.6 + num_groups * 1.5 + (num_groups - 1) * 0.3 + 0.5 + (0.4 + legend_rows * 0.3) + 0.2
        fig, ax = plt.subplots(figsize=(total_width, total_height), dpi=self.dpi)
        
        # Step 5: Plot Groups
        group_y_positions = {gid: num_groups - 1 - idx for idx, gid in enumerate(pp_group_ids)}
        for gid in pp_group_ids:
            norm_dict = normalized_pp_groups[gid]
            y_base = group_y_positions[gid]
            all_segments = self._compute_all_segments(norm_dict, category_order)
            allocation = self._calculate_adaptive_height_allocation(all_segments, category_order)
            
            for seg in all_segments:
                cats = seg['active_categories']
                if not seg['is_overlap']:
                    if cats[0] in allocation['height_map']:
                        self._draw_segment_adaptive(ax, seg['start_norm'], seg['end_norm'], y_base, cats[0],
                            allocation['height_map'][cats[0]], allocation['zorder_map'][cats[0]], MIN_VISIBLE_WIDTH)
                else:
                    for cat in reversed(cats):
                        if cat in allocation['height_map']:
                            self._draw_segment_adaptive(ax, seg['start_norm'], seg['end_norm'], y_base, cat,
                                allocation['height_map'][cat], allocation['zorder_map'][cat], MIN_VISIBLE_WIDTH)
        
        # Step 6: Axes Setup
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, num_groups - 0.5)
        ax.set_yticks(list(group_y_positions.values()))
        ax.set_yticklabels([f"Rank {gid}" for gid in pp_group_ids], fontsize=9)
        ax.set_ylabel('PP Group Rank ID', fontsize=11, fontweight='bold')
        
        time_range_ms = group_time_info[pp_group_ids[0]]['range_ms']
        x_ticks = np.linspace(0, 1, 11)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x * time_range_ms:.1f}" for x in x_ticks], fontsize=8)
        ax.set_xlabel('Wall Clock Time (ms)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.grid(True, axis='y', alpha=0.15, linestyle=':', linewidth=0.3)
        
        # Step 7: Legend
        legend_patches = [mpatches.Patch(color=self.colors.get(cat, self.colors['default']), label=cat) for cat in category_order]
        fig.legend(handles=legend_patches, loc='center', bbox_to_anchor=(0.5, ((0.2 + (0.4 + legend_rows * 0.3) / 2) / total_height)),
                bbox_transform=fig.transFigure, ncol=ncol, fontsize=8, framealpha=1.0, edgecolor='black', fancybox=True, shadow=True, borderpad=0.5)
        
        plt.subplots_adjust(left=self.ylabel_width / total_width, right=(self.ylabel_width + self.plot_width_inches) / total_width,
                           top=1.0 - (0.6 / total_height), bottom=(0.2 + (0.4 + legend_rows * 0.3) + 0.5) / total_height)
        if os.path.dirname(output_path): os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_comm_comp_overlap(self, overlap_df, output_dir, rank_id=None):
        """Plot communication and computation overlap analysis."""
        if overlap_df is None or overlap_df.empty: print("Warning: No overlap data to plot"); return
        has_forward = not overlap_df[overlap_df['phase'] == 'Forward'].empty
        has_backward = not overlap_df[overlap_df['phase'] == 'Backward'].empty
        if not has_forward and not has_backward: print("Warning: No Forward or Backward data to plot"); return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
        has_overlap = (overlap_df['comm_comp_overlap_time_ns'] > 0).any()
        legend_handles, legend_labels = [], []
        
        for idx, phase in enumerate(['Forward', 'Backward']):
            ax = axes[idx]
            phase_df = overlap_df[overlap_df['phase'] == phase].sort_values('mbs_id').reset_index(drop=True)
            if phase_df.empty:
                ax.text(0.5, 0.5, f'No {phase} data available', ha='center', va='center', fontsize=12, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f'{phase} Phase', fontsize=12, fontweight='bold', pad=10); continue
            
            x_pos = np.arange(len(phase_df))
            comp_only = (phase_df['actual_comp_time_ns'].values - phase_df['comm_comp_overlap_time_ns'].values) / 1e6
            overlap = phase_df['comm_comp_overlap_time_ns'].values / 1e6
            comm_only = (phase_df['actual_comm_time_ns'].values - phase_df['comm_comp_overlap_time_ns'].values) / 1e6
            wall_clock = phase_df['wall_clock_ns'].values / 1e6
            
            p1 = ax.bar(x_pos, comp_only, 0.6, label='Computation Time', color='#82AAE7', edgecolor='none')
            p2 = ax.bar(x_pos, overlap, 0.6, bottom=comp_only, label='Comm-Comp Overlap Time', color='#B28FCE', edgecolor='none') if has_overlap else None
            p3 = ax.bar(x_pos, comm_only, 0.6, bottom=comp_only + overlap, label='Communication Time', color='#76E5DF', edgecolor='none')
            
            bar_tops = comp_only + overlap + comm_only
            for i, (x, wc, bt) in enumerate(zip(x_pos, wall_clock, bar_tops)):
                ax.plot([x, x], [bt, wc], color='gray', linewidth=0.8, linestyle=':', alpha=0.8, zorder=5)
            p4 = ax.plot(x_pos, wall_clock, color='#6A0DAD', linewidth=2, marker='o', markersize=4, label='Wall Clock', zorder=10)
            
            if idx == 0:
                legend_handles = [p1, p3, p4[0]]
                legend_labels = ['Computation Execution Time', 'Communication Execution Time', 'Wall Clock']
                if has_overlap:
                    legend_handles.insert(1, p2); legend_labels.insert(1, 'Comm-Comp Overlap Time')
                else:
                    legend_handles.insert(1, mpatches.Patch(facecolor='#B28FCE', edgecolor='none', alpha=0.3))
                    legend_labels.insert(1, 'Comm-Comp Overlap Time (None)')
            
            ax.set_xlabel('MBS ID', fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos); ax.set_xticklabels(phase_df['mbs_id'].values, fontsize=9, rotation=90)
            ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
            ax.set_title(f'Communication & Computation Overlap Analysis (Rank {rank_id})\n{phase} Phase' if rank_id and idx==0 else f'{phase} Phase', fontsize=12, fontweight='bold', pad=10)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5); ax.set_axisbelow(True)
            
        plt.tight_layout(); plt.subplots_adjust(bottom=0.12, hspace=0.25)
        fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=len(legend_labels), fontsize=9, framealpha=0.9, edgecolor='black', borderpad=0.5)
        
        os.makedirs(output_dir, exist_ok=True)
        filename = f'comm_comp_overlap_combined_rank{rank_id}.png' if rank_id else 'comm_comp_overlap_combined.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved overlap analysis: {os.path.join(output_dir, filename)}")