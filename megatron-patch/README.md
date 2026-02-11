# Megatron-TraceViz Patch

This patch integrates **Megatron-TraceViz** support into the NVIDIA Megatron-LM training framework. It is essential for enabling the advanced, distributed-aware profiling capabilities of the TraceViz toolset.

## 1. Purpose

The default PyTorch Profiler in Megatron-LM typically focuses on single-rank performance. This patch extends that functionality to enable **holistic distributed training analysis**.

It performs two key functions:
1.  **Dumps Parallel Group Topology:** It extracts and saves the precise mapping of all communication groups (TP, PP, DP, EP) and model replicas. This allows the analysis tools to reconstruct the global training timeline and correlate low-level kernels with high-level distributed strategies.
2.  **Structured Trace Organization:** It reorganizes the Profiler's output into a clean, iteration-based directory structure, making it possible to run batch analysis across hundreds of ranks automatically.

---

## 2. Captured Parallel Group Information

When the training starts, the patch automatically generates a configuration file (e.g., `unique_parallel_groups_TIMESTAMP.txt`) in your tensorboard directory. This file contains the "Ground Truth" for your distributed setup:

* **Global Configuration:** Captures the exact `TP_SIZE`, `PP_SIZE`, `DP_SIZE`, `EP_SIZE`, `CP_SIZE`, and `WORLD_SIZE` used in the run.
* **Communication Groups:**
    * **TP Groups:** Lists all ranks belonging to each Tensor Parallel group.
    * **PP Groups:** Lists all ranks belonging to each Pipeline Parallel group.
    * **DP Groups:** Lists all ranks belonging to each Data Parallel group.
    * **EP Groups:** (If enabled) Lists all ranks belonging to each Expert Parallel group.
* **Full Model Replicas:** Crucially, it calculates and lists which set of ranks constitute a **complete model replica**. This is vital for visualizing the end-to-end flow of a single model instance across the cluster.

    > [!IMPORTANT]
    > **Fixed Rank Order:** This calculation strictly assumes the specific parallelism rank order of **`tp-cp-ep-dp-pp`**, otherwise the automatically generated model replica groups may be **incorrect**.
---

## 3. Output Directory Structure

After applying this patch, your profiling output (defined by `--tensorboard-dir`) will be organized hierarchically by iteration, rather than a flat list of files.

**Structure:**
```text
/your_tensorboard_dir/
│
├── unique_parallel_groups_timestamp.txt        <-- The Topology Config File
│
├── iteration_10/                               <-- Folder for Iteration 10
│   ├── rank_0000_iter_10.pt.trace.json
│   ├── rank_0001_iter_10.pt.trace.json
│   └── ...
│
├── iteration_20/                               <-- Folder for Iteration 20
│   ├── rank_0000_iter_20.pt.trace.json
│   └── ...
```

---
## 4. How to Apply

> [!IMPORTANT]
> **Required Environment Configuration**
> You **must** set the following environment variables in your training script (or export them in your shell) to ensure the profiler captures the correct distributed data:
>
> * `PROFILE_ALLRANKS=1`: **Critical.** Forces profiling on **all** ranks. Without this, you will only get data for a few ranks, making global pipeline visualization impossible.
> * `PROFILER_WAIT_STEPS`: Number of steps to wait before the profiler initializes (e.g., `10`).
> * `PROFILER_WARMUP_STEPS`: Number of warmup steps before active recording begins (e.g., `3`).

1.  **Copy the Patch:** Place the `enable_traceviz.patch` file into the root directory of your Megatron-LM repository.

2.  **Apply the Patch:** Run the following command in your terminal:
    ```bash
    git apply enable_traceviz.patch
    ```
3.  **Run Training:** Execute your pretraining script as usual. Ensure you include the standard Megatron-LM profiling arguments:
    * `--profile`
    * `--profile-step-start <STEP>`
    * `--profile-step-end <STEP>`
    * `--tensorboard-dir <OUTPUT_PATH>`