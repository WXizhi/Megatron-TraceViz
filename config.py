from typing import List

TRACE_PROCESSOR_BIN_PATH = "C:/Users/Admin/.local/share/perfetto/prebuilts/trace_processor_shell.exe"
# ==========================================
# SQL Query Helpers
# ==========================================
glob_AND = lambda col, pats: ' AND '.join(f"{col} GLOB '{p}'" for p in pats)
glob_OR = lambda col, pats: '(' + ' OR '.join(f"{col} GLOB '{p}'" for p in pats) + ')'

like_all = lambda col, pats: ' AND '.join(f"{col} LIKE '{p}'" for p in pats)
like_any = lambda col, pats: '(' + ' OR '.join(f"{col} LIKE '{p}'" for p in pats) + ')'

# ==========================================
# Pattern Definitions
# ==========================================
# Note: GLOB uses '*' as the wildcard, unlike LIKE which uses '%'.
PATTERNS_GLOB = {
    'kernel_comm': [
        '*nccl*', 
        '*mccl*'
    ],
    
    'kernel_mem': [
        '*memcpy*', 
        '*Memcpy*', 
        '*memset*', 
        '*Memset*'
    ],
    
    'kernel_categories': [
        'kernel', 
        'gpu_memcpy', 
        'gpu_memset'
    ],
    
    'backward_cpu_slice': [
        '*backward_step*'
    ],
    
    'forward_cpu_slice': [
        '*forward_step*'
    ],
    
    'pp_cpu_slice': [
        '*send_forward*',
        '*recv_forward*',
        '*send_backward*', 
        '*recv_backward*',
        '*send_forward_recv_backward*',
        '*send_backward_recv_forward*'
    ],
    
    'dp_cpu_slice': [
        '*start_grad_sync*', 
        '*finish_grad_sync*',  
        '*start_param_sync*',  
        '*finish_param_sync*',
        '*finalize_model_grads*',
        '*optimizer*',
        '*pretrain*loss_func*'
    ],
    
    'optimizer_cpu_slice': [
        '*optimizer*'
    ],
    
    'recompute_cpu_slice': [
        '*recomupte*forward*', 
        '*recompute*forward*', 
        '*custom_forward*'
    ],  
}