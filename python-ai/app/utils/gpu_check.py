"""
GPU Check Utilities
Helper functions for GPU verification and monitoring
"""

import torch


def get_gpu_info():
    """
    Get comprehensive GPU information
    
    Returns:
        dict: GPU information including device details, memory, and capabilities
    """
    
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "message": "CUDA is not available. Check Docker GPU configuration.",
            "help": [
                "Ensure nvidia-container-toolkit is installed on host",
                "Run docker-compose with GPU support enabled",
                "Verify NVIDIA drivers are installed",
                "Check that deploy.resources.reservations.devices is set in docker-compose.yml"
            ]
        }
    
    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        
        # Get memory info
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        
        device_info = {
            "device_id": i,
            "name": torch.cuda.get_device_name(i),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
            "total_memory_gb": round(props.total_memory / (1024**3), 2),
            "free_memory_gb": round(free_mem / (1024**3), 2),
            "used_memory_gb": round((total_mem - free_mem) / (1024**3), 2),
            "memory_utilization_percent": round(((total_mem - free_mem) / total_mem) * 100, 2),
            "is_current_device": i == torch.cuda.current_device()
        }
        devices.append(device_info)
    
    return {
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": devices,
        "pytorch_version": torch.__version__,
        "recommended_settings": {
            "optimal_batch_size_range": "4-32 (adjust based on model size)",
            "fp16_supported": True,
            "notes": "RTX 3050 has 4-6GB VRAM. Use gradient checkpointing for large models."
        }
    }


def test_gpu_computation():
    """
    Perform a simple GPU computation test
    
    Returns:
        dict: Test results including timing and success status
    """
    import time
    
    if not torch.cuda.is_available():
        return {
            "success": False,
            "error": "CUDA not available"
        }
    
    try:
        device = torch.device("cuda:0")
        
        # Matrix multiplication test
        start_time = time.time()
        
        # Create large tensors
        size = 2000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Perform computation
        torch.cuda.synchronize()  # Wait for GPU to finish
        compute_start = time.time()
        
        c = torch.matmul(a, b)
        
        torch.cuda.synchronize()  # Wait for GPU to finish
        compute_time = time.time() - compute_start
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "matrix_size": size,
            "total_time_seconds": round(total_time, 4),
            "compute_time_seconds": round(compute_time, 4),
            "device": str(c.device),
            "result_shape": list(c.shape),
            "flops": round((2 * size**3) / compute_time / 1e9, 2),  # GFLOPS
            "message": "GPU computation test passed successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_optimal_batch_size(model_size_mb, available_memory_gb=4):
    """
    Suggest optimal batch size based on model size and available GPU memory
    
    Args:
        model_size_mb: Model size in megabytes
        available_memory_gb: Available GPU memory in GB (default 4 for RTX 3050)
    
    Returns:
        dict: Recommended batch size and memory estimates
    """
    
    # Convert to consistent units
    model_size_gb = model_size_mb / 1024
    
    # Rule of thumb: Model + Activations + Gradients ≈ 3x model size per sample
    # Leave 20% buffer for other operations
    usable_memory = available_memory_gb * 0.8
    
    # Calculate batch size
    memory_per_sample = model_size_gb * 3
    
    if memory_per_sample > usable_memory:
        return {
            "recommended_batch_size": 1,
            "warning": "Model may not fit in GPU memory. Consider using CPU or gradient checkpointing.",
            "model_size_gb": round(model_size_gb, 2),
            "available_memory_gb": available_memory_gb
        }
    
    batch_size = int(usable_memory / memory_per_sample)
    
    # Cap at reasonable maximum
    batch_size = min(batch_size, 32)
    
    return {
        "recommended_batch_size": max(1, batch_size),
        "model_size_gb": round(model_size_gb, 2),
        "available_memory_gb": available_memory_gb,
        "estimated_memory_usage_gb": round(memory_per_sample * batch_size, 2),
        "notes": "This is an estimate. Actual usage may vary based on sequence length and model architecture."
    }


def clear_gpu_cache():
    """
    Clear GPU cache to free up memory
    
    Returns:
        dict: Cache clearing results
    """
    if not torch.cuda.is_available():
        return {
            "success": False,
            "message": "CUDA not available"
        }
    
    try:
        # Get memory before clearing
        free_before, total = torch.cuda.mem_get_info(0)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Get memory after clearing
        free_after, _ = torch.cuda.mem_get_info(0)
        
        freed_mb = (free_after - free_before) / (1024**2)
        
        return {
            "success": True,
            "freed_memory_mb": round(freed_mb, 2),
            "free_memory_gb": round(free_after / (1024**3), 2),
            "total_memory_gb": round(total / (1024**3), 2),
            "message": f"Successfully freed {round(freed_mb, 2)} MB"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }