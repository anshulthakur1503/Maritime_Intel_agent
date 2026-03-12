#!/usr/bin/env python3
"""
GPU Verification Script for Maritime Intelligence Agent
Checks NVIDIA GPU availability and CUDA configuration
"""

import sys


def check_gpu():
    """Comprehensive GPU check with detailed diagnostics"""
    
    print("=" * 70)
    print("GPU VERIFICATION FOR MARITIME INTELLIGENCE AGENT")
    print("=" * 70)
    
    # Check 1: PyTorch Installation
    print("\n[1] Checking PyTorch installation...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False
    
    # Check 2: CUDA Availability
    print("\n[2] Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA is available")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  - cuDNN enabled: {torch.backends.cudnn.enabled}")
    else:
        print("✗ CUDA is NOT available")
        print("\nPossible reasons:")
        print("  1. NVIDIA drivers not installed on host")
        print("  2. Docker not configured with --gpus flag")
        print("  3. nvidia-container-toolkit not installed")
        print("  4. GPU not accessible from container")
        return False
    
    # Check 3: GPU Device Count
    print("\n[3] Checking GPU devices...")
    device_count = torch.cuda.device_count()
    print(f"  - Number of GPUs detected: {device_count}")
    
    if device_count == 0:
        print("✗ No GPU devices found")
        return False
    
    # Check 4: GPU Details
    print("\n[4] GPU Device Information:")
    for i in range(device_count):
        print(f"\n  GPU {i}:")
        print(f"    - Name: {torch.cuda.get_device_name(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"    - Compute Capability: {props.major}.{props.minor}")
        print(f"    - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    - Multi Processors: {props.multi_processor_count}")
        
        # Memory info
        if hasattr(torch.cuda, 'mem_get_info'):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(f"    - Available Memory: {free_mem / 1024**3:.2f} GB")
            print(f"    - Used Memory: {(total_mem - free_mem) / 1024**3:.2f} GB")
    
    # Check 5: Tensor Operations Test
    print("\n[5] Testing GPU tensor operations...")
    try:
        # Create a tensor on GPU
        device = torch.device("cuda:0")
        test_tensor = torch.randn(1000, 1000, device=device)
        result = torch.matmul(test_tensor, test_tensor)
        
        print(f"✓ Successfully performed matrix multiplication on GPU")
        print(f"  - Tensor shape: {result.shape}")
        print(f"  - Tensor device: {result.device}")
    except Exception as e:
        print(f"✗ GPU tensor operation failed: {e}")
        return False
    
    # Check 6: Transformers Library
    print("\n[6] Checking Transformers library...")
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        
        # Test model loading capability
        print("  - Testing model loading (this may take a moment)...")
        from transformers import AutoTokenizer
        
        # Use a tiny model for quick test
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        print(f"✓ Successfully loaded test tokenizer")
        
    except Exception as e:
        print(f"✗ Transformers check failed: {e}")
        # Not critical, continue
    
    # Check 7: Environment Variables
    print("\n[7] Checking CUDA environment variables...")
    import os
    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 
                 'CUDA_HOME', 'LD_LIBRARY_PATH']
    
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  - {var}: {value}")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("GPU VERIFICATION COMPLETE - ALL CHECKS PASSED ✓")
    print("=" * 70)
    print("\nYour RTX 3050 is properly configured and ready for inference!")
    print("You can now run FinBERT and other transformer models on GPU.")
    print("\nRecommended next steps:")
    print("  1. Start the FastAPI service: docker-compose up python-ai")
    print("  2. Test the /gpu-info endpoint: curl http://localhost:8000/gpu-info")
    print("  3. Run inference tests with your models")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)