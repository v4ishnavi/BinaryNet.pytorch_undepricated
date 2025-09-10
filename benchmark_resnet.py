import torch
import time
import psutil
import os
import numpy as np
from models import resnet, resnet_binary
import gc

def analyze_binary_layers(model):
    """Count binary vs full precision parameters"""
    binary_params = 0
    full_precision_params = 0
    
    for module in model.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            param_count = module.weight.numel()
            if 'Binarize' in type(module).__name__:
                binary_params += param_count
            else:
                full_precision_params += param_count
    
    return binary_params, full_precision_params

def benchmark_model(model, input_tensor, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    binary_params, fp_params = analyze_binary_layers(model)
    
    # Actual storage size (PyTorch checkpoint)
    torch.save(model.state_dict(), 'temp.pth')
    actual_size_mb = os.path.getsize('temp.pth') / 1024 / 1024
    os.remove('temp.pth')
    
    # Theoretical binary size (1-bit weights + 32-bit FP weights)
    theoretical_size_mb = (binary_params * 0.125 + fp_params * 4) / (1024 * 1024)  # bits to MB
    
    # Memory measurement fix
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Baseline memory
        baseline_mem = torch.cuda.memory_allocated() / 1024 / 1024
        
        with torch.no_grad():
            # Forward pass
            output = model(input_tensor)
            torch.cuda.synchronize()
            
            # Memory after forward pass
            forward_mem = torch.cuda.memory_allocated() / 1024 / 1024
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            memory_used = forward_mem - baseline_mem
            
            # Timing measurement
            for _ in range(10):
                _ = model(input_tensor)
            torch.cuda.synchronize()
            
            times = []
            for _ in range(50):
                torch.cuda.synchronize()
                start = time.time()
                _ = model(input_tensor)
                torch.cuda.synchronize()
                times.append((time.time() - start) * 1000)
    else:
        # CPU fallback
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        with torch.no_grad():
            output = model(input_tensor)
            mem_after = process.memory_info().rss / 1024 / 1024
            memory_used = mem_after - mem_before
            
            times = []
            for _ in range(50):
                start = time.time()
                _ = model(input_tensor)
                times.append((time.time() - start) * 1000)
    
    return {
        'name': model_name,
        'params': total_params,
        'binary_params': binary_params,
        'fp_params': fp_params,
        'actual_size_mb': actual_size_mb,
        'theoretical_size_mb': theoretical_size_mb,
        'inference_ms': np.mean(times),
        'memory_mb': max(0, memory_used)  # Ensure non-negative
    }

def compare_resnet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    input_tensor = torch.randn(1, 3, 32, 32)
    
    # Test models
    models = [
        ("Regular", resnet(dataset='cifar10', depth=18, num_classes=10)),
        ("Binary", resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=False)),
        ("Binary+FP", resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=True))
    ]
    
    print(f"\n{'Model':<10} {'Params':<9} {'Binary%':<8} {'Size(MB)':<9} {'Time(ms)':<9} {'Memory(MB)':<10}")
    print("-" * 65)
    
    for name, model in models:
        result = benchmark_model(model, input_tensor, name)
        binary_pct = (result['binary_params'] / result['params'] * 100) if result['params'] > 0 else 0
        
        print(f"{name:<10} {result['params']:>8,} {binary_pct:>6.1f}% {result['actual_size_mb']:>7.1f} {result['inference_ms']:>7.2f} {result['memory_mb']:>8.1f}")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def batch_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = [
        resnet(dataset='cifar10', depth=18, num_classes=10).to(device).eval(),
        resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=False).to(device).eval()
    ]
    
    print(f"\n{'Batch':<6} {'Regular(ms)':<11} {'Binary(ms)':<11} {'Speedup':<8}")
    print("-" * 40)
    
    for batch_size in [1, 32, 128]:
        inputs = torch.randn(batch_size, 3, 32, 32).to(device)
        times = []
        
        for model in models:
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = model(inputs)
                
                # Time measurement
                timings = []
                for _ in range(10):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.time()
                    _ = model(inputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    timings.append((time.time() - start) * 1000)
                times.append(np.mean(timings))
        
        speedup = times[0] / times[1] if times[1] > 0 else 1.0
        print(f"{batch_size:<6} {times[0]:<10.2f} {times[1]:<10.2f} {speedup:<7.2f}x")

def memory_profile():
    """Simple memory analysis"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = [
        ("Regular", resnet(dataset='cifar10', depth=18, num_classes=10)),
        ("Binary", resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=False))
    ]
    
    print(f"\n{'Model':<8} {'Params':<9} {'Actual Size':<12} {'Theoretical':<12}")
    print("-" * 45)
    
    for name, model in models:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        total_params = sum(p.numel() for p in model.parameters())
        binary_params, fp_params = analyze_binary_layers(model)
        
        torch.save(model.state_dict(), 'temp.pth')
        actual_size = os.path.getsize('temp.pth') / 1024 / 1024
        theoretical_size = (binary_params * 0.125 + fp_params * 4) / (1024 * 1024)
        os.remove('temp.pth')
        
        print(f"{name:<8} {total_params:>8,} {actual_size:>9.1f}MB {theoretical_size:>9.1f}MB")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def layer_analysis():
    """Analyze key architectural differences between configurations"""
    print(f"\nARCHITECTURE ANALYSIS:")
    print("=" * 50)
    
    models = {
        'Regular': resnet(dataset='cifar10', depth=18, num_classes=10),
        'Fully Binary': resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=False),
        'Binary + FP First': resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=True)
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Check first conv layer type
        first_conv_type = type(model.conv1).__name__
        first_indicator = "Full Precision" if first_conv_type == "Conv2d" else "Binarized"
        print(f"  First Layer:  {first_indicator}")
        
        # Check final layer type
        final_layer_type = type(model.fc).__name__
        final_indicator = "Full Precision" if final_layer_type == "Linear" else "Binarized"
        print(f"  Final Layer:  {final_indicator}")
        
        # Count layer types
        binary_layers = sum(1 for m in model.modules() if "Binarize" in type(m).__name__)
        total_layers = sum(1 for m in model.modules() if hasattr(m, 'weight') and m.weight is not None)
        
        print(f"  Binary Layers: {binary_layers}/{total_layers}")
        print(f"  Total Params:  {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    print("ResNet Benchmark")
    print("=" * 40)
    
    compare_resnet()
    batch_analysis()
    memory_profile()
    
    print("\nDone! Binary networks slower due to simulation overhead.")

