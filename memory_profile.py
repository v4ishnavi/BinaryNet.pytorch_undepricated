import torch
import torch.nn as nn
import time
import psutil
from models import resnet, resnet_binary
import gc

def compare_memory_and_timing():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    inputs = torch.randn(batch_size, 3, 32, 32).to(device)
    targets = torch.randint(0, 10, (batch_size,)).to(device)
    
    models = [
        ('Regular', lambda: resnet(dataset='cifar10', depth=18, num_classes=10)),
        ('Binary', lambda: resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=False)),
        ('Binary+FP', lambda: resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=True))
    ]
    
    results = {}
    
    for name, model_fn in models:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        model = model_fn().to(device)
        
        # Quick memory and timing measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        
        # Get baseline memory after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            baseline_mem = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            baseline_mem = psutil.Process().memory_info().rss / 1024 / 1024
        
        with torch.no_grad():
            # Memory measurement during inference
            if torch.cuda.is_available():
                _ = model(inputs)  # First forward pass
                torch.cuda.synchronize()
                inference_mem = torch.cuda.memory_allocated() / 1024 / 1024
                memory_used = inference_mem - baseline_mem
            else:
                proc = psutil.Process()
                _ = model(inputs)
                inference_mem = proc.memory_info().rss / 1024 / 1024
                memory_used = inference_mem - baseline_mem
            
            # Quick timing (reduced iterations)
            times = []
            for _ in range(10):  # Reduced from 50
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                _ = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append((time.time() - start) * 1000)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        results[name] = {
            'params': total_params,
            'memory_mb': max(0.1, memory_used),  # Ensure minimum 0.1MB shown
            'time_ms': sum(times) / len(times)
        }
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    print(f"{'Model':<10} {'Params':<9} {'Memory(MB)':<11} {'Time(ms)':<9}")
    print("-" * 45)
    
    for name in ['Regular', 'Binary', 'Binary+FP']:
        if name in results:
            r = results[name]
            print(f"{name:<10} {r['params']:>8,} {r['memory_mb']:>8.1f} {r['time_ms']:>7.1f}")
    
    # Simple comparison
    if 'Regular' in results and 'Binary' in results:
        reg, bin = results['Regular'], results['Binary']
        speedup = reg['time_ms'] / bin['time_ms']
        print(f"\nBinary vs Regular: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")

if __name__ == "__main__":
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    compare_memory_and_timing()
