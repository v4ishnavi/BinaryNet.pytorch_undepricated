import torch
from models import resnet, resnet_binary
import gc

def analyze_binary_layers(model):
    """Count binary vs full precision parameters with detailed breakdown"""
    binary_params = 0
    full_precision_params = 0
    first_layer_fp = False
    
    # Check if first conv layer is full precision
    if hasattr(model, 'conv1'):
        first_layer_fp = 'Binarize' not in type(model.conv1).__name__
    
    for module in model.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            param_count = module.weight.numel()
            if 'Binarize' in type(module).__name__:
                binary_params += param_count
            else:
                full_precision_params += param_count
    
    return binary_params, full_precision_params, first_layer_fp

def calculate_theoretical_size(model):
    """Calculate theoretical storage size in MB"""
    binary_params, fp_params, _ = analyze_binary_layers(model)
    
    # Theoretical binary size: 1-bit for binary weights + 32-bit for FP weights
    theoretical_size_mb = (binary_params * 0.125 + fp_params * 4) / (1024 * 1024)  # bits to MB
    
    return theoretical_size_mb

def calculate_optimal_storage_size(model):
    """Calculate actual storage size if binary weights stored as true 1-bit and FP as 32-bit"""
    binary_params, fp_params, _ = analyze_binary_layers(model)
    
    # Optimal storage: 
    # - Binary weights: 1 bit each (packed efficiently)
    # - FP weights: 32 bits (4 bytes) each
    # - BatchNorm params: 32 bits each (always FP)
    # - Biases: 32 bits each (always FP)
    
    # Count BatchNorm and bias parameters (always FP32)
    bn_and_bias_params = 0
    for module in model.modules():
        # BatchNorm parameters
        if isinstance(module, torch.nn.BatchNorm2d):
            if module.weight is not None:
                bn_and_bias_params += module.weight.numel()
            if module.bias is not None:
                bn_and_bias_params += module.bias.numel()
        
        # Bias parameters in conv/linear layers
        elif hasattr(module, 'bias') and module.bias is not None:
            bn_and_bias_params += module.bias.numel()
    
    # Total FP32 parameters (excluding binary weights)
    total_fp_params = fp_params + bn_and_bias_params
    
    # Optimal storage calculation
    # Binary weights: 1 bit each (packed into bytes, so divide by 8)
    binary_storage_bytes = (binary_params + 7) // 8  # Round up for bit packing
    
    # FP32 weights: 4 bytes each
    fp_storage_bytes = total_fp_params * 4
    
    # Convert to MB
    optimal_size_mb = (binary_storage_bytes + fp_storage_bytes) / (1024 * 1024)
    
    return optimal_size_mb

def benchmark_models():
    """Compare Regular and Binary ResNet models - theoretical size and parameters only"""
    
    models = [
        ("Regular", resnet(dataset='cifar10', depth=18, num_classes=10)),
        ("Binary", resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=False, full_precision_last=False)),
        ("Binary+FP_First", resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=True, full_precision_last=False)),
        ("Binary+FP_Last", resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=False, full_precision_last=True)),
        ("Binary+FP_Both", resnet_binary(dataset='cifar10', depth=18, num_classes=10, full_precision_first=True, full_precision_last=True))
    ]
    
    print(f"{'Model':<10} {'Params':<9} {'First Layer':<11} {'Theoretical (MB)':<15} {'Optimal (MB)':<12}")
    print("-" * 62)
    
    results = {}
    
    for name, model in models:
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Calculate metrics
        total_params = sum(p.numel() for p in model.parameters())
        binary_params, fp_params, first_layer_fp = analyze_binary_layers(model)
        theoretical_size = calculate_theoretical_size(model)
        optimal_size = calculate_optimal_storage_size(model)
        
        # Determine first layer type for display
        first_layer_type = "FP32" if first_layer_fp else "Binary"
        if name == "Regular":
            first_layer_type = "FP32"
        
        # Store and display results
        results[name] = {
            'params': total_params,
            'binary_params': binary_params,
            'fp_params': fp_params,
            'first_layer_fp': first_layer_fp,
            'theoretical_size_mb': theoretical_size,
            'optimal_size_mb': optimal_size
        }
        
        print(f"{name:<10} {total_params:>8,} {first_layer_type:<11} {theoretical_size:>12.2f} {optimal_size:>10.2f}")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return results

if __name__ == "__main__":
    print("Simple ResNet Benchmark - Parameters & Theoretical Size")
    print("=" * 55)
    
    results = benchmark_models()
    
    # Show compression ratios
    if 'Regular' in results and 'Binary' in results:
        reg_theoretical = results['Regular']['theoretical_size_mb']
        bin_theoretical = results['Binary']['theoretical_size_mb']
        
        reg_optimal = results['Regular']['optimal_size_mb']
        bin_optimal = results['Binary']['optimal_size_mb']
        
        theoretical_ratio = reg_theoretical / bin_theoretical
        optimal_ratio = reg_optimal / bin_optimal
        
        print(f"\nCompression Ratios (Regular vs Binary):")
        print(f"  Theoretical: {theoretical_ratio:.1f}x")
        print(f"  Optimal:     {optimal_ratio:.1f}x")
        
        # Show Binary+FP comparisons for all configurations
        fp_configs = [name for name in ['Binary+FP_First', 'Binary+FP_Last', 'Binary+FP_Both'] if name in results]
        
        for config_name in fp_configs:
            config_theoretical = results[config_name]['theoretical_size_mb']
            config_optimal = results[config_name]['optimal_size_mb']
            
            config_theoretical_ratio = reg_theoretical / config_theoretical
            config_optimal_ratio = reg_optimal / config_optimal
            
            print(f"\nCompression Ratios (Regular vs {config_name}):")
            print(f"  Theoretical: {config_theoretical_ratio:.1f}x")
            print(f"  Optimal:     {config_optimal_ratio:.1f}x")
            
            # Impact of FP layers
            fp_impact_theoretical = (config_theoretical / bin_theoretical - 1) * 100
            fp_impact_optimal = (config_optimal / bin_optimal - 1) * 100
            layer_description = config_name.replace('Binary+FP_', '').replace('_', '+')
            print(f"\n{layer_description} FP Layer(s) Impact:")
            print(f"  Theoretical: +{fp_impact_theoretical:.1f}% size increase")
            print(f"  Optimal:     +{fp_impact_optimal:.1f}% size increase")
        
        print(f"\nParameter Breakdown:")
        reg_binary = results['Regular']['binary_params']
        reg_fp = results['Regular']['fp_params'] 
        bin_binary = results['Binary']['binary_params']
        bin_fp = results['Binary']['fp_params']
        
        print(f"  Regular: {reg_binary:,} binary + {reg_fp:,} FP32 = {reg_binary + reg_fp:,} total")
        print(f"  Binary:  {bin_binary:,} binary + {bin_fp:,} FP32 = {bin_binary + bin_fp:,} total")
        
        # Show Binary+FP breakdowns if available
        for config_name in ['Binary+FP_First', 'Binary+FP_Last', 'Binary+FP_Both']:
            if config_name in results:
                config = results[config_name]
                config_binary = config['binary_params']
                config_fp = config['fp_params']
                print(f"  {config_name}: {config_binary:,} binary + {config_fp:,} FP32 = {config_binary + config_fp:,} total")
                
                # Show the difference from full binary
                fp_diff = config_fp - bin_fp
                if fp_diff > 0:
                    layer_type = config_name.split('_')[1:]  # Get 'First', 'Last', or 'Both'
                    print(f"    → {' + '.join(layer_type)} layer(s) add {fp_diff:,} FP32 params vs fully binary")
        
        print(f"\nStorage Analysis:")
        print(f"  Regular model: {reg_optimal:.2f}MB (optimal) vs {reg_theoretical:.2f}MB (theoretical)")
        print(f"  Binary model:  {bin_optimal:.2f}MB (optimal) vs {bin_theoretical:.2f}MB (theoretical)")
        
        savings_optimal = reg_optimal - bin_optimal
        savings_theoretical = reg_theoretical - bin_theoretical
        print(f"\nActual Savings with True 1-bit Storage:")
        print(f"  Optimal: {savings_optimal:.2f}MB saved ({savings_optimal/reg_optimal*100:.1f}% reduction)")
        print(f"  Theoretical: {savings_theoretical:.2f}MB saved ({savings_theoretical/reg_theoretical*100:.1f}% reduction)")
