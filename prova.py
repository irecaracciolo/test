import torch
import time
import numpy as np
 
def check_gpu():
    """Check GPU availability and info"""
    print("=== GPU Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
   
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available, will use CPU")
    print()
 
def matrix_operations_demo():
    """Demonstrate basic GPU operations with timing comparison"""
    print("=== Matrix Operations Demo ===")
   
    # Create large matrices
    size = 2000
    a = torch.randn(size, size)
    b = torch.randn(size, size)
   
    # CPU computation
    print("Computing on CPU...")
    start_time = time.time()
    cpu_result = torch.mm(a, b)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
   
    # GPU computation (if available)
    if torch.cuda.is_available():
        print("Computing on GPU...")
       
        # Move tensors to GPU
        a_gpu = a.cuda()
        b_gpu = b.cuda()
       
        # Warm up GPU
        torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
       
        # Actual timing
        start_time = time.time()
        gpu_result = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
       
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
       
        # Verify results are similar
        gpu_result_cpu = gpu_result.cpu()
        print(f"Results match: {torch.allclose(cpu_result, gpu_result_cpu, rtol=1e-4)}")
   
    print()
 
def simple_neural_network_demo():
    """Simple neural network operations on GPU"""
    print("=== Neural Network Demo ===")
   
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
   
    # Create simple data
    batch_size = 1000
    input_size = 784
    hidden_size = 256
    output_size = 10
   
    # Generate random data
    x = torch.randn(batch_size, input_size).to(device)
    y = torch.randint(0, output_size, (batch_size,)).to(device)
   
    # Simple neural network
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size)
    ).to(device)
   
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
    # Training loop
    print("Training for 100 steps...")
    start_time = time.time()
   
    for step in range(200000):
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
       
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        if (step + 1) % 20 == 0:
            print(f"Step {step+1}/100, Loss: {loss.item():.4f}")
   
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final accuracy: {(outputs.argmax(1) == y).float().mean().item():.2%}")
    print()
 
def gpu_memory_demo():
    """Demonstrate GPU memory management"""
    if not torch.cuda.is_available():
        print("GPU not available, skipping memory demo")
        return
       
    print("=== GPU Memory Demo ===")
   
    # Check initial memory
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Initial GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
   
    # Allocate some memory
    large_tensor = torch.randn(10000, 10000).cuda()
    print(f"After allocation - Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
   
    # Free memory
    del large_tensor
    torch.cuda.empty_cache()
    print(f"After cleanup - Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print()
 
if __name__ == "__main__":
    # Run all demos
    check_gpu()
    matrix_operations_demo()
    simple_neural_network_demo()
    gpu_memory_demo()
   
    print("=== Demo Complete ===")
    print("This script demonstrated:")
    print("- GPU availability checking")
    print("- Basic tensor operations on GPU")
    print("- Performance comparison CPU vs GPU")
    print("- Simple neural network training on GPU")
    print("- GPU memory management")
 