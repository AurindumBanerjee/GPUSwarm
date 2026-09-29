import torch
import time

print("=" * 60)
print("PyTorch GPU Test")
print("=" * 60)

# --------------------------------------------------
# 1. Basic PyTorch information
# --------------------------------------------------

print(f"PyTorch version : {torch.__version__}")
print(f"PyTorch CUDA    : {torch.version.cuda}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\nERROR: CUDA is not available to PyTorch.")
    print("PyTorch is running in CPU-only mode.")
    exit()

# --------------------------------------------------
# 2. GPU information
# --------------------------------------------------

device = torch.device("cuda:0")

print(f"\nGPU device      : {torch.cuda.get_device_name(0)}")
print(f"GPU count       : {torch.cuda.device_count()}")

properties = torch.cuda.get_device_properties(0)

print(f"Total GPU memory: {properties.total_memory / 1024**3:.2f} GB")

# --------------------------------------------------
# 3. Basic GPU tensor test
# --------------------------------------------------

print("\n" + "=" * 60)
print("Basic GPU Tensor Test")
print("=" * 60)

x = torch.rand(1000, 1000, device=device)
y = torch.rand(1000, 1000, device=device)

z = x @ y

torch.cuda.synchronize()

print("Tensor device   :", z.device)
print("Tensor shape    :", z.shape)
print("Tensor checksum :", z.sum().item())

# --------------------------------------------------
# 4. Matrix multiplication benchmark
# --------------------------------------------------

print("\n" + "=" * 60)
print("GPU Matrix Multiplication Test")
print("=" * 60)

N = 4096

A = torch.rand(N, N, device=device)
B = torch.rand(N, N, device=device)

# Warm-up
for _ in range(3):
    C = A @ B

torch.cuda.synchronize()

# Timed execution
torch.cuda.synchronize()
start = time.perf_counter()

for _ in range(10):
    C = A @ B

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"Matrix size     : {N} x {N}")
print(f"Iterations      : 10")
print(f"Total time      : {elapsed:.4f} seconds")
print(f"Average time    : {elapsed / 10:.4f} seconds")
print(f"Result checksum : {C.sum().item():.4f}")

# --------------------------------------------------
# 5. Linear solve test
# --------------------------------------------------

print("\n" + "=" * 60)
print("GPU Linear Solve Test")
print("=" * 60)

N = 1000

A = torch.randn(N, N, device=device)

# Make matrix well-conditioned/invertible
A = A @ A.T + 0.01 * torch.eye(N, device=device)

B = torch.randn(N, 1, device=device)

torch.cuda.synchronize()
start = time.perf_counter()

X = torch.linalg.solve(A, B)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# Check residual: ||AX - B||
residual = torch.linalg.norm(A @ X - B)

print(f"Matrix size     : {N} x {N}")
print(f"Solve time      : {elapsed:.6f} seconds")
print(f"Residual norm   : {residual.item():.6e}")

# --------------------------------------------------
# 6. GPU memory information
# --------------------------------------------------

print("\n" + "=" * 60)
print("GPU Memory")
print("=" * 60)

allocated = torch.cuda.memory_allocated(0) / 1024**3
reserved = torch.cuda.memory_reserved(0) / 1024**3

print(f"Memory allocated: {allocated:.2f} GB")
print(f"Memory reserved : {reserved:.2f} GB")

print("\n" + "=" * 60)
print("GPU TEST COMPLETED SUCCESSFULLY")
print("=" * 60)