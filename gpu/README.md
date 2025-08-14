# GoNP GPU Hardware Integration

This package provides comprehensive GPU acceleration support for the GoNP library using CUDA and OpenCL backends.

## Overview

The GPU package implements a unified interface for GPU hardware acceleration with multiple backend support:

- **CUDA Backend**: NVIDIA GPU acceleration using CUDA Driver API
- **OpenCL Backend**: Cross-platform GPU acceleration (NVIDIA, AMD, Intel)
- **CPU Fallback**: Graceful degradation when GPU hardware is unavailable

## Architecture

```
gpu/
├── types.go              # Common interfaces and error types
├── hardware.go           # Fallback implementation (no GPU)
├── hardware_cuda.go      # CUDA-specific hardware manager
├── hardware_opencl.go    # OpenCL-specific hardware manager
├── cuda.go               # CUDA implementation with CGO bindings
├── opencl.go             # OpenCL implementation with CGO bindings
├── hardware_test.go      # Comprehensive TDD test suite
└── README.md             # This documentation
```

## Build Requirements

### CUDA Support

**Prerequisites:**
- NVIDIA GPU with compute capability ≥ 3.0
- CUDA Toolkit 10.0 or later installed
- `CGO_ENABLED=1` environment variable

**Installation:**

1. **Install CUDA Toolkit:**
   ```bash
   # Ubuntu/Debian
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get install cuda
   
   # CentOS/RHEL
   sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
   sudo dnf install cuda
   
   # macOS (with Homebrew)
   # Note: CUDA support for macOS was discontinued after CUDA 10.2
   brew install --cask cuda
   ```

2. **Set Environment Variables:**
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   export CGO_ENABLED=1
   ```

3. **Build with CUDA Support:**
   ```bash
   go build -tags cuda ./gpu
   go test -tags cuda ./gpu
   ```

### OpenCL Support

**Prerequisites:**
- OpenCL-compatible device (NVIDIA, AMD, Intel GPU/CPU)
- OpenCL SDK/Runtime installed
- `CGO_ENABLED=1` environment variable

**Installation:**

1. **Install OpenCL SDK:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install opencl-headers ocl-icd-opencl-dev
   
   # Install vendor-specific drivers:
   # NVIDIA
   sudo apt-get install nvidia-opencl-dev
   
   # AMD
   sudo apt-get install rocm-opencl-dev
   
   # Intel
   sudo apt-get install intel-opencl-icd
   
   # CentOS/RHEL
   sudo dnf install opencl-headers ocl-icd-devel
   
   # macOS
   # OpenCL is included with macOS, no separate installation needed
   ```

2. **Build with OpenCL Support:**
   ```bash
   go build -tags opencl ./gpu
   go test -tags opencl ./gpu
   ```

### CPU Fallback (Default)

**No additional requirements:**
```bash
go build ./gpu
go test ./gpu
```

## Usage Examples

### Basic GPU Operations

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/julianshen/gonp/gpu"
)

func main() {
    // Create GPU manager (detects available hardware)
    manager, err := gpu.NewHardwareGPUManager()
    if err != nil {
        log.Printf("GPU not available: %v", err)
        return
    }
    
    // Get default GPU device
    device, err := manager.GetDefaultDevice()
    if err != nil {
        log.Fatalf("No GPU devices: %v", err)
    }
    
    fmt.Printf("Using GPU: %s (%s)\n", device.Name(), device.GetBackend())
    fmt.Printf("Memory: %d bytes\n", device.MemorySize())
    
    // Perform vector addition
    a := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
    b := []float64{6.0, 7.0, 8.0, 9.0, 10.0}
    
    result, err := device.ExecuteVectorAdd(a, b)
    if err != nil {
        log.Fatalf("Vector addition failed: %v", err)
    }
    
    fmt.Printf("Result: %v\n", result)
    // Output: Result: [7 9 11 13 15]
}
```

### Memory Management

```go
// Allocate GPU memory
buffer, err := device.AllocateBuffer(1024 * 1024) // 1MB
if err != nil {
    log.Fatalf("Memory allocation failed: %v", err)
}
defer buffer.Free()

// Transfer data to GPU
data := []float64{1, 2, 3, 4, 5}
err = buffer.CopyFromHost(data)
if err != nil {
    log.Fatalf("Host to device copy failed: %v", err)
}

// Transfer data from GPU
var result []float64
err = buffer.CopyToHost(&result)
if err != nil {
    log.Fatalf("Device to host copy failed: %v", err)
}
```

### Asynchronous Operations

```go
// Create stream for asynchronous operations
stream, err := device.CreateStream()
if err != nil {
    log.Fatalf("Stream creation failed: %v", err)
}
defer stream.Destroy()

// Async memory transfer
err = buffer.CopyFromHostAsync(data, stream)
if err != nil {
    log.Fatalf("Async copy failed: %v", err)
}

// Synchronize stream
err = stream.Synchronize()
if err != nil {
    log.Fatalf("Stream sync failed: %v", err)
}
```

### Multi-GPU Operations

```go
// Get all available devices
devices, err := manager.EnumerateDevices()
if err != nil {
    log.Fatalf("Device enumeration failed: %v", err)
}

fmt.Printf("Found %d GPU devices:\n", len(devices))
for i, device := range devices {
    fmt.Printf("  Device %d: %s (%s)\n", i, device.Name(), device.GetBackend())
}

// Use multiple GPUs for computation
largeDataset := make([]float64, 1000000)
// ... initialize data ...

result, err := manager.ExecuteMultiGPUSum(largeDataset, devices)
if err != nil {
    log.Fatalf("Multi-GPU operation failed: %v", err)
}

fmt.Printf("Multi-GPU sum result: %f\n", result)
```

## Performance Considerations

### CUDA Performance Tips

1. **Memory Coalescing**: Ensure contiguous memory access patterns
2. **Occupancy**: Balance thread count with register/shared memory usage
3. **Stream Parallelism**: Use multiple streams for overlapping computation and data transfer
4. **Memory Type**: Use appropriate memory types (global, shared, constant)

### OpenCL Performance Tips

1. **Work-Group Size**: Optimize local work-group size for your device
2. **Memory Access**: Use coalesced memory access patterns
3. **Vector Types**: Utilize vector data types for better throughput
4. **Kernel Optimization**: Minimize divergent branches and memory bank conflicts

### General GPU Tips

1. **Data Size Thresholds**: GPU acceleration becomes beneficial for arrays >1024 elements
2. **Memory Transfer Overhead**: Minimize host-device memory transfers
3. **Batch Operations**: Process large datasets in batches to maximize GPU utilization
4. **Error Handling**: Always check for GPU errors and have CPU fallback strategies

## Troubleshooting

### Common Build Issues

**CUDA compilation fails:**
```
fatal error: cuda.h: No such file or directory
```
**Solution**: Install CUDA Toolkit and ensure PATH includes `/usr/local/cuda/bin`

**OpenCL compilation fails:**
```
fatal error: CL/cl.h: No such file or directory
```
**Solution**: Install OpenCL headers (`opencl-headers` package)

**CGO linking errors:**
```
undefined reference to 'cuInit'
```
**Solution**: Ensure `CGO_ENABLED=1` and proper library paths are set

### Runtime Issues

**No CUDA devices found:**
- Verify NVIDIA driver installation: `nvidia-smi`
- Check device compute capability: Must be ≥ 3.0
- Ensure CUDA runtime is accessible

**OpenCL device detection fails:**
- Verify OpenCL installation: `clinfo` command
- Check device compatibility with OpenCL runtime
- Install vendor-specific OpenCL drivers

**Memory allocation failures:**
- Check available GPU memory
- Reduce allocation size or use streaming
- Implement proper error handling and fallback

### Performance Issues

**GPU slower than CPU:**
- Array size may be too small for GPU benefit
- Memory transfer overhead may dominate
- Consider using larger batch sizes

**Low GPU utilization:**
- Increase parallel work (larger arrays)
- Use multiple streams for overlapping operations
- Check for memory bandwidth limitations

## Testing

### Running Tests

```bash
# Test without GPU (fallback mode)
go test ./gpu

# Test with CUDA (requires CUDA installation)
go test -tags cuda ./gpu

# Test with OpenCL (requires OpenCL installation)  
go test -tags opencl ./gpu

# Verbose testing with performance validation
go test -v -tags cuda ./gpu -run TestGPUPerformanceValidation

# Short testing (skip performance tests)
go test -short ./gpu
```

### Test Categories

- **Device Detection**: Hardware enumeration and capability testing
- **Memory Management**: Buffer allocation and data transfer testing
- **Kernel Execution**: Computation kernel testing with accuracy validation
- **Performance Validation**: Real vs CPU performance comparison
- **Error Handling**: Error condition and recovery testing
- **Multi-GPU Support**: Load balancing and parallel execution testing

## Development

### Adding New GPU Operations

1. **Define Interface**: Add method to `HardwareGPUDevice` interface
2. **Implement CUDA**: Add CUDA kernel and wrapper in `cuda.go`
3. **Implement OpenCL**: Add OpenCL kernel and wrapper in `opencl.go`
4. **Add Tests**: Create comprehensive tests in `hardware_test.go`
5. **Update Documentation**: Document new functionality

### Build Tags Architecture

```go
// Default build (no GPU)
//go:build !cuda && !opencl

// CUDA build
//go:build cuda

// OpenCL build  
//go:build opencl
```

This ensures only relevant code is compiled for each configuration.

## Contributing

1. **Follow TDD**: Write tests before implementation
2. **Test All Backends**: Ensure compatibility across CUDA/OpenCL/fallback
3. **Performance Benchmarks**: Include performance validation for new features
4. **Error Handling**: Implement proper error handling and fallback mechanisms
5. **Documentation**: Update documentation for API changes

## License

This GPU hardware integration is part of the GoNP library and follows the same license terms.

## Support

For GPU-related issues:
1. Check this documentation
2. Run diagnostic tests: `go test -v ./gpu`
3. Verify hardware/driver installation
4. Report issues with hardware configuration and error messages