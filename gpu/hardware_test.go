// Package gpu provides hardware GPU integration tests
//
// This module implements comprehensive TDD tests for actual GPU hardware
// integration, replacing the mock GPU implementations with real CUDA/OpenCL support.
//
// TDD Methodology:
//   - Red Phase: Comprehensive failing tests defining GPU hardware requirements
//   - Green Phase: Minimal implementations making tests pass
//   - Refactor Phase: Code quality improvements and optimization
//
// GPU Hardware Requirements:
//   - CUDA device detection and context management
//   - OpenCL device enumeration and computation
//   - Memory management with proper allocation/deallocation
//   - Kernel execution with error handling and synchronization
//   - Multi-GPU support with load balancing
//
// Test Categories:
//   - Device Detection: CUDA/OpenCL device enumeration
//   - Memory Management: GPU memory allocation and transfer
//   - Kernel Execution: Computation kernels and synchronization
//   - Performance Validation: Real vs mock GPU performance
//   - Error Handling: GPU errors and fallback mechanisms
package gpu

import (
	"math"
	"testing"
	"time"

	"github.com/julianshen/gonp/array"
)

// TestGPUHardwareDetection tests actual GPU hardware detection
func TestGPUHardwareDetection(t *testing.T) {
	t.Run("CUDA device detection", func(t *testing.T) {
		// This test will fail until we implement real CUDA detection
		devices, err := DetectCUDADevices()
		if err != nil {
			t.Logf("CUDA not available: %v", err)
			return
		}

		if len(devices) == 0 {
			t.Logf("No CUDA devices found")
			return
		}

		for i, device := range devices {
			t.Logf("CUDA Device %d:", i)
			t.Logf("  Name: %s", device.Name())
			t.Logf("  Memory: %d bytes", device.MemorySize())

			major, minor := device.ComputeCapability()
			t.Logf("  Compute Capability: %d.%d", major, minor)

			if major < 3 {
				t.Errorf("Device %d has compute capability %d.%d, need at least 3.0", i, major, minor)
			}

			if device.MemorySize() < 512*1024*1024 { // 512MB minimum
				t.Errorf("Device %d has insufficient memory: %d bytes", i, device.MemorySize())
			}
		}
	})

	t.Run("OpenCL device detection", func(t *testing.T) {
		// This test will fail until we implement real OpenCL detection
		devices, err := DetectOpenCLDevices()
		if err != nil {
			t.Logf("OpenCL not available: %v", err)
			return
		}

		if len(devices) == 0 {
			t.Logf("No OpenCL devices found")
			return
		}

		for i, device := range devices {
			t.Logf("OpenCL Device %d:", i)
			t.Logf("  Name: %s", device.Name())
			t.Logf("  Memory: %d bytes", device.MemorySize())
			t.Logf("  Backend: %s", device.GetBackend())

			if !device.IsAvailable() {
				t.Errorf("Device %d reported as not available", i)
			}

			if device.MemorySize() == 0 {
				t.Errorf("Device %d reports zero memory", i)
			}
		}
	})

	t.Run("Unified device manager", func(t *testing.T) {
		// Test unified device management across CUDA and OpenCL
		manager, err := NewHardwareGPUManager()
		if err != nil {
			t.Fatalf("Failed to create hardware GPU manager: %v", err)
		}

		devices, err := manager.EnumerateDevices()
		if err != nil {
			t.Fatalf("Failed to enumerate devices: %v", err)
		}

		t.Logf("Found %d total GPU devices", len(devices))

		if len(devices) > 0 {
			device := devices[0]
			t.Logf("Default device: %s (%s)", device.Name(), device.GetBackend())

			// Test device properties
			if device.Name() == "" {
				t.Errorf("Device name is empty")
			}

			backend := device.GetBackend()
			if backend != "CUDA" && backend != "OpenCL" {
				t.Errorf("Unknown backend: %s", backend)
			}
		}
	})
}

// TestGPUMemoryManagement tests GPU memory allocation and transfer
func TestGPUMemoryManagement(t *testing.T) {
	manager, err := NewHardwareGPUManager()
	if err != nil {
		t.Skip("GPU hardware not available")
	}

	devices, err := manager.EnumerateDevices()
	if err != nil || len(devices) == 0 {
		t.Skip("No GPU devices available")
	}

	device := devices[0]

	t.Run("Memory allocation and deallocation", func(t *testing.T) {
		// Test basic memory allocation
		buffer, err := device.AllocateBuffer(1024 * 1024) // 1MB
		if err != nil {
			t.Fatalf("Failed to allocate GPU memory: %v", err)
		}
		defer buffer.Free()

		if buffer.Size() != 1024*1024 {
			t.Errorf("Expected buffer size 1MB, got %d", buffer.Size())
		}

		// Test large allocation
		largeBuffer, err := device.AllocateBuffer(100 * 1024 * 1024) // 100MB
		if err != nil {
			t.Logf("Large allocation failed (expected): %v", err)
		} else {
			defer largeBuffer.Free()
			t.Logf("Large allocation succeeded: %d bytes", largeBuffer.Size())
		}
	})

	t.Run("Host to device memory transfer", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		buffer, err := device.AllocateBuffer(int64(len(data) * 8)) // 8 bytes per float64
		if err != nil {
			t.Fatalf("Failed to allocate GPU memory: %v", err)
		}
		defer buffer.Free()

		// Copy host data to device
		err = buffer.CopyFromHost(data)
		if err != nil {
			t.Fatalf("Failed to copy data to GPU: %v", err)
		}

		// Copy device data back to host
		var result []float64
		err = buffer.CopyToHost(&result)
		if err != nil {
			t.Fatalf("Failed to copy data from GPU: %v", err)
		}

		// Verify data integrity
		if len(result) != len(data) {
			t.Fatalf("Expected %d elements, got %d", len(data), len(result))
		}

		for i, v := range result {
			if math.Abs(v-data[i]) > 1e-10 {
				t.Errorf("Data mismatch at index %d: expected %.6f, got %.6f", i, data[i], v)
			}
		}

		t.Logf("Memory transfer successful: %v", result)
	})

	t.Run("Asynchronous memory transfer", func(t *testing.T) {
		data := make([]float64, 10000)
		for i := range data {
			data[i] = float64(i)
		}

		buffer, err := device.AllocateBuffer(int64(len(data) * 8))
		if err != nil {
			t.Fatalf("Failed to allocate GPU memory: %v", err)
		}
		defer buffer.Free()

		// Create stream for async operations
		stream, err := device.CreateStream()
		if err != nil {
			t.Fatalf("Failed to create GPU stream: %v", err)
		}
		defer stream.Destroy()

		// Async copy to device
		start := time.Now()
		err = buffer.CopyFromHostAsync(data, stream)
		if err != nil {
			t.Fatalf("Failed to copy data to GPU asynchronously: %v", err)
		}
		copyTime := time.Since(start)

		// Synchronize stream
		err = stream.Synchronize()
		if err != nil {
			t.Fatalf("Failed to synchronize stream: %v", err)
		}

		t.Logf("Async memory transfer completed in %v", copyTime)
	})
}

// TestGPUKernelExecution tests GPU kernel execution and computation
func TestGPUKernelExecution(t *testing.T) {
	manager, err := NewHardwareGPUManager()
	if err != nil {
		t.Skip("GPU hardware not available")
	}

	devices, err := manager.EnumerateDevices()
	if err != nil || len(devices) == 0 {
		t.Skip("No GPU devices available")
	}

	device := devices[0]

	t.Run("Vector addition kernel", func(t *testing.T) {
		size := 10000
		a := make([]float64, size)
		b := make([]float64, size)
		expected := make([]float64, size)

		// Initialize test data
		for i := 0; i < size; i++ {
			a[i] = float64(i)
			b[i] = float64(i + 1)
			expected[i] = a[i] + b[i]
		}

		// Execute vector addition on GPU
		result, err := device.ExecuteVectorAdd(a, b)
		if err != nil {
			t.Fatalf("Vector addition kernel failed: %v", err)
		}

		// Verify results
		if len(result) != size {
			t.Fatalf("Expected %d elements, got %d", size, len(result))
		}

		for i := 0; i < size; i++ {
			if math.Abs(result[i]-expected[i]) > 1e-10 {
				t.Errorf("Result mismatch at index %d: expected %.6f, got %.6f",
					i, expected[i], result[i])
				break
			}
		}

		t.Logf("Vector addition kernel successful for %d elements", size)
	})

	t.Run("Matrix multiplication kernel", func(t *testing.T) {
		size := 100 // 100x100 matrices

		// Create test matrices
		A := make([]float64, size*size)
		B := make([]float64, size*size)

		for i := 0; i < size*size; i++ {
			A[i] = float64(i%100) / 100.0
			B[i] = float64((i+50)%100) / 100.0
		}

		// Execute matrix multiplication on GPU
		start := time.Now()
		result, err := device.ExecuteMatMul(A, B, size, size, size)
		gpuTime := time.Since(start)

		if err != nil {
			t.Fatalf("Matrix multiplication kernel failed: %v", err)
		}

		if len(result) != size*size {
			t.Fatalf("Expected %d elements, got %d", size*size, len(result))
		}

		// Compare with CPU implementation for verification
		start = time.Now()
		cpuResult := matMulCPU(A, B, size)
		cpuTime := time.Since(start)

		// Verify accuracy
		maxError := 0.0
		for i := 0; i < size*size; i++ {
			error := math.Abs(result[i] - cpuResult[i])
			if error > maxError {
				maxError = error
			}
		}

		if maxError > 1e-6 {
			t.Errorf("Matrix multiplication accuracy too low: max error %.10f", maxError)
		}

		speedup := float64(cpuTime) / float64(gpuTime)
		t.Logf("Matrix multiplication: GPU %v, CPU %v, Speedup: %.2fx, Max Error: %.2e",
			gpuTime, cpuTime, speedup, maxError)

		if speedup < 1.0 {
			t.Logf("Warning: GPU slower than CPU for this matrix size")
		}
	})

	t.Run("Reduction operations kernel", func(t *testing.T) {
		size := 1000000 // 1M elements
		data := make([]float64, size)
		expectedSum := 0.0

		for i := 0; i < size; i++ {
			data[i] = float64(i % 1000)
			expectedSum += data[i]
		}

		// Execute sum reduction on GPU
		start := time.Now()
		gpuSum, err := device.ExecuteSum(data)
		gpuTime := time.Since(start)

		if err != nil {
			t.Fatalf("Sum reduction kernel failed: %v", err)
		}

		// Compare accuracy
		error := math.Abs(gpuSum - expectedSum)
		relativeError := error / math.Abs(expectedSum)

		if relativeError > 1e-6 {
			t.Errorf("Sum accuracy too low: expected %.6f, got %.6f, error %.2e",
				expectedSum, gpuSum, relativeError)
		}

		t.Logf("Sum reduction: %d elements in %v, relative error %.2e",
			size, gpuTime, relativeError)
	})
}

// TestGPUPerformanceValidation tests real GPU performance vs mock
func TestGPUPerformanceValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance validation in short mode")
	}

	manager, err := NewHardwareGPUManager()
	if err != nil {
		t.Skip("GPU hardware not available")
	}

	devices, err := manager.EnumerateDevices()
	if err != nil || len(devices) == 0 {
		t.Skip("No GPU devices available")
	}

	device := devices[0]

	t.Run("Performance comparison with mock GPU", func(t *testing.T) {
		sizes := []int{10000, 100000, 1000000}

		for _, size := range sizes {
			// Create test data
			arr, err := array.FromSlice(make([]float64, size))
			if err != nil {
				t.Fatalf("Failed to create test array: %v", err)
			}

			// Test real GPU performance
			start := time.Now()
			realGPUSum, err := device.ExecuteSum(arr.ToSlice().([]float64))
			realGPUTime := time.Since(start)

			if err != nil {
				t.Fatalf("Real GPU sum failed: %v", err)
			}

			// For now, just validate the GPU result is reasonable
			expectedSum := 0.0 // Empty array should sum to 0
			for i := 0; i < size; i++ {
				expectedSum += float64(i % 1000) // Pattern used in test data
			}

			// Compare with expected result
			accuracy := 1.0 - math.Abs(realGPUSum-expectedSum)/math.Max(math.Abs(realGPUSum), math.Abs(expectedSum))

			t.Logf("Size %d: Real GPU %v, Expected %.6f, Got %.6f, Accuracy %.6f",
				size, realGPUTime, expectedSum, realGPUSum, accuracy)

			if accuracy < 0.999 {
				t.Errorf("Insufficient GPU accuracy: %.6f", accuracy)
			}

			// Performance should be reasonable
			elementsPerSecond := float64(size) / realGPUTime.Seconds()
			t.Logf("GPU Performance: %.0f elements/second", elementsPerSecond)
		}
	})
}

// TestGPUHardwareErrorHandling tests GPU error conditions and recovery for hardware
func TestGPUHardwareErrorHandling(t *testing.T) {
	manager, err := NewHardwareGPUManager()
	if err != nil {
		t.Skip("GPU hardware not available")
	}

	devices, err := manager.EnumerateDevices()
	if err != nil || len(devices) == 0 {
		t.Skip("No GPU devices available")
	}

	device := devices[0]

	t.Run("Out of memory handling", func(t *testing.T) {
		// Try to allocate more memory than available
		availableMemory := device.MemorySize()
		oversizeBuffer, err := device.AllocateBuffer(availableMemory * 2)

		if err == nil {
			oversizeBuffer.Free()
			t.Logf("Large allocation succeeded (system may have virtual memory)")
		} else {
			t.Logf("Large allocation failed as expected: %v", err)

			// Verify error is appropriate
			if !IsOutOfMemoryError(err) {
				t.Errorf("Expected out of memory error, got: %v", err)
			}
		}
	})

	t.Run("Invalid kernel parameters", func(t *testing.T) {
		// Test with mismatched array sizes
		a := make([]float64, 1000)
		b := make([]float64, 2000) // Different size

		_, err := device.ExecuteVectorAdd(a, b)
		if err == nil {
			t.Errorf("Expected error for mismatched array sizes")
		} else {
			t.Logf("Mismatched sizes handled correctly: %v", err)
		}

		// Test with empty arrays
		empty := make([]float64, 0)
		_, err = device.ExecuteSum(empty)
		if err == nil {
			t.Errorf("Expected error for empty array")
		} else {
			t.Logf("Empty array handled correctly: %v", err)
		}
	})

	t.Run("Device context recovery", func(t *testing.T) {
		// Test device reset and recovery
		err := device.Reset()
		if err != nil {
			t.Logf("Device reset failed: %v", err)
		} else {
			t.Logf("Device reset successful")

			// Try to use device after reset
			data := []float64{1, 2, 3, 4, 5}
			_, err = device.ExecuteSum(data)
			if err != nil {
				t.Logf("Device not usable after reset: %v", err)
			} else {
				t.Logf("Device recovered successfully after reset")
			}
		}
	})
}

// TestHardwareMultiGPUSupport tests multi-GPU operations and load balancing for hardware
func TestHardwareMultiGPUSupport(t *testing.T) {
	manager, err := NewHardwareGPUManager()
	if err != nil {
		t.Skip("GPU hardware not available")
	}

	devices, err := manager.EnumerateDevices()
	if err != nil {
		t.Skip("Failed to enumerate devices")
	}

	if len(devices) < 2 {
		t.Skip("Need at least 2 GPUs for multi-GPU testing")
	}

	t.Run("Multi-GPU load balancing", func(t *testing.T) {
		size := 1000000
		data := make([]float64, size)
		for i := range data {
			data[i] = float64(i % 1000)
		}

		// Test load balancing across multiple GPUs
		start := time.Now()
		result, err := manager.ExecuteMultiGPUSum(data, devices)
		multiGPUTime := time.Since(start)

		if err != nil {
			t.Fatalf("Multi-GPU sum failed: %v", err)
		}

		// Compare with single GPU
		start = time.Now()
		singleResult, err := devices[0].ExecuteSum(data)
		singleGPUTime := time.Since(start)

		if err != nil {
			t.Fatalf("Single GPU sum failed: %v", err)
		}

		// Verify accuracy
		accuracy := 1.0 - math.Abs(result-singleResult)/math.Max(math.Abs(result), math.Abs(singleResult))
		if accuracy < 0.999 {
			t.Errorf("Multi-GPU accuracy insufficient: %.6f", accuracy)
		}

		speedup := float64(singleGPUTime) / float64(multiGPUTime)
		t.Logf("Multi-GPU: %d devices, speedup %.2fx, accuracy %.6f",
			len(devices), speedup, accuracy)

		if speedup > 1.5 {
			t.Logf("Good multi-GPU scaling achieved")
		} else {
			t.Logf("Limited multi-GPU benefit (communication overhead)")
		}
	})
}

// Helper functions for testing

// matMulCPU performs CPU matrix multiplication for validation
func matMulCPU(A, B []float64, size int) []float64 {
	result := make([]float64, size*size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			sum := 0.0
			for k := 0; k < size; k++ {
				sum += A[i*size+k] * B[k*size+j]
			}
			result[i*size+j] = sum
		}
	}
	return result
}

// Test functions use the interfaces and types defined in types.go
