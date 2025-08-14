package gpu

import (
	"runtime"
	"testing"
)

// TestGPUDeviceDetection tests GPU device enumeration and detection
func TestGPUDeviceDetection(t *testing.T) {
	t.Run("Enumerate available GPU devices", func(t *testing.T) {
		devices, err := EnumerateDevices()
		if err != nil {
			t.Logf("GPU device enumeration failed: %v (may be expected on systems without GPU)", err)
			return
		}

		// Should have at least 0 devices (could be no GPU available)
		if len(devices) < 0 {
			t.Errorf("Invalid device count: %d", len(devices))
		}

		// If devices are available, test their properties
		for i, device := range devices {
			if device == nil {
				t.Errorf("Device %d is nil", i)
				continue
			}

			// Device should have a name
			name := device.Name()
			if name == "" {
				t.Errorf("Device %d has empty name", i)
			}

			// Device should have compute capability
			major, minor := device.ComputeCapability()
			if major < 0 || minor < 0 {
				t.Errorf("Device %d has invalid compute capability: %d.%d", i, major, minor)
			}

			// Device should have memory size
			memSize := device.MemorySize()
			if memSize <= 0 {
				t.Errorf("Device %d has invalid memory size: %d", i, memSize)
			}

			t.Logf("Device %d: %s, Compute %d.%d, Memory %d MB",
				i, name, major, minor, memSize/(1024*1024))
		}
	})

	t.Run("Get default GPU device", func(t *testing.T) {
		device, err := GetDefaultDevice()
		if err != nil {
			t.Logf("No default GPU device available: %v", err)
			return
		}

		if device == nil {
			t.Error("Default device should not be nil when no error")
		}

		// Validate default device properties
		name := device.Name()
		if name == "" {
			t.Error("Default device has empty name")
		}

		memSize := device.MemorySize()
		if memSize <= 0 {
			t.Error("Default device has invalid memory size")
		}

		t.Logf("Default device: %s, Memory: %d MB", name, memSize/(1024*1024))
	})

	t.Run("Detect backend availability", func(t *testing.T) {
		// Test CUDA availability
		cudaAvailable := IsCudaAvailable()
		t.Logf("CUDA available: %v", cudaAvailable)

		// Test OpenCL availability
		openclAvailable := IsOpenCLAvailable()
		t.Logf("OpenCL available: %v", openclAvailable)

		// At least one backend should be available on systems with GPU
		if !cudaAvailable && !openclAvailable {
			t.Logf("No GPU backends available (may be expected on CPU-only systems)")
		}

		// Test backend selection
		backend, err := SelectOptimalBackend()
		if err != nil {
			t.Logf("No optimal backend available: %v", err)
		} else {
			t.Logf("Selected optimal backend: %s", backend)
		}
	})
}

// TestGPUMemoryAllocation tests GPU memory allocation and management
func TestGPUMemoryAllocation(t *testing.T) {
	t.Run("Basic memory allocation and deallocation", func(t *testing.T) {
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for memory test")
		}

		// Test memory allocation
		size := int64(1024 * 1024) // 1MB
		buffer, err := device.AllocateMemory(size)
		if err != nil {
			t.Fatalf("Memory allocation failed: %v", err)
		}
		defer buffer.Free()

		// Validate buffer properties
		if buffer.Size() != size {
			t.Errorf("Buffer size mismatch: expected %d, got %d", size, buffer.Size())
		}

		t.Logf("Successfully allocated %d bytes on GPU", size)
	})

	t.Run("Host-GPU data transfer", func(t *testing.T) {
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for transfer test")
		}

		// Create test data
		hostData := make([]float32, 1024)
		for i := range hostData {
			hostData[i] = float32(i)
		}

		// Allocate GPU memory
		size := int64(len(hostData) * 4) // 4 bytes per float32
		buffer, err := device.AllocateMemory(size)
		if err != nil {
			t.Fatalf("Memory allocation failed: %v", err)
		}
		defer buffer.Free()

		// Copy data to GPU
		err = buffer.CopyFromHost(hostData)
		if err != nil {
			t.Fatalf("Host to GPU copy failed: %v", err)
		}

		// Copy data back from GPU
		resultData := make([]float32, len(hostData))
		err = buffer.CopyToHost(resultData)
		if err != nil {
			t.Fatalf("GPU to host copy failed: %v", err)
		}

		// Validate data integrity
		for i, expected := range hostData {
			if resultData[i] != expected {
				t.Errorf("Data mismatch at index %d: expected %f, got %f",
					i, expected, resultData[i])
			}
		}

		t.Logf("Successfully transferred %d float32 values", len(hostData))
	})

	t.Run("Memory pool allocation", func(t *testing.T) {
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for pool test")
		}

		// Create memory pool
		pool, err := NewMemoryPool(device, 1024*1024*16) // 16MB pool
		if err != nil {
			t.Fatalf("Memory pool creation failed: %v", err)
		}
		defer pool.Destroy()

		// Allocate from pool multiple times
		var buffers []Buffer
		for i := 0; i < 8; i++ {
			buffer, err := pool.Allocate(1024 * 1024 * 2) // 2MB each
			if err != nil {
				t.Fatalf("Pool allocation %d failed: %v", i, err)
			}
			buffers = append(buffers, buffer)
		}

		// Free buffers
		for i, buffer := range buffers {
			err := pool.Free(buffer)
			if err != nil {
				t.Errorf("Pool free %d failed: %v", i, err)
			}
		}

		// Check pool statistics
		stats := pool.GetStats()
		if stats.TotalAllocated != 0 {
			t.Errorf("Pool should have zero allocations after freeing all, got %d",
				stats.TotalAllocated)
		}

		t.Logf("Memory pool test completed successfully")
	})
}

// TestGPUStreamOperations tests GPU stream creation and synchronization
func TestGPUStreamOperations(t *testing.T) {
	t.Run("Stream creation and synchronization", func(t *testing.T) {
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for stream test")
		}

		// Create stream
		stream, err := device.CreateStream()
		if err != nil {
			t.Fatalf("Stream creation failed: %v", err)
		}
		defer stream.Destroy()

		// Test synchronization
		err = stream.Synchronize()
		if err != nil {
			t.Errorf("Stream synchronization failed: %v", err)
		}

		t.Logf("Stream operations completed successfully")
	})

	t.Run("Asynchronous operations", func(t *testing.T) {
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for async test")
		}

		stream, err := device.CreateStream()
		if err != nil {
			t.Fatalf("Stream creation failed: %v", err)
		}
		defer stream.Destroy()

		// Test asynchronous memory operations
		size := int64(1024 * 1024)
		buffer, err := device.AllocateMemory(size)
		if err != nil {
			t.Fatalf("Memory allocation failed: %v", err)
		}
		defer buffer.Free()

		hostData := make([]float32, 256*1024)
		for i := range hostData {
			hostData[i] = float32(i)
		}

		// Start asynchronous copy
		err = buffer.CopyFromHostAsync(hostData, stream)
		if err != nil {
			t.Fatalf("Async copy failed: %v", err)
		}

		// Synchronize to ensure completion
		err = stream.Synchronize()
		if err != nil {
			t.Errorf("Stream sync after async copy failed: %v", err)
		}

		t.Logf("Asynchronous operations completed successfully")
	})
}

// TestGPUErrorHandling tests error conditions and edge cases
func TestGPUErrorHandling(t *testing.T) {
	t.Run("Invalid memory allocation", func(t *testing.T) {
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for error test")
		}

		// Try to allocate impossibly large memory
		_, err = device.AllocateMemory(1024 * 1024 * 1024 * 1024) // 1TB
		if err == nil {
			t.Error("Expected error for excessive memory allocation")
		} else {
			t.Logf("Correctly failed excessive allocation: %v", err)
		}

		// Try to allocate zero memory
		_, err = device.AllocateMemory(0)
		if err == nil {
			t.Error("Expected error for zero memory allocation")
		} else {
			t.Logf("Correctly failed zero allocation: %v", err)
		}

		// Try negative memory
		_, err = device.AllocateMemory(-1024)
		if err == nil {
			t.Error("Expected error for negative memory allocation")
		} else {
			t.Logf("Correctly failed negative allocation: %v", err)
		}
	})

	t.Run("Double free detection", func(t *testing.T) {
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for double-free test")
		}

		buffer, err := device.AllocateMemory(1024)
		if err != nil {
			t.Fatalf("Memory allocation failed: %v", err)
		}

		// Free once (should succeed)
		err = buffer.Free()
		if err != nil {
			t.Errorf("First free failed: %v", err)
		}

		// Free again (should fail or be safe)
		err = buffer.Free()
		if err != nil {
			t.Logf("Double free correctly detected: %v", err)
		} else {
			t.Logf("Double free handled safely")
		}
	})

	t.Run("Invalid data transfer", func(t *testing.T) {
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for transfer error test")
		}

		buffer, err := device.AllocateMemory(1024)
		if err != nil {
			t.Fatalf("Memory allocation failed: %v", err)
		}
		defer buffer.Free()

		// Try to copy nil data
		err = buffer.CopyFromHost(nil)
		if err == nil {
			t.Error("Expected error for nil data transfer")
		} else {
			t.Logf("Correctly failed nil transfer: %v", err)
		}

		// Try to copy mismatched size
		tooLargeData := make([]float32, 1024) // 4KB when buffer is 1KB
		err = buffer.CopyFromHost(tooLargeData)
		if err == nil {
			t.Error("Expected error for oversized data transfer")
		} else {
			t.Logf("Correctly failed oversized transfer: %v", err)
		}
	})
}

// TestGPUPerformanceBasics tests basic performance characteristics
func TestGPUPerformanceBasics(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	t.Run("Memory bandwidth measurement", func(t *testing.T) {
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for performance test")
		}

		// Test various sizes to measure bandwidth
		sizes := []int64{
			1024,             // 1KB
			1024 * 1024,      // 1MB
			16 * 1024 * 1024, // 16MB
		}

		for _, size := range sizes {
			buffer, err := device.AllocateMemory(size)
			if err != nil {
				t.Logf("Cannot allocate %d bytes for bandwidth test: %v", size, err)
				continue
			}

			data := make([]byte, size)

			// Measure upload bandwidth
			uploadBandwidth, err := measureTransferBandwidth(buffer, data, true)
			if err != nil {
				t.Logf("Upload bandwidth measurement failed for size %d: %v", size, err)
			} else {
				t.Logf("Upload bandwidth for %d bytes: %.2f MB/s", size, uploadBandwidth)
			}

			// Measure download bandwidth
			downloadBandwidth, err := measureTransferBandwidth(buffer, data, false)
			if err != nil {
				t.Logf("Download bandwidth measurement failed for size %d: %v", size, err)
			} else {
				t.Logf("Download bandwidth for %d bytes: %.2f MB/s", size, downloadBandwidth)
			}

			buffer.Free()
		}
	})

	t.Run("System resource detection", func(t *testing.T) {
		// Test system info
		cpuCount := runtime.NumCPU()
		t.Logf("CPU cores available: %d", cpuCount)

		devices, err := EnumerateDevices()
		if err != nil {
			t.Logf("GPU enumeration failed: %v", err)
		} else {
			t.Logf("GPU devices available: %d", len(devices))

			totalGPUMemory := int64(0)
			for _, device := range devices {
				totalGPUMemory += device.MemorySize()
			}
			t.Logf("Total GPU memory: %d MB", totalGPUMemory/(1024*1024))
		}

		// Test automatic GPU selection criteria
		shouldUseGPU := ShouldUseGPUForSize(1024 * 1024) // 1MB
		t.Logf("Should use GPU for 1MB data: %v", shouldUseGPU)

		shouldUseGPU = ShouldUseGPUForSize(1024 * 1024 * 100) // 100MB
		t.Logf("Should use GPU for 100MB data: %v", shouldUseGPU)
	})
}

// Note: measureTransferBandwidth is implemented in gpu.go
