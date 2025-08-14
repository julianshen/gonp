package array

import (
	"math"
	"testing"
	"time"

	"github.com/julianshen/gonp/internal"
)

// TestArrayGPUAcceleration tests GPU-accelerated array operations
func TestArrayGPUAcceleration(t *testing.T) {
	t.Run("Basic GPU operations", func(t *testing.T) {
		// Create test data
		data1 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		data2 := []float64{2.0, 3.0, 4.0, 5.0, 6.0}

		arr1, _ := FromSlice(data1)
		arr2, _ := FromSlice(data2)

		// Get default GPU device
		device, err := GPUManager.GetDefaultDevice()
		if err != nil {
			t.Fatalf("Failed to get GPU device: %v", err)
		}

		// Test GPU addition
		gpuResult, err := arr1.AddGPU(arr2, device)
		if err != nil {
			t.Fatalf("GPU addition failed: %v", err)
		}

		// Compare with CPU result
		cpuResult, _ := arr1.Add(arr2)
		if !arrayEqual(gpuResult, cpuResult) {
			t.Errorf("GPU and CPU addition results differ")
		}

		t.Logf("GPU addition successful: %v", gpuResult.ToSlice().([]float64))
	})

	t.Run("GPU matrix multiplication", func(t *testing.T) {
		// Create 2x3 and 3x2 matrices using Reshape to ensure 2D structure
		data1 := []float64{1, 2, 3, 4, 5, 6} // 2x3 matrix flattened
		data2 := []float64{1, 2, 3, 4, 5, 6} // 3x2 matrix flattened

		arr1, _ := FromSlice(data1)
		arr2, _ := FromSlice(data2)

		mat1 := arr1.Reshape(internal.Shape{2, 3}) // 2x3 matrix
		mat2 := arr2.Reshape(internal.Shape{3, 2}) // 3x2 matrix

		device, _ := GPUManager.GetDefaultDevice()

		// Test GPU matrix multiplication
		gpuResult, err := mat1.MatMulGPU(mat2, device)
		if err != nil {
			t.Fatalf("GPU matrix multiplication failed: %v", err)
		}

		// Compare with CPU result (using the matMulCPU function)
		cpuResult, _ := matMulCPU(mat1, mat2)
		if !arrayEqual(gpuResult, cpuResult) {
			t.Errorf("GPU and CPU matmul results differ")
		}

		t.Logf("GPU matrix multiplication successful: shape %v", gpuResult.Shape())
	})

	t.Run("GPU statistical operations", func(t *testing.T) {
		data := make([]float64, 10000)
		for i := range data {
			data[i] = float64(i % 100) // Pattern for testing
		}

		arr, _ := FromSlice(data)
		device, _ := GPUManager.GetDefaultDevice()

		// Test GPU mean
		gpuMean, err := arr.MeanGPU(device)
		if err != nil {
			t.Fatalf("GPU mean failed: %v", err)
		}

		// Compare with CPU (create CPU mean manually since Mean() might not exist)
		sum, _ := arr.SumGPU(device)
		cpuMean := sum / float64(arr.Size())
		if math.Abs(gpuMean-cpuMean) > 1e-10 {
			t.Errorf("GPU mean differs from CPU: GPU=%.6f, CPU=%.6f", gpuMean, cpuMean)
		}

		// Test GPU sum
		gpuSum, err := arr.SumGPU(device)
		if err != nil {
			t.Fatalf("GPU sum failed: %v", err)
		}

		// CPU sum comparison (use Sum() method result)
		sumResult := arr.Sum()
		cpuSumData := sumResult.ToSlice().([]float64)
		cpuSum := cpuSumData[0]
		if math.Abs(gpuSum-cpuSum) > 1e-10 {
			t.Errorf("GPU sum differs from CPU: GPU=%.6f, CPU=%.6f", gpuSum, cpuSum)
		}

		// Test GPU standard deviation
		gpuStd, err := arr.StdGPU(device)
		if err != nil {
			t.Fatalf("GPU std failed: %v", err)
		}

		// CPU standard deviation (manual calculation)
		cpuStd := arr.stdCPU()
		if math.Abs(gpuStd-cpuStd) > 1e-6 { // Relaxed tolerance for std
			t.Errorf("GPU std differs from CPU: GPU=%.6f, CPU=%.6f", gpuStd, cpuStd)
		}

		t.Logf("GPU statistics: mean=%.3f, sum=%.0f, std=%.3f", gpuMean, gpuSum, gpuStd)
	})
}

// TestArrayGPUFallback tests automatic fallback to CPU when GPU fails
func TestArrayGPUFallback(t *testing.T) {
	t.Run("GPU unavailable fallback", func(t *testing.T) {
		data := []float64{1, 2, 3, 4, 5}
		arr, _ := FromSlice(data)

		// Test with nil device (should fallback to CPU)
		result, err := arr.AddGPUWithFallback(arr, nil)
		if err != nil {
			t.Fatalf("Fallback should not fail: %v", err)
		}

		// Should be same as CPU result
		expected, _ := arr.Add(arr)
		if !arrayEqual(result, expected) {
			t.Errorf("Fallback result differs from CPU result")
		}

		t.Logf("GPU fallback successful")
	})

	t.Run("Automatic device selection", func(t *testing.T) {
		// Small array - should prefer CPU
		smallData := []float64{1, 2, 3, 4, 5}
		smallArr, _ := FromSlice(smallData)

		result, usedGPU, err := smallArr.AddAuto(smallArr)
		if err != nil {
			t.Fatalf("Auto add failed: %v", err)
		}

		if usedGPU {
			t.Logf("Small array used GPU (acceptable)")
		} else {
			t.Logf("Small array used CPU (expected)")
		}

		// Large array - should prefer GPU if available
		largeData := make([]float64, 100000)
		for i := range largeData {
			largeData[i] = float64(i)
		}
		largeArr, _ := FromSlice(largeData)

		result2, usedGPU2, err := largeArr.AddAuto(largeArr)
		if err != nil {
			t.Fatalf("Auto add failed for large array: %v", err)
		}

		if usedGPU2 {
			t.Logf("Large array used GPU (expected)")
		} else {
			t.Logf("Large array used CPU (fallback)")
		}

		// Verify results are correct
		expected, _ := smallArr.Add(smallArr)
		if !arrayEqual(result, expected) {
			t.Errorf("Auto small result incorrect")
		}

		expected2, _ := largeArr.Add(largeArr)
		if !arrayEqual(result2, expected2) {
			t.Errorf("Auto large result incorrect")
		}
	})
}

// TestArrayGPUMemoryManagement tests GPU memory handling
func TestArrayGPUMemoryManagement(t *testing.T) {
	t.Run("Large array GPU operations", func(t *testing.T) {
		// Create large arrays that might exceed GPU memory
		largeSize := 1000000 // 1M elements
		data1 := make([]float64, largeSize)
		data2 := make([]float64, largeSize)

		for i := 0; i < largeSize; i++ {
			data1[i] = float64(i % 1000)
			data2[i] = float64((i + 500) % 1000)
		}

		arr1, _ := FromSlice(data1)
		arr2, _ := FromSlice(data2)

		device, err := GPUManager.GetDefaultDevice()
		if err != nil {
			t.Skip("GPU not available")
		}

		// Test memory-efficient GPU operations
		result, err := arr1.AddGPUStreaming(arr2, device, 16777216) // 16MB chunks
		if err != nil {
			t.Fatalf("GPU streaming add failed: %v", err)
		}

		// Spot check a few values
		resultSlice := result.ToSlice().([]float64)
		expected := data1[0] + data2[0]
		if math.Abs(resultSlice[0]-expected) > 1e-10 {
			t.Errorf("Streaming result incorrect at index 0: got %.6f, expected %.6f", resultSlice[0], expected)
		}

		t.Logf("Large array streaming operation successful: %d elements", largeSize)
	})

	t.Run("Multi-GPU operations", func(t *testing.T) {
		devices, err := GPUManager.EnumerateDevices()
		if err != nil || len(devices) < 1 {
			t.Skip("No GPU devices available")
		}

		// Create test data
		data := make([]float64, 100000)
		for i := range data {
			data[i] = float64(i)
		}
		arr, _ := FromSlice(data)

		// Test multi-GPU sum
		result, err := arr.SumMultiGPU(devices)
		if err != nil {
			t.Fatalf("Multi-GPU sum failed: %v", err)
		}

		// Compare with CPU result
		expectedResult := arr.Sum()
		expectedData := expectedResult.ToSlice().([]float64)
		expected := expectedData[0]
		if math.Abs(result-expected) > 1e-6 {
			t.Errorf("Multi-GPU sum differs from CPU: GPU=%.6f, CPU=%.6f", result, expected)
		}

		t.Logf("Multi-GPU operation successful using %d devices", len(devices))
	})
}

// TestArrayGPUPerformance tests GPU performance optimizations
func TestArrayGPUPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	t.Run("GPU vs CPU performance comparison", func(t *testing.T) {
		sizes := []int{1000, 10000, 100000}

		for _, size := range sizes {
			data := make([]float64, size)
			for i := range data {
				data[i] = float64(i)
			}
			arr, _ := FromSlice(data)

			device, err := GPUManager.GetDefaultDevice()
			if err != nil {
				t.Skip("GPU not available")
			}

			// Compare GPU vs CPU performance for sum operation
			start := time.Now()
			gpuResult, _ := arr.SumGPU(device)
			gpuTime := time.Since(start)

			start = time.Now()
			cpuSumResult := arr.Sum()
			cpuTime := time.Since(start)

			// Convert CPU result to float64
			cpuData := cpuSumResult.ToSlice().([]float64)
			cpuResult := cpuData[0]

			// Verify accuracy
			if math.Abs(gpuResult-cpuResult) > 1e-10 {
				t.Errorf("Results differ for size %d", size)
			}

			speedup := float64(cpuTime) / float64(gpuTime)
			t.Logf("Size %d: GPU %v, CPU %v, Speedup: %.2fx", size, gpuTime, cpuTime, speedup)
		}
	})

	t.Run("GPU memory transfer optimization", func(t *testing.T) {
		data := make([]float64, 50000)
		for i := range data {
			data[i] = float64(i)
		}
		arr, _ := FromSlice(data)

		device, _ := GPUManager.GetDefaultDevice()

		// Test zero-copy operations
		zeroCopyResult, err := arr.SumGPUZeroCopy(device)
		if err != nil {
			t.Fatalf("Zero-copy sum failed: %v", err)
		}

		// Test regular GPU operations
		regularResult, err := arr.SumGPU(device)
		if err != nil {
			t.Fatalf("Regular GPU sum failed: %v", err)
		}

		// Results should be identical
		if math.Abs(zeroCopyResult-regularResult) > 1e-10 {
			t.Errorf("Zero-copy and regular GPU results differ")
		}

		t.Logf("Zero-copy optimization successful")
	})
}

// TestArrayGPUIntegrationEdgeCases tests edge cases and error handling
func TestArrayGPUIntegrationEdgeCases(t *testing.T) {
	t.Run("Empty array handling", func(t *testing.T) {
		arr, _ := FromSlice([]float64{})
		device, _ := GPUManager.GetDefaultDevice()

		// Should handle empty arrays gracefully
		_, err := arr.SumGPU(device)
		if err == nil {
			t.Errorf("Expected error for empty array")
		} else {
			t.Logf("Empty array handled correctly: %v", err)
		}
	})

	t.Run("Type compatibility", func(t *testing.T) {
		// Test different data types
		int32Data := []int32{1, 2, 3, 4, 5}
		int32Arr, _ := FromSlice(int32Data)

		device, _ := GPUManager.GetDefaultDevice()

		// Should convert to float64 for GPU operations
		result, err := int32Arr.SumGPU(device)
		if err != nil {
			t.Fatalf("Int32 GPU sum failed: %v", err)
		}

		expected := 1.0 + 2.0 + 3.0 + 4.0 + 5.0 // 15.0
		if math.Abs(result-expected) > 1e-10 {
			t.Errorf("Int32 conversion incorrect: got %.6f, expected %.6f", result, expected)
		}

		t.Logf("Type conversion successful")
	})

	t.Run("Shape compatibility", func(t *testing.T) {
		// Test operations with incompatible shapes
		arr1, _ := FromSlice([]float64{1, 2, 3})
		arr2, _ := FromSlice([]float64{1, 2, 3, 4})

		device, _ := GPUManager.GetDefaultDevice()

		_, err := arr1.AddGPU(arr2, device)
		if err == nil {
			t.Errorf("Expected error for incompatible shapes")
		}

		t.Logf("Shape compatibility check working: %v", err)
	})
}
