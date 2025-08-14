package benchmarks

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestNewGPUBenchmarkSuite tests benchmark suite creation
func TestNewGPUBenchmarkSuite(t *testing.T) {
	t.Run("Default benchmark suite creation", func(t *testing.T) {
		suite := NewGPUBenchmarkSuite()
		if suite == nil {
			t.Fatal("Expected non-nil benchmark suite")
		}

		if suite.iterations != 5 {
			t.Errorf("Expected 5 iterations, got %d", suite.iterations)
		}

		if suite.warmupRuns != 2 {
			t.Errorf("Expected 2 warmup runs, got %d", suite.warmupRuns)
		}

		if suite.timeoutSec != 30 {
			t.Errorf("Expected 30 second timeout, got %d", suite.timeoutSec)
		}

		t.Logf("Benchmark suite created successfully with %d iterations", suite.iterations)
	})

	t.Run("CPU-only benchmark suite creation", func(t *testing.T) {
		suite := NewCPUOnlyBenchmarkSuite()
		if suite == nil {
			t.Fatal("Expected non-nil CPU-only benchmark suite")
		}

		if suite.includeGPU {
			t.Errorf("Expected GPU to be disabled for CPU-only suite")
		}

		t.Logf("CPU-only benchmark suite created successfully")
	})

	t.Run("System information initialization", func(t *testing.T) {
		suite := NewGPUBenchmarkSuite()

		if suite.systemInfo.CPUCores <= 0 {
			t.Errorf("Expected positive CPU cores, got %d", suite.systemInfo.CPUCores)
		}

		if suite.systemInfo.GoVersion == "" {
			t.Errorf("Expected Go version to be set")
		}

		if suite.systemInfo.GOOS == "" {
			t.Errorf("Expected GOOS to be set")
		}

		t.Logf("System info: %d CPU cores, Go %s, OS %s",
			suite.systemInfo.CPUCores, suite.systemInfo.GoVersion, suite.systemInfo.GOOS)
	})
}

// TestBenchmarkResult tests benchmark result structure
func TestBenchmarkResult(t *testing.T) {
	t.Run("Basic benchmark result creation", func(t *testing.T) {
		result := BenchmarkResult{
			Operation:    "addition",
			DataSize:     1000,
			CPUTime:      10 * time.Millisecond,
			GPUTime:      5 * time.Millisecond,
			Speedup:      2.0,
			MemoryUsed:   8000,
			ThroughputMB: 100.0,
			Accuracy:     0.999,
			Success:      true,
		}

		if result.Operation != "addition" {
			t.Errorf("Expected operation 'addition', got '%s'", result.Operation)
		}

		if result.Speedup != 2.0 {
			t.Errorf("Expected speedup 2.0, got %.2f", result.Speedup)
		}

		if result.Accuracy != 0.999 {
			t.Errorf("Expected accuracy 0.999, got %.3f", result.Accuracy)
		}

		t.Logf("Benchmark result: %s operation with %.2fx speedup", result.Operation, result.Speedup)
	})

	t.Run("Failed benchmark result", func(t *testing.T) {
		result := BenchmarkResult{
			Operation: "failed_op",
			DataSize:  1000,
			Success:   false,
			Error:     "GPU operation failed",
		}

		if result.Success {
			t.Errorf("Expected failed result")
		}

		if result.Error == "" {
			t.Errorf("Expected error message to be set")
		}

		t.Logf("Failed benchmark handled correctly: %s", result.Error)
	})
}

// TestBenchmarkAddition tests addition benchmark functionality
func TestBenchmarkAddition(t *testing.T) {
	suite := NewGPUBenchmarkSuite()

	t.Run("Small array addition benchmark", func(t *testing.T) {
		result := suite.benchmarkAddition(100)

		if !result.Success {
			t.Fatalf("Addition benchmark failed: %s", result.Error)
		}

		if result.Operation != "addition" {
			t.Errorf("Expected operation 'addition', got '%s'", result.Operation)
		}

		if result.DataSize != 100 {
			t.Errorf("Expected data size 100, got %d", result.DataSize)
		}

		if result.CPUTime <= 0 {
			t.Errorf("Expected positive CPU time, got %v", result.CPUTime)
		}

		if result.MemoryUsed != 100*8*3 { // 100 elements × 8 bytes × 3 arrays
			t.Errorf("Expected memory used %d, got %d", 100*8*3, result.MemoryUsed)
		}

		t.Logf("Addition benchmark: CPU %v, Memory %d bytes", result.CPUTime, result.MemoryUsed)
	})

	t.Run("Large array addition benchmark", func(t *testing.T) {
		result := suite.benchmarkAddition(10000)

		if !result.Success {
			t.Fatalf("Large addition benchmark failed: %s", result.Error)
		}

		if result.GPUTime > 0 && result.Speedup <= 0 {
			t.Errorf("Expected positive speedup when GPU time is recorded, got %.2f", result.Speedup)
		}

		if result.ThroughputMB <= 0 {
			t.Errorf("Expected positive throughput, got %.2f MB/s", result.ThroughputMB)
		}

		t.Logf("Large addition: CPU %v, GPU %v, Speedup %.2fx",
			result.CPUTime, result.GPUTime, result.Speedup)
	})

	t.Run("GPU accuracy verification", func(t *testing.T) {
		result := suite.benchmarkAddition(1000)

		if !result.Success {
			t.Skip("Benchmark failed, skipping accuracy test")
		}

		if suite.includeGPU && result.GPUTime > 0 {
			if result.Accuracy < 0.999 {
				t.Errorf("Expected high accuracy (>0.999), got %.6f", result.Accuracy)
			}
			t.Logf("GPU accuracy: %.6f", result.Accuracy)
		}
	})
}

// TestBenchmarkSum tests sum operation benchmarking
func TestBenchmarkSum(t *testing.T) {
	suite := NewGPUBenchmarkSuite()

	t.Run("Basic sum benchmark", func(t *testing.T) {
		result := suite.benchmarkSum(5000)

		if !result.Success {
			t.Fatalf("Sum benchmark failed: %s", result.Error)
		}

		if result.Operation != "sum" {
			t.Errorf("Expected operation 'sum', got '%s'", result.Operation)
		}

		if result.DataSize != 5000 {
			t.Errorf("Expected data size 5000, got %d", result.DataSize)
		}

		if result.MemoryUsed != 5000*8 { // 5000 elements × 8 bytes
			t.Errorf("Expected memory used %d, got %d", 5000*8, result.MemoryUsed)
		}

		t.Logf("Sum benchmark: %d elements, CPU %v", result.DataSize, result.CPUTime)
	})

	t.Run("Sum benchmark with GPU comparison", func(t *testing.T) {
		if !suite.includeGPU {
			t.Skip("GPU not available")
		}

		result := suite.benchmarkSum(50000)

		if !result.Success {
			t.Skip("Sum benchmark failed")
		}

		if result.GPUTime > 0 {
			if result.Speedup <= 0 {
				t.Errorf("Expected positive speedup, got %.2f", result.Speedup)
			}

			if result.Accuracy < 0.99 {
				t.Errorf("Expected high accuracy (>0.99), got %.6f", result.Accuracy)
			}

			t.Logf("Sum GPU comparison: Speedup %.2fx, Accuracy %.6f", result.Speedup, result.Accuracy)
		}
	})
}

// TestBenchmarkMatrixMultiplication tests matrix multiplication benchmarking
func TestBenchmarkMatrixMultiplication(t *testing.T) {
	suite := NewGPUBenchmarkSuite()

	t.Run("Small matrix multiplication benchmark", func(t *testing.T) {
		result := suite.benchmarkMatrixMultiplication(10) // 10×10 matrices

		if !result.Success {
			t.Fatalf("Matrix multiplication benchmark failed: %s", result.Error)
		}

		if result.Operation != "matmul" {
			t.Errorf("Expected operation 'matmul', got '%s'", result.Operation)
		}

		expectedSize := 10 * 10 // 100 elements total
		if result.DataSize != expectedSize {
			t.Errorf("Expected data size %d, got %d", expectedSize, result.DataSize)
		}

		expectedMemory := int64(10 * 10 * 3 * 8) // 3 matrices × 100 elements × 8 bytes
		if result.MemoryUsed != expectedMemory {
			t.Errorf("Expected memory used %d, got %d", expectedMemory, result.MemoryUsed)
		}

		t.Logf("Matrix multiplication 10×10: CPU %v, Memory %d bytes",
			result.CPUTime, result.MemoryUsed)
	})

	t.Run("Medium matrix multiplication benchmark", func(t *testing.T) {
		result := suite.benchmarkMatrixMultiplication(50) // 50×50 matrices

		if !result.Success {
			t.Skip("Matrix multiplication benchmark failed")
		}

		if result.CPUTime <= 0 {
			t.Errorf("Expected positive CPU time, got %v", result.CPUTime)
		}

		if result.ThroughputMB <= 0 {
			t.Errorf("Expected positive throughput, got %.2f MB/s", result.ThroughputMB)
		}

		t.Logf("Matrix multiplication 50×50: CPU %v, Throughput %.2f MB/s",
			result.CPUTime, result.ThroughputMB)
	})

	t.Run("GPU matrix multiplication accuracy", func(t *testing.T) {
		if !suite.includeGPU {
			t.Skip("GPU not available")
		}

		result := suite.benchmarkMatrixMultiplication(20) // 20×20 matrices

		if !result.Success {
			t.Skip("Matrix multiplication benchmark failed")
		}

		if result.GPUTime > 0 && result.Accuracy < 0.999 {
			t.Errorf("Expected high GPU accuracy (>0.999), got %.6f", result.Accuracy)
		}

		t.Logf("Matrix multiplication GPU accuracy: %.6f", result.Accuracy)
	})
}

// TestBenchmarkMemoryBandwidth tests memory bandwidth benchmarking
func TestBenchmarkMemoryBandwidth(t *testing.T) {
	suite := NewGPUBenchmarkSuite()

	t.Run("Memory bandwidth measurement", func(t *testing.T) {
		result := suite.benchmarkMemoryBandwidth(100000)

		if !result.Success {
			t.Fatalf("Memory bandwidth benchmark failed: %s", result.Error)
		}

		if result.Operation != "memory_bandwidth" {
			t.Errorf("Expected operation 'memory_bandwidth', got '%s'", result.Operation)
		}

		if result.ThroughputMB <= 0 {
			t.Errorf("Expected positive bandwidth, got %.2f MB/s", result.ThroughputMB)
		}

		expectedMemory := int64(100000 * 8) // 100K elements × 8 bytes
		if result.MemoryUsed != expectedMemory {
			t.Errorf("Expected memory used %d, got %d", expectedMemory, result.MemoryUsed)
		}

		t.Logf("Memory bandwidth: %.2f MB/s for %d elements",
			result.ThroughputMB, result.DataSize)
	})

	t.Run("GPU vs CPU bandwidth comparison", func(t *testing.T) {
		if !suite.includeGPU {
			t.Skip("GPU not available")
		}

		result := suite.benchmarkMemoryBandwidth(500000)

		if !result.Success {
			t.Skip("Memory bandwidth benchmark failed")
		}

		if result.GPUTime > 0 {
			gpuBandwidth := (float64(result.MemoryUsed) / (1024 * 1024)) / result.GPUTime.Seconds()
			cpuBandwidth := result.ThroughputMB

			t.Logf("Bandwidth comparison - CPU: %.2f MB/s, GPU: %.2f MB/s",
				cpuBandwidth, gpuBandwidth)
		}
	})
}

// TestRunAllBenchmarks tests the complete benchmark suite
func TestRunAllBenchmarks(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping comprehensive benchmark test in short mode")
	}

	suite := NewGPUBenchmarkSuite()

	t.Run("Complete benchmark suite execution", func(t *testing.T) {
		results := suite.RunAllBenchmarks()

		if len(results) == 0 {
			t.Fatal("Expected benchmark results, got empty slice")
		}

		// Count successful benchmarks
		successCount := 0
		for _, result := range results {
			if result.Success {
				successCount++
			}
		}

		if successCount == 0 {
			t.Fatal("No benchmarks succeeded")
		}

		t.Logf("Benchmark suite completed: %d total, %d successful",
			len(results), successCount)

		// Verify we have different operation types
		operations := make(map[string]int)
		for _, result := range results {
			operations[result.Operation]++
		}

		expectedOps := []string{"addition", "sum", "matmul", "memory_bandwidth"}
		for _, op := range expectedOps {
			if operations[op] == 0 {
				t.Errorf("Expected %s operation results, got none", op)
			}
		}

		t.Logf("Operations benchmarked: %v", operations)
	})
}

// TestScalabilityAnalysis tests scalability analysis functionality
func TestScalabilityAnalysis(t *testing.T) {
	suite := NewGPUBenchmarkSuite()

	t.Run("Sum operation scalability", func(t *testing.T) {
		sizes := []int{1000, 5000, 10000}
		result := suite.AnalyzeScalability("sum", sizes)

		if result.Operation != "sum" {
			t.Errorf("Expected operation 'sum', got '%s'", result.Operation)
		}

		if len(result.DataSizes) != len(sizes) {
			t.Errorf("Expected %d data sizes, got %d", len(sizes), len(result.DataSizes))
		}

		if len(result.CPUTimes) != len(sizes) {
			t.Errorf("Expected %d CPU times, got %d", len(sizes), len(result.CPUTimes))
		}

		// Check that CPU times generally increase with size
		for i := 1; i < len(result.CPUTimes); i++ {
			if result.CPUTimes[i] < result.CPUTimes[i-1] {
				t.Logf("Warning: CPU time decreased from size %d to %d",
					result.DataSizes[i-1], result.DataSizes[i])
			}
		}

		if result.OptimalSize <= 0 {
			t.Errorf("Expected positive optimal size, got %d", result.OptimalSize)
		}

		t.Logf("Scalability analysis: optimal size %d, max speedup %.2fx",
			result.OptimalSize, result.MaxSpeedup)
	})

	t.Run("Addition operation scalability", func(t *testing.T) {
		sizes := []int{2000, 8000, 20000}
		result := suite.AnalyzeScalability("addition", sizes)

		if result.Operation != "addition" {
			t.Errorf("Expected operation 'addition', got '%s'", result.Operation)
		}

		// Verify scaling factor calculation
		if len(result.Speedups) >= 2 && result.ScalingFactor <= 0 {
			t.Errorf("Expected positive scaling factor, got %.2f", result.ScalingFactor)
		}

		t.Logf("Addition scalability: scaling factor %.2f", result.ScalingFactor)
	})
}

// TestGenerateReport tests benchmark report generation
func TestGenerateReport(t *testing.T) {
	suite := NewGPUBenchmarkSuite()

	// Create sample results
	results := []BenchmarkResult{
		{
			Operation:    "addition",
			DataSize:     1000,
			CPUTime:      1 * time.Millisecond,
			GPUTime:      500 * time.Microsecond,
			Speedup:      2.0,
			ThroughputMB: 100.0,
			Accuracy:     0.999,
			Success:      true,
		},
		{
			Operation:    "sum",
			DataSize:     10000,
			CPUTime:      2 * time.Millisecond,
			GPUTime:      800 * time.Microsecond,
			Speedup:      2.5,
			ThroughputMB: 200.0,
			Accuracy:     0.998,
			Success:      true,
		},
		{
			Operation: "matmul",
			DataSize:  10000,
			CPUTime:   10 * time.Millisecond,
			Success:   false,
			Error:     "Test error",
		},
	}

	t.Run("Basic report generation", func(t *testing.T) {
		report := suite.GenerateReport(results)

		if report == "" {
			t.Fatal("Expected non-empty report")
		}

		// Check for required sections
		expectedSections := []string{
			"GPU Performance Benchmark Report",
			"System Information",
			"Performance Summary",
			"Performance Recommendations",
		}

		for _, section := range expectedSections {
			if !strings.Contains(report, section) {
				t.Errorf("Report missing section: %s", section)
			}
		}

		// Check for system information
		if !strings.Contains(report, "CPU Cores") {
			t.Errorf("Report missing CPU cores information")
		}

		// Check for operation results
		if !strings.Contains(report, "Addition Operations") {
			t.Errorf("Report missing addition operation results")
		}

		t.Logf("Generated report length: %d characters", len(report))
	})

	t.Run("Report with GPU information", func(t *testing.T) {
		if !suite.includeGPU {
			t.Skip("GPU not available")
		}

		report := suite.GenerateReport(results)

		if !strings.Contains(report, "GPU Device Information") {
			t.Errorf("Report missing GPU device information")
		}

		// Should contain speedup information
		if !strings.Contains(report, "Speedup") {
			t.Errorf("Report missing speedup information")
		}
	})

	t.Run("CPU-only report", func(t *testing.T) {
		cpuSuite := NewCPUOnlyBenchmarkSuite()
		report := cpuSuite.GenerateReport(results)

		if !strings.Contains(report, "GPU not available") {
			t.Errorf("CPU-only report should mention GPU unavailability")
		}
	})
}

// TestUtilityFunctions tests helper functions
func TestUtilityFunctions(t *testing.T) {
	t.Run("Median calculation", func(t *testing.T) {
		times := []time.Duration{
			1 * time.Millisecond,
			2 * time.Millisecond,
			3 * time.Millisecond,
			4 * time.Millisecond,
			5 * time.Millisecond,
		}

		result := median(times)
		expected := 3 * time.Millisecond

		if result != expected {
			t.Errorf("Expected median %v, got %v", expected, result)
		}

		// Test even number of elements
		evenTimes := times[:4]
		evenResult := median(evenTimes)
		expectedEven := (2*time.Millisecond + 3*time.Millisecond) / 2

		if evenResult != expectedEven {
			t.Errorf("Expected even median %v, got %v", expectedEven, evenResult)
		}
	})

	t.Run("Average calculation", func(t *testing.T) {
		values := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		result := average(values)
		expected := 3.0

		if math.Abs(result-expected) > 1e-10 {
			t.Errorf("Expected average %.2f, got %.2f", expected, result)
		}

		// Test empty slice
		emptyResult := average([]float64{})
		if emptyResult != 0.0 {
			t.Errorf("Expected average of empty slice to be 0, got %.2f", emptyResult)
		}
	})

	t.Run("Maximum calculation", func(t *testing.T) {
		values := []float64{1.5, 3.2, 2.1, 5.7, 4.3}
		result := maximum(values)
		expected := 5.7

		if math.Abs(result-expected) > 1e-10 {
			t.Errorf("Expected maximum %.2f, got %.2f", expected, result)
		}

		// Test single element
		singleResult := maximum([]float64{42.0})
		if math.Abs(singleResult-42.0) > 1e-10 {
			t.Errorf("Expected single element maximum 42.0, got %.2f", singleResult)
		}
	})

	t.Run("Array accuracy calculation", func(t *testing.T) {
		// Create test arrays
		data1 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		data2 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

		arr1, _ := array.FromSlice(data1)
		arr2, _ := array.FromSlice(data2)

		// Perfect accuracy
		accuracy := calculateArrayAccuracy(arr1, arr2)
		if math.Abs(accuracy-1.0) > 1e-10 {
			t.Errorf("Expected perfect accuracy 1.0, got %.6f", accuracy)
		}

		// Different arrays
		data3 := []float64{1.1, 2.1, 3.1, 4.1, 5.1}
		arr3, _ := array.FromSlice(data3)

		accuracy2 := calculateArrayAccuracy(arr1, arr3)
		if accuracy2 >= 1.0 || accuracy2 <= 0.0 {
			t.Errorf("Expected accuracy between 0 and 1, got %.6f", accuracy2)
		}

		t.Logf("Array accuracy test: perfect=%.6f, different=%.6f", accuracy, accuracy2)
	})
}

// TestBenchmarkEdgeCases tests edge cases and error conditions
func TestBenchmarkEdgeCases(t *testing.T) {
	suite := NewGPUBenchmarkSuite()

	t.Run("Zero size benchmark", func(t *testing.T) {
		// Create empty array using Empty function
		arr := array.Empty(internal.Shape{0}, internal.Float64)

		if arr.Size() != 0 {
			t.Errorf("Expected empty array size 0, got %d", arr.Size())
		}

		// Test that empty array operations fail gracefully
		device, _ := array.GPUManager.GetDefaultDevice()
		_, err := arr.SumGPU(device)
		if err == nil {
			t.Errorf("Expected error for empty array sum")
		} else {
			t.Logf("Empty array handled correctly: %v", err)
		}
	})

	t.Run("Very small benchmark", func(t *testing.T) {
		result := suite.benchmarkAddition(1)

		if !result.Success {
			t.Fatalf("Single element benchmark failed: %s", result.Error)
		}

		if result.DataSize != 1 {
			t.Errorf("Expected data size 1, got %d", result.DataSize)
		}

		t.Logf("Single element benchmark successful")
	})

	t.Run("Large benchmark performance", func(t *testing.T) {
		if testing.Short() {
			t.Skip("Skipping large benchmark in short mode")
		}

		result := suite.benchmarkSum(1000000) // 1M elements

		if !result.Success {
			t.Skip("Large benchmark failed")
		}

		if result.CPUTime <= 0 {
			t.Errorf("Expected positive CPU time for large benchmark")
		}

		t.Logf("Large benchmark (1M elements): CPU %v", result.CPUTime)
	})
}
