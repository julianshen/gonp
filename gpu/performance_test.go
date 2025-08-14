package gpu

import (
	"context"
	"math"
	"runtime"
	"testing"
	"time"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	stats "github.com/julianshen/gonp/stats"
)

// TestMemoryOptimization tests advanced memory optimization strategies
func TestMemoryOptimization(t *testing.T) {
	t.Run("Zero-copy operations", func(t *testing.T) {
		// Test zero-copy data transfer between CPU and GPU
		size := 1024 * 1024 // 1M elements
		data := make([]float64, size)
		for i := range data {
			data[i] = float64(i)
		}
		arr, _ := array.FromSlice(data)

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for zero-copy test")
		}

		// Test zero-copy buffer creation
		zeroCopyBuffer, err := CreateZeroCopyBuffer(device, arr)
		if err != nil {
			t.Fatalf("Zero-copy buffer creation failed: %v", err)
		}
		defer zeroCopyBuffer.Free()

		// Verify data integrity with zero-copy
		retrievedData, err := zeroCopyBuffer.GetData()
		if err != nil {
			t.Fatalf("Zero-copy data retrieval failed: %v", err)
		}

		if len(retrievedData) != len(data) {
			t.Errorf("Zero-copy data size mismatch: expected %d, got %d", len(data), len(retrievedData))
		}

		// Verify first and last elements to check data integrity
		if math.Abs(retrievedData[0]-data[0]) > 1e-10 {
			t.Errorf("Zero-copy data corruption at start: expected %.1f, got %.6f", data[0], retrievedData[0])
		}

		if math.Abs(retrievedData[size-1]-data[size-1]) > 1e-10 {
			t.Errorf("Zero-copy data corruption at end: expected %.1f, got %.6f", data[size-1], retrievedData[size-1])
		}

		t.Logf("Zero-copy operations successful for %d elements", size)
	})

	t.Run("Memory-mapped file operations", func(t *testing.T) {
		// Test memory-mapped files for very large datasets
		if testing.Short() {
			t.Skip("Skipping memory-mapped file test in short mode")
		}

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for memory-mapped test")
		}

		// Test creation of memory-mapped array
		size := 50 * 1024 * 1024 // 50MB worth of float64s
		mmapArray, err := CreateMemoryMappedArray(size, internal.Float64)
		if err != nil {
			t.Fatalf("Memory-mapped array creation failed: %v", err)
		}
		defer mmapArray.Close()

		// Test GPU operations on memory-mapped data
		arr := mmapArray.ToArray()
		result, err := SumGPU(arr, device)
		if err != nil {
			t.Fatalf("GPU sum on memory-mapped data failed: %v", err)
		}

		// Memory-mapped arrays should be initialized to zero
		expectedSum := 0.0
		if math.Abs(result-expectedSum) > 1e-6 {
			t.Errorf("Memory-mapped array sum incorrect: expected %.1f, got %.6f", expectedSum, result)
		}

		t.Logf("Memory-mapped operations successful for %d elements (%.1f MB)", size, float64(size*8)/(1024*1024))
	})

	t.Run("Streaming operations for large datasets", func(t *testing.T) {
		// Test streaming operations for datasets larger than GPU memory
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for streaming test")
		}

		// Simulate very large dataset (larger than typical GPU memory)
		totalElements := 128 * 1024 * 1024 // 128M elements (1GB of float64)
		chunkSize := 16 * 1024 * 1024      // 16M elements per chunk (128MB)

		// Create streaming processor
		processor, err := NewStreamingProcessor(device, chunkSize)
		if err != nil {
			t.Fatalf("Streaming processor creation failed: %v", err)
		}
		defer processor.Close()

		// Test streaming mean computation
		sum := 0.0
		chunks := totalElements / chunkSize

		for i := 0; i < chunks; i++ {
			// Generate chunk data
			chunkData := make([]float64, chunkSize)
			for j := range chunkData {
				chunkData[j] = float64(i*chunkSize + j)
			}
			chunk, _ := array.FromSlice(chunkData)

			// Process chunk on GPU
			chunkSum, err := processor.ProcessChunk(chunk, "sum")
			if err != nil {
				t.Fatalf("Chunk processing failed for chunk %d: %v", i, err)
			}
			sum += chunkSum
		}

		// Calculate expected sum: sum of 0 to (totalElements-1)
		expectedSum := float64(totalElements-1) * float64(totalElements) / 2.0
		accuracy := math.Abs(sum-expectedSum) / expectedSum

		if accuracy > 1e-10 {
			t.Errorf("Streaming sum accuracy insufficient: expected %.1f, got %.1f, accuracy: %.2e", expectedSum, sum, accuracy)
		}

		t.Logf("Streaming operations successful: processed %d elements in %d chunks", totalElements, chunks)
	})

	t.Run("Compression-based data transfer", func(t *testing.T) {
		// Test compressed data transfer to reduce bandwidth requirements
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for compression test")
		}

		// Create compressible data (many repeated values)
		size := 1024 * 1024
		data := make([]float64, size)
		for i := range data {
			data[i] = float64(i % 100) // Repeating pattern for good compression
		}
		arr, _ := array.FromSlice(data)

		// Test compressed transfer
		transferStats, err := TransferDataCompressed(device, arr, "lz4")
		if err != nil {
			t.Fatalf("Compressed data transfer failed: %v", err)
		}

		// Verify compression efficiency
		originalSize := int64(len(data) * 8) // 8 bytes per float64
		compressionRatio := float64(originalSize) / float64(transferStats.CompressedSize)

		if compressionRatio < 2.0 {
			t.Errorf("Compression ratio too low: %.2fx (expected >2x for repetitive data)", compressionRatio)
		}

		// Verify transfer time improvement
		speedup := float64(transferStats.UncompressedTransferTime) / float64(transferStats.CompressedTransferTime)
		if speedup < 1.2 {
			t.Logf("Warning: compression speedup only %.2fx (may indicate fast connection)", speedup)
		}

		t.Logf("Compressed transfer: %.2fx compression, %.2fx speedup", compressionRatio, speedup)
	})

	t.Run("Asynchronous computation overlap", func(t *testing.T) {
		// Test overlapping computation with data transfer
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for async test")
		}

		// Create multiple datasets for pipeline testing
		numDatasets := 4
		datasetSize := 512 * 1024 // 512K elements each
		datasets := make([]*array.Array, numDatasets)

		for i := 0; i < numDatasets; i++ {
			data := make([]float64, datasetSize)
			for j := range data {
				data[j] = float64(i*datasetSize + j)
			}
			datasets[i], _ = array.FromSlice(data)
		}

		// Test asynchronous pipeline
		pipeline, err := NewAsyncPipeline(device, 2) // 2 streams for overlap
		if err != nil {
			t.Fatalf("Async pipeline creation failed: %v", err)
		}
		defer pipeline.Close()

		// Submit all operations asynchronously
		results := make([]<-chan float64, numDatasets)
		for i, dataset := range datasets {
			resultChan, err := pipeline.SubmitMeanOperation(dataset)
			if err != nil {
				t.Fatalf("Async operation submission failed for dataset %d: %v", i, err)
			}
			results[i] = resultChan
		}

		// Collect results
		for i, resultChan := range results {
			result := <-resultChan
			// Mean of dataset i: values from i*datasetSize to (i+1)*datasetSize-1
			start := float64(i * datasetSize)
			end := float64((i+1)*datasetSize - 1)
			expectedMean := (start + end) / 2.0

			if math.Abs(result-expectedMean) > 1e-6 {
				t.Errorf("Async result incorrect for dataset %d: expected %.1f, got %.6f", i, expectedMean, result)
			}
		}

		// Check pipeline efficiency
		stats := pipeline.GetStats()
		efficiency := stats.ComputeTime / (stats.ComputeTime + stats.IdleTime)

		if efficiency < 0.7 {
			t.Logf("Warning: pipeline efficiency %.1f%% (expected >70%%)", efficiency*100)
		}

		t.Logf("Asynchronous pipeline: %.1f%% efficiency, %d operations completed", efficiency*100, numDatasets)
	})
}

// TestMultiGPUSupport tests multi-GPU functionality and load balancing
func TestMultiGPUSupport(t *testing.T) {
	t.Run("Multi-GPU device enumeration", func(t *testing.T) {
		// Test detection and enumeration of multiple GPU devices
		devices, err := EnumerateDevices()
		if err != nil {
			t.Fatalf("Device enumeration failed: %v", err)
		}

		if len(devices) == 0 {
			t.Skip("No GPU devices available for multi-GPU test")
		}

		// Test capabilities of each device
		for i, device := range devices {
			if !device.IsAvailable() {
				continue
			}

			major, minor := device.ComputeCapability()
			memory := device.MemorySize()
			backend := device.GetBackend()

			t.Logf("Device %d: %s, Compute: %d.%d, Memory: %.1f GB, Backend: %s",
				i, device.Name(), major, minor,
				float64(memory)/(1024*1024*1024), backend)

			// Test basic operation on each device
			testData := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
			arr, _ := array.FromSlice(testData)

			result, err := MeanGPU(arr, device)
			if err != nil {
				t.Errorf("Device %d basic operation failed: %v", i, err)
			} else if math.Abs(result-3.0) > 1e-10 {
				t.Errorf("Device %d mean incorrect: expected 3.0, got %.6f", i, result)
			}
		}
	})

	t.Run("Data parallelism across multiple GPUs", func(t *testing.T) {
		// Test splitting data across multiple GPUs for parallel processing
		devices, err := EnumerateDevices()
		if err != nil {
			t.Fatalf("Device enumeration failed: %v", err)
		}

		// Filter available GPU devices
		gpuDevices := make([]Device, 0)
		for _, device := range devices {
			if device.IsAvailable() && device.GetBackend() != BackendCPU {
				gpuDevices = append(gpuDevices, device)
			}
		}

		if len(gpuDevices) < 2 {
			t.Skip("Need at least 2 GPUs for data parallelism test")
		}

		// Create large dataset for parallel processing
		totalSize := 2 * 1024 * 1024 // 2M elements
		data := make([]float64, totalSize)
		for i := range data {
			data[i] = float64(i)
		}
		arr, _ := array.FromSlice(data)

		// Test multi-GPU parallel sum
		multiGPUManager, err := NewMultiGPUManager(gpuDevices)
		if err != nil {
			t.Fatalf("Multi-GPU manager creation failed: %v", err)
		}
		defer multiGPUManager.Close()

		// Time multi-GPU parallel operation
		start := time.Now()
		parallelResult, err := multiGPUManager.ParallelSum(arr)
		if err != nil {
			t.Fatalf("Multi-GPU parallel sum failed: %v", err)
		}
		parallelTime := time.Since(start)

		// Time single GPU operation for comparison
		start = time.Now()
		singleResult, err := SumGPU(arr, gpuDevices[0])
		if err != nil {
			t.Fatalf("Single GPU sum failed: %v", err)
		}
		singleTime := time.Since(start)

		// Verify accuracy
		if math.Abs(parallelResult-singleResult) > 1e-6 {
			t.Errorf("Multi-GPU result differs from single GPU: %.6f vs %.6f", parallelResult, singleResult)
		}

		// Calculate speedup
		speedup := float64(singleTime) / float64(parallelTime)

		t.Logf("Multi-GPU vs Single GPU: %.2fx speedup (%v vs %v)", speedup, parallelTime, singleTime)

		// Multi-GPU should provide some speedup (at least 1.2x for overhead compensation)
		if speedup < 1.2 && len(gpuDevices) > 1 {
			t.Logf("Warning: multi-GPU speedup %.2fx lower than expected", speedup)
		}
	})

	t.Run("Dynamic load balancing", func(t *testing.T) {
		// Test dynamic load balancing across heterogeneous GPUs
		devices, err := EnumerateDevices()
		if err != nil {
			t.Fatalf("Device enumeration failed: %v", err)
		}

		// Filter available GPU devices
		gpuDevices := make([]Device, 0)
		for _, device := range devices {
			if device.IsAvailable() && device.GetBackend() != BackendCPU {
				gpuDevices = append(gpuDevices, device)
			}
		}

		if len(gpuDevices) < 2 {
			t.Skip("Need at least 2 GPUs for load balancing test")
		}

		// Create load balancer
		loadBalancer, err := NewDynamicLoadBalancer(gpuDevices)
		if err != nil {
			t.Fatalf("Load balancer creation failed: %v", err)
		}
		defer loadBalancer.Close()

		// Submit multiple tasks of varying sizes
		taskSizes := []int{64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024}
		tasks := make([]*LoadBalancedTask, len(taskSizes))

		for i, size := range taskSizes {
			data := make([]float64, size)
			for j := range data {
				data[j] = float64(j)
			}
			arr, _ := array.FromSlice(data)

			task, err := loadBalancer.SubmitSumTask(arr)
			if err != nil {
				t.Fatalf("Task submission failed for size %d: %v", size, err)
			}
			tasks[i] = task
		}

		// Wait for all tasks to complete and verify results
		for i, task := range tasks {
			result, err := task.GetResult()
			if err != nil {
				t.Fatalf("Task %d completion failed: %v", i, err)
			}

			// Verify result accuracy
			expectedSum := float64(taskSizes[i]-1) * float64(taskSizes[i]) / 2.0
			if math.Abs(result-expectedSum) > 1e-6 {
				t.Errorf("Task %d result incorrect: expected %.1f, got %.6f", i, expectedSum, result)
			}
		}

		// Check load balancing statistics
		stats := loadBalancer.GetStats()

		// Verify workload distribution is reasonably balanced
		maxDeviceTime := 0.0
		minDeviceTime := math.Inf(1)

		for deviceID, deviceStats := range stats.DeviceStats {
			deviceTime := deviceStats.TotalComputeTime
			if deviceTime > maxDeviceTime {
				maxDeviceTime = deviceTime
			}
			if deviceTime < minDeviceTime {
				minDeviceTime = deviceTime
			}

			t.Logf("Device %d: %.2fms compute time, %d tasks", deviceID, deviceTime*1000, deviceStats.TasksCompleted)
		}

		// Load balance efficiency: ratio of min to max device utilization
		balanceEfficiency := minDeviceTime / maxDeviceTime
		if balanceEfficiency < 0.5 {
			t.Logf("Warning: load balance efficiency %.1f%% (expected >50%%)", balanceEfficiency*100)
		}

		t.Logf("Load balancing: %.1f%% efficiency, %d tasks completed", balanceEfficiency*100, len(tasks))
	})

	t.Run("GPU cluster support", func(t *testing.T) {
		// Test distributed computing across GPU cluster (simulated)
		if testing.Short() {
			t.Skip("Skipping GPU cluster test in short mode")
		}

		// Simulate cluster nodes (using local devices as proxy)
		devices, err := EnumerateDevices()
		if err != nil {
			t.Fatalf("Device enumeration failed: %v", err)
		}

		if len(devices) < 1 {
			t.Skip("Need at least 1 GPU for cluster simulation")
		}

		// Create cluster manager
		cluster, err := NewGPUCluster(devices)
		if err != nil {
			t.Fatalf("GPU cluster creation failed: %v", err)
		}
		defer cluster.Shutdown()

		// Test large-scale distributed operation
		totalElements := 8 * 1024 * 1024 // 8M elements
		data := make([]float64, totalElements)
		for i := range data {
			data[i] = float64(i % 1000) // Repeating pattern
		}
		arr, _ := array.FromSlice(data)

		// Submit distributed computation
		job, err := cluster.SubmitDistributedJob("mean", arr)
		if err != nil {
			t.Fatalf("Distributed job submission failed: %v", err)
		}

		// Wait for completion
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		result, err := job.WaitForCompletion(ctx)
		if err != nil {
			t.Fatalf("Distributed job completion failed: %v", err)
		}

		// Verify result accuracy
		expectedMean := float64(999) / 2.0        // Mean of 0-999 pattern
		if math.Abs(result-expectedMean) > 1e-1 { // Relaxed tolerance for distributed computation
			t.Errorf("Distributed result incorrect: expected %.1f, got %.6f", expectedMean, result)
		}

		// Check cluster utilization
		clusterStats := cluster.GetStats()
		totalNodes := len(clusterStats.NodeStats)
		activeNodes := 0

		for nodeID, nodeStats := range clusterStats.NodeStats {
			if nodeStats.TasksProcessed > 0 {
				activeNodes++
			}
			t.Logf("Node %d: %d tasks, %.2fms total time", nodeID, nodeStats.TasksProcessed, nodeStats.TotalTime*1000)
		}

		utilizationRate := float64(activeNodes) / float64(totalNodes)
		if utilizationRate < 0.8 {
			t.Logf("Warning: cluster utilization %.1f%% (expected >80%%)", utilizationRate*100)
		}

		t.Logf("Distributed computing: %.1f%% node utilization, result = %.6f", utilizationRate*100, result)
	})
}

// TestHybridCPUGPUComputing tests CPU-GPU hybrid computing strategies
func TestHybridCPUGPUComputing(t *testing.T) {
	t.Run("Automatic workload distribution", func(t *testing.T) {
		// Test automatic distribution of work between CPU and GPU
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for hybrid test")
		}

		// Create hybrid computing manager
		hybridManager, err := NewHybridComputeManager(device, runtime.NumCPU())
		if err != nil {
			t.Fatalf("Hybrid compute manager creation failed: %v", err)
		}
		defer hybridManager.Close()

		// Test with workloads of different sizes
		workloadSizes := []int{
			1024,        // Small - should prefer CPU
			64 * 1024,   // Medium - may use either
			1024 * 1024, // Large - should prefer GPU
		}

		for _, size := range workloadSizes {
			data := make([]float64, size)
			for i := range data {
				data[i] = float64(i)
			}
			arr, _ := array.FromSlice(data)

			// Submit to hybrid manager for automatic distribution
			result, deviceUsed, err := hybridManager.ComputeMean(arr)
			if err != nil {
				t.Fatalf("Hybrid computation failed for size %d: %v", size, err)
			}

			// Verify result accuracy
			expectedMean := float64(size-1) / 2.0
			if math.Abs(result-expectedMean) > 1e-10 {
				t.Errorf("Hybrid result incorrect for size %d: expected %.1f, got %.6f", size, expectedMean, result)
			}

			t.Logf("Size %d: used %s, result = %.1f", size, deviceUsed, result)
		}

		// Check distribution statistics
		stats := hybridManager.GetStats()
		t.Logf("CPU tasks: %d (%.1f%%), GPU tasks: %d (%.1f%%)",
			stats.CPUTasks, float64(stats.CPUTasks)*100/float64(stats.TotalTasks),
			stats.GPUTasks, float64(stats.GPUTasks)*100/float64(stats.TotalTasks))
	})

	t.Run("Pipeline optimization", func(t *testing.T) {
		// Test optimized pipeline for mixed CPU-GPU workflows
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for pipeline test")
		}

		// Create pipeline optimizer
		pipeline, err := NewHybridPipeline(device, runtime.NumCPU())
		if err != nil {
			t.Fatalf("Hybrid pipeline creation failed: %v", err)
		}
		defer pipeline.Close()

		// Define complex workflow: preprocess (CPU) -> compute (GPU) -> postprocess (CPU)
		dataSize := 512 * 1024
		inputData := make([]float64, dataSize)
		for i := range inputData {
			inputData[i] = float64(i) + 0.5 // Add offset for preprocessing
		}
		arr, _ := array.FromSlice(inputData)

		// Submit pipeline workflow
		workflowResult, err := pipeline.ExecuteWorkflow(arr, &HybridWorkflow{
			PreprocessCPU:  "normalize", // CPU normalization
			ComputeGPU:     "variance",  // GPU variance computation
			PostprocessCPU: "sqrt",      // CPU square root (standard deviation)
		})
		if err != nil {
			t.Fatalf("Pipeline workflow execution failed: %v", err)
		}

		// Verify workflow result (should be standard deviation)
		if workflowResult.FinalResult < 0 {
			t.Errorf("Pipeline result invalid: %.6f (expected positive standard deviation)", workflowResult.FinalResult)
		}

		// Check pipeline efficiency
		totalTime := workflowResult.PreprocessTime + workflowResult.ComputeTime + workflowResult.PostprocessTime
		overhead := workflowResult.TotalTime - totalTime
		efficiency := totalTime / workflowResult.TotalTime

		if efficiency < 0.8 {
			t.Logf("Warning: pipeline efficiency %.1f%% (expected >80%%)", efficiency*100)
		}

		t.Logf("Pipeline: %.1f%% efficiency, %.2fms total (%.2fms overhead)",
			efficiency*100, workflowResult.TotalTime*1000, overhead*1000)
	})

	t.Run("Energy efficiency optimization", func(t *testing.T) {
		// Test energy-efficient computing mode for battery-powered devices
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for energy test")
		}

		// Create energy optimizer
		energyOptimizer, err := NewEnergyOptimizer(device)
		if err != nil {
			t.Fatalf("Energy optimizer creation failed: %v", err)
		}
		defer energyOptimizer.Close()

		// Test different power modes
		powerModes := []string{"performance", "balanced", "power_save"}
		testSize := 256 * 1024

		data := make([]float64, testSize)
		for i := range data {
			data[i] = float64(i)
		}
		arr, _ := array.FromSlice(data)

		for _, mode := range powerModes {
			err := energyOptimizer.SetPowerMode(mode)
			if err != nil {
				t.Logf("Power mode %s not supported: %v", mode, err)
				continue
			}

			// Measure energy consumption (simulated)
			start := time.Now()
			result, err := energyOptimizer.ComputeMean(arr)
			duration := time.Since(start)

			if err != nil {
				t.Errorf("Energy-optimized computation failed in mode %s: %v", mode, err)
				continue
			}

			// Verify result accuracy
			expectedMean := float64(testSize-1) / 2.0
			if math.Abs(result-expectedMean) > 1e-10 {
				t.Errorf("Energy-optimized result incorrect in mode %s: expected %.1f, got %.6f", mode, expectedMean, result)
			}

			// Get energy statistics
			energyStats := energyOptimizer.GetEnergyStats()

			t.Logf("Mode %s: %.2fms, %.1fmJ estimated energy, %.1f%% efficiency",
				mode, float64(duration)/1e6, energyStats.EstimatedEnergyMJ, energyStats.EfficiencyPercent)
		}
	})

	t.Run("Fault tolerance and graceful degradation", func(t *testing.T) {
		// Test graceful degradation when GPU operations fail
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for fault tolerance test")
		}

		// Create fault-tolerant manager
		faultTolerant, err := NewFaultTolerantManager(device)
		if err != nil {
			t.Fatalf("Fault-tolerant manager creation failed: %v", err)
		}
		defer faultTolerant.Close()

		// Test with normal operation
		testData := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		arr, _ := array.FromSlice(testData)

		result, usedGPU, err := faultTolerant.ComputeMeanWithFallback(arr)
		if err != nil {
			t.Fatalf("Fault-tolerant computation failed: %v", err)
		}

		expectedMean := 3.0
		if math.Abs(result-expectedMean) > 1e-10 {
			t.Errorf("Fault-tolerant result incorrect: expected %.1f, got %.6f", expectedMean, result)
		}

		// Test with simulated GPU failure
		faultTolerant.SimulateGPUFailure(true)

		result2, usedGPU2, err := faultTolerant.ComputeMeanWithFallback(arr)
		if err != nil {
			t.Fatalf("Fault-tolerant fallback failed: %v", err)
		}

		if math.Abs(result2-expectedMean) > 1e-10 {
			t.Errorf("Fault-tolerant fallback result incorrect: expected %.1f, got %.6f", expectedMean, result2)
		}

		// GPU should have been used first time, CPU second time
		if !usedGPU {
			t.Logf("Warning: GPU not used in normal operation")
		}
		if usedGPU2 {
			t.Errorf("GPU used despite simulated failure")
		}

		// Check fault tolerance statistics
		stats := faultTolerant.GetStats()

		t.Logf("Fault tolerance: %d GPU attempts, %d fallbacks, %.1f%% success rate",
			stats.GPUAttempts, stats.CPUFallbacks, float64(stats.SuccessfulOperations)*100/float64(stats.TotalOperations))
	})
}

// TestAdvancedBenchmarking tests comprehensive performance benchmarking
func TestAdvancedBenchmarking(t *testing.T) {
	t.Run("Comprehensive GPU vs CPU benchmarks", func(t *testing.T) {
		if testing.Short() {
			t.Skip("Skipping comprehensive benchmarks in short mode")
		}

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for benchmarking")
		}

		// Create benchmark suite
		benchmarkSuite, err := NewBenchmarkSuite(device)
		if err != nil {
			t.Fatalf("Benchmark suite creation failed: %v", err)
		}
		defer benchmarkSuite.Close()

		// Test different operation types and sizes
		operations := []string{"mean", "sum", "std", "correlation", "matmul"}
		sizes := []int{1024, 16 * 1024, 256 * 1024, 4 * 1024 * 1024}

		results := make(map[string]map[int]*BenchmarkResult)

		for _, op := range operations {
			results[op] = make(map[int]*BenchmarkResult)

			for _, size := range sizes {
				// Run benchmark
				result, err := benchmarkSuite.BenchmarkOperation(op, size, 10) // 10 iterations
				if err != nil {
					t.Errorf("Benchmark failed for %s size %d: %v", op, size, err)
					continue
				}

				results[op][size] = result

				t.Logf("%s[%d]: GPU %.2fms (%.2fx vs CPU %.2fms)",
					op, size, result.GPUTime*1000, result.Speedup, result.CPUTime*1000)
			}
		}

		// Analyze benchmark results
		for op, opResults := range results {
			var avgSpeedup float64
			validResults := 0

			for size, result := range opResults {
				if result != nil && result.Speedup > 0 {
					avgSpeedup += result.Speedup
					validResults++

					// Check if GPU provides reasonable speedup for large sizes
					if size > 256*1024 && result.Speedup < 1.5 {
						t.Logf("Warning: %s size %d has low GPU speedup: %.2fx", op, size, result.Speedup)
					}
				}
			}

			if validResults > 0 {
				avgSpeedup /= float64(validResults)
				t.Logf("Operation %s: average %.2fx speedup across %d tests", op, avgSpeedup, validResults)
			}
		}
	})

	t.Run("Memory bandwidth benchmarks", func(t *testing.T) {
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for memory benchmark")
		}

		// Test memory transfer bandwidth
		sizes := []int{1024 * 1024, 16 * 1024 * 1024, 256 * 1024 * 1024} // 1MB, 16MB, 256MB

		for _, size := range sizes {
			data := make([]float64, size)
			for i := range data {
				data[i] = float64(i)
			}

			// Measure host-to-device transfer
			buffer, err := device.AllocateMemory(int64(size * 8))
			if err != nil {
				t.Errorf("Memory allocation failed for size %d: %v", size, err)
				continue
			}

			start := time.Now()
			err = buffer.CopyFromHost(data)
			h2dTime := time.Since(start)

			if err != nil {
				buffer.Free()
				t.Errorf("Host-to-device transfer failed for size %d: %v", size, err)
				continue
			}

			// Measure device-to-host transfer
			readData := make([]float64, size)
			start = time.Now()
			err = buffer.CopyToHost(readData)
			d2hTime := time.Since(start)

			buffer.Free()

			if err != nil {
				t.Errorf("Device-to-host transfer failed for size %d: %v", size, err)
				continue
			}

			// Calculate bandwidth
			sizeBytes := float64(size * 8)
			h2dBandwidth := sizeBytes / h2dTime.Seconds() / (1024 * 1024 * 1024) // GB/s
			d2hBandwidth := sizeBytes / d2hTime.Seconds() / (1024 * 1024 * 1024) // GB/s

			t.Logf("Size %.1fMB: H2D %.1f GB/s, D2H %.1f GB/s",
				sizeBytes/(1024*1024), h2dBandwidth, d2hBandwidth)

			// Verify data integrity
			for i := 0; i < 100; i++ { // Check first 100 elements
				if math.Abs(readData[i]-data[i]) > 1e-10 {
					t.Errorf("Data corruption detected at index %d: expected %.1f, got %.6f", i, data[i], readData[i])
					break
				}
			}
		}
	})

	t.Run("Scalability analysis", func(t *testing.T) {
		// Test how GPU performance scales with problem size
		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for scalability test")
		}

		_ = "sum" // Use sum as representative operation
		baseSizes := []int{1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}

		var gpuTimes, cpuTimes []float64
		var sizes []int

		for _, baseSize := range baseSizes {
			// Create test data
			data := make([]float64, baseSize)
			for i := range data {
				data[i] = float64(i)
			}
			arr, _ := array.FromSlice(data)

			// Measure GPU time
			start := time.Now()
			_, err := SumGPU(arr, device)
			gpuTime := time.Since(start).Seconds()

			if err != nil {
				t.Logf("GPU operation failed for size %d: %v", baseSize, err)
				continue
			}

			// Measure CPU time
			start = time.Now()
			_, err = stats.Sum(arr)
			cpuTime := time.Since(start).Seconds()

			if err != nil {
				t.Logf("CPU operation failed for size %d: %v", baseSize, err)
				continue
			}

			sizes = append(sizes, baseSize)
			gpuTimes = append(gpuTimes, gpuTime)
			cpuTimes = append(cpuTimes, cpuTime)

			speedup := cpuTime / gpuTime
			t.Logf("Size %d: GPU %.3fms, CPU %.3fms, speedup %.2fx",
				baseSize, gpuTime*1000, cpuTime*1000, speedup)
		}

		// Analyze scalability trends
		if len(sizes) >= 3 {
			// Calculate performance improvement trend
			earlySpeedup := cpuTimes[0] / gpuTimes[0]
			lateSpeedup := cpuTimes[len(sizes)-1] / gpuTimes[len(sizes)-1]

			scalabilityFactor := lateSpeedup / earlySpeedup

			t.Logf("Scalability analysis: early speedup %.2fx, late speedup %.2fx, factor %.2fx",
				earlySpeedup, lateSpeedup, scalabilityFactor)

			if scalabilityFactor > 1.5 {
				t.Logf("Good scalability: GPU performance improves with problem size")
			} else if scalabilityFactor < 0.7 {
				t.Logf("Warning: GPU scalability declining with problem size")
			}
		}
	})
}
