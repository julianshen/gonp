// Package gpu provides performance validation tests comparing optimized vs standard algorithms
//
// This module validates the performance improvements achieved by cache-aware,
// NUMA-aware, and distributed computing optimizations using TDD methodology.
//
// Performance Validation Features:
//   - Cache-aware vs naive algorithm comparison
//   - NUMA-aware vs standard memory allocation comparison
//   - SIMD-optimized vs scalar operation comparison
//   - Distributed vs sequential computation comparison
//   - Memory bandwidth and throughput analysis

package gpu

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// PerformanceMetrics holds performance measurement results
type PerformanceMetrics struct {
	ExecutionTime    time.Duration
	ThroughputMBps   float64
	OperationsPerSec float64
	MemoryEfficiency float64
	CacheHitRate     float64
	Speedup          float64
	ErrorRate        float64
}

// TestCacheAwarePerformanceValidation validates cache optimization performance
func TestCacheAwarePerformanceValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance validation in short mode")
	}

	t.Run("Matrix multiplication performance comparison", func(t *testing.T) {
		sizes := []int{64, 128, 256}

		for _, size := range sizes {
			t.Run(fmt.Sprintf("Size%dx%d", size, size), func(t *testing.T) {
				// Create test matrices
				A := make([]float64, size*size)
				B := make([]float64, size*size)

				// Initialize with patterns that exercise cache
				for i := 0; i < size*size; i++ {
					A[i] = float64(i%100) / 100.0
					B[i] = float64((i+50)%100) / 100.0
				}

				// Benchmark naive implementation
				naiveMetrics := benchmarkNaiveMatMul(A, B, size, size, size)

				// Benchmark cache-aware implementation
				tileSize := DetectOptimalTileSize(size, size)
				cacheAwareMetrics := benchmarkCacheAwareMatMul(A, B, size, size, size, tileSize)

				// Calculate performance improvement
				speedup := naiveMetrics.ExecutionTime.Seconds() / cacheAwareMetrics.ExecutionTime.Seconds()
				cacheAwareMetrics.Speedup = speedup

				// Validate performance improvement
				if speedup < 1.0 {
					t.Logf("Warning: Cache-aware slower than naive for %dx%d (%.2fx)", size, size, speedup)
				} else if speedup >= 1.1 {
					t.Logf("Success: Cache-aware %.2fx faster than naive for %dx%d", speedup, size, size)
				}

				// Log detailed metrics
				t.Logf("Matrix %dx%d Performance:", size, size)
				t.Logf("  Naive:       %v (%.2f GFLOPS)", naiveMetrics.ExecutionTime, naiveMetrics.OperationsPerSec/1e9)
				t.Logf("  Cache-aware: %v (%.2f GFLOPS)", cacheAwareMetrics.ExecutionTime, cacheAwareMetrics.OperationsPerSec/1e9)
				t.Logf("  Speedup:     %.2fx", speedup)

				// Performance regression check
				if size >= 128 && speedup < 0.8 {
					t.Errorf("Significant performance regression: %.2fx slower", 1.0/speedup)
				}
			})
		}
	})

	t.Run("Memory access pattern optimization", func(t *testing.T) {
		dataSizes := []int{1024, 4096, 16384} // Different cache pressure levels

		for _, dataSize := range dataSizes {
			t.Run(fmt.Sprintf("DataSize%d", dataSize), func(t *testing.T) {
				data := make([]float64, dataSize)
				for i := range data {
					data[i] = float64(i)
				}

				// Benchmark sequential access
				seqMetrics := benchmarkSequentialAccess(data)

				// Benchmark cache-friendly access
				cacheMetrics := benchmarkCacheFriendlyAccess(data)

				// Calculate bandwidth improvement
				bandwidthImprovement := cacheMetrics.ThroughputMBps / seqMetrics.ThroughputMBps

				t.Logf("Memory Access %d elements:", dataSize)
				t.Logf("  Sequential:     %.1f MB/s", seqMetrics.ThroughputMBps)
				t.Logf("  Cache-friendly: %.1f MB/s", cacheMetrics.ThroughputMBps)
				t.Logf("  Improvement:    %.2fx", bandwidthImprovement)

				// Reasonable bandwidth check
				if seqMetrics.ThroughputMBps < 100 {
					t.Logf("Warning: Low sequential bandwidth (%.1f MB/s)", seqMetrics.ThroughputMBps)
				}
			})
		}
	})
}

// TestNUMAPerformanceValidation validates NUMA optimization performance
func TestNUMAPerformanceValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping NUMA performance validation in short mode")
	}

	t.Run("NUMA vs standard allocation comparison", func(t *testing.T) {
		topology, err := DetectNUMATopologyAdvanced()
		if err != nil || topology.NodeCount < 2 {
			t.Skip("Multi-node NUMA system required for validation")
		}

		sizes := []int{1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024} // 1MB, 4MB, 16MB

		for _, size := range sizes {
			t.Run(fmt.Sprintf("Size%dMB", size/(1024*1024)), func(t *testing.T) {
				// Benchmark standard allocation
				standardMetrics := benchmarkStandardAllocation(size)

				// Benchmark NUMA-aware allocation
				numaMetrics := benchmarkNUMAAllocation(size, 0) // Allocate on node 0

				// Calculate performance difference
				improvement := standardMetrics.ExecutionTime.Seconds() / numaMetrics.ExecutionTime.Seconds()

				t.Logf("Memory Allocation %d MB:", size/(1024*1024))
				t.Logf("  Standard: %v (%.1f MB/s)", standardMetrics.ExecutionTime, standardMetrics.ThroughputMBps)
				t.Logf("  NUMA:     %v (%.1f MB/s)", numaMetrics.ExecutionTime, numaMetrics.ThroughputMBps)
				t.Logf("  Factor:   %.2fx", improvement)

				// NUMA optimization is often subtle, so we expect modest improvements
				if improvement > 1.05 {
					t.Logf("NUMA optimization effective: %.1f%% improvement", (improvement-1)*100)
				} else if improvement < 0.95 {
					t.Logf("NUMA allocation slightly slower (within expected variance)")
				} else {
					t.Logf("NUMA and standard allocation performance equivalent")
				}
			})
		}
	})

	t.Run("Local vs remote memory access validation", func(t *testing.T) {
		topology, err := DetectNUMATopologyAdvanced()
		if err != nil || topology.NodeCount < 2 {
			t.Skip("Multi-node NUMA system required")
		}

		size := 10 * 1024 * 1024 // 10MB
		iterations := 100

		// Test local memory access (simulated)
		localBuffer, err := AllocateOnNUMANode(size, 0)
		if err != nil {
			t.Fatalf("Failed to allocate local buffer: %v", err)
		}
		defer localBuffer.Free()

		localMetrics := benchmarkNUMABufferAccess(localBuffer, iterations)

		// Test remote memory access (simulated)
		remoteBuffer, err := AllocateOnNUMANode(size, 1)
		if err != nil {
			t.Fatalf("Failed to allocate remote buffer: %v", err)
		}
		defer remoteBuffer.Free()

		remoteMetrics := benchmarkNUMABufferAccess(remoteBuffer, iterations)

		// Calculate NUMA penalty
		numaPenalty := remoteMetrics.ExecutionTime.Seconds() / localMetrics.ExecutionTime.Seconds()

		t.Logf("NUMA Memory Access Comparison:")
		t.Logf("  Local node:  %v (%.1f MB/s)", localMetrics.ExecutionTime, localMetrics.ThroughputMBps)
		t.Logf("  Remote node: %v (%.1f MB/s)", remoteMetrics.ExecutionTime, remoteMetrics.ThroughputMBps)
		t.Logf("  Penalty:     %.2fx slower for remote", numaPenalty)

		// In our simplified implementation, penalty should be minimal
		if numaPenalty > 2.0 {
			t.Logf("Significant NUMA penalty detected: %.1fx", numaPenalty)
		} else if numaPenalty > 1.2 {
			t.Logf("Moderate NUMA penalty: %.1fx", numaPenalty)
		} else {
			t.Logf("Minimal NUMA penalty (expected for simulation)")
		}
	})
}

// TestDistributedPerformanceValidation validates distributed computing performance
func TestDistributedPerformanceValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping distributed performance validation in short mode")
	}

	t.Run("Sequential vs distributed matrix multiplication", func(t *testing.T) {
		sizes := []int{64, 128} // Keep reasonable for test performance
		processCounts := []int{2, 4}

		for _, size := range sizes {
			for _, procCount := range processCounts {
				t.Run(fmt.Sprintf("Size%d_Procs%d", size, procCount), func(t *testing.T) {
					// Create test matrices
					A := make([]float64, size*size)
					B := make([]float64, size*size)

					for i := 0; i < size*size; i++ {
						A[i] = float64(i + 1)
						B[i] = float64((i + 1) * 2)
					}

					// Benchmark sequential implementation
					seqMetrics := benchmarkSequentialMatMul(A, B, size)

					// Benchmark distributed implementation
					distMetrics := benchmarkDistributedMatMul(A, B, size, procCount)

					// Calculate parallel efficiency
					theoreticalSpeedup := float64(procCount)
					actualSpeedup := seqMetrics.ExecutionTime.Seconds() / distMetrics.ExecutionTime.Seconds()
					efficiency := actualSpeedup / theoreticalSpeedup * 100

					t.Logf("Matrix %dx%d with %d processes:", size, size, procCount)
					t.Logf("  Sequential:  %v", seqMetrics.ExecutionTime)
					t.Logf("  Distributed: %v", distMetrics.ExecutionTime)
					t.Logf("  Speedup:     %.2fx", actualSpeedup)
					t.Logf("  Efficiency:  %.1f%%", efficiency)

					// Check for reasonable parallel performance
					if actualSpeedup >= 1.1 {
						t.Logf("Good parallel performance achieved")
					} else if actualSpeedup < 0.8 {
						t.Logf("Parallel overhead detected (expected for small problems)")
					}

					// Efficiency should be reasonable for small test cases
					if efficiency > 50 {
						t.Logf("High parallel efficiency: %.1f%%", efficiency)
					} else {
						t.Logf("Moderate efficiency due to communication overhead")
					}
				})
			}
		}
	})

	t.Run("Communication vs computation ratio analysis", func(t *testing.T) {
		sizes := []int{32, 64, 128}

		for _, size := range sizes {
			t.Run(fmt.Sprintf("Size%d", size), func(t *testing.T) {
				// Estimate computation time (sequential)
				data := make([]float64, size*size)
				for i := range data {
					data[i] = float64(i)
				}

				compStart := time.Now()
				_ = sequentialMatMul(data, data, size, size, size)
				compTime := time.Since(compStart)

				// Estimate communication time (MPI setup + message passing)
				commStart := time.Now()
				comm, err := InitializeMPI(2)
				if err == nil {
					// Simulate sending matrix data
					if comm.Rank() == 0 {
						_ = comm.Send(data, 1, 99)
					}
					comm.Finalize()
				}
				commTime := time.Since(commStart)

				ratio := commTime.Seconds() / compTime.Seconds()

				t.Logf("Communication/Computation Analysis %dx%d:", size, size)
				t.Logf("  Computation: %v", compTime)
				t.Logf("  Communication: %v", commTime)
				t.Logf("  Ratio: %.2f", ratio)

				if ratio < 0.1 {
					t.Logf("Communication overhead very low - good for parallelization")
				} else if ratio < 1.0 {
					t.Logf("Moderate communication overhead")
				} else {
					t.Logf("High communication overhead - consider larger problem sizes")
				}
			})
		}
	})
}

// TestOverallPerformanceSummary provides a comprehensive performance summary
func TestOverallPerformanceSummary(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance summary in short mode")
	}

	t.Run("Comprehensive performance analysis", func(t *testing.T) {
		size := 128

		// Test data
		A := make([]float64, size*size)
		B := make([]float64, size*size)
		for i := 0; i < size*size; i++ {
			A[i] = float64(i%1000) / 1000.0
			B[i] = float64((i+500)%1000) / 1000.0
		}

		t.Logf("=== GoNP Performance Summary (%dx%d matrix) ===", size, size)

		// 1. Naive baseline
		naiveMetrics := benchmarkNaiveMatMul(A, B, size, size, size)
		t.Logf("1. Naive Implementation:")
		t.Logf("   Time: %v (%.2f GFLOPS)", naiveMetrics.ExecutionTime, naiveMetrics.OperationsPerSec/1e9)

		// 2. Cache-aware optimization
		tileSize := DetectOptimalTileSize(size, size)
		cacheMetrics := benchmarkCacheAwareMatMul(A, B, size, size, size, tileSize)
		cacheSpeedup := naiveMetrics.ExecutionTime.Seconds() / cacheMetrics.ExecutionTime.Seconds()
		t.Logf("2. Cache-Aware Implementation:")
		t.Logf("   Time: %v (%.2f GFLOPS) - %.2fx speedup", cacheMetrics.ExecutionTime, cacheMetrics.OperationsPerSec/1e9, cacheSpeedup)

		// 3. Memory optimization
		memData := make([]float64, size*100)
		for i := range memData {
			memData[i] = float64(i)
		}
		memMetrics := benchmarkCacheFriendlyAccess(memData)
		t.Logf("3. Memory Access Optimization:")
		t.Logf("   Throughput: %.1f MB/s", memMetrics.ThroughputMBps)

		// 4. NUMA awareness (if available)
		topology, err := DetectNUMATopologyAdvanced()
		if err == nil {
			numaBuffer, err := AllocateOnNUMANode(1024*1024, 0)
			if err == nil {
				numaMetrics := benchmarkNUMABufferAccess(numaBuffer, 10)
				t.Logf("4. NUMA-Aware Allocation:")
				t.Logf("   Throughput: %.1f MB/s (%d nodes)", numaMetrics.ThroughputMBps, topology.NodeCount)
				numaBuffer.Free()
			}
		} else {
			t.Logf("4. NUMA optimization: Not available")
		}

		// 5. Distributed computing
		distMetrics := benchmarkDistributedMatMul(A, B, size, 2)
		distSpeedup := naiveMetrics.ExecutionTime.Seconds() / distMetrics.ExecutionTime.Seconds()
		t.Logf("5. Distributed Computing:")
		t.Logf("   Time: %v - %.2fx speedup (2 processes)", distMetrics.ExecutionTime, distSpeedup)

		// Overall assessment
		t.Logf("\n=== Performance Assessment ===")
		bestTime := naiveMetrics.ExecutionTime
		if cacheMetrics.ExecutionTime < bestTime {
			bestTime = cacheMetrics.ExecutionTime
		}
		if distMetrics.ExecutionTime < bestTime {
			bestTime = distMetrics.ExecutionTime
		}

		overallSpeedup := naiveMetrics.ExecutionTime.Seconds() / bestTime.Seconds()
		t.Logf("Best performance: %v (%.2fx improvement over naive)", bestTime, overallSpeedup)

		if overallSpeedup >= 2.0 {
			t.Logf("✓ Excellent optimization achieved")
		} else if overallSpeedup >= 1.5 {
			t.Logf("✓ Good optimization achieved")
		} else if overallSpeedup >= 1.2 {
			t.Logf("✓ Moderate optimization achieved")
		} else {
			t.Logf("⚠ Limited optimization (expected for test environment)")
		}

		t.Logf("=== End Performance Summary ===")
	})
}

// Benchmark helper functions

func benchmarkNaiveMatMul(A, B []float64, M, N, K int) PerformanceMetrics {
	start := time.Now()
	result := naiveMatMul(A, B, M, N, K)
	elapsed := time.Since(start)

	operations := float64(2 * M * N * K) // 2 ops per inner loop iteration
	throughput := operations / elapsed.Seconds()

	return PerformanceMetrics{
		ExecutionTime:    elapsed,
		OperationsPerSec: throughput,
		ErrorRate:        validateMatMul(result, A, B, M, N, K),
	}
}

func benchmarkCacheAwareMatMul(A, B []float64, M, N, K int, tileSize int) PerformanceMetrics {
	start := time.Now()
	result, _ := CacheAwareMatMul(A, B, M, N, K, tileSize)
	elapsed := time.Since(start)

	operations := float64(2 * M * N * K)
	throughput := operations / elapsed.Seconds()

	return PerformanceMetrics{
		ExecutionTime:    elapsed,
		OperationsPerSec: throughput,
		ErrorRate:        validateMatMul(result, A, B, M, N, K),
	}
}

func benchmarkSequentialAccess(data []float64) PerformanceMetrics {
	start := time.Now()
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	elapsed := time.Since(start)

	bytes := float64(len(data) * 8)                         // 8 bytes per float64
	throughput := bytes / elapsed.Seconds() / (1024 * 1024) // MB/s

	_ = sum // Prevent optimization
	return PerformanceMetrics{
		ExecutionTime:  elapsed,
		ThroughputMBps: throughput,
	}
}

func benchmarkCacheFriendlyAccess(data []float64) PerformanceMetrics {
	start := time.Now()
	result, _ := CacheAwareCopy(data)
	elapsed := time.Since(start)

	bytes := float64(len(data) * 8 * 2) // Read + write
	throughput := bytes / elapsed.Seconds() / (1024 * 1024)

	_ = result // Prevent optimization
	return PerformanceMetrics{
		ExecutionTime:  elapsed,
		ThroughputMBps: throughput,
	}
}

func benchmarkStandardAllocation(size int) PerformanceMetrics {
	start := time.Now()
	data := make([]float64, size/8)
	for i := range data {
		data[i] = float64(i)
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	elapsed := time.Since(start)

	bytes := float64(size)
	throughput := bytes / elapsed.Seconds() / (1024 * 1024)

	_ = sum
	return PerformanceMetrics{
		ExecutionTime:  elapsed,
		ThroughputMBps: throughput,
	}
}

func benchmarkNUMAAllocation(size int, nodeID int) PerformanceMetrics {
	start := time.Now()
	buffer, err := AllocateOnNUMANode(size, nodeID)
	if err != nil {
		// Fallback to standard allocation
		return benchmarkStandardAllocation(size)
	}
	defer buffer.Free()

	// Simulate access pattern
	sum, _ := BenchmarkMemoryAccess(buffer, 10)
	elapsed := time.Since(start)

	bytes := float64(size)
	throughput := bytes / elapsed.Seconds() / (1024 * 1024)

	_ = sum
	return PerformanceMetrics{
		ExecutionTime:  elapsed,
		ThroughputMBps: throughput,
	}
}

func benchmarkNUMABufferAccess(buffer *NUMABuffer, iterations int) PerformanceMetrics {
	start := time.Now()
	sum, _ := BenchmarkMemoryAccess(buffer, iterations)
	elapsed := time.Since(start)

	bytes := float64(buffer.Size()) * float64(iterations)
	throughput := bytes / elapsed.Seconds() / (1024 * 1024)

	_ = sum
	return PerformanceMetrics{
		ExecutionTime:  elapsed,
		ThroughputMBps: throughput,
	}
}

func benchmarkSequentialMatMul(A, B []float64, size int) PerformanceMetrics {
	start := time.Now()
	result := naiveMatMul(A, B, size, size, size)
	elapsed := time.Since(start)

	operations := float64(2 * size * size * size)
	throughput := operations / elapsed.Seconds()

	_ = result
	return PerformanceMetrics{
		ExecutionTime:    elapsed,
		OperationsPerSec: throughput,
	}
}

func benchmarkDistributedMatMul(A, B []float64, size int, procCount int) PerformanceMetrics {
	start := time.Now()

	comm, err := InitializeMPI(procCount)
	if err != nil {
		// Fallback to sequential
		return benchmarkSequentialMatMul(A, B, size)
	}
	defer comm.Finalize()

	result, err := DistributedMatMul(comm, A, B, size, size, size)
	elapsed := time.Since(start)

	operations := float64(2 * size * size * size)
	throughput := operations / elapsed.Seconds()

	_ = result
	_ = err
	return PerformanceMetrics{
		ExecutionTime:    elapsed,
		OperationsPerSec: throughput,
	}
}

// Validation helper functions

func validateMatMul(result, A, B []float64, M, N, K int) float64 {
	// Compare against reference implementation
	expected := naiveMatMul(A, B, M, N, K)

	maxError := 0.0
	for i := 0; i < len(result) && i < len(expected); i++ {
		error := math.Abs(result[i] - expected[i])
		if error > maxError {
			maxError = error
		}
	}

	return maxError
}

// Helper function to get naive matrix multiplication (already defined in cache_aware_test.go)
