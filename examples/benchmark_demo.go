// Package main demonstrates the GPU benchmarking framework
//
// This program shows how to use the comprehensive performance benchmarking
// suite to compare CPU vs GPU performance across different operations.
//
// Usage:
//
//	go run examples/benchmark_demo.go
package main

import (
	"fmt"

	"github.com/julianshen/gonp/benchmarks"
)

func main() {
	fmt.Println("=== GoNP GPU Performance Benchmarking Demo ===\n")

	// Create benchmark suite
	suite := benchmarks.NewGPUBenchmarkSuite()

	// Quick individual benchmark examples
	fmt.Println("Running individual benchmark examples...")

	// Addition benchmark
	fmt.Println("\n1. Array Addition Benchmark:")
	addResult := suite.BenchmarkAddition(50000)
	if addResult.Success {
		fmt.Printf("   Data Size: %d elements\n", addResult.DataSize)
		fmt.Printf("   CPU Time: %v\n", addResult.CPUTime)
		if addResult.GPUTime > 0 {
			fmt.Printf("   GPU Time: %v\n", addResult.GPUTime)
			fmt.Printf("   Speedup: %.2fx\n", addResult.Speedup)
			fmt.Printf("   Accuracy: %.6f\n", addResult.Accuracy)
		}
		fmt.Printf("   Memory Used: %.1f KB\n", float64(addResult.MemoryUsed)/1024)
		fmt.Printf("   Throughput: %.2f MB/s\n", addResult.ThroughputMB)
	} else {
		fmt.Printf("   Failed: %s\n", addResult.Error)
	}

	// Sum benchmark
	fmt.Println("\n2. Array Sum Benchmark:")
	sumResult := suite.BenchmarkSum(100000)
	if sumResult.Success {
		fmt.Printf("   Data Size: %d elements\n", sumResult.DataSize)
		fmt.Printf("   CPU Time: %v\n", sumResult.CPUTime)
		if sumResult.GPUTime > 0 {
			fmt.Printf("   GPU Time: %v\n", sumResult.GPUTime)
			fmt.Printf("   Speedup: %.2fx\n", sumResult.Speedup)
			fmt.Printf("   Accuracy: %.6f\n", sumResult.Accuracy)
		}
		fmt.Printf("   Memory Used: %.1f KB\n", float64(sumResult.MemoryUsed)/1024)
		fmt.Printf("   Throughput: %.2f MB/s\n", sumResult.ThroughputMB)
	} else {
		fmt.Printf("   Failed: %s\n", sumResult.Error)
	}

	// Matrix multiplication benchmark
	fmt.Println("\n3. Matrix Multiplication Benchmark (100×100):")
	matResult := suite.BenchmarkMatrixMultiplication(100)
	if matResult.Success {
		fmt.Printf("   Matrix Size: 100×100 (%d total elements)\n", matResult.DataSize)
		fmt.Printf("   CPU Time: %v\n", matResult.CPUTime)
		if matResult.GPUTime > 0 {
			fmt.Printf("   GPU Time: %v\n", matResult.GPUTime)
			fmt.Printf("   Speedup: %.2fx\n", matResult.Speedup)
			fmt.Printf("   Accuracy: %.6f\n", matResult.Accuracy)
		}
		fmt.Printf("   Memory Used: %.1f KB\n", float64(matResult.MemoryUsed)/1024)
		fmt.Printf("   Throughput: %.2f MB/s\n", matResult.ThroughputMB)
	} else {
		fmt.Printf("   Failed: %s\n", matResult.Error)
	}

	// Scalability analysis example
	fmt.Println("\n4. Scalability Analysis (Sum Operation):")
	sizes := []int{1000, 10000, 50000, 100000}
	scalabilityResult := suite.AnalyzeScalability("sum", sizes)
	fmt.Printf("   Operation: %s\n", scalabilityResult.Operation)
	fmt.Printf("   Data Sizes: %v\n", scalabilityResult.DataSizes)
	fmt.Printf("   CPU Times: %v\n", scalabilityResult.CPUTimes)
	if len(scalabilityResult.GPUTimes) > 0 && scalabilityResult.GPUTimes[0] > 0 {
		fmt.Printf("   GPU Times: %v\n", scalabilityResult.GPUTimes)
		fmt.Printf("   Speedups: %.2f\n", scalabilityResult.Speedups)
	}
	fmt.Printf("   Optimal Size: %d elements\n", scalabilityResult.OptimalSize)
	fmt.Printf("   Max Speedup: %.2fx\n", scalabilityResult.MaxSpeedup)
	fmt.Printf("   Scaling Factor: %.2f\n", scalabilityResult.ScalingFactor)

	// Generate comprehensive report
	fmt.Println("\n=== Comprehensive Performance Report ===")
	fmt.Println("\nRunning complete benchmark suite (this may take a moment)...")

	results := suite.RunAllBenchmarks()
	report := suite.GenerateReport(results)

	fmt.Println(report)

	// Summary statistics
	successfulResults := 0
	totalSpeedup := 0.0
	speedupCount := 0

	for _, result := range results {
		if result.Success {
			successfulResults++
			if result.Speedup > 0 {
				totalSpeedup += result.Speedup
				speedupCount++
			}
		}
	}

	fmt.Println("\n=== Summary Statistics ===")
	fmt.Printf("Total Benchmarks: %d\n", len(results))
	fmt.Printf("Successful: %d (%.1f%%)\n", successfulResults, float64(successfulResults)/float64(len(results))*100)

	if speedupCount > 0 {
		avgSpeedup := totalSpeedup / float64(speedupCount)
		fmt.Printf("Average GPU Speedup: %.2fx\n", avgSpeedup)

		if avgSpeedup > 1.0 {
			fmt.Printf("\n✅ GPU acceleration provides %.1f%% performance improvement on average\n", (avgSpeedup-1.0)*100)
		} else {
			fmt.Printf("\n⚠️  CPU performance is better for these workloads (%.1f%% faster)\n", (1.0-avgSpeedup)*100)
		}
	} else {
		fmt.Println("No GPU performance data available (CPU-only system)")
	}

	fmt.Println("\n=== Recommendations ===")
	if speedupCount > 0 {
		fmt.Println("• Use GPU acceleration for operations showing >1.5x speedup")
		fmt.Println("• Small arrays (< 65K elements) generally perform better on CPU")
		fmt.Println("• Large arrays (≥ 65K elements) benefit from GPU parallel processing")
		fmt.Println("• Consider using AddAuto() for automatic CPU/GPU selection")
	} else {
		fmt.Println("• GPU not available - using CPU fallback for all operations")
		fmt.Println("• Consider installing GPU drivers/CUDA for potential acceleration")
		fmt.Println("• CPU-only performance is still highly optimized")
	}

	fmt.Println("\n✨ Benchmark demo completed successfully!")
}
