//go:build ignore
// +build ignore

// Package examples demonstrates GoNP SIMD acceleration capabilities
//
// This example shows how SIMD acceleration provides significant performance
// improvements for mathematical operations on large arrays.
//
// Run with debug to see SIMD usage:
//
//	GONP_DEBUG=1 GONP_DEBUG_LEVEL=VERBOSE go run examples/simd_example.go
package main

import (
	"fmt"
	"math/rand"
	"time"
	"unsafe"

	"github.com/julianshen/gonp/internal"
)

func main() {
	fmt.Println("=== GoNP SIMD Acceleration Example ===")

	// Enable debug mode to see SIMD usage
	internal.EnableDebugMode()
	internal.SetLogLevel(internal.DebugLevelInfo)

	// Show CPU capabilities
	fmt.Println("\n1. CPU SIMD Capabilities:")
	internal.PrintSIMDInfo()

	// Show SIMD statistics
	fmt.Println("\n2. SIMD Configuration:")
	stats := internal.GetSIMDStatistics()
	for key, value := range stats {
		fmt.Printf("   %s: %v\n", key, value)
	}

	// Example 3: Direct SIMD operations
	fmt.Println("\n3. Direct SIMD Operations:")
	demonstrateSIMDOperations()

	// Example 4: Performance comparison
	fmt.Println("\n4. Performance Comparison:")
	performanceBenchmark()

	// Example 5: Memory alignment importance
	fmt.Println("\n5. Memory Alignment:")
	demonstrateAlignment()

	fmt.Println("\nSIMD acceleration provides significant performance improvements!")
	fmt.Println("Key benefits:")
	fmt.Println("  - 2-4x faster mathematical operations")
	fmt.Println("  - Automatic CPU feature detection")
	fmt.Println("  - Transparent fallback to scalar operations")
	fmt.Println("  - Zero API changes required")
}

func demonstrateSIMDOperations() {
	size := 1024

	// Create test data
	a := make([]float64, size)
	b := make([]float64, size)
	result := make([]float64, size)

	// Initialize with random data
	for i := 0; i < size; i++ {
		a[i] = rand.Float64() * 100
		b[i] = rand.Float64() * 100
	}

	fmt.Println("   Testing SIMD addition...")
	start := time.Now()
	internal.SIMDAddFloat64(a, b, result)
	duration := time.Since(start)

	fmt.Printf("   Added %d float64 values in %v\n", size, duration)
	fmt.Printf("   First few results: [%.2f, %.2f, %.2f, %.2f]\n",
		result[0], result[1], result[2], result[3])

	// Test scalar addition
	fmt.Println("   Testing SIMD scalar addition...")
	scalar := 42.0
	start = time.Now()
	internal.SIMDAddScalarFloat64(a, scalar, result)
	duration = time.Since(start)

	fmt.Printf("   Added scalar %.1f to %d values in %v\n", scalar, size, duration)
	fmt.Printf("   First few results: [%.2f, %.2f, %.2f, %.2f]\n",
		result[0], result[1], result[2], result[3])

	// Test sum
	fmt.Println("   Testing SIMD sum...")
	start = time.Now()
	sum := internal.SIMDSumFloat64(a)
	duration = time.Since(start)

	fmt.Printf("   Summed %d values = %.2f in %v\n", size, sum, duration)
}

func performanceBenchmark() {
	sizes := []int{64, 256, 1024, 4096}
	iterations := 1000

	fmt.Println("   Size    | Scalar Time | SIMD Time | Speedup")
	fmt.Println("   --------|-------------|-----------|--------")

	for _, size := range sizes {
		// Prepare data
		a := make([]float64, size)
		b := make([]float64, size)
		result := make([]float64, size)

		for i := 0; i < size; i++ {
			a[i] = float64(i + 1)
			b[i] = float64(i + 2)
		}

		// Test scalar performance
		scalarProvider := internal.NewScalarProvider()
		start := time.Now()
		for i := 0; i < iterations; i++ {
			scalarProvider.AddFloat64(a, b, result, size)
		}
		scalarTime := time.Since(start)

		// Test SIMD performance
		start = time.Now()
		for i := 0; i < iterations; i++ {
			internal.SIMDAddFloat64(a, b, result)
		}
		simdTime := time.Since(start)

		// Calculate speedup
		speedup := float64(scalarTime) / float64(simdTime)

		fmt.Printf("   %-7d | %-11v | %-9v | %.2fx\n",
			size, scalarTime, simdTime, speedup)
	}
}

func demonstrateAlignment() {
	provider := internal.GetSIMDProvider()

	fmt.Printf("   Required alignment: %d bytes\n", provider.AlignmentRequirement())
	fmt.Printf("   Vector width: %d bytes\n", provider.VectorWidth())

	// Test with different data
	data1 := make([]float64, 100)
	data2 := make([]float64, 100)

	// Check alignment
	aligned1 := provider.IsAligned(unsafe.Pointer(&data1[0]))
	aligned2 := provider.IsAligned(unsafe.Pointer(&data2[0]))

	fmt.Printf("   Array 1 aligned: %v\n", aligned1)
	fmt.Printf("   Array 2 aligned: %v\n", aligned2)

	// Show SIMD threshold
	fmt.Printf("   SIMD threshold: %d elements\n", internal.SIMDThreshold)

	// Test with small vs large arrays
	smallArray := make([]float64, 16)
	largeArray := make([]float64, 1000)

	shouldUseSIMDSmall := internal.ShouldUseSIMD(len(smallArray), unsafe.Pointer(&smallArray[0]))
	shouldUseSIMDLarge := internal.ShouldUseSIMD(len(largeArray), unsafe.Pointer(&largeArray[0]))

	fmt.Printf("   Small array (%d elements) should use SIMD: %v\n",
		len(smallArray), shouldUseSIMDSmall)
	fmt.Printf("   Large array (%d elements) should use SIMD: %v\n",
		len(largeArray), shouldUseSIMDLarge)
}
