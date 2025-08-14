//go:build ignore
// +build ignore

// Package examples demonstrates GoNP debug and profiling capabilities
//
// This example shows how to use debug modes and profiling hooks to
// monitor performance and troubleshoot issues.
//
// Run with debug enabled:
//
//	GONP_DEBUG=1 GONP_DEBUG_LEVEL=INFO GONP_PROFILE_MEMORY=1 GONP_PROFILE_TIMING=1 go run examples/debug_example.go
//
// Or programmatically enable debug mode and run:
//
//	go run examples/debug_example.go
package main

import (
	"fmt"
	"log"
	"time"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/stats"
)

func main() {
	fmt.Println("=== GoNP Debug and Profiling Example ===")

	// Example 1: Environment variable debug mode
	fmt.Println("\n1. Current Debug Configuration:")
	config := internal.GetDebugConfig()
	fmt.Printf("   Debug Enabled: %v\n", config.Enabled)
	fmt.Printf("   Log Level: %s\n", config.LogLevel.String())
	fmt.Printf("   Memory Profiling: %v\n", config.ProfileMemory)
	fmt.Printf("   Timing Profiling: %v\n", config.ProfileTiming)

	// Example 2: Programmatically enable debug mode
	fmt.Println("\n2. Enabling Debug Mode Programmatically:")
	internal.EnableDebugMode()
	internal.EnableMemoryProfiling()
	internal.EnableTimingProfiling()
	internal.SetLogLevel(internal.DebugLevelInfo)

	// Reset statistics for clean demo
	internal.ResetStatistics()

	// Example 3: Array operations with debug tracking
	fmt.Println("\n3. Performing Array Operations (check debug output above):")

	// Create arrays - this will be profiled
	data1 := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	data2 := []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}

	arr1, err := array.FromSlice(data1)
	if err != nil {
		log.Fatal("Failed to create array1:", err)
	}

	arr2, err := array.FromSlice(data2)
	if err != nil {
		log.Fatal("Failed to create array2:", err)
	}

	// Perform operations - these will be profiled
	sum, err := arr1.Add(arr2)
	if err != nil {
		log.Fatal("Failed to add arrays:", err)
	}

	product, err := arr1.Mul(arr2)
	if err != nil {
		log.Fatal("Failed to multiply arrays:", err)
	}

	// Example 4: Custom profiling
	fmt.Println("\n4. Custom Operation Profiling:")
	ctx := internal.StartProfiler("statistical_analysis")

	// Simulate some work
	time.Sleep(10 * time.Millisecond)

	// Calculate statistics
	mean, _ := stats.Mean(sum)
	std, _ := stats.Std(product)

	ctx.EndProfiler()

	fmt.Printf("   Mean of sum: %.2f\n", mean)
	fmt.Printf("   Std of product: %.2f\n", std)

	// Example 5: Debug statistics
	fmt.Println("\n5. Debug Statistics:")
	internal.PrintStatistics()

	// Example 6: Different debug levels
	fmt.Println("\n6. Testing Different Debug Levels:")

	internal.SetLogLevel(internal.DebugLevelVerbose)
	internal.DebugVerbose("This is a verbose message")

	internal.SetLogLevel(internal.DebugLevelWarn)
	internal.DebugInfo("This info message will not appear")
	internal.DebugWarn("This warning message will appear")

	internal.SetLogLevel(internal.DebugLevelError)
	internal.DebugWarn("This warning will not appear")
	internal.DebugError("This error message will appear")

	// Example 7: Manual statistics access
	fmt.Println("\n7. Accessing Statistics Programmatically:")
	stats := internal.GetStatistics()
	for key, value := range stats {
		fmt.Printf("   %s: %d\n", key, value)
	}

	// Example 8: Performance comparison
	fmt.Println("\n8. Performance Comparison (Debug vs No Debug):")
	runPerformanceComparison()

	fmt.Println("\nDebug example completed!")
	fmt.Println("\nTo enable debug mode via environment variables:")
	fmt.Println("  export GONP_DEBUG=1")
	fmt.Println("  export GONP_DEBUG_LEVEL=INFO")
	fmt.Println("  export GONP_PROFILE_MEMORY=1")
	fmt.Println("  export GONP_PROFILE_TIMING=1")
	fmt.Println("  export GONP_LOG_FILE=gonp_debug.log  # Optional: log to file")
}

func runPerformanceComparison() {
	data := make([]float64, 1000)
	for i := range data {
		data[i] = float64(i)
	}

	// Test with debug enabled
	internal.EnableDebugMode()
	internal.EnableMemoryProfiling()
	internal.EnableTimingProfiling()

	start := time.Now()
	for i := 0; i < 100; i++ {
		arr, _ := array.FromSlice(data)
		_, _ = arr.AddScalar(1.0)
	}
	debugTime := time.Since(start)

	// Test with debug disabled
	internal.DisableDebugMode()

	start = time.Now()
	for i := 0; i < 100; i++ {
		arr, _ := array.FromSlice(data)
		_, _ = arr.AddScalar(1.0)
	}
	normalTime := time.Since(start)

	fmt.Printf("   With Debug: %v\n", debugTime)
	fmt.Printf("   Without Debug: %v\n", normalTime)

	if debugTime > 0 && normalTime > 0 {
		overhead := float64(debugTime-normalTime) / float64(normalTime) * 100
		fmt.Printf("   Debug Overhead: %.1f%%\n", overhead)
	}
}
