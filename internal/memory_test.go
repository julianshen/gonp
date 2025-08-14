// Package internal provides tests for production-grade memory management
//
// This module tests advanced memory pool optimization, leak detection,
// garbage collection integration, and resource cleanup using TDD methodology.
//
// TDD Methodology:
//   - Red Phase: Write failing tests defining memory management requirements
//   - Green Phase: Implement minimal memory management functionality
//   - Refactor Phase: Optimize for performance and production reliability

package internal

import (
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestProductionMemoryPool tests enterprise-grade memory pool functionality
func TestProductionMemoryPool(t *testing.T) {
	t.Run("Memory pool with leak detection", func(t *testing.T) {
		// Initialize production memory pool with leak detection
		pool, err := NewProductionMemoryPool(MemoryPoolConfig{
			MaxPoolSize:    10 * 1024 * 1024, // 10MB
			MaxBlockCount:  1000,             // Max blocks per size
			LeakDetection:  true,             // Enable leak detection
			GCIntegration:  true,             // Integrate with Go GC
			MonitoringMode: true,             // Enable monitoring
		})
		if err != nil {
			t.Fatalf("Failed to create production memory pool: %v", err)
		}
		defer pool.Shutdown()

		// Test basic allocation and deallocation
		block, err := pool.Allocate(1024)
		if err != nil {
			t.Errorf("Failed to allocate memory: %v", err)
		}

		if block.Size() != 1024 {
			t.Errorf("Wrong block size: expected 1024, got %d", block.Size())
		}

		// Track allocation
		stats := pool.GetDetailedStats()
		if stats.ActiveAllocations != 1 {
			t.Errorf("Wrong active allocation count: expected 1, got %d", stats.ActiveAllocations)
		}

		// Test deallocation
		err = pool.Deallocate(block)
		if err != nil {
			t.Errorf("Failed to deallocate memory: %v", err)
		}

		// Verify deallocation
		stats = pool.GetDetailedStats()
		if stats.ActiveAllocations != 0 {
			t.Errorf("Memory leak detected: expected 0 active allocations, got %d", stats.ActiveAllocations)
		}

		if stats.TotalLeaks > 0 {
			t.Errorf("Memory leak detected: %d leaks found", stats.TotalLeaks)
		}
	})

	t.Run("Automatic leak detection and reporting", func(t *testing.T) {
		pool, err := NewProductionMemoryPool(MemoryPoolConfig{
			MaxPoolSize:   1024 * 1024,
			LeakDetection: true,
			LeakTimeout:   100 * time.Millisecond,
		})
		if err != nil {
			t.Fatalf("Failed to create memory pool: %v", err)
		}
		defer pool.Shutdown()

		// Allocate memory but don't deallocate (simulate leak)
		block, err := pool.Allocate(512)
		if err != nil {
			t.Errorf("Failed to allocate memory: %v", err)
		}

		// Wait for leak detection timeout
		time.Sleep(200 * time.Millisecond)

		// Check for detected leaks
		leaks := pool.GetLeakReport()
		if len(leaks) == 0 {
			t.Errorf("Expected leak detection, but no leaks found")
		}

		// Verify leak details
		leak := leaks[0]
		if leak.Size != 512 {
			t.Errorf("Wrong leak size: expected 512, got %d", leak.Size)
		}

		if leak.Age < 100*time.Millisecond {
			t.Errorf("Wrong leak age: expected >100ms, got %v", leak.Age)
		}

		// Clean up the leaked block
		err = pool.Deallocate(block)
		if err != nil {
			t.Errorf("Failed to deallocate leaked block: %v", err)
		}
	})

	t.Run("Memory pool resizing and adaptive management", func(t *testing.T) {
		pool, err := NewProductionMemoryPool(MemoryPoolConfig{
			MaxPoolSize:     1024 * 1024, // 1MB initial
			AdaptiveResize:  true,
			ResizeThreshold: 0.8, // Resize when 80% full
		})
		if err != nil {
			t.Fatalf("Failed to create adaptive memory pool: %v", err)
		}
		defer pool.Shutdown()

		// Fill pool to trigger resize
		blocks := make([]MemoryBlock, 0)
		for i := 0; i < 100; i++ {
			block, err := pool.Allocate(10240) // 10KB each
			if err != nil {
				t.Errorf("Allocation %d failed: %v", i, err)
				break
			}
			blocks = append(blocks, block)
		}

		// Check if pool resized
		stats := pool.GetDetailedStats()
		if stats.PoolSize <= 1024*1024 {
			t.Errorf("Pool should have resized: current size %d", stats.PoolSize)
		}

		t.Logf("Pool resized to %d bytes (efficiency: %.1f%%)", stats.PoolSize, stats.Efficiency)

		// Clean up
		for _, block := range blocks {
			pool.Deallocate(block)
		}
	})
}

// TestMemoryPressureHandling tests memory pressure and GC integration
func TestMemoryPressureHandling(t *testing.T) {
	t.Run("Memory pressure detection and response", func(t *testing.T) {
		pool, err := NewProductionMemoryPool(MemoryPoolConfig{
			MaxPoolSize:       5 * 1024 * 1024, // 5MB
			PressureHandling:  true,
			PressureThreshold: 0.85, // Trigger at 85% system memory
		})
		if err != nil {
			t.Fatalf("Failed to create memory pool: %v", err)
		}
		defer pool.Shutdown()

		// Get initial memory stats
		var m1 runtime.MemStats
		runtime.ReadMemStats(&m1)

		// Simulate memory pressure by allocating large blocks
		var blocks []MemoryBlock
		for i := 0; i < 20; i++ {
			block, err := pool.Allocate(256 * 1024) // 256KB each
			if err != nil {
				break // Expected when memory pressure kicks in
			}
			blocks = append(blocks, block)
		}

		// Check memory pressure response
		stats := pool.GetDetailedStats()
		if stats.PressureEvents == 0 && len(blocks) > 15 {
			t.Logf("Warning: No pressure events detected with %d allocations", len(blocks))
		}

		// Get memory stats after allocation
		var m2 runtime.MemStats
		runtime.ReadMemStats(&m2)

		t.Logf("Memory usage: %d -> %d bytes (pressure events: %d)",
			m1.Alloc, m2.Alloc, stats.PressureEvents)

		// Clean up
		for _, block := range blocks {
			pool.Deallocate(block)
		}

		// Force GC and check cleanup
		runtime.GC()
		runtime.ReadMemStats(&m2)
		t.Logf("After cleanup: %d bytes", m2.Alloc)
	})

	t.Run("GC integration and finalizer cleanup", func(t *testing.T) {
		pool, err := NewProductionMemoryPool(MemoryPoolConfig{
			MaxPoolSize:             2 * 1024 * 1024,
			GCIntegration:           true,
			UseFinalizersForCleanup: true,
		})
		if err != nil {
			t.Fatalf("Failed to create GC-integrated memory pool: %v", err)
		}
		defer pool.Shutdown()

		// Allocate blocks and let them go out of scope
		initialStats := pool.GetDetailedStats()

		func() {
			var blocks []MemoryBlock
			for i := 0; i < 10; i++ {
				block, err := pool.Allocate(1024)
				if err != nil {
					t.Errorf("Failed to allocate block %d: %v", i, err)
					continue
				}
				blocks = append(blocks, block)
			}
			// blocks go out of scope here
		}()

		// Force GC to trigger finalizers
		runtime.GC()
		runtime.GC() // Double GC to ensure finalizers run
		time.Sleep(10 * time.Millisecond)

		// Check if GC cleaned up unreferenced blocks
		finalStats := pool.GetDetailedStats()

		if finalStats.FinalizerCleanups == 0 {
			t.Logf("Warning: No finalizer cleanups detected")
		}

		t.Logf("GC Integration: Initial allocations=%d, Final=%d, Finalizer cleanups=%d",
			initialStats.ActiveAllocations, finalStats.ActiveAllocations, finalStats.FinalizerCleanups)
	})
}

// TestConcurrentMemoryOperations tests thread safety and concurrent access
func TestConcurrentMemoryOperations(t *testing.T) {
	t.Run("Concurrent allocation and deallocation", func(t *testing.T) {
		pool, err := NewProductionMemoryPool(MemoryPoolConfig{
			MaxPoolSize:   10 * 1024 * 1024,
			MaxBlockCount: 1000,
			LeakDetection: true,
			ThreadSafety:  true,
		})
		if err != nil {
			t.Fatalf("Failed to create thread-safe memory pool: %v", err)
		}
		defer pool.Shutdown()

		const numGoroutines = 10
		const allocationsPerGoroutine = 100

		var wg sync.WaitGroup
		errors := make(chan error, numGoroutines*allocationsPerGoroutine)

		// Launch concurrent allocators
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(goroutineID int) {
				defer wg.Done()

				var blocks []MemoryBlock
				// Allocate
				for j := 0; j < allocationsPerGoroutine; j++ {
					size := 512 + (j % 1024) // Variable sizes
					block, err := pool.Allocate(size)
					if err != nil {
						errors <- err
						return
					}
					blocks = append(blocks, block)
				}

				// Deallocate
				for _, block := range blocks {
					err := pool.Deallocate(block)
					if err != nil {
						errors <- err
						return
					}
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		// Check for errors
		for err := range errors {
			t.Errorf("Concurrent operation error: %v", err)
		}

		// Verify no memory leaks
		stats := pool.GetDetailedStats()
		if stats.ActiveAllocations != 0 {
			t.Errorf("Memory leaks detected: %d active allocations remaining", stats.ActiveAllocations)
		}

		if stats.TotalLeaks > 0 {
			t.Errorf("Leak detector found %d leaks", stats.TotalLeaks)
		}

		t.Logf("Concurrent test completed: %d total allocations, efficiency: %.1f%%",
			stats.TotalAllocations, stats.Efficiency)
	})

	t.Run("Memory pool contention under high load", func(t *testing.T) {
		pool, err := NewProductionMemoryPool(MemoryPoolConfig{
			MaxPoolSize:        5 * 1024 * 1024,
			ContentionHandling: true,
			LockFreePaths:      true, // Enable lock-free optimization
		})
		if err != nil {
			t.Fatalf("Failed to create contention-optimized pool: %v", err)
		}
		defer pool.Shutdown()

		const duration = 100 * time.Millisecond
		const numWorkers = 20

		var wg sync.WaitGroup
		start := time.Now()

		// Launch high-contention workers
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				localOps := 0

				for time.Since(start) < duration {
					block, err := pool.Allocate(256)
					if err != nil {
						continue
					}

					// Immediately deallocate to create contention
					pool.Deallocate(block)
					localOps++
				}

				// Atomic add would be used in real implementation
				_ = localOps
			}()
		}

		wg.Wait()

		stats := pool.GetDetailedStats()
		throughput := float64(stats.TotalAllocations) / duration.Seconds()

		t.Logf("Contention test: %.0f allocs/sec, contention events: %d, efficiency: %.1f%%",
			throughput, stats.ContentionEvents, stats.Efficiency)

		if stats.ContentionEvents > stats.TotalAllocations/2 {
			t.Logf("High contention detected (%d events)", stats.ContentionEvents)
		}
	})
}

// TestMemoryPoolProfiling tests performance monitoring and profiling
func TestMemoryPoolProfiling(t *testing.T) {
	t.Run("Memory allocation profiling and metrics", func(t *testing.T) {
		pool, err := NewProductionMemoryPool(MemoryPoolConfig{
			MaxPoolSize:      2 * 1024 * 1024,
			ProfilingEnabled: true,
			MetricsInterval:  10 * time.Millisecond,
		})
		if err != nil {
			t.Fatalf("Failed to create profiling-enabled pool: %v", err)
		}
		defer pool.Shutdown()

		// Perform various allocations to generate profiling data
		sizes := []int{64, 128, 256, 512, 1024, 2048}
		var blocks []MemoryBlock

		for _, size := range sizes {
			for i := 0; i < 20; i++ {
				block, err := pool.Allocate(size)
				if err != nil {
					t.Errorf("Failed to allocate %d bytes: %v", size, err)
					continue
				}
				blocks = append(blocks, block)
			}
		}

		// Wait for metrics collection
		time.Sleep(50 * time.Millisecond)

		// Get profiling data
		profile := pool.GetAllocationProfile()
		if len(profile.SizeDistribution) == 0 {
			t.Errorf("No profiling data collected")
		}

		// Check size distribution
		for size, count := range profile.SizeDistribution {
			t.Logf("Size %d bytes: %d allocations", size, count)
			if count == 0 {
				t.Errorf("No allocations recorded for size %d", size)
			}
		}

		// Check performance metrics
		if profile.AverageAllocationTime == 0 {
			t.Errorf("No allocation time metrics collected")
		}

		if profile.PeakMemoryUsage == 0 {
			t.Errorf("No memory usage metrics collected")
		}

		t.Logf("Profiling results: Avg alloc time: %v, Peak usage: %d bytes",
			profile.AverageAllocationTime, profile.PeakMemoryUsage)

		// Clean up
		for _, block := range blocks {
			pool.Deallocate(block)
		}
	})

	t.Run("Memory usage monitoring and alerts", func(t *testing.T) {
		pool, err := NewProductionMemoryPool(MemoryPoolConfig{
			MaxPoolSize:         1024 * 1024, // 1MB
			MonitoringMode:      true,
			UsageAlertThreshold: 0.75, // Alert at 75% usage
		})
		if err != nil {
			t.Fatalf("Failed to create monitoring pool: %v", err)
		}
		defer pool.Shutdown()

		// Set up alert handler
		alerts := make(chan MemoryAlert, 10)
		pool.SetAlertHandler(func(alert MemoryAlert) {
			alerts <- alert
		})

		// Allocate memory to trigger alerts
		var blocks []MemoryBlock
		for i := 0; i < 100; i++ {
			block, err := pool.Allocate(8192) // 8KB each
			if err != nil {
				break
			}
			blocks = append(blocks, block)
		}

		// Check for alerts
		select {
		case alert := <-alerts:
			if alert.Type != AlertHighMemoryUsage {
				t.Errorf("Wrong alert type: expected %d, got %d", AlertHighMemoryUsage, alert.Type)
			}
			t.Logf("Received memory usage alert: %s (usage: %.1f%%)", alert.Message, alert.MemoryUsage*100)
		case <-time.After(100 * time.Millisecond):
			t.Logf("No memory usage alerts received (may be expected if allocation failed)")
		}

		// Clean up
		for _, block := range blocks {
			pool.Deallocate(block)
		}

		// Check for cleanup alerts
		stats := pool.GetDetailedStats()
		if stats.AlertsTriggered == 0 {
			t.Logf("No alerts were triggered during test")
		} else {
			t.Logf("Total alerts triggered: %d", stats.AlertsTriggered)
		}
	})
}

// TestMemoryPoolErrorHandling tests error conditions and recovery
func TestMemoryPoolErrorHandling(t *testing.T) {
	t.Run("Out of memory handling and recovery", func(t *testing.T) {
		pool, err := NewProductionMemoryPool(MemoryPoolConfig{
			MaxPoolSize:         100 * 1024, // Very small: 100KB
			OOMHandling:         true,
			GracefulDegradation: true,
		})
		if err != nil {
			t.Fatalf("Failed to create OOM-handling pool: %v", err)
		}
		defer pool.Shutdown()

		// Try to allocate more than pool capacity
		var blocks []MemoryBlock
		var oomErrors int

		for i := 0; i < 50; i++ {
			block, err := pool.Allocate(10240) // 10KB each, should exceed 100KB limit
			if err != nil {
				if IsOutOfMemoryError(err) {
					oomErrors++
				} else {
					t.Errorf("Unexpected error type: %v", err)
				}
				break
			}
			blocks = append(blocks, block)
		}

		if oomErrors == 0 && len(blocks) > 10 {
			t.Errorf("Expected out of memory errors, but allocated %d blocks", len(blocks))
		}

		t.Logf("OOM handling: allocated %d blocks, %d OOM errors", len(blocks), oomErrors)

		// Test recovery by freeing some memory
		if len(blocks) > 0 {
			for i := 0; i < len(blocks)/2; i++ {
				pool.Deallocate(blocks[i])
			}

			// Try allocation again
			block, err := pool.Allocate(5120)
			if err != nil {
				t.Errorf("Failed to recover from OOM: %v", err)
			} else {
				t.Logf("Successfully recovered from OOM condition")
				pool.Deallocate(block)
			}
		}

		// Clean up remaining blocks
		for i := len(blocks) / 2; i < len(blocks); i++ {
			pool.Deallocate(blocks[i])
		}
	})

	t.Run("Invalid operation error handling", func(t *testing.T) {
		pool, err := NewProductionMemoryPool(MemoryPoolConfig{
			MaxPoolSize:         1024 * 1024,
			StrictErrorHandling: true,
		})
		if err != nil {
			t.Fatalf("Failed to create strict error handling pool: %v", err)
		}
		defer pool.Shutdown()

		// Test invalid size allocation
		_, err = pool.Allocate(0)
		if err == nil {
			t.Errorf("Expected error for zero-size allocation")
		} else if !IsInvalidSizeError(err) {
			t.Errorf("Wrong error type for invalid size: %v", err)
		}

		// Test negative size allocation
		_, err = pool.Allocate(-1024)
		if err == nil {
			t.Errorf("Expected error for negative size allocation")
		}

		// Test double deallocation
		block, err := pool.Allocate(1024)
		if err != nil {
			t.Fatalf("Failed to allocate test block: %v", err)
		}

		err = pool.Deallocate(block)
		if err != nil {
			t.Errorf("First deallocation failed: %v", err)
		}

		err = pool.Deallocate(block)
		if err == nil {
			t.Errorf("Expected error for double deallocation")
		} else if !IsDoubleDeallocationError(err) {
			t.Errorf("Wrong error type for double deallocation: %v", err)
		}

		// Test null block deallocation
		err = pool.Deallocate(nil)
		if err == nil {
			t.Errorf("Expected error for null block deallocation")
		}
	})
}
