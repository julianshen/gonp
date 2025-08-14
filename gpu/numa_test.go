// Package gpu provides tests for NUMA-aware optimizations
//
// This module tests NUMA topology detection, memory allocation affinity,
// thread scheduling, and distributed memory management using TDD methodology.
//
// TDD Methodology:
//   - Red Phase: Write failing tests defining NUMA requirements
//   - Green Phase: Implement minimal NUMA functionality
//   - Refactor Phase: Optimize for performance and code quality

package gpu

import (
	"testing"
	"time"
)

// TestNUMATopologyDetection tests NUMA system topology discovery
func TestNUMATopologyDetection(t *testing.T) {
	t.Run("Basic NUMA topology detection", func(t *testing.T) {
		// This test will fail until we implement NUMA detection
		topology, err := DetectNUMATopologyAdvanced()
		if err != nil {
			t.Fatalf("NUMA topology detection failed: %v", err)
		}

		// Validate topology structure
		if topology.NodeCount <= 0 {
			t.Errorf("Invalid node count: %d", topology.NodeCount)
		}

		if len(topology.CPUAffinity) != topology.NodeCount {
			t.Errorf("CPU affinity mismatch: nodes=%d, affinity=%d",
				topology.NodeCount, len(topology.CPUAffinity))
		}

		if len(topology.MemorySize) != topology.NodeCount {
			t.Errorf("Memory size mismatch: nodes=%d, memory=%d",
				topology.NodeCount, len(topology.MemorySize))
		}

		// Log detected topology
		t.Logf("Detected NUMA topology: %d nodes", topology.NodeCount)
		for i := 0; i < topology.NodeCount; i++ {
			t.Logf("  Node %d: %d CPUs, %d MB memory",
				i, len(topology.CPUAffinity[i]), topology.MemorySize[i]/(1024*1024))
		}
	})

	t.Run("NUMA distance matrix calculation", func(t *testing.T) {
		topology, err := DetectNUMATopologyAdvanced()
		if err != nil {
			t.Skip("NUMA topology not available")
		}

		// Test distance matrix
		distances, err := topology.GetDistanceMatrix()
		if err != nil {
			t.Fatalf("Distance matrix calculation failed: %v", err)
		}

		// Validate distance matrix properties
		expectedSize := topology.NodeCount * topology.NodeCount
		if len(distances) != expectedSize {
			t.Errorf("Distance matrix size mismatch: expected %d, got %d",
				expectedSize, len(distances))
		}

		// Distance from node to itself should be 0 or minimal
		for i := 0; i < topology.NodeCount; i++ {
			selfDistance := distances[i*topology.NodeCount+i]
			if selfDistance < 0 || selfDistance > 50 { // Reasonable bounds
				t.Errorf("Invalid self-distance for node %d: %d", i, selfDistance)
			}
		}

		// Distance should be symmetric
		for i := 0; i < topology.NodeCount; i++ {
			for j := 0; j < topology.NodeCount; j++ {
				dist1 := distances[i*topology.NodeCount+j]
				dist2 := distances[j*topology.NodeCount+i]
				if dist1 != dist2 {
					t.Errorf("Distance matrix not symmetric: (%d,%d)=%d, (%d,%d)=%d",
						i, j, dist1, j, i, dist2)
				}
			}
		}
	})

	t.Run("Current NUMA node detection", func(t *testing.T) {
		currentNode, err := GetCurrentNUMANode()
		if err != nil {
			t.Skip("Current NUMA node detection not supported")
		}

		topology, _ := DetectNUMATopologyAdvanced()
		if topology != nil && (currentNode < 0 || currentNode >= topology.NodeCount) {
			t.Errorf("Invalid current NUMA node: %d (max: %d)", currentNode, topology.NodeCount-1)
		}

		t.Logf("Current NUMA node: %d", currentNode)
	})
}

// TestNUMAMemoryAllocation tests NUMA-aware memory allocation
func TestNUMAMemoryAllocation(t *testing.T) {
	t.Run("Node-specific memory allocation", func(t *testing.T) {
		topology, err := DetectNUMATopologyAdvanced()
		if err != nil || topology.NodeCount < 2 {
			t.Skip("Multi-node NUMA system required")
		}

		size := 1024 * 1024 // 1MB allocation

		// Test allocation on each NUMA node
		for nodeID := 0; nodeID < topology.NodeCount; nodeID++ {
			memory, err := AllocateOnNUMANode(size, nodeID)
			if err != nil {
				t.Errorf("Failed to allocate on NUMA node %d: %v", nodeID, err)
				continue
			}
			defer memory.Free()

			// Verify allocation properties
			if memory.Size() != int64(size) {
				t.Errorf("Allocated size mismatch: expected %d, got %d", size, memory.Size())
			}

			// Verify memory is on correct NUMA node
			actualNode, err := memory.GetNUMANode()
			if err != nil {
				t.Errorf("Failed to get NUMA node for allocation: %v", err)
			} else if actualNode != nodeID {
				t.Errorf("Memory allocated on wrong node: expected %d, got %d", nodeID, actualNode)
			}

			t.Logf("Successfully allocated %d bytes on NUMA node %d", size, nodeID)
		}
	})

	t.Run("Interleaved memory allocation", func(t *testing.T) {
		topology, err := DetectNUMATopologyAdvanced()
		if err != nil || topology.NodeCount < 2 {
			t.Skip("Multi-node NUMA system required")
		}

		size := 4 * 1024 * 1024 // 4MB allocation

		// Test interleaved allocation across all nodes
		memory, err := AllocateInterleaved(size, topology.GetAllNodes())
		if err != nil {
			t.Fatalf("Interleaved allocation failed: %v", err)
		}
		defer memory.Free()

		// Verify allocation
		if memory.Size() != int64(size) {
			t.Errorf("Interleaved allocation size mismatch: expected %d, got %d", size, memory.Size())
		}

		// Verify interleaving (this is implementation-specific)
		policy, err := memory.GetAllocationPolicy()
		if err != nil {
			t.Errorf("Failed to get allocation policy: %v", err)
		} else if policy != "interleaved" {
			t.Errorf("Expected interleaved policy, got: %s", policy)
		}

		t.Logf("Successfully allocated %d bytes with interleaved policy", size)
	})

	t.Run("Memory migration between nodes", func(t *testing.T) {
		topology, err := DetectNUMATopologyAdvanced()
		if err != nil || topology.NodeCount < 2 {
			t.Skip("Multi-node NUMA system required")
		}

		size := 2 * 1024 * 1024 // 2MB allocation

		// Allocate on first node
		memory, err := AllocateOnNUMANode(size, 0)
		if err != nil {
			t.Fatalf("Initial allocation failed: %v", err)
		}
		defer memory.Free()

		// Migrate to second node
		targetNode := 1
		err = memory.MigrateToNode(targetNode)
		if err != nil {
			t.Errorf("Memory migration failed: %v", err)
			return
		}

		// Verify migration
		currentNode, err := memory.GetNUMANode()
		if err != nil {
			t.Errorf("Failed to get NUMA node after migration: %v", err)
		} else if currentNode != targetNode {
			t.Errorf("Migration failed: expected node %d, memory on node %d", targetNode, currentNode)
		}

		t.Logf("Successfully migrated memory from node 0 to node %d", targetNode)
	})
}

// TestNUMAThreadAffinity tests CPU affinity and thread scheduling
func TestNUMAThreadAffinity(t *testing.T) {
	t.Run("CPU affinity setting", func(t *testing.T) {
		topology, err := DetectNUMATopologyAdvanced()
		if err != nil || topology.NodeCount == 0 {
			t.Skip("NUMA topology required")
		}

		// Test setting affinity to each NUMA node
		for nodeID := 0; nodeID < topology.NodeCount; nodeID++ {
			cpus := topology.CPUAffinity[nodeID]
			if len(cpus) == 0 {
				t.Errorf("No CPUs found for NUMA node %d", nodeID)
				continue
			}

			// Set thread affinity to this node
			err := SetThreadAffinity(cpus)
			if err != nil {
				t.Errorf("Failed to set thread affinity for node %d: %v", nodeID, err)
				continue
			}

			// Verify affinity was set
			currentAffinity, err := GetThreadAffinity()
			if err != nil {
				t.Errorf("Failed to get thread affinity: %v", err)
				continue
			}

			// Check if at least one CPU from the target node is in current affinity
			found := false
			for _, cpu := range cpus {
				for _, current := range currentAffinity {
					if cpu == current {
						found = true
						break
					}
				}
				if found {
					break
				}
			}

			if !found {
				t.Errorf("Thread affinity not set correctly for node %d", nodeID)
			}

			t.Logf("Set thread affinity to node %d (CPUs: %v)", nodeID, cpus[:minTest(len(cpus), 4)])
		}

		// Reset affinity to all CPUs
		allCPUs := topology.GetAllCPUs()
		err = SetThreadAffinity(allCPUs)
		if err != nil {
			t.Errorf("Failed to reset thread affinity: %v", err)
		}
	})

	t.Run("NUMA-aware goroutine scheduling", func(t *testing.T) {
		topology, err := DetectNUMATopologyAdvanced()
		if err != nil || topology.NodeCount < 2 {
			t.Skip("Multi-node NUMA system required")
		}

		// Test launching goroutines on specific NUMA nodes
		nodeCount := minTest(topology.NodeCount, 4) // Limit for test performance
		done := make(chan int, nodeCount)

		for nodeID := 0; nodeID < nodeCount; nodeID++ {
			go func(node int) {
				// Set goroutine affinity to specific NUMA node
				err := SetGoroutineNUMANode(node)
				if err != nil {
					t.Errorf("Failed to set goroutine NUMA affinity: %v", err)
					done <- -1
					return
				}

				// Verify we're running on the correct node
				currentNode, err := GetCurrentNUMANode()
				if err != nil {
					t.Logf("Warning: Cannot verify current NUMA node: %v", err)
				} else if currentNode != node {
					t.Errorf("Goroutine not on expected node: expected %d, got %d", node, currentNode)
				}

				// Simulate some work
				time.Sleep(10 * time.Millisecond)
				done <- node
			}(nodeID)
		}

		// Wait for all goroutines to complete
		for i := 0; i < nodeCount; i++ {
			select {
			case result := <-done:
				if result >= 0 {
					t.Logf("Goroutine completed on NUMA node %d", result)
				}
			case <-time.After(5 * time.Second):
				t.Errorf("Goroutine %d timed out", i)
			}
		}
	})
}

// TestNUMAPerformanceOptimization tests performance benefits of NUMA awareness
func TestNUMAPerformanceOptimization(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping NUMA performance tests in short mode")
	}

	t.Run("Local vs remote memory access performance", func(t *testing.T) {
		topology, err := DetectNUMATopologyAdvanced()
		if err != nil || topology.NodeCount < 2 {
			t.Skip("Multi-node NUMA system required")
		}

		size := 10 * 1024 * 1024 // 10MB for meaningful measurement

		// Test local memory access
		localNode := 0
		err = SetThreadAffinity(topology.CPUAffinity[localNode])
		if err != nil {
			t.Fatalf("Failed to set thread affinity: %v", err)
		}

		localMemory, err := AllocateOnNUMANode(size, localNode)
		if err != nil {
			t.Fatalf("Failed to allocate local memory: %v", err)
		}
		defer localMemory.Free()

		// Benchmark local access
		start := time.Now()
		localSum, err := BenchmarkMemoryAccess(localMemory, 1000)
		localTime := time.Since(start)
		if err != nil {
			t.Fatalf("Local memory benchmark failed: %v", err)
		}

		// Test remote memory access
		remoteNode := 1
		remoteMemory, err := AllocateOnNUMANode(size, remoteNode)
		if err != nil {
			t.Fatalf("Failed to allocate remote memory: %v", err)
		}
		defer remoteMemory.Free()

		// Benchmark remote access (still from localNode CPU)
		start = time.Now()
		remoteSum, err := BenchmarkMemoryAccess(remoteMemory, 1000)
		remoteTime := time.Since(start)
		if err != nil {
			t.Fatalf("Remote memory benchmark failed: %v", err)
		}

		// Verify correctness
		if localSum != remoteSum {
			t.Errorf("Benchmark results differ: local=%.1f, remote=%.1f", localSum, remoteSum)
		}

		// Calculate performance difference
		if localTime > 0 && remoteTime > 0 {
			ratio := float64(remoteTime) / float64(localTime)
			t.Logf("Memory access performance: Local %v, Remote %v, Ratio: %.2fx",
				localTime, remoteTime, ratio)

			// Remote access should typically be slower (ratio > 1.0)
			// But this is system-dependent
			if ratio > 2.0 {
				t.Logf("Significant NUMA penalty detected: %.2fx slower for remote access", ratio)
			} else if ratio < 0.9 {
				t.Logf("Unexpected: Remote access faster than local (may indicate measurement noise)")
			}
		}
	})

	t.Run("NUMA-aware vs NUMA-oblivious allocation performance", func(t *testing.T) {
		topology, err := DetectNUMATopologyAdvanced()
		if err != nil || topology.NodeCount < 2 {
			t.Skip("Multi-node NUMA system required")
		}

		size := 20 * 1024 * 1024 // 20MB
		iterations := 100

		// Test NUMA-aware allocation (local to current CPU)
		currentNode, _ := GetCurrentNUMANode()
		if currentNode < 0 {
			currentNode = 0
		}

		numaMemory, err := AllocateOnNUMANode(size, currentNode)
		if err != nil {
			t.Fatalf("NUMA-aware allocation failed: %v", err)
		}
		defer numaMemory.Free()

		start := time.Now()
		numaSum, err := BenchmarkMemoryAccess(numaMemory, iterations)
		numaTime := time.Since(start)
		if err != nil {
			t.Fatalf("NUMA-aware benchmark failed: %v", err)
		}

		// Test standard allocation (no NUMA awareness)
		standardMemory, err := AllocateStandard(size)
		if err != nil {
			t.Fatalf("Standard allocation failed: %v", err)
		}
		defer standardMemory.Free()

		start = time.Now()
		standardSum, err := BenchmarkMemoryAccess(standardMemory, iterations)
		standardTime := time.Since(start)
		if err != nil {
			t.Fatalf("Standard benchmark failed: %v", err)
		}

		// Verify correctness
		if absTest(numaSum-standardSum) > 0.1 {
			t.Errorf("Benchmark results differ significantly: NUMA=%.1f, Standard=%.1f",
				numaSum, standardSum)
		}

		// Calculate performance improvement
		if numaTime > 0 && standardTime > 0 {
			improvement := float64(standardTime) / float64(numaTime)
			t.Logf("NUMA optimization: NUMA-aware %v, Standard %v, Improvement: %.2fx",
				numaTime, standardTime, improvement)

			// NUMA-aware should typically be faster (improvement > 1.0)
			if improvement > 1.1 {
				t.Logf("NUMA optimization effective: %.1f%% performance improvement",
					(improvement-1.0)*100)
			} else if improvement < 0.95 {
				t.Logf("Warning: NUMA-aware allocation slower than standard")
			} else {
				t.Logf("NUMA effect minimal (within measurement noise)")
			}
		}
	})
}

// Helper functions for testing

func minTest(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func absTest(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
