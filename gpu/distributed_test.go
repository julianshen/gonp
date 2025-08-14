// Package gpu provides tests for distributed computing with MPI-style communication
//
// This module tests distributed processing, message passing, parallel operations,
// and cluster coordination using TDD methodology.
//
// TDD Methodology:
//   - Red Phase: Write failing tests defining distributed computing requirements
//   - Green Phase: Implement minimal distributed functionality
//   - Refactor Phase: Optimize for performance and fault tolerance

package gpu

import (
	"testing"
	"time"
)

// TestDistributedProcessInitialization tests MPI-style process initialization
func TestDistributedProcessInitialization(t *testing.T) {
	t.Run("Basic process initialization", func(t *testing.T) {
		// Initialize distributed environment
		comm, err := InitializeMPI(4) // 4 processes
		if err != nil {
			t.Fatalf("MPI initialization failed: %v", err)
		}
		defer comm.Finalize()

		// Verify process properties
		rank := comm.Rank()
		size := comm.Size()

		if rank < 0 || rank >= size {
			t.Errorf("Invalid rank: %d (size: %d)", rank, size)
		}

		if size != 4 {
			t.Errorf("Expected 4 processes, got %d", size)
		}

		t.Logf("Process %d of %d initialized successfully", rank, size)
	})

	t.Run("Process name and identification", func(t *testing.T) {
		comm, err := InitializeMPI(2)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		name, err := comm.GetProcessorName()
		if err != nil {
			t.Errorf("Failed to get processor name: %v", err)
		}

		if len(name) == 0 {
			t.Errorf("Processor name should not be empty")
		}

		t.Logf("Running on processor: %s", name)
	})

	t.Run("Distributed environment validation", func(t *testing.T) {
		comm, err := InitializeMPI(3)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		// All processes should have same size but different ranks
		size := comm.Size()
		rank := comm.Rank()

		if size != 3 {
			t.Errorf("All processes should report same size: expected 3, got %d", size)
		}

		// Check that rank is unique per process (simplified test)
		if rank < 0 || rank >= size {
			t.Errorf("Rank %d out of bounds for size %d", rank, size)
		}
	})
}

// TestMessagePassing tests point-to-point communication
func TestMessagePassing(t *testing.T) {
	t.Run("Basic send and receive", func(t *testing.T) {
		comm, err := InitializeMPI(2)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()

		if rank == 0 {
			// Process 0: Send message to process 1
			message := []float64{1.0, 2.0, 3.0, 4.0}
			err := comm.Send(message, 1, 100) // tag=100
			if err != nil {
				t.Errorf("Send failed: %v", err)
			}
			t.Logf("Process 0 sent: %v", message)
		} else if rank == 1 {
			// Process 1: Receive message from process 0
			var received []float64
			err := comm.Recv(&received, 0, 100) // from rank 0, tag=100
			if err != nil {
				t.Errorf("Recv failed: %v", err)
			}

			expected := []float64{1.0, 2.0, 3.0, 4.0}
			if len(received) != len(expected) {
				t.Errorf("Received wrong length: expected %d, got %d", len(expected), len(received))
			}

			for i, v := range expected {
				if i < len(received) && received[i] != v {
					t.Errorf("Received[%d] = %f, expected %f", i, received[i], v)
				}
			}
			t.Logf("Process 1 received: %v", received)
		}
	})

	t.Run("Non-blocking communication", func(t *testing.T) {
		comm, err := InitializeMPI(2)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()

		if rank == 0 {
			// Non-blocking send
			message := []float64{10.0, 20.0, 30.0}
			request, err := comm.ISend(message, 1, 200)
			if err != nil {
				t.Errorf("ISend failed: %v", err)
			}

			// Wait for completion
			err = request.Wait()
			if err != nil {
				t.Errorf("Send wait failed: %v", err)
			}
			t.Logf("Process 0 completed non-blocking send")
		} else if rank == 1 {
			// Non-blocking receive
			var received []float64
			request, err := comm.IRecv(&received, 0, 200)
			if err != nil {
				t.Errorf("IRecv failed: %v", err)
			}

			// Wait for completion
			err = request.Wait()
			if err != nil {
				t.Errorf("Recv wait failed: %v", err)
			}

			expected := []float64{10.0, 20.0, 30.0}
			if len(received) != len(expected) {
				t.Errorf("Non-blocking receive failed: expected %v, got %v", expected, received)
			}
			t.Logf("Process 1 completed non-blocking receive: %v", received)
		}
	})

	t.Run("Message status and probing", func(t *testing.T) {
		comm, err := InitializeMPI(2)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()

		if rank == 0 {
			// Send with delay to test probing
			time.Sleep(100 * time.Millisecond)
			message := []float64{100.0, 200.0}
			err := comm.Send(message, 1, 300)
			if err != nil {
				t.Errorf("Send failed: %v", err)
			}
		} else if rank == 1 {
			// Probe for incoming message
			status, err := comm.Probe(0, 300)
			if err != nil {
				t.Errorf("Probe failed: %v", err)
			}

			if status.Source != 0 {
				t.Errorf("Wrong source: expected 0, got %d", status.Source)
			}

			if status.Tag != 300 {
				t.Errorf("Wrong tag: expected 300, got %d", status.Tag)
			}

			// Now receive the probed message
			var received []float64
			err = comm.Recv(&received, 0, 300)
			if err != nil {
				t.Errorf("Recv after probe failed: %v", err)
			}
			t.Logf("Probed and received: %v", received)
		}
	})
}

// TestCollectiveCommunication tests collective operations
func TestCollectiveCommunication(t *testing.T) {
	t.Run("Broadcast operation", func(t *testing.T) {
		comm, err := InitializeMPI(4)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()
		var data []float64

		if rank == 0 {
			// Root process broadcasts data
			data = []float64{1.5, 2.5, 3.5, 4.5}
		}

		// All processes participate in broadcast
		err = comm.Broadcast(&data, 0) // root=0
		if err != nil {
			t.Errorf("Broadcast failed: %v", err)
		}

		// Verify all processes have the same data
		expected := []float64{1.5, 2.5, 3.5, 4.5}
		if len(data) != len(expected) {
			t.Errorf("Process %d: wrong data length: expected %d, got %d", rank, len(expected), len(data))
		}

		for i, v := range expected {
			if i < len(data) && data[i] != v {
				t.Errorf("Process %d: data[%d] = %f, expected %f", rank, i, data[i], v)
			}
		}

		t.Logf("Process %d received broadcast: %v", rank, data)
	})

	t.Run("Reduce operation", func(t *testing.T) {
		comm, err := InitializeMPI(3)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()

		// Each process contributes its rank as data
		localData := []float64{float64(rank), float64(rank + 10)}

		var result []float64
		// Sum reduction: root=0
		err = comm.Reduce(localData, &result, ReduceSum, 0)
		if err != nil {
			t.Errorf("Reduce failed: %v", err)
		}

		if rank == 0 {
			// Root should have sum of all contributions
			// Process 0: [0, 10], Process 1: [1, 11], Process 2: [2, 12]
			// Expected sum: [3, 33]
			expected := []float64{3.0, 33.0}
			if len(result) != len(expected) {
				t.Errorf("Wrong result length: expected %d, got %d", len(expected), len(result))
			}

			for i, v := range expected {
				if i < len(result) && result[i] != v {
					t.Errorf("result[%d] = %f, expected %f", i, result[i], v)
				}
			}
			t.Logf("Reduce result: %v", result)
		}
	})

	t.Run("All-reduce operation", func(t *testing.T) {
		comm, err := InitializeMPI(3)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()

		// Each process contributes different data
		localData := []float64{float64(rank + 1)} // 1, 2, 3

		var result []float64
		// All processes get the sum
		err = comm.AllReduce(localData, &result, ReduceSum)
		if err != nil {
			t.Errorf("AllReduce failed: %v", err)
		}

		// All processes should have sum = 6
		expected := []float64{6.0}
		if len(result) != len(expected) {
			t.Errorf("Process %d: wrong result length: expected %d, got %d", rank, len(expected), len(result))
		}

		if len(result) > 0 && result[0] != expected[0] {
			t.Errorf("Process %d: result[0] = %f, expected %f", rank, result[0], expected[0])
		}

		t.Logf("Process %d AllReduce result: %v", rank, result)
	})

	t.Run("Gather operation", func(t *testing.T) {
		comm, err := InitializeMPI(3)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()

		// Each process contributes its rank squared
		localData := []float64{float64(rank * rank)}

		var gathered [][]float64
		err = comm.Gather(localData, &gathered, 0) // root=0
		if err != nil {
			t.Errorf("Gather failed: %v", err)
		}

		if rank == 0 {
			// Root should have gathered all data
			if len(gathered) != 3 {
				t.Errorf("Wrong gather length: expected 3, got %d", len(gathered))
			}

			// Expected: [[0], [1], [4]] from processes 0, 1, 2
			expected := [][]float64{{0.0}, {1.0}, {4.0}}
			for i, expectedSlice := range expected {
				if i < len(gathered) {
					if len(gathered[i]) != len(expectedSlice) {
						t.Errorf("gathered[%d] length wrong: expected %d, got %d", i, len(expectedSlice), len(gathered[i]))
						continue
					}
					for j, expectedVal := range expectedSlice {
						if gathered[i][j] != expectedVal {
							t.Errorf("gathered[%d][%d] = %f, expected %f", i, j, gathered[i][j], expectedVal)
						}
					}
				}
			}
			t.Logf("Gathered data: %v", gathered)
		}
	})
}

// TestParallelOperations tests distributed parallel computations
func TestParallelOperations(t *testing.T) {
	t.Run("Distributed matrix multiplication", func(t *testing.T) {
		comm, err := InitializeMPI(2)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()
		size := comm.Size()

		// Simple 4x4 matrix multiplication distributed across processes
		matrixSize := 4

		var A, B []float64

		if rank == 0 {
			// Initialize matrices on root
			A = make([]float64, matrixSize*matrixSize)
			B = make([]float64, matrixSize*matrixSize)

			// Initialize with simple pattern
			for i := 0; i < matrixSize*matrixSize; i++ {
				A[i] = float64(i + 1)
				B[i] = float64((i + 1) * 2)
			}
		}

		// Perform distributed matrix multiplication
		result, err := DistributedMatMul(comm, A, B, matrixSize, matrixSize, matrixSize)
		if err != nil {
			t.Errorf("Distributed MatMul failed: %v", err)
		}

		if rank == 0 {
			// Verify result on root
			if len(result) != matrixSize*matrixSize {
				t.Errorf("Wrong result size: expected %d, got %d", matrixSize*matrixSize, len(result))
			}

			// Compare with sequential version for correctness
			expected := sequentialMatMul(A, B, matrixSize, matrixSize, matrixSize)
			maxError := 0.0
			for i := 0; i < len(result); i++ {
				if i < len(expected) {
					error := absTest(result[i] - expected[i])
					if error > maxError {
						maxError = error
					}
				}
			}

			if maxError > 1e-10 {
				t.Errorf("Distributed result differs from sequential: max error = %e", maxError)
			}

			t.Logf("Distributed MatMul completed with %d processes", size)
		}
	})

	t.Run("Parallel array operations", func(t *testing.T) {
		comm, err := InitializeMPI(4)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()
		size := comm.Size()

		// Create distributed array
		totalSize := 100
		localSize := totalSize / size

		// Each process handles a chunk of the array
		localData := make([]float64, localSize)
		for i := 0; i < localSize; i++ {
			localData[i] = float64(rank*localSize + i)
		}

		// Parallel sum operation
		localSum := 0.0
		for _, v := range localData {
			localSum += v
		}

		var totalSum []float64
		err = comm.AllReduce([]float64{localSum}, &totalSum, ReduceSum)
		if err != nil {
			t.Errorf("Parallel sum failed: %v", err)
		}

		// Expected sum: 0+1+2+...+99 = 4950
		expected := 4950.0
		if len(totalSum) > 0 && totalSum[0] != expected {
			t.Errorf("Process %d: parallel sum = %f, expected %f", rank, totalSum[0], expected)
		}

		t.Logf("Process %d: local sum = %f, total sum = %f", rank, localSum, totalSum[0])
	})

	t.Run("Load balancing and work distribution", func(t *testing.T) {
		comm, err := InitializeMPI(3)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()
		size := comm.Size()

		// Simulate uneven workload
		totalWork := 100
		workItems := make([]int, totalWork)
		for i := range workItems {
			workItems[i] = i + 1
		}

		// Distribute work among processes
		distributor := NewWorkDistributor(comm)
		localWork, err := distributor.DistributeWork(workItems)
		if err != nil {
			t.Errorf("Work distribution failed: %v", err)
		}

		// Process local work (simulate computation)
		localResult := 0
		for _, work := range localWork {
			localResult += work * work // Square each work item
		}

		// Gather results
		var allResults []float64
		err = comm.AllReduce([]float64{float64(localResult)}, &allResults, ReduceSum)
		if err != nil {
			t.Errorf("Result gathering failed: %v", err)
		}

		if rank == 0 {
			// Expected: sum of squares 1²+2²+...+100² = 338350
			expected := 338350.0
			if len(allResults) > 0 && allResults[0] != expected {
				t.Errorf("Total result = %f, expected %f", allResults[0], expected)
			}
			t.Logf("Work distributed across %d processes: total = %f", size, allResults[0])
		}
	})
}

// TestFaultTolerance tests error handling and recovery
func TestFaultTolerance(t *testing.T) {
	t.Run("Communication timeout handling", func(t *testing.T) {
		comm, err := InitializeMPI(2)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()

		if rank == 0 {
			// Don't send anything, let receiver timeout
			time.Sleep(100 * time.Millisecond)
		} else if rank == 1 {
			// Try to receive with timeout
			var received []float64
			err := comm.RecvWithTimeout(&received, 0, 400, 50*time.Millisecond)
			if err == nil {
				t.Errorf("Expected timeout error, got nil")
			}

			if !IsTimeoutError(err) {
				t.Errorf("Expected timeout error, got: %v", err)
			}

			t.Logf("Correctly handled timeout: %v", err)
		}
	})

	t.Run("Invalid rank handling", func(t *testing.T) {
		comm, err := InitializeMPI(2)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		// Try to send to invalid rank
		message := []float64{1.0, 2.0}
		err = comm.Send(message, 99, 100) // rank 99 doesn't exist
		if err == nil {
			t.Errorf("Expected error for invalid rank, got nil")
		}

		t.Logf("Correctly handled invalid rank: %v", err)
	})

	t.Run("Process failure simulation", func(t *testing.T) {
		// This test simulates what happens when a process becomes unavailable
		comm, err := InitializeMPI(3)
		if err != nil {
			t.Skip("MPI not available")
		}
		defer comm.Finalize()

		rank := comm.Rank()

		// Simulate process 1 failing (not participating in collective)
		if rank != 1 {
			var data []float64
			if rank == 0 {
				data = []float64{10.0, 20.0}
			}

			// This should timeout or fail gracefully
			err = comm.BroadcastWithTimeout(&data, 0, 100*time.Millisecond)
			if err != nil && !IsTimeoutError(err) {
				t.Errorf("Unexpected error type: %v", err)
			}
		}
		// Process 1 doesn't participate, simulating failure
	})
}

// Helper functions for testing

// sequentialMatMul performs standard matrix multiplication for comparison
func sequentialMatMul(A, B []float64, M, N, K int) []float64 {
	C := make([]float64, M*N)

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := 0.0
			for k := 0; k < K; k++ {
				sum += A[i*K+k] * B[k*N+j]
			}
			C[i*N+j] = sum
		}
	}

	return C
}

// Use absTest function from numa_test.go (already defined)
