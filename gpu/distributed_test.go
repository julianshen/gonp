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
    "sync"
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
        // Initialize both ranks in the same test run
        comm0, err := InitializeMPI(2)
        if err != nil { t.Skip("MPI not available") }
        comm1, err := InitializeMPI(2)
        if err != nil { t.Skip("MPI not available") }
        defer comm0.Finalize()
        defer comm1.Finalize()

        var wg sync.WaitGroup
        var received []float64
        wg.Add(2)

        go func() {
            defer wg.Done()
            message := []float64{1.0, 2.0, 3.0, 4.0}
            if err := comm0.Send(message, 1, 100); err != nil {
                t.Errorf("Send failed: %v", err)
            }
        }()

        go func() {
            defer wg.Done()
            if err := comm1.Recv(&received, 0, 100); err != nil {
                t.Errorf("Recv failed: %v", err)
            }
        }()

        wg.Wait()

        expected := []float64{1.0, 2.0, 3.0, 4.0}
        if len(received) != len(expected) {
            t.Errorf("Received wrong length: expected %d, got %d", len(expected), len(received))
        }
        for i, v := range expected {
            if i < len(received) && received[i] != v {
                t.Errorf("Received[%d] = %f, expected %f", i, received[i], v)
            }
        }
    })

    t.Run("Non-blocking communication", func(t *testing.T) {
        comm0, err := InitializeMPI(2)
        if err != nil { t.Skip("MPI not available") }
        comm1, err := InitializeMPI(2)
        if err != nil { t.Skip("MPI not available") }
        defer comm0.Finalize()
        defer comm1.Finalize()

        message := []float64{10.0, 20.0, 30.0}
        sendReq, err := comm0.ISend(message, 1, 200)
        if err != nil { t.Fatalf("ISend failed: %v", err) }

        var received []float64
        recvReq, err := comm1.IRecv(&received, 0, 200)
        if err != nil { t.Fatalf("IRecv failed: %v", err) }

        if err := sendReq.Wait(); err != nil { t.Errorf("Send wait failed: %v", err) }
        if err := recvReq.Wait(); err != nil { t.Errorf("Recv wait failed: %v", err) }

        expected := []float64{10.0, 20.0, 30.0}
        if len(received) != len(expected) {
            t.Errorf("Non-blocking receive failed: expected %v, got %v", expected, received)
        }
    })

    t.Run("Message status and probing", func(t *testing.T) {
        comm0, err := InitializeMPI(2)
        if err != nil { t.Skip("MPI not available") }
        comm1, err := InitializeMPI(2)
        if err != nil { t.Skip("MPI not available") }
        defer comm0.Finalize()
        defer comm1.Finalize()

        // Sender after a delay
        go func() {
            time.Sleep(100 * time.Millisecond)
            msg := []float64{100.0, 200.0}
            if err := comm0.Send(msg, 1, 300); err != nil {
                t.Errorf("Send failed: %v", err)
            }
        }()

        // Probe with retries until message appears
        var status *MessageStatus
        var perr error
        deadline := time.Now().Add(2 * time.Second)
        for time.Now().Before(deadline) {
            status, perr = comm1.Probe(0, 300)
            if perr == nil { break }
            time.Sleep(20 * time.Millisecond)
        }
        if perr != nil {
            t.Fatalf("Probe failed: %v", perr)
        }
        if status.Source != 0 { t.Errorf("Wrong source: expected 0, got %d", status.Source) }
        if status.Tag != 300 { t.Errorf("Wrong tag: expected 300, got %d", status.Tag) }

        var received []float64
        if err := comm1.Recv(&received, 0, 300); err != nil {
            t.Errorf("Recv after probe failed: %v", err)
        }
    })
}

// TestCollectiveCommunication tests collective operations
func TestCollectiveCommunication(t *testing.T) {
    t.Run("Broadcast operation", func(t *testing.T) {
        size := 4
        comms := make([]*MPICommunicator, size)
        for i := 0; i < size; i++ {
            c, err := InitializeMPI(size)
            if err != nil { t.Skip("MPI not available") }
            comms[i] = c
        }
        defer func() { for _, c := range comms { _ = c.Finalize() } }()

        data := make([][]float64, size)
        data[0] = []float64{1.5, 2.5, 3.5, 4.5} // root payload

        var wg sync.WaitGroup
        wg.Add(size)
        for r := 0; r < size; r++ {
            r := r
            go func() {
                defer wg.Done()
                if err := comms[r].Broadcast(&data[r], 0); err != nil {
                    t.Errorf("rank %d broadcast failed: %v", r, err)
                }
            }()
        }
        wg.Wait()

        expected := []float64{1.5, 2.5, 3.5, 4.5}
        for r := 0; r < size; r++ {
            if len(data[r]) != len(expected) {
                t.Errorf("rank %d: wrong data length: expected %d, got %d", r, len(expected), len(data[r]))
                continue
            }
            for i, v := range expected {
                if data[r][i] != v {
                    t.Errorf("rank %d: data[%d] = %f, expected %f", r, i, data[r][i], v)
                }
            }
        }
    })

    t.Run("Reduce operation", func(t *testing.T) {
        size := 3
        comms := make([]*MPICommunicator, size)
        for i := 0; i < size; i++ {
            c, err := InitializeMPI(size)
            if err != nil { t.Skip("MPI not available") }
            comms[i] = c
        }
        defer func() { for _, c := range comms { _ = c.Finalize() } }()

        results := make([][]float64, size)
        var wg sync.WaitGroup
        wg.Add(size)
        for r := 0; r < size; r++ {
            r := r
            go func() {
                defer wg.Done()
                local := []float64{float64(r), float64(r + 10)}
                if err := comms[r].Reduce(local, &results[r], ReduceSum, 0); err != nil {
                    t.Errorf("rank %d reduce failed: %v", r, err)
                }
            }()
        }
        wg.Wait()

        expected := []float64{3.0, 33.0}
        if len(results[0]) != len(expected) {
            t.Errorf("root: wrong result length: expected %d, got %d", len(expected), len(results[0]))
        } else {
            for i, v := range expected {
                if results[0][i] != v {
                    t.Errorf("root: result[%d] = %f, expected %f", i, results[0][i], v)
                }
            }
        }
    })

    t.Run("All-reduce operation", func(t *testing.T) {
        size := 3
        comms := make([]*MPICommunicator, size)
        for i := 0; i < size; i++ {
            c, err := InitializeMPI(size)
            if err != nil { t.Skip("MPI not available") }
            comms[i] = c
        }
        defer func() { for _, c := range comms { _ = c.Finalize() } }()

        results := make([][]float64, size)
        var wg sync.WaitGroup
        wg.Add(size)
        for r := 0; r < size; r++ {
            r := r
            go func() {
                defer wg.Done()
                local := []float64{float64(r + 1)}
                if err := comms[r].AllReduce(local, &results[r], ReduceSum); err != nil {
                    t.Errorf("rank %d allreduce failed: %v", r, err)
                }
            }()
        }
        wg.Wait()

        expected := []float64{6.0}
        for r := 0; r < size; r++ {
            if len(results[r]) != len(expected) {
                t.Errorf("rank %d: wrong result length: expected %d, got %d", r, len(expected), len(results[r]))
                continue
            }
            if results[r][0] != expected[0] {
                t.Errorf("rank %d: result[0] = %f, expected %f", r, results[r][0], expected[0])
            }
        }
    })

    t.Run("Gather operation", func(t *testing.T) {
        size := 3
        comms := make([]*MPICommunicator, size)
        for i := 0; i < size; i++ {
            c, err := InitializeMPI(size)
            if err != nil { t.Skip("MPI not available") }
            comms[i] = c
        }
        defer func() { for _, c := range comms { _ = c.Finalize() } }()

        gathered := make([][][]float64, size)
        var wg sync.WaitGroup
        wg.Add(size)
        for r := 0; r < size; r++ {
            r := r
            go func() {
                defer wg.Done()
                local := []float64{float64(r * r)}
                if err := comms[r].Gather(local, &gathered[r], 0); err != nil {
                    t.Errorf("rank %d gather failed: %v", r, err)
                }
            }()
        }
        wg.Wait()

        if len(gathered[0]) != 3 {
            t.Errorf("root: wrong gather length: expected 3, got %d", len(gathered[0]))
        }
        expected := [][]float64{{0.0}, {1.0}, {4.0}}
        for i, expectedSlice := range expected {
            if i < len(gathered[0]) {
                if len(gathered[0][i]) != len(expectedSlice) {
                    t.Errorf("gathered[%d] length wrong: expected %d, got %d", i, len(expectedSlice), len(gathered[0][i]))
                    continue
                }
                for j, expectedVal := range expectedSlice {
                    if gathered[0][i][j] != expectedVal {
                        t.Errorf("gathered[%d][%d] = %f, expected %f", i, j, gathered[0][i][j], expectedVal)
                    }
                }
            }
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
