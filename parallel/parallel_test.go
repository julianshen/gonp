package parallel

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/julianshen/gonp/array"
)

func TestWorkerPool(t *testing.T) {
	t.Run("Basic Worker Pool", func(t *testing.T) {
		pool := NewWorkerPool(4)
		defer pool.Close()

		var results []int
		var mu sync.Mutex

		// Submit jobs
		for i := 0; i < 10; i++ {
			val := i
			pool.Submit(func() {
				result := val * val
				mu.Lock()
				results = append(results, result)
				mu.Unlock()
			})
		}

		pool.Wait()

		if len(results) != 10 {
			t.Errorf("Expected 10 results, got %d", len(results))
		}

		// Verify all squares are present (order doesn't matter)
		expected := make(map[int]bool)
		for i := 0; i < 10; i++ {
			expected[i*i] = true
		}

		for _, result := range results {
			if !expected[result] {
				t.Errorf("Unexpected result: %d", result)
			}
		}
	})

	t.Run("Worker Pool with Results", func(t *testing.T) {
		pool := NewWorkerPool(2)
		defer pool.Close()

		// Test with result collection
		jobs := make([]Job, 5)
		for i := 0; i < 5; i++ {
			val := i
			jobs[i] = func() interface{} {
				return val * 2
			}
		}

		results := pool.SubmitAndCollect(jobs)

		if len(results) != 5 {
			t.Errorf("Expected 5 results, got %d", len(results))
		}

		// Results should be in order
		for i, result := range results {
			expected := i * 2
			if result.(int) != expected {
				t.Errorf("Result %d: expected %d, got %v", i, expected, result)
			}
		}
	})
}

func TestParallelArray(t *testing.T) {
	t.Run("Parallel Element-wise Operations", func(t *testing.T) {
		size := 10000
		data1 := make([]float64, size)
		data2 := make([]float64, size)

		for i := 0; i < size; i++ {
			data1[i] = float64(i)
			data2[i] = float64(i * 2)
		}

		arr1, _ := array.FromSlice(data1)
		arr2, _ := array.FromSlice(data2)

		// Test parallel addition
		start := time.Now()
		result, err := ParallelAdd(arr1, arr2, nil)
		parallelTime := time.Since(start)

		if err != nil {
			t.Fatalf("Parallel addition failed: %v", err)
		}

		if result.Size() != size {
			t.Errorf("Expected size %d, got %d", size, result.Size())
		}

		// Verify correctness
		for i := 0; i < parallelMin(100, size); i++ {
			expected := float64(i + i*2)
			actual := result.At(i).(float64)
			if math.Abs(actual-expected) > 1e-10 {
				t.Errorf("At index %d: expected %f, got %f", i, expected, actual)
			}
		}

		t.Logf("Parallel addition of %d elements took %v", size, parallelTime)
	})

	t.Run("Parallel Reduction", func(t *testing.T) {
		size := 100000
		data := make([]float64, size)
		for i := 0; i < size; i++ {
			data[i] = float64(i + 1) // 1, 2, 3, ..., 100000
		}

		arr, _ := array.FromSlice(data)

		// Test parallel sum
		start := time.Now()
		result, err := ParallelSum(arr, nil)
		parallelTime := time.Since(start)

		if err != nil {
			t.Fatalf("Parallel sum failed: %v", err)
		}

		// Expected sum: n(n+1)/2 = 100000*100001/2 = 5000050000
		expected := float64(size * (size + 1) / 2)
		if math.Abs(result-expected) > 1e-6 {
			t.Errorf("Expected sum %f, got %f", expected, result)
		}

		t.Logf("Parallel sum of %d elements took %v", size, parallelTime)
	})
}

func TestParallelMatrixOps(t *testing.T) {
	t.Run("Parallel Matrix Multiplication", func(t *testing.T) {
		// Create test matrices
		m, n, p := 100, 80, 60

		dataA := make([]float64, m*n)
		dataB := make([]float64, n*p)

		for i := 0; i < m*n; i++ {
			dataA[i] = float64(i % 10)
		}
		for i := 0; i < n*p; i++ {
			dataB[i] = float64((i % 5) + 1)
		}

		arrA, _ := array.FromSlice(dataA)
		arrA = arrA.Reshape([]int{m, n})
		arrB, _ := array.FromSlice(dataB)
		arrB = arrB.Reshape([]int{n, p})

		start := time.Now()
		result, err := ParallelMatMul(arrA, arrB, nil)
		parallelTime := time.Since(start)

		if err != nil {
			t.Fatalf("Parallel matrix multiplication failed: %v", err)
		}

		expectedShape := []int{m, p}
		if !equalSlices(result.Shape(), expectedShape) {
			t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape())
		}

		t.Logf("Parallel matrix multiplication (%dx%d)×(%dx%d) took %v", m, n, n, p, parallelTime)

		// Verify a few elements
		if result.Size() > 0 {
			firstElement := result.At(0, 0).(float64)
			if firstElement < 0 {
				t.Errorf("Unexpected negative result: %f", firstElement)
			}
		}
	})
}

func TestConcurrentSafetyOperations(t *testing.T) {
	t.Run("Thread Safety Test", func(t *testing.T) {
		size := 1000
		data := make([]float64, size)
		for i := 0; i < size; i++ {
			data[i] = float64(i)
		}

		arr, _ := array.FromSlice(data)
		numGoroutines := 10

		var wg sync.WaitGroup
		results := make([]float64, numGoroutines)

		// Run parallel sum operations
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(index int) {
				defer wg.Done()
				result, err := ParallelSum(arr, &ParallelOptions{Workers: 2})
				if err != nil {
					t.Errorf("Goroutine %d failed: %v", index, err)
					return
				}
				results[index] = result
			}(i)
		}

		wg.Wait()

		// All results should be the same
		expected := results[0]
		for i, result := range results {
			if math.Abs(result-expected) > 1e-10 {
				t.Errorf("Result %d differs: expected %f, got %f", i, expected, result)
			}
		}
	})
}

func BenchmarkParallelVsSequential(b *testing.B) {
	sizes := []int{1000, 10000, 100000}

	for _, size := range sizes {
		// Setup test data
		data := make([]float64, size)
		for i := 0; i < size; i++ {
			data[i] = float64(i) * 0.001
		}
		arr, _ := array.FromSlice(data)

		b.Run(fmt.Sprintf("Sequential_Sum_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sum := 0.0
				for j := 0; j < size; j++ {
					sum += data[j]
				}
				_ = sum
			}
		})

		b.Run(fmt.Sprintf("Parallel_Sum_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result, err := ParallelSum(arr, nil)
				if err != nil {
					b.Fatalf("Parallel sum failed: %v", err)
				}
				_ = result
			}
		})
	}
}

func BenchmarkWorkerPoolOverhead(b *testing.B) {
	pool := NewWorkerPool(runtime.NumCPU())
	defer pool.Close()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		pool.Submit(func() {
			// Minimal work to measure overhead
			_ = 1 + 1
		})
	}

	pool.Wait()
}

func TestParallelOptions(t *testing.T) {
	t.Run("Custom Parallel Options", func(t *testing.T) {
		size := 10000
		data := make([]float64, size)
		for i := 0; i < size; i++ {
			data[i] = 1.0
		}

		arr, _ := array.FromSlice(data)

		options := &ParallelOptions{
			Workers:   2,
			ChunkSize: 1000,
		}

		result, err := ParallelSum(arr, options)
		if err != nil {
			t.Fatalf("Parallel sum with options failed: %v", err)
		}

		expected := float64(size)
		if math.Abs(result-expected) > 1e-10 {
			t.Errorf("Expected sum %f, got %f", expected, result)
		}
	})

	t.Run("Adaptive Chunk Sizing", func(t *testing.T) {
		sizes := []int{100, 1000, 10000, 100000}

		for _, size := range sizes {
			data := make([]float64, size)
			for i := 0; i < size; i++ {
				data[i] = 1.0
			}

			arr, _ := array.FromSlice(data)

			options := AdaptiveParallelOptions(size)
			result, err := ParallelSum(arr, options)

			if err != nil {
				t.Errorf("Adaptive parallel sum failed for size %d: %v", size, err)
				continue
			}

			expected := float64(size)
			if math.Abs(result-expected) > 1e-10 {
				t.Errorf("Size %d: expected sum %f, got %f", size, expected, result)
			}

			t.Logf("Size %d: Workers=%d, ChunkSize=%d", size, options.Workers, options.ChunkSize)
		}
	})
}

// Helper functions

func equalSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
