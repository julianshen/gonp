package parallel

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// ParallelOptions contains options for parallel processing
type ParallelOptions struct {
	// Number of worker goroutines
	Workers int

	// Size of work chunks
	ChunkSize int

	// Timeout for operations
	Timeout time.Duration

	// Enable NUMA-aware processing
	NUMAAware bool

	// Memory threshold for enabling parallel processing
	MemoryThreshold int64
}

// DefaultParallelOptions returns default parallel processing options
func DefaultParallelOptions() *ParallelOptions {
	return &ParallelOptions{
		Workers:         runtime.NumCPU(),
		ChunkSize:       1000,
		Timeout:         30 * time.Second,
		NUMAAware:       false,
		MemoryThreshold: 1024 * 1024, // 1MB
	}
}

// AdaptiveParallelOptions returns optimized options based on data size
func AdaptiveParallelOptions(dataSize int) *ParallelOptions {
	numCPU := runtime.NumCPU()

	// Adaptive worker count
	workers := numCPU
	if dataSize < 1000 {
		workers = 1 // Sequential for small data
	} else if dataSize < 10000 {
		workers = parallelMin(2, numCPU) // Limited parallelism for medium data
	}

	// Adaptive chunk size
	chunkSize := parallelMax(100, dataSize/workers/4) // At least 100, target 4 chunks per worker
	if chunkSize > 10000 {
		chunkSize = 10000 // Cap chunk size
	}

	return &ParallelOptions{
		Workers:         workers,
		ChunkSize:       chunkSize,
		Timeout:         30 * time.Second,
		NUMAAware:       false,
		MemoryThreshold: 1024 * 1024,
	}
}

// Job represents a unit of work
type Job func() interface{}

// Task represents a unit of work without return value
type Task func()

// WorkerPool manages a pool of worker goroutines
type WorkerPool struct {
	workers    int
	taskQueue  chan Task
	jobQueue   chan Job
	resultChan chan interface{}
	wg         sync.WaitGroup
	ctx        context.Context
	cancel     context.CancelFunc
	closed     bool
	mutex      sync.RWMutex
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(workers int) *WorkerPool {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	ctx, cancel := context.WithCancel(context.Background())

	wp := &WorkerPool{
		workers:    workers,
		taskQueue:  make(chan Task, workers*2),
		jobQueue:   make(chan Job, workers*2),
		resultChan: make(chan interface{}, workers*2),
		ctx:        ctx,
		cancel:     cancel,
	}

	wp.start()
	return wp
}

// start initializes the worker goroutines
func (wp *WorkerPool) start() {
	for i := 0; i < wp.workers; i++ {
		wp.wg.Add(1)
		go wp.worker(i)
	}
}

// worker is the main worker routine
func (wp *WorkerPool) worker(id int) {
	defer wp.wg.Done()

	for {
		select {
		case task, ok := <-wp.taskQueue:
			if !ok {
				return
			}
			task()

		case job, ok := <-wp.jobQueue:
			if !ok {
				return
			}
			result := job()
			select {
			case wp.resultChan <- result:
			case <-wp.ctx.Done():
				return
			}

		case <-wp.ctx.Done():
			return
		}
	}
}

// Submit submits a task to the worker pool
func (wp *WorkerPool) Submit(task Task) error {
	wp.mutex.RLock()
	defer wp.mutex.RUnlock()

	if wp.closed {
		return fmt.Errorf("worker pool is closed")
	}

	select {
	case wp.taskQueue <- task:
		return nil
	case <-wp.ctx.Done():
		return fmt.Errorf("worker pool context cancelled")
	}
}

// SubmitJob submits a job that returns a result
func (wp *WorkerPool) SubmitJob(job Job) error {
	wp.mutex.RLock()
	defer wp.mutex.RUnlock()

	if wp.closed {
		return fmt.Errorf("worker pool is closed")
	}

	select {
	case wp.jobQueue <- job:
		return nil
	case <-wp.ctx.Done():
		return fmt.Errorf("worker pool context cancelled")
	}
}

// SubmitAndCollect submits multiple jobs and collects results in order
func (wp *WorkerPool) SubmitAndCollect(jobs []Job) []interface{} {
	results := make([]interface{}, len(jobs))
	var wg sync.WaitGroup

	for i, job := range jobs {
		wg.Add(1)
		index := i
		jobFunc := job

		wp.Submit(func() {
			defer wg.Done()
			result := jobFunc()
			results[index] = result
		})
	}

	wg.Wait()
	return results
}

// Wait waits for all submitted tasks to complete
func (wp *WorkerPool) Wait() {
	// Close task channels to signal no more work
	close(wp.taskQueue)
	close(wp.jobQueue)

	// Wait for all workers to finish
	wp.wg.Wait()
}

// Close closes the worker pool
func (wp *WorkerPool) Close() {
	wp.mutex.Lock()
	defer wp.mutex.Unlock()

	if wp.closed {
		return
	}

	wp.closed = true
	wp.cancel()

	// Close channels if not already closed
	select {
	case <-wp.taskQueue:
	default:
		close(wp.taskQueue)
	}

	select {
	case <-wp.jobQueue:
	default:
		close(wp.jobQueue)
	}

	close(wp.resultChan)
}

// Parallel Array Operations

// ParallelAdd performs element-wise addition of two arrays in parallel
func ParallelAdd(a, b *array.Array, options *ParallelOptions) (*array.Array, error) {
	if options == nil {
		options = DefaultParallelOptions()
	}

	if a.Size() != b.Size() {
		return nil, fmt.Errorf("array size mismatch: %d vs %d", a.Size(), b.Size())
	}

	ctx := internal.StartProfiler("Parallel.Add")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	size := a.Size()

	// For small arrays, use sequential processing
	if int64(size) < options.MemoryThreshold || options.Workers == 1 {
		return sequentialAdd(a, b)
	}

	// Create result array
	result := array.Zeros(a.Shape(), a.DType())

	// Create worker pool
	pool := NewWorkerPool(options.Workers)
	defer pool.Close()

	// Process in chunks
	chunkSize := options.ChunkSize
	if chunkSize <= 0 {
		chunkSize = parallelMax(1, size/options.Workers) // Ensure at least 1
	}
	var wg sync.WaitGroup

	for start := 0; start < size; start += chunkSize {
		end := parallelMin(start+chunkSize, size)

		wg.Add(1)
		pool.Submit(func() {
			defer wg.Done()

			for i := start; i < end; i++ {
				// Convert flat index to multi-dimensional indices
				indices := flatIndexToIndices(i, a.Shape())
				aVal := a.At(indices...)
				bVal := b.At(indices...)

				// Perform addition based on type
				switch aVal.(type) {
				case float64:
					sum := aVal.(float64) + bVal.(float64)
					result.Set(sum, indices...)
				case float32:
					sum := aVal.(float32) + bVal.(float32)
					result.Set(sum, indices...)
				case int64:
					sum := aVal.(int64) + bVal.(int64)
					result.Set(sum, indices...)
				case int:
					sum := aVal.(int) + bVal.(int)
					result.Set(sum, indices...)
				default:
					// Fallback to interface{} addition
					result.Set(aVal, indices...)
				}
			}
		})
	}

	wg.Wait()

	internal.DebugVerbose("Parallel addition completed: %d elements", size)
	return result, nil
}

// sequentialAdd performs sequential addition (fallback)
func sequentialAdd(a, b *array.Array) (*array.Array, error) {
	result := array.Zeros(a.Shape(), a.DType())

	for i := 0; i < a.Size(); i++ {
		indices := flatIndexToIndices(i, a.Shape())
		aVal := a.At(indices...)
		bVal := b.At(indices...)

		switch aVal.(type) {
		case float64:
			sum := aVal.(float64) + bVal.(float64)
			result.Set(sum, indices...)
		case float32:
			sum := aVal.(float32) + bVal.(float32)
			result.Set(sum, indices...)
		case int64:
			sum := aVal.(int64) + bVal.(int64)
			result.Set(sum, indices...)
		case int:
			sum := aVal.(int) + bVal.(int)
			result.Set(sum, indices...)
		default:
			result.Set(aVal, indices...)
		}
	}

	return result, nil
}

// ParallelSum computes the sum of array elements in parallel
func ParallelSum(arr *array.Array, options *ParallelOptions) (float64, error) {
	if options == nil {
		options = DefaultParallelOptions()
	}

	ctx := internal.StartProfiler("Parallel.Sum")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	size := arr.Size()
	if size == 0 {
		return 0, nil
	}

	// For small arrays, use sequential processing
	if int64(size) < options.MemoryThreshold || options.Workers == 1 {
		return sequentialSum(arr), nil
	}

	// Parallel reduction
	chunkSize := options.ChunkSize
	if chunkSize <= 0 {
		chunkSize = parallelMax(1, size/options.Workers) // Ensure at least 1
	}
	numChunks := (size + chunkSize - 1) / chunkSize

	partialSums := make([]float64, numChunks)
	var wg sync.WaitGroup

	pool := NewWorkerPool(options.Workers)
	defer pool.Close()

	for i := 0; i < numChunks; i++ {
		start := i * chunkSize
		end := parallelMin(start+chunkSize, size)
		chunkIndex := i

		wg.Add(1)
		pool.Submit(func() {
			defer wg.Done()

			sum := 0.0
			for j := start; j < end; j++ {
				indices := flatIndexToIndices(j, arr.Shape())
				val := arr.At(indices...)
				switch v := val.(type) {
				case float64:
					sum += v
				case float32:
					sum += float64(v)
				case int64:
					sum += float64(v)
				case int:
					sum += float64(v)
				case int32:
					sum += float64(v)
				}
			}
			partialSums[chunkIndex] = sum
		})
	}

	wg.Wait()

	// Final reduction
	totalSum := 0.0
	for _, partial := range partialSums {
		totalSum += partial
	}

	internal.DebugVerbose("Parallel sum completed: %d elements", size)
	return totalSum, nil
}

// sequentialSum computes sum sequentially (fallback)
func sequentialSum(arr *array.Array) float64 {
	sum := 0.0
	for i := 0; i < arr.Size(); i++ {
		indices := flatIndexToIndices(i, arr.Shape())
		val := arr.At(indices...)
		switch v := val.(type) {
		case float64:
			sum += v
		case float32:
			sum += float64(v)
		case int64:
			sum += float64(v)
		case int:
			sum += float64(v)
		case int32:
			sum += float64(v)
		}
	}
	return sum
}

// ParallelMatMul performs parallel matrix multiplication
func ParallelMatMul(a, b *array.Array, options *ParallelOptions) (*array.Array, error) {
	if options == nil {
		options = DefaultParallelOptions()
	}

	ctx := internal.StartProfiler("Parallel.MatMul")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	// Validate matrix dimensions
	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) != 2 || len(bShape) != 2 {
		return nil, fmt.Errorf("matrices must be 2D")
	}

	if aShape[1] != bShape[0] {
		return nil, fmt.Errorf("incompatible matrix dimensions: (%d,%d) × (%d,%d)",
			aShape[0], aShape[1], bShape[0], bShape[1])
	}

	m, k, n := aShape[0], aShape[1], bShape[1]

	// Create result matrix
	resultShape := []int{m, n}
	result := array.Zeros(resultShape, a.DType())

	// For small matrices, use sequential multiplication
	if int64(m*n) < options.MemoryThreshold {
		return sequentialMatMul(a, b, result, m, k, n), nil
	}

	// Parallel matrix multiplication by rows
	pool := NewWorkerPool(options.Workers)
	defer pool.Close()

	var wg sync.WaitGroup
	rowsPerWorker := parallelMax(1, m/options.Workers)

	for startRow := 0; startRow < m; startRow += rowsPerWorker {
		endRow := parallelMin(startRow+rowsPerWorker, m)

		wg.Add(1)
		pool.Submit(func() {
			defer wg.Done()

			for i := startRow; i < endRow; i++ {
				for j := 0; j < n; j++ {
					sum := 0.0

					for l := 0; l < k; l++ {
						aVal := a.At(i, l)
						bVal := b.At(l, j)

						// Convert to float64 for calculation
						aFloat := convertToFloat64(aVal)
						bFloat := convertToFloat64(bVal)

						sum += aFloat * bFloat
					}

					// Convert result back to original type
					switch a.DType() {
					case internal.Float64:
						result.Set(sum, i, j)
					case internal.Float32:
						result.Set(float32(sum), i, j)
					case internal.Int64:
						result.Set(int64(sum), i, j)
					case internal.Int32:
						result.Set(int32(sum), i, j)
					default:
						result.Set(sum, i, j)
					}
				}
			}
		})
	}

	wg.Wait()

	internal.DebugVerbose("Parallel matrix multiplication completed: (%d×%d)×(%d×%d)", m, k, k, n)
	return result, nil
}

// sequentialMatMul performs sequential matrix multiplication
func sequentialMatMul(a, b, result *array.Array, m, k, n int) *array.Array {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0

			for l := 0; l < k; l++ {
				aVal := a.At(i, l)
				bVal := b.At(l, j)

				aFloat := convertToFloat64(aVal)
				bFloat := convertToFloat64(bVal)

				sum += aFloat * bFloat
			}

			switch a.DType() {
			case internal.Float64:
				result.Set(sum, i, j)
			case internal.Float32:
				result.Set(float32(sum), i, j)
			case internal.Int64:
				result.Set(int64(sum), i, j)
			case internal.Int32:
				result.Set(int32(sum), i, j)
			default:
				result.Set(sum, i, j)
			}
		}
	}

	return result
}

// ParallelApply applies a function to each element in parallel
func ParallelApply(arr *array.Array, fn func(interface{}) interface{}, options *ParallelOptions) (*array.Array, error) {
	if options == nil {
		options = DefaultParallelOptions()
	}

	ctx := internal.StartProfiler("Parallel.Apply")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	size := arr.Size()
	if size == 0 {
		return array.Empty(arr.Shape(), arr.DType()), nil
	}

	result := array.Empty(arr.Shape(), arr.DType())

	// For small arrays, use sequential processing
	if int64(size) < options.MemoryThreshold {
		for i := 0; i < size; i++ {
			indices := flatIndexToIndices(i, arr.Shape())
			val := arr.At(indices...)
			newVal := fn(val)
			result.Set(newVal, indices...)
		}
		return result, nil
	}

	// Parallel processing
	pool := NewWorkerPool(options.Workers)
	defer pool.Close()

	chunkSize := options.ChunkSize
	if chunkSize <= 0 {
		chunkSize = parallelMax(1, size/options.Workers) // Ensure at least 1
	}
	var wg sync.WaitGroup

	for start := 0; start < size; start += chunkSize {
		end := parallelMin(start+chunkSize, size)

		wg.Add(1)
		pool.Submit(func() {
			defer wg.Done()

			for i := start; i < end; i++ {
				indices := flatIndexToIndices(i, arr.Shape())
				val := arr.At(indices...)
				newVal := fn(val)
				result.Set(newVal, indices...)
			}
		})
	}

	wg.Wait()

	internal.DebugVerbose("Parallel apply completed: %d elements", size)
	return result, nil
}

// Helper functions

// flatIndexToIndices converts a flat index to multi-dimensional indices
func flatIndexToIndices(flatIndex int, shape internal.Shape) []int {
	ndim := len(shape)
	indices := make([]int, ndim)

	for i := ndim - 1; i >= 0; i-- {
		indices[i] = flatIndex % shape[i]
		flatIndex /= shape[i]
	}

	return indices
}

// parallelMin returns the minimum of two integers (renamed to avoid conflicts)
func parallelMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// parallelMax returns the maximum of two integers
func parallelMax(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func convertToFloat64(val interface{}) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int64:
		return float64(v)
	case int:
		return float64(v)
	case int32:
		return float64(v)
	case int16:
		return float64(v)
	case int8:
		return float64(v)
	default:
		return 0.0
	}
}

// GetOptimalWorkerCount returns the optimal number of workers for given data size
func GetOptimalWorkerCount(dataSize int) int {
	numCPU := runtime.NumCPU()

	if dataSize < 1000 {
		return 1
	} else if dataSize < 10000 {
		return parallelMin(2, numCPU)
	} else if dataSize < 100000 {
		return parallelMin(numCPU/2, numCPU)
	} else {
		return numCPU
	}
}

// GetSystemInfo returns information about the system for optimization
type SystemInfo struct {
	NumCPU        int
	NumGoroutines int
	MemStats      runtime.MemStats
}

// GetSystemInfo returns current system information
func GetSystemInfo() SystemInfo {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	return SystemInfo{
		NumCPU:        runtime.NumCPU(),
		NumGoroutines: runtime.NumGoroutine(),
		MemStats:      memStats,
	}
}
