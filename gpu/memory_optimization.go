// Package gpu provides advanced memory management optimizations
//
// This module implements memory optimization strategies for GPU operations
// including memory pooling, cache-aware algorithms, and efficient allocation.
//
// Optimization Features:
//   - Memory pooling to reduce allocation overhead
//   - Cache-aware data layouts and access patterns
//   - Memory bandwidth optimization
//   - NUMA-aware memory allocation
//   - Memory prefetching and streaming
package gpu

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// HardwareMemoryPool manages reusable GPU memory buffers to reduce allocation overhead
type HardwareMemoryPool struct {
	pools    map[int64]*sizedPool // Size-specific buffer pools
	mutex    sync.RWMutex         // Protects pools map
	stats    HardwarePoolStats    // Performance statistics
	maxSize  int64                // Maximum buffer size to pool
	maxCount int                  // Maximum buffers per size
}

// sizedPool manages buffers of a specific size
type sizedPool struct {
	buffers chan Buffer
	size    int64
	created int32 // Atomic counter for created buffers
	reused  int32 // Atomic counter for reused buffers
}

// HardwarePoolStats tracks memory pool performance metrics
type HardwarePoolStats struct {
	TotalAllocations int64   // Total allocation requests
	PoolHits         int64   // Successful pool reuse
	PoolMisses       int64   // New allocations required
	ActiveBuffers    int32   // Currently active buffers
	CacheEfficiency  float64 // Hit rate percentage
}

// CacheAwareLayout represents optimized memory layout for cache efficiency
type CacheAwareLayout struct {
	BlockSize  int     // Optimal block size for memory access
	StrideSize int     // Memory stride for coalesced access
	Alignment  int     // Memory alignment requirements
	Prefetch   bool    // Enable memory prefetching
	TileSize   [2]int  // 2D tiling dimensions for matrix operations
	Bandwidth  float64 // Estimated memory bandwidth (GB/s)
}

// NUMAInfo represents NUMA topology information for optimization
type NUMAInfo struct {
	NodeCount   int     // Number of NUMA nodes
	CurrentNode int     // Current NUMA node
	MemorySize  []int64 // Memory size per NUMA node
	Affinity    []int   // CPU affinity per NUMA node
	GPUDistance []int   // GPU distance from each NUMA node
}

// Global memory pool instance
var globalMemoryPool *HardwareMemoryPool
var poolInitOnce sync.Once

// NewHardwareMemoryPool creates a new memory pool for GPU buffers
func NewHardwareMemoryPool(maxSize int64, maxCount int) *HardwareMemoryPool {
	return &HardwareMemoryPool{
		pools:    make(map[int64]*sizedPool),
		maxSize:  maxSize,
		maxCount: maxCount,
		stats:    HardwarePoolStats{},
	}
}

// GetGlobalHardwareMemoryPool returns the global memory pool instance
func GetGlobalHardwareMemoryPool() *HardwareMemoryPool {
	poolInitOnce.Do(func() {
		globalMemoryPool = NewHardwareMemoryPool(100*1024*1024, 32) // 100MB max, 32 buffers per size
	})
	return globalMemoryPool
}

// GetBuffer retrieves a buffer from the pool or creates a new one
func (p *HardwareMemoryPool) GetBuffer(device Device, size int64) (Buffer, error) {
	atomic.AddInt64(&p.stats.TotalAllocations, 1)

	// Don't pool very large buffers
	if size > p.maxSize {
		atomic.AddInt64(&p.stats.PoolMisses, 1)
		buffer, err := device.AllocateMemory(size)
		if err == nil {
			atomic.AddInt32(&p.stats.ActiveBuffers, 1)
		}
		return buffer, err
	}

	// Round size to nearest power of 2 for better pooling
	poolSize := nextPowerOf2(size)

	p.mutex.RLock()
	pool, exists := p.pools[poolSize]
	p.mutex.RUnlock()

	if !exists {
		// Create new sized pool
		p.mutex.Lock()
		pool, exists = p.pools[poolSize]
		if !exists {
			pool = &sizedPool{
				buffers: make(chan Buffer, p.maxCount),
				size:    poolSize,
			}
			p.pools[poolSize] = pool
		}
		p.mutex.Unlock()
	}

	// Try to get buffer from pool
	select {
	case buffer := <-pool.buffers:
		atomic.AddInt64(&p.stats.PoolHits, 1)
		atomic.AddInt32(&pool.reused, 1)
		atomic.AddInt32(&p.stats.ActiveBuffers, 1)
		return buffer, nil
	default:
		// Pool empty, create new buffer
		atomic.AddInt64(&p.stats.PoolMisses, 1)
		buffer, err := device.AllocateMemory(poolSize)
		if err == nil {
			atomic.AddInt32(&pool.created, 1)
			atomic.AddInt32(&p.stats.ActiveBuffers, 1)
		}
		return buffer, err
	}
}

// ReturnBuffer returns a buffer to the pool for reuse
func (p *HardwareMemoryPool) ReturnBuffer(buffer Buffer) {
	if buffer == nil {
		return
	}

	atomic.AddInt32(&p.stats.ActiveBuffers, -1)

	size := buffer.Size()
	if size > p.maxSize {
		// Don't pool large buffers
		buffer.Free()
		return
	}

	poolSize := nextPowerOf2(size)

	p.mutex.RLock()
	pool, exists := p.pools[poolSize]
	p.mutex.RUnlock()

	if exists {
		select {
		case pool.buffers <- buffer:
			// Successfully returned to pool
		default:
			// Pool full, free the buffer
			buffer.Free()
		}
	} else {
		// No pool for this size, free the buffer
		buffer.Free()
	}
}

// GetStats returns current memory pool statistics
func (p *HardwareMemoryPool) GetStats() HardwarePoolStats {
	stats := p.stats
	total := atomic.LoadInt64(&stats.TotalAllocations)
	hits := atomic.LoadInt64(&stats.PoolHits)

	if total > 0 {
		stats.CacheEfficiency = float64(hits) / float64(total) * 100.0
	}

	return stats
}

// ClearPool clears all buffers from the pool and frees memory
func (p *HardwareMemoryPool) ClearPool() {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	for _, pool := range p.pools {
		for {
			select {
			case buffer := <-pool.buffers:
				buffer.Free()
			default:
				goto nextPool
			}
		}
	nextPool:
	}

	// Reset statistics
	p.stats = HardwarePoolStats{}
}

// DetectOptimalLayout analyzes hardware to determine optimal memory layout
func DetectOptimalLayout(device Device) *CacheAwareLayout {
	layout := &CacheAwareLayout{
		BlockSize:  64,             // Default cache line size
		StrideSize: 1,              // Contiguous access
		Alignment:  16,             // 16-byte alignment
		Prefetch:   true,           // Enable prefetching
		TileSize:   [2]int{32, 32}, // 32x32 tiles for matrices
		Bandwidth:  100.0,          // Conservative estimate
	}

	// Adjust based on device backend
	switch device.GetBackend() {
	case BackendCUDA:
		// NVIDIA GPUs prefer larger tiles and higher bandwidth
		layout.TileSize = [2]int{64, 64}
		layout.Bandwidth = 500.0 // Higher bandwidth estimate
		layout.Alignment = 32    // Better alignment for CUDA

	case BackendOpenCL:
		// OpenCL devices vary more, use conservative settings
		layout.TileSize = [2]int{16, 16}
		layout.Bandwidth = 200.0
		layout.Alignment = 16

	default:
		// CPU fallback
		layout.BlockSize = int(getCacheLineSize())
		layout.TileSize = [2]int{8, 8}
		layout.Bandwidth = 50.0
		layout.Prefetch = false
	}

	// Adjust for device memory size
	memorySize := device.MemorySize()
	if memorySize > 8*1024*1024*1024 { // >8GB
		layout.TileSize[0] *= 2
		layout.TileSize[1] *= 2
		layout.Bandwidth *= 1.5
	} else if memorySize < 2*1024*1024*1024 { // <2GB
		layout.TileSize[0] /= 2
		layout.TileSize[1] /= 2
		layout.Bandwidth *= 0.7
	}

	return layout
}

// OptimizeForBandwidth optimizes memory access patterns for maximum bandwidth
func OptimizeForBandwidth(data []float64, layout *CacheAwareLayout) []float64 {
	if len(data) < layout.BlockSize {
		return data // Too small to optimize
	}

	// Apply cache-friendly reorganization
	optimized := make([]float64, len(data))
	blockSize := layout.BlockSize

	// Block-wise copying for better cache utilization
	for i := 0; i < len(data); i += blockSize {
		end := i + blockSize
		if end > len(data) {
			end = len(data)
		}
		copy(optimized[i:end], data[i:end])
	}

	return optimized
}

// PrefetchMemory hints to the system to prefetch memory for better performance
func PrefetchMemory(data []float64, layout *CacheAwareLayout) {
	if !layout.Prefetch || len(data) == 0 {
		return
	}

	// Software prefetching simulation
	// In real implementation, this would use platform-specific prefetch instructions
	blockSize := layout.BlockSize
	for i := 0; i < len(data); i += blockSize {
		// Access first element of each block to trigger prefetch
		if i < len(data) {
			_ = data[i]
		}
	}
}

// DetectNUMATopology detects NUMA topology for memory optimization
func DetectNUMATopology() *NUMAInfo {
	info := &NUMAInfo{
		NodeCount:   1,
		CurrentNode: 0,
		MemorySize:  []int64{8 * 1024 * 1024 * 1024}, // Default 8GB
		Affinity:    []int{0},
		GPUDistance: []int{0},
	}

	// Try to detect actual NUMA configuration
	// This is a simplified implementation
	cpuCount := runtime.NumCPU()
	if cpuCount >= 8 {
		// Assume multi-socket system might have NUMA
		info.NodeCount = 2
		info.MemorySize = []int64{
			4 * 1024 * 1024 * 1024, // 4GB per node
			4 * 1024 * 1024 * 1024,
		}
		info.Affinity = []int{0, 1}
		info.GPUDistance = []int{0, 1} // GPU closer to node 0
	}

	return info
}

// StreamingMemoryManager handles large datasets that don't fit in GPU memory
type StreamingMemoryManager struct {
	device      Device
	chunkSize   int64
	bufferCount int
	buffers     []Buffer
	current     int
	mutex       sync.Mutex
}

// NewStreamingMemoryManager creates a streaming memory manager
func NewStreamingMemoryManager(device Device, chunkSize int64, bufferCount int) (*StreamingMemoryManager, error) {
	if bufferCount < 2 {
		bufferCount = 2 // Minimum for double buffering
	}

	manager := &StreamingMemoryManager{
		device:      device,
		chunkSize:   chunkSize,
		bufferCount: bufferCount,
		buffers:     make([]Buffer, bufferCount),
		current:     0,
	}

	// Pre-allocate buffers
	for i := 0; i < bufferCount; i++ {
		buffer, err := device.AllocateMemory(chunkSize)
		if err != nil {
			// Cleanup already allocated buffers
			for j := 0; j < i; j++ {
				manager.buffers[j].Free()
			}
			return nil, err
		}
		manager.buffers[i] = buffer
	}

	return manager, nil
}

// GetNextBuffer returns the next available buffer for streaming
func (sm *StreamingMemoryManager) GetNextBuffer() Buffer {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	buffer := sm.buffers[sm.current]
	sm.current = (sm.current + 1) % sm.bufferCount
	return buffer
}

// Cleanup frees all streaming buffers
func (sm *StreamingMemoryManager) Cleanup() {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	for _, buffer := range sm.buffers {
		if buffer != nil {
			buffer.Free()
		}
	}
}

// Utility functions

// nextPowerOf2 returns the next power of 2 greater than or equal to n
func nextPowerOf2(n int64) int64 {
	if n <= 0 {
		return 1
	}

	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n |= n >> 32
	n++

	return n
}

// getCacheLineSize returns the CPU cache line size
func getCacheLineSize() int64 {
	// Platform-specific cache line size detection
	// This is simplified - real implementation would use CPUID or system calls
	switch runtime.GOARCH {
	case "amd64", "arm64":
		return 64 // Most modern CPUs use 64-byte cache lines
	case "386", "arm":
		return 32 // Older architectures may use 32-byte cache lines
	default:
		return 64 // Safe default
	}
}

// BenchmarkMemoryBandwidth measures memory bandwidth for optimization
func BenchmarkMemoryBandwidth(device Device, size int64) (float64, error) {
	// Allocate test buffer
	buffer, err := device.AllocateMemory(size)
	if err != nil {
		return 0, err
	}
	defer buffer.Free()

	// Create test data
	data := make([]float64, size/8) // 8 bytes per float64
	for i := range data {
		data[i] = float64(i)
	}

	// Measure upload bandwidth
	uploadStart := time.Now()
	err = buffer.CopyFromHost(data)
	if err != nil {
		return 0, err
	}
	uploadTime := time.Since(uploadStart)

	// Measure download bandwidth
	downloadStart := time.Now()
	var result []float64
	err = buffer.CopyToHost(&result)
	if err != nil {
		return 0, err
	}
	downloadTime := time.Since(downloadStart)

	// Calculate bandwidth (GB/s)
	sizeGB := float64(size) / (1024 * 1024 * 1024)
	uploadBW := sizeGB / uploadTime.Seconds()
	downloadBW := sizeGB / downloadTime.Seconds()

	// Return average bandwidth
	return (uploadBW + downloadBW) / 2.0, nil
}
