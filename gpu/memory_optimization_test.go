// Package gpu provides tests for memory optimization features
//
// This module tests memory pooling, cache-aware algorithms, and streaming
// memory management using TDD methodology.

package gpu

import (
	"errors"
	"testing"
)

// TestMemoryPool tests memory pool functionality
func TestMemoryPool(t *testing.T) {
	pool := NewHardwareMemoryPool(1024*1024, 4) // 1MB max, 4 buffers per size

	// Create mock device for testing
	mockDevice := &MockGPUDevice{
		name:    "Test GPU",
		memory:  1024 * 1024 * 1024,
		backend: BackendCUDA,
		buffers: make(map[int64]*MockGPUBuffer),
	}

	t.Run("Basic buffer allocation and return", func(t *testing.T) {
		// Get buffer from empty pool
		buffer1, err := pool.GetBuffer(mockDevice, 1024)
		if err != nil {
			t.Fatalf("Failed to get buffer: %v", err)
		}

		stats := pool.GetStats()
		if stats.TotalAllocations != 1 {
			t.Errorf("Expected 1 allocation, got %d", stats.TotalAllocations)
		}
		if stats.PoolMisses != 1 {
			t.Errorf("Expected 1 miss, got %d", stats.PoolMisses)
		}

		// Return buffer to pool
		pool.ReturnBuffer(buffer1)

		// Get buffer again (should reuse)
		buffer2, err := pool.GetBuffer(mockDevice, 1024)
		if err != nil {
			t.Fatalf("Failed to reuse buffer: %v", err)
		}

		stats = pool.GetStats()
		if stats.TotalAllocations != 2 {
			t.Errorf("Expected 2 allocations, got %d", stats.TotalAllocations)
		}
		if stats.PoolHits != 1 {
			t.Errorf("Expected 1 hit, got %d", stats.PoolHits)
		}
		if stats.CacheEfficiency != 50.0 {
			t.Errorf("Expected 50%% efficiency, got %.1f%%", stats.CacheEfficiency)
		}

		pool.ReturnBuffer(buffer2)
	})

	t.Run("Power of 2 size rounding", func(t *testing.T) {
		// Request odd size, should be rounded to next power of 2
		buffer, err := pool.GetBuffer(mockDevice, 1000)
		if err != nil {
			t.Fatalf("Failed to get buffer: %v", err)
		}

		expectedSize := nextPowerOf2(1000) // Should be 1024
		if buffer.Size() != expectedSize {
			t.Errorf("Expected buffer size %d, got %d", expectedSize, buffer.Size())
		}

		pool.ReturnBuffer(buffer)
	})

	t.Run("Large buffer handling", func(t *testing.T) {
		// Request buffer larger than max pool size
		largeBuffer, err := pool.GetBuffer(mockDevice, 2*1024*1024) // 2MB > 1MB max
		if err != nil {
			t.Fatalf("Failed to get large buffer: %v", err)
		}

		stats := pool.GetStats()
		initialMisses := stats.PoolMisses

		// Return large buffer (should not be pooled)
		pool.ReturnBuffer(largeBuffer)

		// Get another large buffer (should create new one)
		largeBuffer2, err := pool.GetBuffer(mockDevice, 2*1024*1024)
		if err != nil {
			t.Fatalf("Failed to get second large buffer: %v", err)
		}

		stats = pool.GetStats()
		if stats.PoolMisses != initialMisses+1 {
			t.Errorf("Large buffer should not be pooled")
		}

		pool.ReturnBuffer(largeBuffer2)
	})

	t.Run("Pool capacity limits", func(t *testing.T) {
		// Fill pool to capacity
		size := int64(512)
		buffers := make([]Buffer, 6) // More than max count (4)

		for i := 0; i < 6; i++ {
			buffer, err := pool.GetBuffer(mockDevice, size)
			if err != nil {
				t.Fatalf("Failed to get buffer %d: %v", i, err)
			}
			buffers[i] = buffer
		}

		// Return all buffers (only 4 should be pooled)
		for _, buffer := range buffers {
			pool.ReturnBuffer(buffer)
		}

		// Get 4 buffers (should all be reused)
		hitsBefore := pool.GetStats().PoolHits
		for i := 0; i < 4; i++ {
			buffer, err := pool.GetBuffer(mockDevice, size)
			if err != nil {
				t.Fatalf("Failed to reuse buffer %d: %v", i, err)
			}
			pool.ReturnBuffer(buffer)
		}

		hitsAfter := pool.GetStats().PoolHits
		if hitsAfter-hitsBefore != 4 {
			t.Errorf("Expected 4 pool hits, got %d", hitsAfter-hitsBefore)
		}
	})

	// Cleanup
	pool.ClearPool()
}

// TestCacheAwareLayout tests cache-aware memory layout optimization
func TestCacheAwareLayout(t *testing.T) {
	mockDevice := &MockGPUDevice{
		name:    "Test GPU",
		memory:  4 * 1024 * 1024 * 1024,
		backend: BackendCUDA,
	}

	t.Run("CUDA layout optimization", func(t *testing.T) {
		layout := DetectOptimalLayout(mockDevice)

		if layout.TileSize[0] != 64 || layout.TileSize[1] != 64 {
			t.Errorf("Expected 64x64 tiles for CUDA, got %dx%d",
				layout.TileSize[0], layout.TileSize[1])
		}

		if layout.Bandwidth != 500.0 {
			t.Errorf("Expected 500.0 bandwidth for CUDA, got %.1f", layout.Bandwidth)
		}

		if layout.Alignment != 32 {
			t.Errorf("Expected 32-byte alignment for CUDA, got %d", layout.Alignment)
		}
	})

	t.Run("OpenCL layout optimization", func(t *testing.T) {
		mockDevice.backend = BackendOpenCL
		layout := DetectOptimalLayout(mockDevice)

		if layout.TileSize[0] != 16 || layout.TileSize[1] != 16 {
			t.Errorf("Expected 16x16 tiles for OpenCL, got %dx%d",
				layout.TileSize[0], layout.TileSize[1])
		}

		if layout.Bandwidth != 200.0 {
			t.Errorf("Expected 200.0 bandwidth for OpenCL, got %.1f", layout.Bandwidth)
		}
	})

	t.Run("Memory size adjustment", func(t *testing.T) {
		// Test with small memory device
		smallDevice := &MockGPUDevice{
			name:    "Small GPU",
			memory:  1 * 1024 * 1024 * 1024, // 1GB
			backend: BackendCUDA,
		}

		layout := DetectOptimalLayout(smallDevice)
		if layout.TileSize[0] != 32 || layout.TileSize[1] != 32 {
			t.Errorf("Expected reduced tile size for small memory, got %dx%d",
				layout.TileSize[0], layout.TileSize[1])
		}
	})
}

// TestBandwidthOptimization tests memory bandwidth optimization
func TestBandwidthOptimization(t *testing.T) {
	layout := &CacheAwareLayout{
		BlockSize:  64,
		StrideSize: 1,
		Alignment:  16,
		Prefetch:   true,
		TileSize:   [2]int{32, 32},
		Bandwidth:  100.0,
	}

	t.Run("Block-wise optimization", func(t *testing.T) {
		data := make([]float64, 1000)
		for i := range data {
			data[i] = float64(i)
		}

		optimized := OptimizeForBandwidth(data, layout)

		if len(optimized) != len(data) {
			t.Errorf("Optimized data size mismatch: expected %d, got %d",
				len(data), len(optimized))
		}

		// Verify data integrity
		for i, v := range optimized {
			if v != data[i] {
				t.Errorf("Data corruption at index %d: expected %.1f, got %.1f",
					i, data[i], v)
				break
			}
		}
	})

	t.Run("Small data handling", func(t *testing.T) {
		smallData := []float64{1, 2, 3}
		optimized := OptimizeForBandwidth(smallData, layout)

		// Should return original data for small arrays
		if len(optimized) != len(smallData) {
			t.Errorf("Small data should not be modified")
		}
	})

	t.Run("Memory prefetching", func(t *testing.T) {
		data := make([]float64, 1000)
		for i := range data {
			data[i] = float64(i)
		}

		// Should not panic or error
		PrefetchMemory(data, layout)

		// Test with prefetch disabled
		layout.Prefetch = false
		PrefetchMemory(data, layout)

		// Test with empty data
		PrefetchMemory([]float64{}, layout)
	})
}

// TestStreamingMemoryManager tests streaming memory management
func TestStreamingMemoryManager(t *testing.T) {
	mockDevice := &MockGPUDevice{
		name:    "Test GPU",
		memory:  1024 * 1024 * 1024,
		backend: "Mock",
		buffers: make(map[int64]*MockGPUBuffer),
	}

	t.Run("Basic streaming operations", func(t *testing.T) {
		chunkSize := int64(1024 * 1024) // 1MB chunks
		bufferCount := 3

		manager, err := NewStreamingMemoryManager(mockDevice, chunkSize, bufferCount)
		if err != nil {
			t.Fatalf("Failed to create streaming manager: %v", err)
		}
		defer manager.Cleanup()

		// Get buffers in round-robin fashion
		for i := 0; i < bufferCount*2; i++ {
			buffer := manager.GetNextBuffer()
			if buffer == nil {
				t.Errorf("Got nil buffer at iteration %d", i)
			}

			if buffer.Size() != chunkSize {
				t.Errorf("Expected buffer size %d, got %d", chunkSize, buffer.Size())
			}
		}
	})

	t.Run("Minimum buffer count", func(t *testing.T) {
		// Request 1 buffer, should get 2 (minimum for double buffering)
		manager, err := NewStreamingMemoryManager(mockDevice, 1024, 1)
		if err != nil {
			t.Fatalf("Failed to create streaming manager: %v", err)
		}
		defer manager.Cleanup()

		if len(manager.buffers) != 2 {
			t.Errorf("Expected minimum 2 buffers, got %d", len(manager.buffers))
		}
	})

	t.Run("Allocation failure handling", func(t *testing.T) {
		// Create device that fails allocation
		failDevice := &MockGPUDevice{
			name:      "Fail GPU",
			memory:    1024,
			backend:   BackendCPU,
			buffers:   make(map[int64]*MockGPUBuffer),
			failAlloc: true,
		}

		_, err := NewStreamingMemoryManager(failDevice, 1024, 2)
		if err == nil {
			t.Errorf("Expected error from failing device")
		}
	})
}

// TestNUMADetection tests NUMA topology detection
func TestNUMADetection(t *testing.T) {
	info := DetectNUMATopology()

	if info == nil {
		t.Fatalf("NUMA info should not be nil")
	}

	if info.NodeCount < 1 {
		t.Errorf("Should have at least 1 NUMA node, got %d", info.NodeCount)
	}

	if len(info.MemorySize) != info.NodeCount {
		t.Errorf("Memory size array length mismatch: nodes=%d, memory=%d",
			info.NodeCount, len(info.MemorySize))
	}

	if len(info.Affinity) != info.NodeCount {
		t.Errorf("Affinity array length mismatch: nodes=%d, affinity=%d",
			info.NodeCount, len(info.Affinity))
	}

	totalMemory := int64(0)
	for _, mem := range info.MemorySize {
		totalMemory += mem
	}

	if totalMemory == 0 {
		t.Errorf("Total memory should be greater than 0")
	}
}

// TestUtilityFunctions tests utility functions
func TestUtilityFunctions(t *testing.T) {
	t.Run("nextPowerOf2", func(t *testing.T) {
		tests := []struct {
			input    int64
			expected int64
		}{
			{0, 1},
			{1, 1},
			{2, 2},
			{3, 4},
			{7, 8},
			{8, 8},
			{9, 16},
			{1023, 1024},
			{1024, 1024},
			{1025, 2048},
		}

		for _, test := range tests {
			result := nextPowerOf2(test.input)
			if result != test.expected {
				t.Errorf("nextPowerOf2(%d) = %d, expected %d",
					test.input, result, test.expected)
			}
		}
	})

	t.Run("getCacheLineSize", func(t *testing.T) {
		size := getCacheLineSize()

		// Should be a reasonable cache line size
		if size != 32 && size != 64 {
			t.Errorf("Unexpected cache line size: %d", size)
		}
	})
}

// TestGlobalMemoryPool tests the global memory pool instance
func TestGlobalMemoryPool(t *testing.T) {
	pool1 := GetGlobalHardwareMemoryPool()
	pool2 := GetGlobalHardwareMemoryPool()

	if pool1 != pool2 {
		t.Errorf("Global memory pool should be a singleton")
	}

	if pool1 == nil {
		t.Errorf("Global memory pool should not be nil")
	}

	// Test basic functionality
	stats := pool1.GetStats()
	if stats.CacheEfficiency < 0 || stats.CacheEfficiency > 100 {
		t.Errorf("Cache efficiency should be between 0-100%%, got %.1f%%", stats.CacheEfficiency)
	}
}

// Mock implementation for testing

type MockGPUDevice struct {
	name      string
	memory    int64
	backend   BackendType
	buffers   map[int64]*MockGPUBuffer
	failAlloc bool
}

func (d *MockGPUDevice) Name() string {
	return d.name
}

func (d *MockGPUDevice) ComputeCapability() (major, minor int) {
	return 3, 5
}

func (d *MockGPUDevice) MemorySize() int64 {
	return d.memory
}

func (d *MockGPUDevice) IsAvailable() bool {
	return true
}

func (d *MockGPUDevice) GetBackend() BackendType {
	return d.backend
}

func (d *MockGPUDevice) SupportsMixedPrecision() bool {
	return true
}

func (d *MockGPUDevice) AllocateMemory(size int64) (Buffer, error) {
	if d.failAlloc {
		return nil, errors.New("Mock allocation failure")
	}

	buffer := &MockGPUBuffer{
		size:   size,
		device: d,
		data:   make([]byte, size),
	}
	d.buffers[size] = buffer
	return buffer, nil
}

func (d *MockGPUDevice) CreateStream() (Stream, error) {
	return &MockGPUStream{device: d}, nil
}

func (d *MockGPUDevice) Synchronize() error {
	return nil
}

type MockGPUBuffer struct {
	size   int64
	device *MockGPUDevice
	data   []byte
	freed  bool
}

func (b *MockGPUBuffer) Size() int64 {
	return b.size
}

func (b *MockGPUBuffer) CopyFromHost(data interface{}) error {
	if b.freed {
		return errors.New("Buffer already freed")
	}
	return nil
}

func (b *MockGPUBuffer) GetDevice() Device {
	return b.device
}

func (b *MockGPUBuffer) CopyFromHostAsync(data interface{}, stream Stream) error {
	return b.CopyFromHost(data)
}

func (b *MockGPUBuffer) CopyToHostAsync(data interface{}, stream Stream) error {
	return b.CopyToHost(data)
}

func (b *MockGPUBuffer) CopyToHost(data interface{}) error {
	if b.freed {
		return errors.New("Buffer already freed")
	}
	return nil
}

func (b *MockGPUBuffer) Free() error {
	b.freed = true
	return nil
}

type MockGPUStream struct {
	device Device
}

func (s *MockGPUStream) Synchronize() error {
	return nil
}

func (s *MockGPUStream) IsComplete() bool {
	return true
}

func (s *MockGPUStream) Destroy() error {
	return nil
}

func (s *MockGPUStream) GetDevice() Device {
	return s.device
}
