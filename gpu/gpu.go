// Package gpu provides GPU acceleration interfaces for GoNP.
//
// This package implements a unified GPU abstraction layer supporting multiple
// backends (CUDA, OpenCL) for high-performance numerical computing operations.
//
// The design follows the TDD methodology with comprehensive interface testing
// and fallback mechanisms for systems without GPU support.
//
// Key Features:
//   - Device detection and enumeration across multiple GPU backends
//   - Unified memory management with automatic pooling
//   - Asynchronous stream operations for overlapped compute and transfer
//   - Performance monitoring and automatic CPU/GPU selection
//   - Thread-safe operations with proper resource cleanup
//
// Basic Usage:
//
//	// Detect and use default GPU device
//	device, err := gpu.GetDefaultDevice()
//	if err != nil {
//		log.Printf("No GPU available: %v", err)
//		return
//	}
//
//	// Allocate GPU memory
//	buffer, err := device.AllocateMemory(1024 * 1024) // 1MB
//	if err != nil {
//		log.Fatalf("GPU allocation failed: %v", err)
//	}
//	defer buffer.Free()
//
//	// Transfer data to/from GPU
//	data := make([]float32, 256*1024)
//	err = buffer.CopyFromHost(data)
//	if err != nil {
//		log.Fatalf("Data transfer failed: %v", err)
//	}
//
// Performance Considerations:
//   - GPU operations are most efficient for large datasets (>1MB)
//   - Memory transfers have overhead - batch operations when possible
//   - Use streams for asynchronous operations and overlapping compute/transfer
//   - Memory pools reduce allocation overhead for frequent operations
package gpu

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// Device represents a GPU device with computation and memory capabilities
type Device interface {
	// Name returns the device name (e.g., "NVIDIA GeForce RTX 3080")
	Name() string

	// ComputeCapability returns the compute capability version (major, minor)
	ComputeCapability() (major, minor int)

	// MemorySize returns total device memory in bytes
	MemorySize() int64

	// AllocateMemory allocates GPU memory and returns a buffer handle
	AllocateMemory(size int64) (Buffer, error)

	// CreateStream creates a new computation stream for asynchronous operations
	CreateStream() (Stream, error)

	// Synchronize waits for all operations on the device to complete
	Synchronize() error

	// GetBackend returns the backend type (CUDA, OpenCL, etc.)
	GetBackend() BackendType

	// IsAvailable returns true if the device is currently available
	IsAvailable() bool

	// SupportsMixedPrecision returns true if the device supports FP16/FP32 mixed precision
	SupportsMixedPrecision() bool
}

// Buffer represents GPU memory allocation with data transfer capabilities
type Buffer interface {
	// Size returns the allocated buffer size in bytes
	Size() int64

	// CopyFromHost transfers data from host memory to GPU
	CopyFromHost(data interface{}) error

	// CopyToHost transfers data from GPU to host memory
	CopyToHost(data interface{}) error

	// CopyFromHostAsync transfers data asynchronously using a stream
	CopyFromHostAsync(data interface{}, stream Stream) error

	// CopyToHostAsync transfers data asynchronously using a stream
	CopyToHostAsync(data interface{}, stream Stream) error

	// Free deallocates the GPU memory (safe to call multiple times)
	Free() error

	// GetDevice returns the device this buffer belongs to
	GetDevice() Device
}

// Stream represents a GPU computation stream for asynchronous operations
type Stream interface {
	// Synchronize waits for all operations in this stream to complete
	Synchronize() error

	// IsComplete returns true if all operations in the stream are finished
	IsComplete() bool

	// Destroy releases the stream resources
	Destroy() error

	// GetDevice returns the device this stream belongs to
	GetDevice() Device
}

// MemoryPool manages GPU memory allocations with reuse and pooling
type MemoryPool interface {
	// Allocate gets a buffer from the pool or creates a new one
	Allocate(size int64) (Buffer, error)

	// Free returns a buffer to the pool for reuse
	Free(buffer Buffer) error

	// GetStats returns memory pool statistics
	GetStats() MemoryPoolStats

	// Destroy releases all pool resources
	Destroy() error
}

// MemoryPoolStats contains memory pool usage statistics
type MemoryPoolStats struct {
	TotalAllocated  int64 // Currently allocated bytes
	TotalPooled     int64 // Bytes available in pool
	AllocationCount int64 // Number of active allocations
	PoolHits        int64 // Number of pool reuses
	PoolMisses      int64 // Number of new allocations
}

// BackendType represents the GPU backend implementation
type BackendType string

const (
	BackendCUDA   BackendType = "CUDA"
	BackendOpenCL BackendType = "OpenCL"
	BackendMetal  BackendType = "Metal"
	BackendCPU    BackendType = "CPU" // Fallback for CPU-only systems
)

// DeviceInfo contains detailed device information
type DeviceInfo struct {
	Name              string
	BackendType       BackendType
	ComputeCapability [2]int
	MemorySize        int64
	MaxWorkGroupSize  int
	MaxComputeUnits   int
	Available         bool
}

// Global variables for device management
var (
	deviceMutex   sync.RWMutex
	deviceCache   []Device
	defaultDevice Device
	backendInited bool
	initError     error
)

// EnumerateDevices discovers and returns all available GPU devices
func EnumerateDevices() ([]Device, error) {
	deviceMutex.Lock()
	defer deviceMutex.Unlock()

	// Initialize backends if not already done
	if !backendInited {
		err := initializeBackends()
		if err != nil {
			initError = err
			backendInited = true
		}
	}

	if initError != nil {
		return nil, fmt.Errorf("GPU backend initialization failed: %w", initError)
	}

	// Return cached devices if available
	if deviceCache != nil {
		return deviceCache, nil
	}

	var devices []Device

	// Try CUDA backend first (typically highest performance)
	if IsCudaAvailable() {
		cudaDevices, err := enumerateCudaDevices()
		if err == nil {
			devices = append(devices, cudaDevices...)
		}
	}

	// Try OpenCL backend
	if IsOpenCLAvailable() {
		openclDevices, err := enumerateOpenCLDevices()
		if err == nil {
			devices = append(devices, openclDevices...)
		}
	}

	// Cache the results
	deviceCache = devices

	return devices, nil
}

// GetDefaultDevice returns the best available GPU device, or error if none
func GetDefaultDevice() (Device, error) {
	deviceMutex.RLock()
	if defaultDevice != nil {
		deviceMutex.RUnlock()
		return defaultDevice, nil
	}
	deviceMutex.RUnlock()

	// Find and select the best device
	devices, err := EnumerateDevices()
	if err != nil {
		return nil, fmt.Errorf("failed to enumerate devices: %w", err)
	}

	if len(devices) == 0 {
		return nil, errors.New("no GPU devices available")
	}

	// Select the device with most memory (simple heuristic)
	deviceMutex.Lock()
	defer deviceMutex.Unlock()

	if defaultDevice != nil {
		return defaultDevice, nil // Another goroutine set it
	}

	var bestDevice Device
	var maxMemory int64

	for _, device := range devices {
		if device.IsAvailable() && device.MemorySize() > maxMemory {
			maxMemory = device.MemorySize()
			bestDevice = device
		}
	}

	if bestDevice == nil {
		return nil, errors.New("no available GPU devices found")
	}

	defaultDevice = bestDevice
	return defaultDevice, nil
}

// IsCudaAvailable returns true if CUDA backend is available
func IsCudaAvailable() bool {
    // Ask the CUDA backend (build-tagged) whether devices can be detected.
    if devices, err := DetectCUDADevices(); err == nil && len(devices) > 0 {
        return true
    }
    return false
}

// IsOpenCLAvailable returns true if OpenCL backend is available
func IsOpenCLAvailable() bool {
	// Placeholder - would check for OpenCL runtime
	return false // Set to false for now since we don't have OpenCL implementation
}

// SelectOptimalBackend returns the best available GPU backend
func SelectOptimalBackend() (BackendType, error) {
	if IsCudaAvailable() {
		return BackendCUDA, nil
	}

	if IsOpenCLAvailable() {
		return BackendOpenCL, nil
	}

	return BackendCPU, errors.New("no GPU backends available, falling back to CPU")
}

// ShouldUseGPUForSize determines if GPU should be used for given data size
func ShouldUseGPUForSize(sizeBytes int64) bool {
	// GPU is typically beneficial for larger datasets due to transfer overhead
	const minGPUThreshold = 1024 * 1024 // 1MB minimum

	if sizeBytes < minGPUThreshold {
		return false
	}

	// Check if GPU is available
	_, err := GetDefaultDevice()
	return err == nil
}

// NewMemoryPool creates a new GPU memory pool for efficient allocation reuse
func NewMemoryPool(device Device, poolSizeBytes int64) (MemoryPool, error) {
	if device == nil {
		return nil, errors.New("device cannot be nil")
	}

	if poolSizeBytes <= 0 {
		return nil, errors.New("pool size must be positive")
	}

	pool := &memoryPoolImpl{
		device:    device,
		poolSize:  poolSizeBytes,
		buffers:   make(map[int64][]Buffer),
		allocated: make(map[Buffer]bool),
		stats:     MemoryPoolStats{},
	}

	return pool, nil
}

// Implementation types (simplified for now)

// cpuDevice implements Device interface for CPU fallback
type cpuDevice struct {
	name       string
	memorySize int64
	available  bool
}

func (d *cpuDevice) Name() string {
	return d.name
}

func (d *cpuDevice) ComputeCapability() (major, minor int) {
	return 0, 0 // CPU has no compute capability version
}

func (d *cpuDevice) MemorySize() int64 {
	return d.memorySize
}

func (d *cpuDevice) AllocateMemory(size int64) (Buffer, error) {
	if size <= 0 {
		return nil, errors.New("allocation size must be positive")
	}

	if size > d.memorySize {
		return nil, fmt.Errorf("requested size %d exceeds device memory %d", size, d.memorySize)
	}

	buffer := &cpuBuffer{
		size:   size,
		data:   make([]byte, size),
		device: d,
		freed:  false,
	}

	return buffer, nil
}

func (d *cpuDevice) CreateStream() (Stream, error) {
	return &cpuStream{device: d}, nil
}

func (d *cpuDevice) Synchronize() error {
	// CPU operations are always synchronous
	return nil
}

func (d *cpuDevice) GetBackend() BackendType {
	return BackendCPU
}

func (d *cpuDevice) IsAvailable() bool {
	return d.available
}

func (d *cpuDevice) SupportsMixedPrecision() bool {
	return false // CPU fallback doesn't support GPU mixed precision
}

// cpuBuffer implements Buffer interface for CPU fallback
type cpuBuffer struct {
	size   int64
	data   []byte
	device Device
	freed  bool
	mutex  sync.RWMutex
}

func (b *cpuBuffer) Size() int64 {
	return b.size
}

func (b *cpuBuffer) CopyFromHost(data interface{}) error {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	if b.freed {
		return errors.New("buffer has been freed")
	}

	if data == nil {
		return errors.New("data cannot be nil")
	}

	// Convert data to bytes and validate size
	bytes, err := interfaceToBytes(data)
	if err != nil {
		return fmt.Errorf("data conversion failed: %w", err)
	}

	if int64(len(bytes)) > b.size {
		return fmt.Errorf("data size %d exceeds buffer size %d", len(bytes), b.size)
	}

	copy(b.data, bytes)
	return nil
}

func (b *cpuBuffer) CopyToHost(data interface{}) error {
	b.mutex.RLock()
	defer b.mutex.RUnlock()

	if b.freed {
		return errors.New("buffer has been freed")
	}

	if data == nil {
		return errors.New("data cannot be nil")
	}

	// Convert buffer data back to requested type
	return bytesToInterface(b.data, data)
}

func (b *cpuBuffer) CopyFromHostAsync(data interface{}, stream Stream) error {
	// CPU operations are synchronous, so this is the same as sync copy
	return b.CopyFromHost(data)
}

func (b *cpuBuffer) CopyToHostAsync(data interface{}, stream Stream) error {
	// CPU operations are synchronous, so this is the same as sync copy
	return b.CopyToHost(data)
}

func (b *cpuBuffer) Free() error {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	if b.freed {
		return nil // Safe to call multiple times
	}

	b.freed = true
	b.data = nil
	return nil
}

func (b *cpuBuffer) GetDevice() Device {
	return b.device
}

// cpuStream implements Stream interface for CPU fallback
type cpuStream struct {
	device Device
}

func (s *cpuStream) Synchronize() error {
	// CPU operations are always synchronous
	return nil
}

func (s *cpuStream) IsComplete() bool {
	return true // CPU operations complete immediately
}

func (s *cpuStream) Destroy() error {
	// No resources to clean up for CPU
	return nil
}

func (s *cpuStream) GetDevice() Device {
	return s.device
}

// memoryPoolImpl implements MemoryPool interface
type memoryPoolImpl struct {
	device    Device
	poolSize  int64
	buffers   map[int64][]Buffer // Size -> available buffers
	allocated map[Buffer]bool    // Track allocated buffers
	stats     MemoryPoolStats
	mutex     sync.RWMutex
}

func (p *memoryPoolImpl) Allocate(size int64) (Buffer, error) {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// Check if we have a buffer of this size in the pool
	if buffers, exists := p.buffers[size]; exists && len(buffers) > 0 {
		// Reuse from pool
		buffer := buffers[len(buffers)-1]
		p.buffers[size] = buffers[:len(buffers)-1]
		p.allocated[buffer] = true
		p.stats.PoolHits++
		p.stats.AllocationCount++
		p.stats.TotalAllocated += size
		return buffer, nil
	}

	// Create new buffer
	buffer, err := p.device.AllocateMemory(size)
	if err != nil {
		return nil, err
	}

	p.allocated[buffer] = true
	p.stats.PoolMisses++
	p.stats.AllocationCount++
	p.stats.TotalAllocated += size

	return buffer, nil
}

func (p *memoryPoolImpl) Free(buffer Buffer) error {
	if buffer == nil {
		return errors.New("buffer cannot be nil")
	}

	p.mutex.Lock()
	defer p.mutex.Unlock()

	if !p.allocated[buffer] {
		return errors.New("buffer not allocated from this pool")
	}

	size := buffer.Size()
	delete(p.allocated, buffer)
	p.stats.AllocationCount--
	p.stats.TotalAllocated -= size

	// Return to pool for reuse
	p.buffers[size] = append(p.buffers[size], buffer)
	p.stats.TotalPooled += size

	return nil
}

func (p *memoryPoolImpl) GetStats() MemoryPoolStats {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	return p.stats
}

func (p *memoryPoolImpl) Destroy() error {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// Free all pooled buffers
	for _, buffers := range p.buffers {
		for _, buffer := range buffers {
			buffer.Free()
		}
	}

	// Free all allocated buffers (shouldn't happen in normal usage)
	for buffer := range p.allocated {
		buffer.Free()
	}

	p.buffers = nil
	p.allocated = nil

	return nil
}

// Helper functions

// initializeBackends sets up available GPU backends
func initializeBackends() error {
	// This would initialize CUDA/OpenCL runtime
	// For now, just return no error
	return nil
}

// enumerateCudaDevices discovers CUDA devices
func enumerateCudaDevices() ([]Device, error) {
	// Placeholder - would use CUDA runtime API
	return nil, errors.New("CUDA not implemented yet")
}

// enumerateOpenCLDevices discovers OpenCL devices
func enumerateOpenCLDevices() ([]Device, error) {
	// Placeholder - would use OpenCL API
	return nil, errors.New("OpenCL not implemented yet")
}

// interfaceToBytes converts various data types to byte slice
func interfaceToBytes(data interface{}) ([]byte, error) {
	switch v := data.(type) {
	case []byte:
		return v, nil
	case []float32:
		return (*[1 << 30]byte)(unsafe.Pointer(&v[0]))[: len(v)*4 : len(v)*4], nil
	case []float64:
		return (*[1 << 30]byte)(unsafe.Pointer(&v[0]))[: len(v)*8 : len(v)*8], nil
	case []int32:
		return (*[1 << 30]byte)(unsafe.Pointer(&v[0]))[: len(v)*4 : len(v)*4], nil
	case []int64:
		return (*[1 << 30]byte)(unsafe.Pointer(&v[0]))[: len(v)*8 : len(v)*8], nil
	default:
		return nil, fmt.Errorf("unsupported data type: %T", data)
	}
}

// bytesToInterface converts byte slice back to requested data type
func bytesToInterface(bytes []byte, data interface{}) error {
	switch v := data.(type) {
	case []byte:
		copy(v, bytes)
	case []float32:
		if len(bytes) < len(v)*4 {
			return errors.New("insufficient data for float32 slice")
		}
		src := (*[1 << 30]float32)(unsafe.Pointer(&bytes[0]))[:len(v):len(v)]
		copy(v, src)
	case []float64:
		if len(bytes) < len(v)*8 {
			return errors.New("insufficient data for float64 slice")
		}
		src := (*[1 << 30]float64)(unsafe.Pointer(&bytes[0]))[:len(v):len(v)]
		copy(v, src)
	case []int32:
		if len(bytes) < len(v)*4 {
			return errors.New("insufficient data for int32 slice")
		}
		src := (*[1 << 30]int32)(unsafe.Pointer(&bytes[0]))[:len(v):len(v)]
		copy(v, src)
	case []int64:
		if len(bytes) < len(v)*8 {
			return errors.New("insufficient data for int64 slice")
		}
		src := (*[1 << 30]int64)(unsafe.Pointer(&bytes[0]))[:len(v):len(v)]
		copy(v, src)
	default:
		return fmt.Errorf("unsupported data type: %T", data)
	}
	return nil
}

// measureTransferBandwidth measures GPU memory transfer performance
func measureTransferBandwidth(buffer Buffer, data []byte, upload bool) (float64, error) {
	const iterations = 10
	var totalTime time.Duration

	for i := 0; i < iterations; i++ {
		start := time.Now()

		if upload {
			err := buffer.CopyFromHost(data)
			if err != nil {
				return 0, err
			}
		} else {
			err := buffer.CopyToHost(data)
			if err != nil {
				return 0, err
			}
		}

		totalTime += time.Since(start)
	}

	avgTime := totalTime / iterations
	bytesPerSec := float64(len(data)) / avgTime.Seconds()
	mbPerSec := bytesPerSec / (1024 * 1024)

	return mbPerSec, nil
}

// init function creates a fallback CPU device for systems without GPU
func init() {
	// Create a CPU fallback device
	cpuFallback := &cpuDevice{
		name:       fmt.Sprintf("CPU Fallback (%d cores)", runtime.NumCPU()),
		memorySize: 1024 * 1024 * 1024, // 1GB virtual limit
		available:  true,
	}

	// Set as default if no GPU found
	deviceMutex.Lock()
	deviceCache = []Device{cpuFallback}
	deviceMutex.Unlock()
}
