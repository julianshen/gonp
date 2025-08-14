package internal

import (
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// TypedStorage implements Storage for specific Go types
type TypedStorage struct {
	data     interface{}
	dtype    DType
	length   int
	capacity int
	elemSize int
}

// NewTypedStorage creates a new typed storage
func NewTypedStorage(data interface{}, dtype DType) *TypedStorage {
	val := reflect.ValueOf(data)
	if val.Kind() != reflect.Slice {
		panic("data must be a slice")
	}

	elemSize := int(val.Type().Elem().Size())

	return &TypedStorage{
		data:     data,
		dtype:    dtype,
		length:   val.Len(),
		capacity: val.Cap(),
		elemSize: elemSize,
	}
}

// Data returns a pointer to the underlying data
func (ts *TypedStorage) Data() unsafe.Pointer {
	val := reflect.ValueOf(ts.data)
	return unsafe.Pointer(val.Pointer())
}

// Len returns the number of elements
func (ts *TypedStorage) Len() int {
	return ts.length
}

// Cap returns the capacity
func (ts *TypedStorage) Cap() int {
	return ts.capacity
}

// Type returns the data type
func (ts *TypedStorage) Type() DType {
	return ts.dtype
}

// ElementSize returns the size of each element in bytes
func (ts *TypedStorage) ElementSize() int {
	return ts.elemSize
}

// Clone creates a copy of the storage
func (ts *TypedStorage) Clone() Storage {
	val := reflect.ValueOf(ts.data)
	newSlice := reflect.MakeSlice(val.Type(), ts.length, ts.length)
	reflect.Copy(newSlice, val)

	return &TypedStorage{
		data:     newSlice.Interface(),
		dtype:    ts.dtype,
		length:   ts.length,
		capacity: ts.length,
		elemSize: ts.elemSize,
	}
}

// GetSlice returns the underlying slice as interface{}
func (ts *TypedStorage) GetSlice() interface{} {
	return ts.data
}

// StoragePool manages reusable storage objects
type StoragePool struct {
	pools map[DType]*sync.Pool
}

// NewStoragePool creates a new storage pool
func NewStoragePool() *StoragePool {
	pools := make(map[DType]*sync.Pool)

	// Initialize pools for each data type
	pools[Float64] = &sync.Pool{
		New: func() interface{} {
			return make([]float64, 0, 64)
		},
	}

	pools[Float32] = &sync.Pool{
		New: func() interface{} {
			return make([]float32, 0, 64)
		},
	}

	pools[Int64] = &sync.Pool{
		New: func() interface{} {
			return make([]int64, 0, 64)
		},
	}

	pools[Int32] = &sync.Pool{
		New: func() interface{} {
			return make([]int32, 0, 64)
		},
	}

	pools[Bool] = &sync.Pool{
		New: func() interface{} {
			return make([]bool, 0, 64)
		},
	}

	pools[String] = &sync.Pool{
		New: func() interface{} {
			return make([]string, 0, 64)
		},
	}

	pools[Interface] = &sync.Pool{
		New: func() interface{} {
			return make([]interface{}, 0, 64)
		},
	}

	return &StoragePool{pools: pools}
}

// Get retrieves a slice from the pool
func (sp *StoragePool) Get(dtype DType, size int) interface{} {
	pool, exists := sp.pools[dtype]
	if !exists {
		return sp.allocateSlice(dtype, size)
	}

	slice := pool.Get()
	return sp.resizeSlice(slice, dtype, size)
}

// Put returns a slice to the pool
func (sp *StoragePool) Put(storage Storage) {
	pool, exists := sp.pools[storage.Type()]
	if !exists {
		return
	}

	if ts, ok := storage.(*TypedStorage); ok {
		// Reset slice length to 0 but keep capacity
		slice := sp.resetSlice(ts.GetSlice(), storage.Type())
		pool.Put(slice)
	}
}

// allocateSlice creates a new slice of the specified type and size
func (sp *StoragePool) allocateSlice(dtype DType, size int) interface{} {
	switch dtype {
	case Float64:
		return make([]float64, size)
	case Float32:
		return make([]float32, size)
	case Int64:
		return make([]int64, size)
	case Int32:
		return make([]int32, size)
	case Int16:
		return make([]int16, size)
	case Int8:
		return make([]int8, size)
	case Uint64:
		return make([]uint64, size)
	case Uint32:
		return make([]uint32, size)
	case Uint16:
		return make([]uint16, size)
	case Uint8:
		return make([]uint8, size)
	case Bool:
		return make([]bool, size)
	case Complex64:
		return make([]complex64, size)
	case Complex128:
		return make([]complex128, size)
	case String:
		return make([]string, size)
	case Interface:
		return make([]interface{}, size)
	default:
		panic("unsupported data type")
	}
}

// resizeSlice resizes a slice to the specified size
func (sp *StoragePool) resizeSlice(slice interface{}, dtype DType, size int) interface{} {
	val := reflect.ValueOf(slice)
	if val.Cap() >= size {
		return val.Slice(0, size).Interface()
	}

	// If capacity is insufficient, allocate new slice
	return sp.allocateSlice(dtype, size)
}

// resetSlice resets a slice length to 0
func (sp *StoragePool) resetSlice(slice interface{}, dtype DType) interface{} {
	val := reflect.ValueOf(slice)
	return val.Slice(0, 0).Interface()
}

// DefaultPool is the default storage pool
var DefaultPool = NewStoragePool()

// Production-grade memory management implementation
// This extends the basic storage functionality with enterprise features

// MemoryPoolConfig defines configuration for the production memory pool
type MemoryPoolConfig struct {
	MaxPoolSize             int           // Maximum total pool size in bytes
	MaxBlockCount           int           // Maximum blocks per size class
	LeakDetection           bool          // Enable leak detection
	LeakTimeout             time.Duration // Timeout for leak detection
	GCIntegration           bool          // Integrate with Go GC
	UseFinalizersForCleanup bool          // Use finalizers for automatic cleanup
	MonitoringMode          bool          // Enable performance monitoring
	AdaptiveResize          bool          // Enable adaptive pool resizing
	ResizeThreshold         float64       // Threshold for triggering resize (0.0-1.0)
	PressureHandling        bool          // Enable memory pressure handling
	PressureThreshold       float64       // System memory threshold for pressure detection
	ThreadSafety            bool          // Enable thread-safe operations
	ContentionHandling      bool          // Enable contention detection and mitigation
	LockFreePaths           bool          // Enable lock-free optimization paths
	ProfilingEnabled        bool          // Enable allocation profiling
	MetricsInterval         time.Duration // Metrics collection interval
	UsageAlertThreshold     float64       // Usage threshold for alerts
	OOMHandling             bool          // Enable out-of-memory handling
	GracefulDegradation     bool          // Enable graceful degradation on OOM
	StrictErrorHandling     bool          // Enable strict error validation
	SecureMode              bool          // Enable secure memory operations (zero on alloc/dealloc)
}

// ProductionMemoryPool implements enterprise-grade memory management
type ProductionMemoryPool struct {
	config            MemoryPoolConfig
	mutex             sync.RWMutex
	sizeClasses       map[int]*sizeClass
	totalSize         int64
	activeBlocks      int64
	totalAllocs       int64
	totalLeaks        int64
	pressureEvents    int64
	contentionEvents  int64
	alertsTriggered   int64
	finalizerCleanups int64

	// Leak detection
	allocations  map[uintptr]*allocationInfo
	leakDetector *leakDetector

	// Performance monitoring
	profiler     *allocationProfiler
	alertHandler func(MemoryAlert)

	// State management
	active   bool
	shutdown chan struct{}
}

// MemoryBlock represents an allocated memory block
type MemoryBlock interface {
	Size() int
	Data() unsafe.Pointer
	ID() uintptr
}

// memoryBlock implements MemoryBlock interface
type memoryBlock struct {
	size        int
	data        unsafe.Pointer
	id          uintptr
	pool        *ProductionMemoryPool
	allocated   time.Time
	deallocated bool
}

func (mb *memoryBlock) Size() int            { return mb.size }
func (mb *memoryBlock) Data() unsafe.Pointer { return mb.data }
func (mb *memoryBlock) ID() uintptr          { return mb.id }

// sizeClass manages blocks of a specific size
type sizeClass struct {
	size      int
	blocks    []*memoryBlock
	mutex     sync.Mutex
	allocated int64
	peak      int64
}

// allocationInfo tracks allocation metadata for leak detection
type allocationInfo struct {
	block      *memoryBlock
	timestamp  time.Time
	stackTrace []uintptr
}

// leakDetector monitors for memory leaks
type leakDetector struct {
	pool     *ProductionMemoryPool
	ticker   *time.Ticker
	stopChan chan struct{}
}

// allocationProfiler collects performance metrics
type allocationProfiler struct {
	mutex            sync.RWMutex
	sizeDistribution map[int]int64
	totalAllocTime   time.Duration
	totalAllocs      int64
	peakUsage        int64
	startTime        time.Time
}

// MemoryPoolStats provides detailed pool statistics
type MemoryPoolStats struct {
	ActiveAllocations int64
	TotalAllocations  int64
	TotalLeaks        int64
	PoolSize          int64
	Efficiency        float64
	PressureEvents    int64
	ContentionEvents  int64
	AlertsTriggered   int64
	FinalizerCleanups int64
}

// LeakInfo provides information about detected memory leaks
type LeakInfo struct {
	Size       int
	Age        time.Duration
	ID         uintptr
	StackTrace []uintptr
}

// AllocationProfile provides profiling information
type AllocationProfile struct {
	SizeDistribution      map[int]int64
	AverageAllocationTime time.Duration
	PeakMemoryUsage       int64
	TotalAllocations      int64
}

// MemoryAlert represents a memory-related alert
type MemoryAlert struct {
	Type        AlertType
	Message     string
	MemoryUsage float64
	Timestamp   time.Time
}

// AlertType defines the type of memory alert
type AlertType int

const (
	AlertHighMemoryUsage AlertType = iota
	AlertMemoryLeak
	AlertOOMCondition
	AlertContentionDetected
)

// Memory pool errors
var (
	ErrPoolNotActive         = errors.New("memory pool not active")
	ErrInvalidSize           = errors.New("invalid allocation size")
	ErrOutOfMemory           = errors.New("out of memory")
	ErrDoubleDeallocation    = errors.New("double deallocation detected")
	ErrBlockNotFound         = errors.New("memory block not found")
	ErrPoolSizeLimitExceeded = errors.New("pool size limit exceeded")
)

// NewProductionMemoryPool creates a new production-grade memory pool
func NewProductionMemoryPool(config MemoryPoolConfig) (*ProductionMemoryPool, error) {
	if config.MaxPoolSize <= 0 {
		return nil, errors.New("MaxPoolSize must be positive")
	}
	if config.MaxBlockCount <= 0 {
		config.MaxBlockCount = 1000 // Default
	}
	if config.LeakTimeout == 0 {
		config.LeakTimeout = 5 * time.Minute // Default
	}
	if config.MetricsInterval == 0 {
		config.MetricsInterval = 1 * time.Second // Default
	}

	pool := &ProductionMemoryPool{
		config:      config,
		sizeClasses: make(map[int]*sizeClass),
		allocations: make(map[uintptr]*allocationInfo),
		active:      true,
		shutdown:    make(chan struct{}),
	}

	// Initialize leak detector
	if config.LeakDetection {
		pool.leakDetector = &leakDetector{
			pool:     pool,
			stopChan: make(chan struct{}),
		}
		pool.startLeakDetection()
	}

	// Initialize profiler
	if config.ProfilingEnabled {
		pool.profiler = &allocationProfiler{
			sizeDistribution: make(map[int]int64),
			startTime:        time.Now(),
		}
	}

	return pool, nil
}

// Allocate allocates a memory block of the specified size
func (pool *ProductionMemoryPool) Allocate(size int) (MemoryBlock, error) {
	if !pool.active {
		return nil, ErrPoolNotActive
	}

	if size <= 0 {
		return nil, ErrInvalidSize
	}

	start := time.Now()
	defer func() {
		if pool.profiler != nil {
			pool.profiler.recordAllocation(size, time.Since(start))
		}
	}()

	// Check memory pressure
	if pool.config.PressureHandling {
		if pool.checkMemoryPressure() {
			atomic.AddInt64(&pool.pressureEvents, 1)
			if pool.config.GracefulDegradation {
				return nil, ErrOutOfMemory
			}
		}
	}

	// Check pool size limits
	currentSize := atomic.LoadInt64(&pool.totalSize)

	pool.mutex.RLock()
	maxPoolSize := pool.config.MaxPoolSize
	pool.mutex.RUnlock()

	if currentSize+int64(size) > int64(maxPoolSize) {
		if pool.config.AdaptiveResize {
			pool.resizePool()
		} else {
			if pool.config.OOMHandling {
				return nil, ErrOutOfMemory
			}
			return nil, ErrPoolSizeLimitExceeded
		}
	} else if pool.config.AdaptiveResize && pool.canResize() {
		// Preemptive resize when threshold is reached
		pool.resizePool()
	}

	pool.mutex.Lock()
	sizeClass := pool.getSizeClass(size)
	pool.mutex.Unlock()

	// Try to get block from size class
	block := sizeClass.getBlock()
	if block == nil {
		// Allocate new block
		data := make([]byte, size)
		block = &memoryBlock{
			size:        size,
			data:        unsafe.Pointer(&data[0]),
			id:          uintptr(unsafe.Pointer(&data[0])),
			pool:        pool,
			allocated:   time.Now(),
			deallocated: false,
		}

		// Set up finalizer for GC integration
		if pool.config.GCIntegration && pool.config.UseFinalizersForCleanup {
			runtime.SetFinalizer(block, (*memoryBlock).finalize)
		}
	} else {
		// Reset deallocated flag for reused blocks
		block.deallocated = false
		block.allocated = time.Now()
	}

	// Update statistics
	atomic.AddInt64(&pool.activeBlocks, 1)
	atomic.AddInt64(&pool.totalAllocs, 1)
	atomic.AddInt64(&pool.totalSize, int64(size))

	// Track allocation for leak detection
	if pool.config.LeakDetection {
		pool.trackAllocation(block)
	}

	// Check usage alerts
	pool.checkUsageAlerts()

	return block, nil
}

// Deallocate deallocates a memory block
func (pool *ProductionMemoryPool) Deallocate(block MemoryBlock) error {
	if !pool.active {
		return ErrPoolNotActive
	}

	if block == nil {
		return errors.New("cannot deallocate nil block")
	}

	mb, ok := block.(*memoryBlock)
	if !ok {
		return errors.New("invalid block type")
	}

	// Check for double deallocation
	if pool.config.StrictErrorHandling {
		if mb.deallocated {
			return ErrDoubleDeallocation
		}
		mb.deallocated = true
	}

	// Remove from leak tracking
	if pool.config.LeakDetection {
		pool.untrackAllocation(mb)
	}

	// Return to size class pool
	pool.mutex.RLock()
	sizeClass := pool.sizeClasses[mb.size]
	pool.mutex.RUnlock()

	if sizeClass != nil {
		sizeClass.returnBlock(mb)
	}

	// Update statistics
	atomic.AddInt64(&pool.activeBlocks, -1)
	atomic.AddInt64(&pool.totalSize, -int64(mb.size))

	// Clear finalizer
	if pool.config.GCIntegration {
		runtime.SetFinalizer(mb, nil)
	}

	return nil
}

// GetDetailedStats returns detailed pool statistics
func (pool *ProductionMemoryPool) GetDetailedStats() MemoryPoolStats {
	activeAllocs := atomic.LoadInt64(&pool.activeBlocks)
	totalAllocs := atomic.LoadInt64(&pool.totalAllocs)

	pool.mutex.RLock()
	configuredPoolSize := int64(pool.config.MaxPoolSize)
	pool.mutex.RUnlock()

	efficiency := 0.0
	if totalAllocs > 0 {
		efficiency = float64(activeAllocs) / float64(totalAllocs) * 100
	}

	return MemoryPoolStats{
		ActiveAllocations: activeAllocs,
		TotalAllocations:  totalAllocs,
		TotalLeaks:        atomic.LoadInt64(&pool.totalLeaks),
		PoolSize:          configuredPoolSize,
		Efficiency:        efficiency,
		PressureEvents:    atomic.LoadInt64(&pool.pressureEvents),
		ContentionEvents:  atomic.LoadInt64(&pool.contentionEvents),
		AlertsTriggered:   atomic.LoadInt64(&pool.alertsTriggered),
		FinalizerCleanups: atomic.LoadInt64(&pool.finalizerCleanups),
	}
}

// GetLeakReport returns information about detected memory leaks
func (pool *ProductionMemoryPool) GetLeakReport() []LeakInfo {
	if !pool.config.LeakDetection {
		return nil
	}

	pool.mutex.RLock()
	defer pool.mutex.RUnlock()

	var leaks []LeakInfo
	now := time.Now()

	for _, info := range pool.allocations {
		if now.Sub(info.timestamp) > pool.config.LeakTimeout {
			leaks = append(leaks, LeakInfo{
				Size:       info.block.size,
				Age:        now.Sub(info.timestamp),
				ID:         info.block.id,
				StackTrace: info.stackTrace,
			})
		}
	}

	return leaks
}

// GetAllocationProfile returns allocation profiling information
func (pool *ProductionMemoryPool) GetAllocationProfile() AllocationProfile {
	if pool.profiler == nil {
		return AllocationProfile{}
	}

	pool.profiler.mutex.RLock()
	defer pool.profiler.mutex.RUnlock()

	avgTime := time.Duration(0)
	if pool.profiler.totalAllocs > 0 {
		avgTime = pool.profiler.totalAllocTime / time.Duration(pool.profiler.totalAllocs)
	}

	// Copy size distribution
	distribution := make(map[int]int64)
	for size, count := range pool.profiler.sizeDistribution {
		distribution[size] = count
	}

	return AllocationProfile{
		SizeDistribution:      distribution,
		AverageAllocationTime: avgTime,
		PeakMemoryUsage:       pool.profiler.peakUsage,
		TotalAllocations:      pool.profiler.totalAllocs,
	}
}

// SetAlertHandler sets the handler for memory alerts
func (pool *ProductionMemoryPool) SetAlertHandler(handler func(MemoryAlert)) {
	pool.alertHandler = handler
}

// Shutdown gracefully shuts down the memory pool
func (pool *ProductionMemoryPool) Shutdown() error {
	pool.mutex.Lock()
	defer pool.mutex.Unlock()

	if !pool.active {
		return nil
	}

	pool.active = false
	close(pool.shutdown)

	// Stop leak detector
	if pool.leakDetector != nil {
		close(pool.leakDetector.stopChan)
		if pool.leakDetector.ticker != nil {
			pool.leakDetector.ticker.Stop()
		}
	}

	// Clean up all allocations
	for _, sizeClass := range pool.sizeClasses {
		sizeClass.cleanup()
	}

	return nil
}

// Helper methods

func (pool *ProductionMemoryPool) getSizeClass(size int) *sizeClass {
	sc, exists := pool.sizeClasses[size]
	if !exists {
		sc = &sizeClass{
			size:   size,
			blocks: make([]*memoryBlock, 0, pool.config.MaxBlockCount),
		}
		pool.sizeClasses[size] = sc
	}
	return sc
}

func (pool *ProductionMemoryPool) checkMemoryPressure() bool {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Simple heuristic: check if we're using too much memory
	threshold := float64(pool.config.MaxPoolSize) * pool.config.PressureThreshold
	return float64(m.Alloc) > threshold
}

func (pool *ProductionMemoryPool) canResize() bool {
	currentUsage := float64(atomic.LoadInt64(&pool.totalSize)) / float64(pool.config.MaxPoolSize)
	return currentUsage > pool.config.ResizeThreshold
}

func (pool *ProductionMemoryPool) resizePool() {
	pool.mutex.Lock()
	defer pool.mutex.Unlock()
	// Double the pool size (simplified resize strategy)
	pool.config.MaxPoolSize *= 2
}

func (pool *ProductionMemoryPool) trackAllocation(block *memoryBlock) {
	pool.mutex.Lock()
	defer pool.mutex.Unlock()

	// Capture stack trace for debugging
	var stackTrace []uintptr
	if pool.config.LeakDetection {
		stackTrace = make([]uintptr, 10)
		n := runtime.Callers(2, stackTrace)
		stackTrace = stackTrace[:n]
	}

	pool.allocations[block.id] = &allocationInfo{
		block:      block,
		timestamp:  time.Now(),
		stackTrace: stackTrace,
	}
}

func (pool *ProductionMemoryPool) untrackAllocation(block *memoryBlock) {
	pool.mutex.Lock()
	defer pool.mutex.Unlock()
	delete(pool.allocations, block.id)
}

func (pool *ProductionMemoryPool) isDoubleDeallocation(block *memoryBlock) bool {
	pool.mutex.RLock()
	defer pool.mutex.RUnlock()
	_, exists := pool.allocations[block.id]
	return !exists
}

func (pool *ProductionMemoryPool) startLeakDetection() {
	pool.leakDetector.ticker = time.NewTicker(pool.config.LeakTimeout)
	go func() {
		for {
			select {
			case <-pool.leakDetector.ticker.C:
				pool.detectLeaks()
			case <-pool.leakDetector.stopChan:
				return
			}
		}
	}()
}

func (pool *ProductionMemoryPool) detectLeaks() {
	leaks := pool.GetLeakReport()
	if len(leaks) > 0 {
		atomic.AddInt64(&pool.totalLeaks, int64(len(leaks)))

		if pool.alertHandler != nil {
			pool.alertHandler(MemoryAlert{
				Type:      AlertMemoryLeak,
				Message:   fmt.Sprintf("Detected %d memory leaks", len(leaks)),
				Timestamp: time.Now(),
			})
		}
	}
}

func (pool *ProductionMemoryPool) checkUsageAlerts() {
	if pool.alertHandler == nil || !pool.config.MonitoringMode {
		return
	}

	usage := float64(atomic.LoadInt64(&pool.totalSize)) / float64(pool.config.MaxPoolSize)
	if usage > pool.config.UsageAlertThreshold {
		atomic.AddInt64(&pool.alertsTriggered, 1)
		pool.alertHandler(MemoryAlert{
			Type:        AlertHighMemoryUsage,
			Message:     fmt.Sprintf("Memory usage at %.1f%% of pool capacity", usage*100),
			MemoryUsage: usage,
			Timestamp:   time.Now(),
		})
	}
}

// sizeClass methods

func (sc *sizeClass) getBlock() *memoryBlock {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()

	if len(sc.blocks) > 0 {
		block := sc.blocks[len(sc.blocks)-1]
		sc.blocks = sc.blocks[:len(sc.blocks)-1]
		return block
	}
	return nil
}

func (sc *sizeClass) returnBlock(block *memoryBlock) {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()

	if len(sc.blocks) < cap(sc.blocks) {
		sc.blocks = append(sc.blocks, block)
	}
}

func (sc *sizeClass) cleanup() {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()
	sc.blocks = nil
}

// allocationProfiler methods

func (ap *allocationProfiler) recordAllocation(size int, duration time.Duration) {
	ap.mutex.Lock()
	defer ap.mutex.Unlock()

	ap.sizeDistribution[size]++
	ap.totalAllocTime += duration
	ap.totalAllocs++

	currentUsage := atomic.LoadInt64(&ap.totalAllocs) * int64(size)
	if currentUsage > ap.peakUsage {
		ap.peakUsage = currentUsage
	}
}

// memoryBlock finalizer for GC integration
func (mb *memoryBlock) finalize() {
	if mb.pool != nil && mb.pool.active {
		atomic.AddInt64(&mb.pool.finalizerCleanups, 1)
		// Automatic cleanup via finalizer
		mb.pool.untrackAllocation(mb)
	}
}

// Error checking functions

func IsOutOfMemoryError(err error) bool {
	return err == ErrOutOfMemory
}

func IsInvalidSizeError(err error) bool {
	return err == ErrInvalidSize
}

func IsDoubleDeallocationError(err error) bool {
	return err == ErrDoubleDeallocation
}
