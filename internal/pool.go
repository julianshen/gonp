package internal

import (
	"sync"
)

// ArrayPool provides a pool of reusable array storage to reduce allocations
type ArrayPool struct {
	float64Pool sync.Pool
	int64Pool   sync.Pool
	float32Pool sync.Pool
	int32Pool   sync.Pool
	boolPool    sync.Pool
}

// Global array pool instance
var GlobalArrayPool = &ArrayPool{
	float64Pool: sync.Pool{
		New: func() interface{} {
			// Start with a reasonable default size
			return make([]float64, 0, 64)
		},
	},
	int64Pool: sync.Pool{
		New: func() interface{} {
			return make([]int64, 0, 64)
		},
	},
	float32Pool: sync.Pool{
		New: func() interface{} {
			return make([]float32, 0, 64)
		},
	},
	int32Pool: sync.Pool{
		New: func() interface{} {
			return make([]int32, 0, 64)
		},
	},
	boolPool: sync.Pool{
		New: func() interface{} {
			return make([]bool, 0, 64)
		},
	},
}

// GetSlice retrieves a slice from the pool with the requested capacity
func (p *ArrayPool) GetSlice(dtype DType, size int) interface{} {
	switch dtype {
	case Float64:
		slice := p.float64Pool.Get().([]float64)
		if cap(slice) < size {
			// If pooled slice is too small, create a new one
			return make([]float64, size)
		}
		// Resize to requested size and clear contents
		slice = slice[:size]
		for i := range slice {
			slice[i] = 0
		}
		return slice
	case Int64:
		slice := p.int64Pool.Get().([]int64)
		if cap(slice) < size {
			return make([]int64, size)
		}
		slice = slice[:size]
		for i := range slice {
			slice[i] = 0
		}
		return slice
	case Float32:
		slice := p.float32Pool.Get().([]float32)
		if cap(slice) < size {
			return make([]float32, size)
		}
		slice = slice[:size]
		for i := range slice {
			slice[i] = 0
		}
		return slice
	case Int32:
		slice := p.int32Pool.Get().([]int32)
		if cap(slice) < size {
			return make([]int32, size)
		}
		slice = slice[:size]
		for i := range slice {
			slice[i] = 0
		}
		return slice
	case Bool:
		slice := p.boolPool.Get().([]bool)
		if cap(slice) < size {
			return make([]bool, size)
		}
		slice = slice[:size]
		for i := range slice {
			slice[i] = false
		}
		return slice
	default:
		// For unsupported types, allocate normally
		return nil
	}
}

// PutSlice returns a slice to the pool for reuse
func (p *ArrayPool) PutSlice(slice interface{}, dtype DType) {
	switch dtype {
	case Float64:
		if s, ok := slice.([]float64); ok && cap(s) <= 8192 { // Don't pool very large slices
			p.float64Pool.Put(s[:0]) // Reset length but keep capacity
		}
	case Int64:
		if s, ok := slice.([]int64); ok && cap(s) <= 8192 {
			p.int64Pool.Put(s[:0])
		}
	case Float32:
		if s, ok := slice.([]float32); ok && cap(s) <= 8192 {
			p.float32Pool.Put(s[:0])
		}
	case Int32:
		if s, ok := slice.([]int32); ok && cap(s) <= 8192 {
			p.int32Pool.Put(s[:0])
		}
	case Bool:
		if s, ok := slice.([]bool); ok && cap(s) <= 8192 {
			p.boolPool.Put(s[:0])
		}
	}
}

// ShapePool provides a pool for Shape objects to reduce allocations
type ShapePool struct {
	pool sync.Pool
}

// Global shape pool instance
var GlobalShapePool = &ShapePool{
	pool: sync.Pool{
		New: func() interface{} {
			return make(Shape, 0, 4) // Most arrays are 1-4 dimensions
		},
	},
}

// GetShape retrieves a Shape from the pool
func (p *ShapePool) GetShape(dims int) Shape {
	shape := p.pool.Get().(Shape)
	if cap(shape) < dims {
		return make(Shape, dims)
	}
	return shape[:dims]
}

// PutShape returns a Shape to the pool
func (p *ShapePool) PutShape(shape Shape) {
	if cap(shape) <= 8 { // Don't pool very large shapes
		p.pool.Put(shape[:0])
	}
}

// BufferPool provides a pool for temporary byte buffers
type BufferPool struct {
	pool sync.Pool
}

// Global buffer pool instance
var GlobalBufferPool = &BufferPool{
	pool: sync.Pool{
		New: func() interface{} {
			return make([]byte, 0, 1024) // 1KB default buffer
		},
	},
}

// GetBuffer retrieves a buffer from the pool
func (p *BufferPool) GetBuffer(minSize int) []byte {
	buf := p.pool.Get().([]byte)
	if cap(buf) < minSize {
		return make([]byte, 0, minSize)
	}
	return buf[:0] // Reset length but keep capacity
}

// PutBuffer returns a buffer to the pool
func (p *BufferPool) PutBuffer(buf []byte) {
	if cap(buf) <= 64*1024 { // Don't pool very large buffers (>64KB)
		p.pool.Put(buf[:0])
	}
}

// PooledAllocation tracks allocations that can be returned to pools
type PooledAllocation struct {
	Data  interface{}
	DType DType
	Pool  *ArrayPool
}

// Release returns the allocation to the appropriate pool
func (pa *PooledAllocation) Release() {
	if pa.Pool != nil {
		pa.Pool.PutSlice(pa.Data, pa.DType)
	}
}

// GetPooledSlice is a convenience function to get a slice from the global pool
func GetPooledSlice(dtype DType, size int) *PooledAllocation {
	slice := GlobalArrayPool.GetSlice(dtype, size)
	if slice == nil {
		return nil
	}
	return &PooledAllocation{
		Data:  slice,
		DType: dtype,
		Pool:  GlobalArrayPool,
	}
}

// MemoryStats provides statistics about pool usage
type MemoryStats struct {
	Float64PoolSize int
	Int64PoolSize   int
	Float32PoolSize int
	Int32PoolSize   int
	BoolPoolSize    int
	ShapePoolSize   int
	BufferPoolSize  int
}

// GetMemoryStats returns current pool statistics (for debugging)
func GetMemoryStats() MemoryStats {
	// Note: sync.Pool doesn't expose size information directly
	// This is a placeholder for potential future monitoring
	return MemoryStats{
		// These would need custom tracking if detailed stats are needed
	}
}
