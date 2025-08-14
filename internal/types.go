package internal

import (
	"unsafe"
)

// DType represents the data type of array elements
type DType int

const (
	Float32 DType = iota
	Float64
	Int8
	Int16
	Int32
	Int64
	Uint8
	Uint16
	Uint32
	Uint64
	Bool
	Complex64
	Complex128
	String    // String data type for text processing
	Interface // interface{} for heterogeneous data
)

// Shape represents the dimensions of an array
type Shape []int

// Ndim returns the number of dimensions
func (s Shape) Ndim() int {
	return len(s)
}

// Size returns the total number of elements
func (s Shape) Size() int {
	if len(s) == 0 {
		return 1 // Empty shape (scalar) has 1 element
	}
	size := 1
	for _, dim := range s {
		size *= dim
	}
	return size
}

// Equal checks if two shapes are equal
func (s Shape) Equal(other Shape) bool {
	if len(s) != len(other) {
		return false
	}
	for i, dim := range s {
		if dim != other[i] {
			return false
		}
	}
	return true
}

// Copy creates a copy of the shape
func (s Shape) Copy() Shape {
	result := make(Shape, len(s))
	copy(result, s)
	return result
}

// Stride represents the memory strides for each dimension
type Stride []int

// Copy creates a copy of the stride
func (s Stride) Copy() Stride {
	result := make(Stride, len(s))
	copy(result, s)
	return result
}

// Storage interface for array data storage
type Storage interface {
	Data() unsafe.Pointer
	Len() int
	Cap() int
	Type() DType
	ElementSize() int
	Clone() Storage
}

// Range represents a slice range for array indexing
type Range struct {
	Start int
	Stop  int
	Step  int
}

// NewRange creates a new range with default step of 1
func NewRange(start, stop int) Range {
	return Range{Start: start, Stop: stop, Step: 1}
}

// Length returns the number of elements in the range
func (r Range) Length() int {
	if r.Step == 0 {
		return 0
	}
	if r.Step > 0 {
		if r.Stop <= r.Start {
			return 0
		}
		return (r.Stop - r.Start + r.Step - 1) / r.Step
	}
	if r.Start <= r.Stop {
		return 0
	}
	return (r.Start - r.Stop - r.Step - 1) / (-r.Step)
}

// Size returns the size in bytes for each data type
func (d DType) Size() int {
	switch d {
	case Float32:
		return 4
	case Float64:
		return 8
	case Int8:
		return 1
	case Int16:
		return 2
	case Int32:
		return 4
	case Int64:
		return 8
	case Uint8:
		return 1
	case Uint16:
		return 2
	case Uint32:
		return 4
	case Uint64:
		return 8
	case Bool:
		return 1
	case Complex64:
		return 8
	case Complex128:
		return 16
	case String:
		return int(unsafe.Sizeof("")) // Size of string header
	case Interface:
		return int(unsafe.Sizeof((*interface{})(nil))) // Size of interface header
	default:
		return 0
	}
}

// String returns a string representation of the data type
func (d DType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	case Int8:
		return "int8"
	case Int16:
		return "int16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Uint8:
		return "uint8"
	case Uint16:
		return "uint16"
	case Uint32:
		return "uint32"
	case Uint64:
		return "uint64"
	case Bool:
		return "bool"
	case Complex64:
		return "complex64"
	case Complex128:
		return "complex128"
	case String:
		return "string"
	case Interface:
		return "interface{}"
	default:
		return "unknown"
	}
}

// InferDType infers the DType from a Go value
func InferDType(value interface{}) DType {
	switch value.(type) {
	case float32:
		return Float32
	case float64:
		return Float64
	case int:
		return Int64 // Map int to int64 for consistency
	case int8:
		return Int8
	case int16:
		return Int16
	case int32:
		return Int32
	case int64:
		return Int64
	case uint:
		return Uint64 // Map uint to uint64 for consistency
	case uint8:
		return Uint8
	case uint16:
		return Uint16
	case uint32:
		return Uint32
	case uint64:
		return Uint64
	case bool:
		return Bool
	case complex64:
		return Complex64
	case complex128:
		return Complex128
	case string:
		return String
	default:
		return Interface
	}
}
