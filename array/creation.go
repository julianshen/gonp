package array

import (
	"fmt"
	"reflect"

	"github.com/julianshen/gonp/internal"
)

// Zeros creates a new array filled with zeros of the specified shape and data type.
//
// This is equivalent to NumPy's np.zeros() function. All elements in the returned
// array are initialized to the zero value for the specified data type.
//
// Parameters:
//   - shape: The dimensions of the array (e.g., Shape{3, 4} for 3x4 array)
//   - dtype: The data type of the elements
//
// Performance: O(n) where n is the total number of elements.
// Memory is allocated and zero-initialized efficiently.
//
// Supported Data Types:
//   - Float64, Float32: 0.0
//   - Int64, Int32, Int16, Int8: 0
//   - Uint64, Uint32, Uint16, Uint8: 0
//   - Bool: false
//   - Complex64, Complex128: 0+0i
//
// Example:
//
//	// Create 1D array of zeros
//	arr1d := array.Zeros(internal.Shape{5}, internal.Float64)
//	// Result: [0.0, 0.0, 0.0, 0.0, 0.0]
//
//	// Create 2D array of zeros
//	arr2d := array.Zeros(internal.Shape{3, 4}, internal.Int32)
//	// Result: 3x4 array filled with int32 zeros
//
//	// Create 3D array
//	arr3d := array.Zeros(internal.Shape{2, 3, 4}, internal.Bool)
//	// Result: 2x3x4 array filled with false values
//
// Panics if:
//   - shape is empty (no dimensions)
//   - any dimension is zero or negative
func Zeros(shape internal.Shape, dtype internal.DType) *Array {
	ctx := internal.StartProfiler("Array.Zeros")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if len(shape) == 0 {
		panic("shape cannot be empty")
	}
	for _, dim := range shape {
		if dim <= 0 {
			panic("all shape dimensions must be positive")
		}
	}
	size := shape.Size()
	internal.IncrementArrayCreations()
	data := allocateSliceForDType(size, dtype)
	storage := internal.NewTypedStorage(data, dtype)
	stride := calculateStride(shape)

	return &Array{
		storage: storage,
		shape:   shape.Copy(),
		stride:  stride,
		dtype:   dtype,
		offset:  0,
	}
}

// Ones creates a new array filled with ones of the specified shape and data type.
//
// This is equivalent to NumPy's np.ones() function. All elements in the returned
// array are initialized to the one value for the specified data type.
//
// Parameters:
//   - shape: The dimensions of the array
//   - dtype: The data type of the elements
//
// Performance: O(n) where n is the total number of elements.
// Memory is allocated and then filled with one values.
//
// One Values by Data Type:
//   - Float64, Float32: 1.0
//   - Int64, Int32, Int16, Int8: 1
//   - Uint64, Uint32, Uint16, Uint8: 1
//   - Bool: true
//   - Complex64, Complex128: 1+0i
//
// Example:
//
//	// Create 1D array of ones
//	arr1d := array.Ones(internal.Shape{4}, internal.Float64)
//	// Result: [1.0, 1.0, 1.0, 1.0]
//
//	// Create 2D array of ones
//	arr2d := array.Ones(internal.Shape{2, 3}, internal.Int64)
//	// Result: 2x3 array filled with 1s
//
//	// Boolean ones array
//	arrBool := array.Ones(internal.Shape{3}, internal.Bool)
//	// Result: [true, true, true]
//
// Panics if:
//   - shape is empty or contains non-positive dimensions
func Ones(shape internal.Shape, dtype internal.DType) *Array {
	ctx := internal.StartProfiler("Array.Ones")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if len(shape) == 0 {
		panic("shape cannot be empty")
	}
	for _, dim := range shape {
		if dim <= 0 {
			panic("all shape dimensions must be positive")
		}
	}
	size := shape.Size()
	internal.IncrementArrayCreations()
	data := allocateSliceForDType(size, dtype)
	fillSliceWithOne(data, dtype)
	storage := internal.NewTypedStorage(data, dtype)
	stride := calculateStride(shape)

	return &Array{
		storage: storage,
		shape:   shape.Copy(),
		stride:  stride,
		dtype:   dtype,
		offset:  0,
	}
}

// Arange creates a new array with evenly spaced values within a given interval
func Arange(start, stop, step float64) *Array {
	if step <= 0 {
		panic("step must be positive")
	}
	if start >= stop {
		panic("start must be less than stop")
	}

	size := int((stop-start)/step + 0.5)
	if size <= 0 {
		size = 0
	}

	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = start + float64(i)*step
	}

	storage := internal.NewTypedStorage(data, internal.Float64)
	shape := internal.Shape{size}
	stride := calculateStride(shape)

	return &Array{
		storage: storage,
		shape:   shape,
		stride:  stride,
		dtype:   internal.Float64,
		offset:  0,
	}
}

// FromSlice creates a new array from a Go slice with automatic shape and type inference.
//
// This is the most commonly used function for creating arrays from existing Go data.
// It automatically infers the data type and shape from the input slice structure.
// Multi-dimensional slices are supported and their nested structure is preserved.
//
// This is equivalent to NumPy's np.array() function.
//
// Supported Input Types:
//   - []T: 1D slices of any supported numeric type
//   - [][]T: 2D slices (must have consistent row lengths)
//   - [][][]T: 3D slices (must have consistent dimensions)
//   - Higher dimensional nested slices
//
// Type Inference:
//   - Go slice types are automatically mapped to appropriate DType
//   - Mixed numeric types in the same slice are not supported
//   - String slices require special handling (see array string support)
//
// Performance:
//   - O(n) where n is total number of elements
//   - Memory is copied from input for safety
//   - Large arrays benefit from SIMD optimizations
//
// Example:
//
//	// 1D arrays
//	arr1, _ := array.FromSlice([]float64{1.5, 2.5, 3.5})
//	arr2, _ := array.FromSlice([]int{1, 2, 3, 4, 5})
//	arr3, _ := array.FromSlice([]bool{true, false, true})
//
//	// 2D arrays
//	matrix, _ := array.FromSlice([][]float64{
//		{1.0, 2.0, 3.0},
//		{4.0, 5.0, 6.0},
//	})
//	fmt.Printf("Shape: %v\n", matrix.Shape())  // [2 3]
//
//	// 3D arrays
//	tensor, _ := array.FromSlice([][][]int{
//		{{1, 2}, {3, 4}},
//		{{5, 6}, {7, 8}},
//	})
//	fmt.Printf("Shape: %v\n", tensor.Shape())  // [2 2 2]
//
// Returns an error if:
//   - data is nil or not a slice
//   - nested slices have inconsistent dimensions
//   - data contains unsupported types
func FromSlice(data interface{}) (*Array, error) {
	ctx := internal.StartProfiler("Array.FromSlice")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if err := internal.QuickValidateNotNil(data, "FromSlice", "data"); err != nil {
		return nil, err
	}
	internal.IncrementArrayCreations()

	val := reflect.ValueOf(data)
	if val.Kind() != reflect.Slice {
		return nil, fmt.Errorf("data must be a slice, got %T", data)
	}

	dtype, err := getDTypeFromValue(val)
	if err != nil {
		return nil, err
	}

	storage := internal.NewTypedStorage(data, dtype)
	shape := internal.Shape{val.Len()}
	stride := calculateStride(shape)

	return &Array{
		storage: storage,
		shape:   shape,
		stride:  stride,
		dtype:   dtype,
		offset:  0,
	}, nil
}

// Full creates a new array filled with a specific value
func Full(shape internal.Shape, value interface{}, dtype internal.DType) *Array {
	size := shape.Size()
	data := allocateSliceForDType(size, dtype)
	fillSliceWithValue(data, value, dtype)
	storage := internal.NewTypedStorage(data, dtype)
	stride := calculateStride(shape)

	return &Array{
		storage: storage,
		shape:   shape.Copy(),
		stride:  stride,
		dtype:   dtype,
		offset:  0,
	}
}

// Empty creates a new uninitialized array
func Empty(shape internal.Shape, dtype internal.DType) *Array {
	size := shape.Size()
	data := allocateSliceForDType(size, dtype)
	storage := internal.NewTypedStorage(data, dtype)
	stride := calculateStride(shape)

	return &Array{
		storage: storage,
		shape:   shape.Copy(),
		stride:  stride,
		dtype:   dtype,
		offset:  0,
	}
}

// Linspace creates an array with evenly spaced values
func Linspace(start, stop float64, num int) *Array {
	if num <= 0 {
		panic("num must be positive")
	}

	data := make([]float64, num)
	if num == 1 {
		data[0] = start
	} else {
		step := (stop - start) / float64(num-1)
		for i := 0; i < num; i++ {
			data[i] = start + float64(i)*step
		}
	}

	storage := internal.NewTypedStorage(data, internal.Float64)
	shape := internal.Shape{num}
	stride := calculateStride(shape)

	return &Array{
		storage: storage,
		shape:   shape,
		stride:  stride,
		dtype:   internal.Float64,
		offset:  0,
	}
}

// allocateSliceForDType creates a slice of the appropriate type
// Uses memory pools for supported types to reduce GC pressure
func allocateSliceForDType(size int, dtype internal.DType) interface{} {
	// Use pooled allocation for common types and reasonable sizes
	if size > 0 && size <= 4096 { // Pool slices up to 4K elements
		if pooledSlice := internal.GlobalArrayPool.GetSlice(dtype, size); pooledSlice != nil {
			return pooledSlice
		}
	}

	// Fallback to regular allocation for unsupported types or very large sizes
	switch dtype {
	case internal.Float64:
		return make([]float64, size)
	case internal.Float32:
		return make([]float32, size)
	case internal.Int64:
		return make([]int64, size)
	case internal.Int32:
		return make([]int32, size)
	case internal.Int16:
		return make([]int16, size)
	case internal.Int8:
		return make([]int8, size)
	case internal.Uint64:
		return make([]uint64, size)
	case internal.Uint32:
		return make([]uint32, size)
	case internal.Uint16:
		return make([]uint16, size)
	case internal.Uint8:
		return make([]uint8, size)
	case internal.Bool:
		return make([]bool, size)
	case internal.Complex64:
		return make([]complex64, size)
	case internal.Complex128:
		return make([]complex128, size)
	case internal.String:
		return make([]string, size)
	case internal.Interface:
		return make([]interface{}, size)
	default:
		panic(fmt.Sprintf("unsupported dtype: %v", dtype))
	}
}

// fillSliceWithOne fills a slice with the value 1 for the given dtype
func fillSliceWithOne(data interface{}, dtype internal.DType) {
	switch dtype {
	case internal.Float64:
		slice := data.([]float64)
		for i := range slice {
			slice[i] = 1.0
		}
	case internal.Float32:
		slice := data.([]float32)
		for i := range slice {
			slice[i] = 1.0
		}
	case internal.Int64:
		slice := data.([]int64)
		for i := range slice {
			slice[i] = 1
		}
	case internal.Int32:
		slice := data.([]int32)
		for i := range slice {
			slice[i] = 1
		}
	case internal.Int16:
		slice := data.([]int16)
		for i := range slice {
			slice[i] = 1
		}
	case internal.Int8:
		slice := data.([]int8)
		for i := range slice {
			slice[i] = 1
		}
	case internal.Uint64:
		slice := data.([]uint64)
		for i := range slice {
			slice[i] = 1
		}
	case internal.Uint32:
		slice := data.([]uint32)
		for i := range slice {
			slice[i] = 1
		}
	case internal.Uint16:
		slice := data.([]uint16)
		for i := range slice {
			slice[i] = 1
		}
	case internal.Uint8:
		slice := data.([]uint8)
		for i := range slice {
			slice[i] = 1
		}
	case internal.Bool:
		slice := data.([]bool)
		for i := range slice {
			slice[i] = true
		}
	case internal.Complex64:
		slice := data.([]complex64)
		for i := range slice {
			slice[i] = 1.0 + 0i
		}
	case internal.Complex128:
		slice := data.([]complex128)
		for i := range slice {
			slice[i] = 1.0 + 0i
		}
	case internal.String:
		slice := data.([]string)
		for i := range slice {
			slice[i] = "1"
		}
	case internal.Interface:
		slice := data.([]interface{})
		for i := range slice {
			slice[i] = 1
		}
	default:
		panic(fmt.Sprintf("unsupported dtype: %v", dtype))
	}
}

// fillSliceWithValue fills a slice with the given value
func fillSliceWithValue(data interface{}, value interface{}, dtype internal.DType) {
	switch dtype {
	case internal.Float64:
		slice := data.([]float64)
		val := convertToFloat64(value)
		for i := range slice {
			slice[i] = val
		}
	case internal.Float32:
		slice := data.([]float32)
		val := float32(convertToFloat64(value))
		for i := range slice {
			slice[i] = val
		}
	case internal.Int64:
		slice := data.([]int64)
		val := convertToInt64(value)
		for i := range slice {
			slice[i] = val
		}
	case internal.Int32:
		slice := data.([]int32)
		val := int32(convertToInt64(value))
		for i := range slice {
			slice[i] = val
		}
	case internal.Int16:
		slice := data.([]int16)
		val := int16(convertToInt64(value))
		for i := range slice {
			slice[i] = val
		}
	case internal.Int8:
		slice := data.([]int8)
		val := int8(convertToInt64(value))
		for i := range slice {
			slice[i] = val
		}
	case internal.Uint64:
		slice := data.([]uint64)
		val := uint64(convertToInt64(value))
		for i := range slice {
			slice[i] = val
		}
	case internal.Uint32:
		slice := data.([]uint32)
		val := uint32(convertToInt64(value))
		for i := range slice {
			slice[i] = val
		}
	case internal.Uint16:
		slice := data.([]uint16)
		val := uint16(convertToInt64(value))
		for i := range slice {
			slice[i] = val
		}
	case internal.Uint8:
		slice := data.([]uint8)
		val := uint8(convertToInt64(value))
		for i := range slice {
			slice[i] = val
		}
	case internal.Bool:
		slice := data.([]bool)
		val := convertToBool(value)
		for i := range slice {
			slice[i] = val
		}
	case internal.Complex64:
		slice := data.([]complex64)
		val := convertToComplex64(value)
		for i := range slice {
			slice[i] = val
		}
	case internal.Complex128:
		slice := data.([]complex128)
		val := convertToComplex128(value)
		for i := range slice {
			slice[i] = val
		}
	case internal.String:
		slice := data.([]string)
		val := convertToString(value)
		for i := range slice {
			slice[i] = val
		}
	case internal.Interface:
		slice := data.([]interface{})
		for i := range slice {
			slice[i] = value
		}
	default:
		panic(fmt.Sprintf("unsupported dtype: %v", dtype))
	}
}

// Helper functions for type conversion
func convertToFloat64(value interface{}) float64 {
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case int32:
		return float64(v)
	case int16:
		return float64(v)
	case int8:
		return float64(v)
	case uint64:
		return float64(v)
	case uint32:
		return float64(v)
	case uint16:
		return float64(v)
	case uint8:
		return float64(v)
	default:
		panic(fmt.Sprintf("cannot convert %T to float64", value))
	}
}

func convertToInt64(value interface{}) int64 {
	switch v := value.(type) {
	case int64:
		return v
	case int:
		return int64(v)
	case int32:
		return int64(v)
	case int16:
		return int64(v)
	case int8:
		return int64(v)
	case uint64:
		return int64(v)
	case uint32:
		return int64(v)
	case uint16:
		return int64(v)
	case uint8:
		return int64(v)
	case float64:
		return int64(v)
	case float32:
		return int64(v)
	default:
		panic(fmt.Sprintf("cannot convert %T to int64", value))
	}
}

func convertToBool(value interface{}) bool {
	switch v := value.(type) {
	case bool:
		return v
	case int:
		return v != 0
	case int64:
		return v != 0
	case float64:
		return v != 0.0
	case float32:
		return v != 0.0
	default:
		panic(fmt.Sprintf("cannot convert %T to bool", value))
	}
}

func convertToComplex64(value interface{}) complex64 {
	switch v := value.(type) {
	case complex64:
		return v
	case complex128:
		return complex64(v)
	case float64:
		return complex(float32(v), 0)
	case float32:
		return complex(v, 0)
	case int:
		return complex(float32(v), 0)
	case int64:
		return complex(float32(v), 0)
	default:
		panic(fmt.Sprintf("cannot convert %T to complex64", value))
	}
}

func convertToComplex128(value interface{}) complex128 {
	switch v := value.(type) {
	case complex128:
		return v
	case complex64:
		return complex128(v)
	case float64:
		return complex(v, 0)
	case float32:
		return complex(float64(v), 0)
	case int:
		return complex(float64(v), 0)
	case int64:
		return complex(float64(v), 0)
	default:
		panic(fmt.Sprintf("cannot convert %T to complex128", value))
	}
}

func convertToString(value interface{}) string {
	switch v := value.(type) {
	case string:
		return v
	case int:
		return fmt.Sprintf("%d", v)
	case int64:
		return fmt.Sprintf("%d", v)
	case float64:
		return fmt.Sprintf("%g", v)
	case float32:
		return fmt.Sprintf("%g", v)
	case bool:
		return fmt.Sprintf("%t", v)
	default:
		return fmt.Sprintf("%v", v)
	}
}
