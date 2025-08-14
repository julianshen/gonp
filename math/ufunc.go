package math

import (
	"fmt"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// UnaryFunc represents a universal function that operates on a single array
type UnaryFunc func(x interface{}) interface{}

// BinaryFunc represents a universal function that operates on two arrays
type BinaryFunc func(x, y interface{}) interface{}

// ApplyUnary applies a unary universal function to an array
func ApplyUnary(arr *array.Array, fn UnaryFunc) (*array.Array, error) {
	if err := internal.QuickValidateNotNil(arr, "ApplyUnary", "array"); err != nil {
		return nil, err
	}
	if fn == nil {
		return nil, internal.NewValidationErrorWithMsg("ApplyUnary", "function cannot be nil")
	}
	// Create result array with same shape and dtype
	result := array.Empty(arr.Shape(), arr.DType())

	// Apply function element-wise using nested loops over each dimension
	return applyUnaryIterative(arr, result, fn)
}

// applyUnaryIterative applies a unary function iteratively over all array elements
func applyUnaryIterative(src, dst *array.Array, fn UnaryFunc) (*array.Array, error) {
	shape := src.Shape()

	// Use recursive approach to handle arbitrary dimensions
	indices := make([]int, len(shape))
	return applyUnaryRecursive(src, dst, fn, indices, 0)
}

// applyUnaryRecursive recursively applies function over each dimension
func applyUnaryRecursive(src, dst *array.Array, fn UnaryFunc, indices []int, dim int) (*array.Array, error) {
	shape := src.Shape()

	if dim == len(shape) {
		// Base case: apply function at this position
		val := src.At(indices...)
		resultVal := fn(val)
		err := dst.Set(resultVal, indices...)
		if err != nil {
			return nil, fmt.Errorf("failed to set result value: %v", err)
		}
		return dst, nil
	}

	// Recursive case: iterate over this dimension
	for i := 0; i < shape[dim]; i++ {
		indices[dim] = i
		_, err := applyUnaryRecursive(src, dst, fn, indices, dim+1)
		if err != nil {
			return nil, err
		}
	}

	return dst, nil
}

// applyBinaryIterative applies a binary function iteratively over all array elements
func applyBinaryIterative(srcA, srcB, dst *array.Array, fn BinaryFunc) (*array.Array, error) {
	shape := dst.Shape()

	// Use recursive approach to handle arbitrary dimensions
	indices := make([]int, len(shape))
	return applyBinaryRecursive(srcA, srcB, dst, fn, indices, 0)
}

// applyBinaryRecursive recursively applies binary function over each dimension
func applyBinaryRecursive(srcA, srcB, dst *array.Array, fn BinaryFunc, indices []int, dim int) (*array.Array, error) {
	shape := dst.Shape()

	if dim == len(shape) {
		// Base case: apply function at this position
		valA := srcA.At(indices...)
		valB := srcB.At(indices...)
		resultVal := fn(valA, valB)
		err := dst.Set(resultVal, indices...)
		if err != nil {
			return nil, fmt.Errorf("failed to set result value: %v", err)
		}
		return dst, nil
	}

	// Recursive case: iterate over this dimension
	for i := 0; i < shape[dim]; i++ {
		indices[dim] = i
		_, err := applyBinaryRecursive(srcA, srcB, dst, fn, indices, dim+1)
		if err != nil {
			return nil, err
		}
	}

	return dst, nil
}

// ApplyBinary applies a binary universal function to two arrays with broadcasting
func ApplyBinary(a, b *array.Array, fn BinaryFunc) (*array.Array, error) {
	// Broadcast arrays to compatible shapes
	broadcastA, broadcastB, err := array.BroadcastArrays(a, b)
	if err != nil {
		return nil, fmt.Errorf("broadcasting failed: %v", err)
	}

	// Create result array with the broadcasted shape
	resultShape := broadcastA.Shape()
	result := array.Empty(resultShape, a.DType()) // Use first array's dtype as default

	// Apply function element-wise using iterative approach
	return applyBinaryIterative(broadcastA, broadcastB, result, fn)
}

// ApplyUnaryInPlace applies a unary universal function to an array in-place
func ApplyUnaryInPlace(arr *array.Array, fn UnaryFunc) error {
	// Apply function element-wise directly to the array using iterative approach
	_, err := applyUnaryIterative(arr, arr, fn)
	return err
}

// makeFloatFunc creates a unary function that converts input to float64, applies the function, and converts back
func makeFloatFunc(mathFunc func(float64) float64) UnaryFunc {
	return func(x interface{}) interface{} {
		switch v := x.(type) {
		case float64:
			return mathFunc(v)
		case float32:
			return float32(mathFunc(float64(v)))
		case int64:
			return int64(mathFunc(float64(v)))
		case int32:
			return int32(mathFunc(float64(v)))
		case int16:
			return int16(mathFunc(float64(v)))
		case int8:
			return int8(mathFunc(float64(v)))
		case uint64:
			return uint64(mathFunc(float64(v)))
		case uint32:
			return uint32(mathFunc(float64(v)))
		case uint16:
			return uint16(mathFunc(float64(v)))
		case uint8:
			return uint8(mathFunc(float64(v)))
		case complex64:
			// For complex numbers, apply to real part and set imaginary to 0
			real := float32(mathFunc(float64(real(v))))
			return complex(real, 0)
		case complex128:
			// For complex numbers, apply to real part and set imaginary to 0
			real := mathFunc(real(v))
			return complex(real, 0)
		default:
			panic(fmt.Sprintf("unsupported type for mathematical function: %T", x))
		}
	}
}

// makeComplexFunc creates a unary function that handles complex numbers properly
func makeComplexFunc(realFunc func(float64) float64, complexFunc func(complex128) complex128) UnaryFunc {
	return func(x interface{}) interface{} {
		switch v := x.(type) {
		case complex64:
			result := complexFunc(complex128(v))
			return complex64(result)
		case complex128:
			return complexFunc(v)
		default:
			// For real numbers, use the real function
			return makeFloatFunc(realFunc)(x)
		}
	}
}

// makeBinaryFloatFunc creates a binary function for mathematical operations
func makeBinaryFloatFunc(mathFunc func(x, y float64) float64) BinaryFunc {
	return func(x, y interface{}) interface{} {
		// Convert both inputs to float64, apply function, convert back to x's type
		xFloat := convertToFloat64(x)
		yFloat := convertToFloat64(y)
		result := mathFunc(xFloat, yFloat)

		switch x.(type) {
		case float64:
			return result
		case float32:
			return float32(result)
		case int64:
			return int64(result)
		case int32:
			return int32(result)
		case int16:
			return int16(result)
		case int8:
			return int8(result)
		case uint64:
			return uint64(result)
		case uint32:
			return uint32(result)
		case uint16:
			return uint16(result)
		case uint8:
			return uint8(result)
		default:
			return result // Default to float64
		}
	}
}

// Helper function to convert any numeric type to float64
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
	case complex64:
		return float64(real(v))
	case complex128:
		return real(v)
	default:
		panic(fmt.Sprintf("cannot convert %T to float64", value))
	}
}
