package math

import (
	"math"
	"math/cmplx"

	"github.com/julianshen/gonp/array"
)

// Sqrt computes the square root of array elements
func Sqrt(arr *array.Array) (*array.Array, error) {
	sqrtFunc := makeComplexFunc(math.Sqrt, cmplx.Sqrt)
	return ApplyUnary(arr, sqrtFunc)
}

// Cbrt computes the cube root of array elements
func Cbrt(arr *array.Array) (*array.Array, error) {
	cbrtFunc := makeFloatFunc(math.Cbrt)
	return ApplyUnary(arr, cbrtFunc)
}

// Square computes the square of array elements
func Square(arr *array.Array) (*array.Array, error) {
	squareFunc := func(x interface{}) interface{} {
		switch v := x.(type) {
		case float64:
			return v * v
		case float32:
			return v * v
		case int64:
			return v * v
		case int32:
			return v * v
		case int16:
			return v * v
		case int8:
			return v * v
		case uint64:
			return v * v
		case uint32:
			return v * v
		case uint16:
			return v * v
		case uint8:
			return v * v
		case complex64:
			return v * v
		case complex128:
			return v * v
		default:
			return makeFloatFunc(func(x float64) float64 { return x * x })(x)
		}
	}
	return ApplyUnary(arr, squareFunc)
}

// Power computes x^y element-wise
func Power(x, y *array.Array) (*array.Array, error) {
	powerFunc := func(a, b interface{}) interface{} {
		// Handle complex numbers properly
		switch va := a.(type) {
		case complex64:
			vb := convertToComplex64(b)
			return complex64(cmplx.Pow(complex128(va), complex128(vb)))
		case complex128:
			vb := convertToComplex128(b)
			return cmplx.Pow(va, vb)
		default:
			// For real numbers, use math.Pow
			return makeBinaryFloatFunc(math.Pow)(a, b)
		}
	}
	return ApplyBinary(x, y, powerFunc)
}

// PowerScalar computes x^scalar for array elements
func PowerScalar(arr *array.Array, scalar float64) (*array.Array, error) {
	powerFunc := makeFloatFunc(func(x float64) float64 {
		return math.Pow(x, scalar)
	})
	return ApplyUnary(arr, powerFunc)
}

// Reciprocal computes 1/x for array elements
func Reciprocal(arr *array.Array) (*array.Array, error) {
	reciprocalFunc := func(x interface{}) interface{} {
		switch v := x.(type) {
		case float64:
			if v == 0 {
				return math.Inf(1)
			}
			return 1.0 / v
		case float32:
			if v == 0 {
				return float32(math.Inf(1))
			}
			return 1.0 / v
		case complex64:
			if v == 0 {
				return complex64(complex(float32(math.Inf(1)), float32(math.Inf(1))))
			}
			return 1.0 / v
		case complex128:
			if v == 0 {
				return complex(math.Inf(1), math.Inf(1))
			}
			return 1.0 / v
		default:
			floatVal := convertToFloat64(x)
			if floatVal == 0 {
				return math.Inf(1)
			}
			return 1.0 / floatVal
		}
	}
	return ApplyUnary(arr, reciprocalFunc)
}

// In-place versions

// SqrtInPlace computes the square root of array elements in-place
func SqrtInPlace(arr *array.Array) error {
	sqrtFunc := makeComplexFunc(math.Sqrt, cmplx.Sqrt)
	return ApplyUnaryInPlace(arr, sqrtFunc)
}

// CbrtInPlace computes the cube root of array elements in-place
func CbrtInPlace(arr *array.Array) error {
	cbrtFunc := makeFloatFunc(math.Cbrt)
	return ApplyUnaryInPlace(arr, cbrtFunc)
}

// SquareInPlace computes the square of array elements in-place
func SquareInPlace(arr *array.Array) error {
	squareFunc := func(x interface{}) interface{} {
		switch v := x.(type) {
		case float64:
			return v * v
		case float32:
			return v * v
		case int64:
			return v * v
		case int32:
			return v * v
		case int16:
			return v * v
		case int8:
			return v * v
		case uint64:
			return v * v
		case uint32:
			return v * v
		case uint16:
			return v * v
		case uint8:
			return v * v
		case complex64:
			return v * v
		case complex128:
			return v * v
		default:
			return makeFloatFunc(func(x float64) float64 { return x * x })(x)
		}
	}
	return ApplyUnaryInPlace(arr, squareFunc)
}

// PowerScalarInPlace computes x^scalar for array elements in-place
func PowerScalarInPlace(arr *array.Array, scalar float64) error {
	powerFunc := makeFloatFunc(func(x float64) float64 {
		return math.Pow(x, scalar)
	})
	return ApplyUnaryInPlace(arr, powerFunc)
}

// ReciprocalInPlace computes 1/x for array elements in-place
func ReciprocalInPlace(arr *array.Array) error {
	reciprocalFunc := func(x interface{}) interface{} {
		switch v := x.(type) {
		case float64:
			if v == 0 {
				return math.Inf(1)
			}
			return 1.0 / v
		case float32:
			if v == 0 {
				return float32(math.Inf(1))
			}
			return 1.0 / v
		case complex64:
			if v == 0 {
				return complex64(complex(float32(math.Inf(1)), float32(math.Inf(1))))
			}
			return 1.0 / v
		case complex128:
			if v == 0 {
				return complex(math.Inf(1), math.Inf(1))
			}
			return 1.0 / v
		default:
			floatVal := convertToFloat64(x)
			if floatVal == 0 {
				return math.Inf(1)
			}
			return 1.0 / floatVal
		}
	}
	return ApplyUnaryInPlace(arr, reciprocalFunc)
}

// Helper functions for complex number conversion
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
		return complex(float32(convertToFloat64(v)), 0)
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
		return complex(convertToFloat64(v), 0)
	}
}
