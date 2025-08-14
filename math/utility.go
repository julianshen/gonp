package math

import (
	"math"

	"github.com/julianshen/gonp/array"
)

// Abs computes the absolute value of array elements
func Abs(arr *array.Array) (*array.Array, error) {
	absFunc := func(x interface{}) interface{} {
		switch v := x.(type) {
		case float64:
			return math.Abs(v)
		case float32:
			return float32(math.Abs(float64(v)))
		case int64:
			if v < 0 {
				return -v
			}
			return v
		case int32:
			if v < 0 {
				return -v
			}
			return v
		case int16:
			if v < 0 {
				return -v
			}
			return v
		case int8:
			if v < 0 {
				return -v
			}
			return v
		case uint64, uint32, uint16, uint8:
			return v // Unsigned types are always positive
		case complex64:
			return complex64(complex(float32(math.Abs(float64(real(v)*real(v)+imag(v)*imag(v)))), 0))
		case complex128:
			return complex(math.Sqrt(real(v)*real(v)+imag(v)*imag(v)), 0)
		default:
			return makeFloatFunc(math.Abs)(x)
		}
	}
	return ApplyUnary(arr, absFunc)
}

// Sign returns the sign of array elements (-1, 0, or 1)
func Sign(arr *array.Array) (*array.Array, error) {
	signFunc := func(x interface{}) interface{} {
		switch v := x.(type) {
		case float64:
			if v > 0 {
				return 1.0
			} else if v < 0 {
				return -1.0
			}
			return 0.0
		case float32:
			if v > 0 {
				return float32(1.0)
			} else if v < 0 {
				return float32(-1.0)
			}
			return float32(0.0)
		case int64:
			if v > 0 {
				return int64(1)
			} else if v < 0 {
				return int64(-1)
			}
			return int64(0)
		case int32:
			if v > 0 {
				return int32(1)
			} else if v < 0 {
				return int32(-1)
			}
			return int32(0)
		case int16:
			if v > 0 {
				return int16(1)
			} else if v < 0 {
				return int16(-1)
			}
			return int16(0)
		case int8:
			if v > 0 {
				return int8(1)
			} else if v < 0 {
				return int8(-1)
			}
			return int8(0)
		case uint64, uint32, uint16, uint8:
			// Unsigned types: 0 -> 0, positive -> 1
			floatVal := convertToFloat64(v)
			if floatVal == 0 {
				return v // Keep original type for 0
			}
			// Return 1 in the same type
			switch v.(type) {
			case uint64:
				return uint64(1)
			case uint32:
				return uint32(1)
			case uint16:
				return uint16(1)
			case uint8:
				return uint8(1)
			}
			return v
		case complex64:
			// For complex numbers, return the unit vector in the same direction
			mag := math.Sqrt(float64(real(v)*real(v) + imag(v)*imag(v)))
			if mag == 0 {
				return complex64(0)
			}
			return complex64(complex(float32(float64(real(v))/mag), float32(float64(imag(v))/mag)))
		case complex128:
			mag := math.Sqrt(real(v)*real(v) + imag(v)*imag(v))
			if mag == 0 {
				return complex128(0)
			}
			return complex(real(v)/mag, imag(v)/mag)
		default:
			floatVal := convertToFloat64(x)
			if floatVal > 0 {
				return 1.0
			} else if floatVal < 0 {
				return -1.0
			}
			return 0.0
		}
	}
	return ApplyUnary(arr, signFunc)
}

// Maximum computes the element-wise maximum of two arrays
func Maximum(a, b *array.Array) (*array.Array, error) {
	maxFunc := func(x, y interface{}) interface{} {
		xFloat := convertToFloat64(x)
		yFloat := convertToFloat64(y)

		var result float64
		if math.IsNaN(xFloat) || math.IsNaN(yFloat) {
			result = math.NaN()
		} else {
			result = math.Max(xFloat, yFloat)
		}

		// Convert back to original type (using x's type)
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
			return result
		}
	}
	return ApplyBinary(a, b, maxFunc)
}

// Minimum computes the element-wise minimum of two arrays
func Minimum(a, b *array.Array) (*array.Array, error) {
	minFunc := func(x, y interface{}) interface{} {
		xFloat := convertToFloat64(x)
		yFloat := convertToFloat64(y)

		var result float64
		if math.IsNaN(xFloat) || math.IsNaN(yFloat) {
			result = math.NaN()
		} else {
			result = math.Min(xFloat, yFloat)
		}

		// Convert back to original type (using x's type)
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
			return result
		}
	}
	return ApplyBinary(a, b, minFunc)
}

// Clip clips array values to be within [min, max] range
func Clip(arr *array.Array, min, max float64) (*array.Array, error) {
	clipFunc := func(x interface{}) interface{} {
		xFloat := convertToFloat64(x)

		var result float64
		if xFloat < min {
			result = min
		} else if xFloat > max {
			result = max
		} else {
			result = xFloat
		}

		// Convert back to original type
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
			return result
		}
	}
	return ApplyUnary(arr, clipFunc)
}

// Fmod computes the floating-point remainder of x/y element-wise
func Fmod(x, y *array.Array) (*array.Array, error) {
	fmodFunc := makeBinaryFloatFunc(math.Mod)
	return ApplyBinary(x, y, fmodFunc)
}

// Remainder computes the IEEE remainder of x/y element-wise
func Remainder(x, y *array.Array) (*array.Array, error) {
	remainderFunc := makeBinaryFloatFunc(math.Remainder)
	return ApplyBinary(x, y, remainderFunc)
}

// Copysign returns array with elements having the magnitude of x and the sign of y
func Copysign(x, y *array.Array) (*array.Array, error) {
	copysignFunc := makeBinaryFloatFunc(math.Copysign)
	return ApplyBinary(x, y, copysignFunc)
}

// In-place versions

// AbsInPlace computes the absolute value of array elements in-place
func AbsInPlace(arr *array.Array) error {
	absFunc := func(x interface{}) interface{} {
		switch v := x.(type) {
		case float64:
			return math.Abs(v)
		case float32:
			return float32(math.Abs(float64(v)))
		case int64:
			if v < 0 {
				return -v
			}
			return v
		case int32:
			if v < 0 {
				return -v
			}
			return v
		case int16:
			if v < 0 {
				return -v
			}
			return v
		case int8:
			if v < 0 {
				return -v
			}
			return v
		case uint64, uint32, uint16, uint8:
			return v // Unsigned types are always positive
		case complex64:
			return complex64(complex(float32(math.Abs(float64(real(v)*real(v)+imag(v)*imag(v)))), 0))
		case complex128:
			return complex(math.Sqrt(real(v)*real(v)+imag(v)*imag(v)), 0)
		default:
			return makeFloatFunc(math.Abs)(x)
		}
	}
	return ApplyUnaryInPlace(arr, absFunc)
}

// SignInPlace returns the sign of array elements in-place
func SignInPlace(arr *array.Array) error {
	signFunc := func(x interface{}) interface{} {
		switch v := x.(type) {
		case float64:
			if v > 0 {
				return 1.0
			} else if v < 0 {
				return -1.0
			}
			return 0.0
		case float32:
			if v > 0 {
				return float32(1.0)
			} else if v < 0 {
				return float32(-1.0)
			}
			return float32(0.0)
		case int64:
			if v > 0 {
				return int64(1)
			} else if v < 0 {
				return int64(-1)
			}
			return int64(0)
		case int32:
			if v > 0 {
				return int32(1)
			} else if v < 0 {
				return int32(-1)
			}
			return int32(0)
		case int16:
			if v > 0 {
				return int16(1)
			} else if v < 0 {
				return int16(-1)
			}
			return int16(0)
		case int8:
			if v > 0 {
				return int8(1)
			} else if v < 0 {
				return int8(-1)
			}
			return int8(0)
		case uint64, uint32, uint16, uint8:
			// Unsigned types: 0 -> 0, positive -> 1
			floatVal := convertToFloat64(v)
			if floatVal == 0 {
				return v // Keep original type for 0
			}
			// Return 1 in the same type
			switch v.(type) {
			case uint64:
				return uint64(1)
			case uint32:
				return uint32(1)
			case uint16:
				return uint16(1)
			case uint8:
				return uint8(1)
			}
			return v
		case complex64:
			// For complex numbers, return the unit vector in the same direction
			mag := math.Sqrt(float64(real(v)*real(v) + imag(v)*imag(v)))
			if mag == 0 {
				return complex64(0)
			}
			return complex64(complex(float32(float64(real(v))/mag), float32(float64(imag(v))/mag)))
		case complex128:
			mag := math.Sqrt(real(v)*real(v) + imag(v)*imag(v))
			if mag == 0 {
				return complex128(0)
			}
			return complex(real(v)/mag, imag(v)/mag)
		default:
			floatVal := convertToFloat64(x)
			if floatVal > 0 {
				return 1.0
			} else if floatVal < 0 {
				return -1.0
			}
			return 0.0
		}
	}
	return ApplyUnaryInPlace(arr, signFunc)
}

// ClipInPlace clips array values to be within [min, max] range in-place
func ClipInPlace(arr *array.Array, min, max float64) error {
	clipFunc := func(x interface{}) interface{} {
		xFloat := convertToFloat64(x)

		var result float64
		if xFloat < min {
			result = min
		} else if xFloat > max {
			result = max
		} else {
			result = xFloat
		}

		// Convert back to original type
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
			return result
		}
	}
	return ApplyUnaryInPlace(arr, clipFunc)
}
