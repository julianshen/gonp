package math

import (
	"math"
	"math/cmplx"

	"github.com/julianshen/gonp/array"
)

// Exp computes the exponential of array elements
func Exp(arr *array.Array) (*array.Array, error) {
	expFunc := makeComplexFunc(math.Exp, cmplx.Exp)
	return ApplyUnary(arr, expFunc)
}

// Exp2 computes 2^x for array elements
func Exp2(arr *array.Array) (*array.Array, error) {
	exp2Func := makeFloatFunc(math.Exp2)
	return ApplyUnary(arr, exp2Func)
}

// Exp10 computes 10^x for array elements
func Exp10(arr *array.Array) (*array.Array, error) {
	exp10Func := makeFloatFunc(func(x float64) float64 {
		return math.Pow(10, x)
	})
	return ApplyUnary(arr, exp10Func)
}

// Expm1 computes exp(x) - 1 for array elements
func Expm1(arr *array.Array) (*array.Array, error) {
	expm1Func := makeFloatFunc(math.Expm1)
	return ApplyUnary(arr, expm1Func)
}

// Log computes the natural logarithm of array elements
func Log(arr *array.Array) (*array.Array, error) {
	logFunc := makeComplexFunc(math.Log, cmplx.Log)
	return ApplyUnary(arr, logFunc)
}

// Log2 computes the base-2 logarithm of array elements
func Log2(arr *array.Array) (*array.Array, error) {
	log2Func := makeFloatFunc(math.Log2)
	return ApplyUnary(arr, log2Func)
}

// Log10 computes the base-10 logarithm of array elements
func Log10(arr *array.Array) (*array.Array, error) {
	log10Func := makeComplexFunc(math.Log10, cmplx.Log10)
	return ApplyUnary(arr, log10Func)
}

// Log1p computes log(1 + x) for array elements
func Log1p(arr *array.Array) (*array.Array, error) {
	log1pFunc := makeFloatFunc(math.Log1p)
	return ApplyUnary(arr, log1pFunc)
}

// Logb computes the binary logarithm of array elements
func Logb(arr *array.Array) (*array.Array, error) {
	logbFunc := makeFloatFunc(math.Logb)
	return ApplyUnary(arr, logbFunc)
}

// In-place versions

// ExpInPlace computes the exponential of array elements in-place
func ExpInPlace(arr *array.Array) error {
	expFunc := makeComplexFunc(math.Exp, cmplx.Exp)
	return ApplyUnaryInPlace(arr, expFunc)
}

// Exp2InPlace computes 2^x for array elements in-place
func Exp2InPlace(arr *array.Array) error {
	exp2Func := makeFloatFunc(math.Exp2)
	return ApplyUnaryInPlace(arr, exp2Func)
}

// Exp10InPlace computes 10^x for array elements in-place
func Exp10InPlace(arr *array.Array) error {
	exp10Func := makeFloatFunc(func(x float64) float64 {
		return math.Pow(10, x)
	})
	return ApplyUnaryInPlace(arr, exp10Func)
}

// Expm1InPlace computes exp(x) - 1 for array elements in-place
func Expm1InPlace(arr *array.Array) error {
	expm1Func := makeFloatFunc(math.Expm1)
	return ApplyUnaryInPlace(arr, expm1Func)
}

// LogInPlace computes the natural logarithm of array elements in-place
func LogInPlace(arr *array.Array) error {
	logFunc := makeComplexFunc(math.Log, cmplx.Log)
	return ApplyUnaryInPlace(arr, logFunc)
}

// Log2InPlace computes the base-2 logarithm of array elements in-place
func Log2InPlace(arr *array.Array) error {
	log2Func := makeFloatFunc(math.Log2)
	return ApplyUnaryInPlace(arr, log2Func)
}

// Log10InPlace computes the base-10 logarithm of array elements in-place
func Log10InPlace(arr *array.Array) error {
	log10Func := makeComplexFunc(math.Log10, cmplx.Log10)
	return ApplyUnaryInPlace(arr, log10Func)
}

// Log1pInPlace computes log(1 + x) for array elements in-place
func Log1pInPlace(arr *array.Array) error {
	log1pFunc := makeFloatFunc(math.Log1p)
	return ApplyUnaryInPlace(arr, log1pFunc)
}

// LogbInPlace computes the binary logarithm of array elements in-place
func LogbInPlace(arr *array.Array) error {
	logbFunc := makeFloatFunc(math.Logb)
	return ApplyUnaryInPlace(arr, logbFunc)
}
