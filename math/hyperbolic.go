package math

import (
	"math"
	"math/cmplx"

	"github.com/julianshen/gonp/array"
)

// Sinh computes the hyperbolic sine of array elements
func Sinh(arr *array.Array) (*array.Array, error) {
	sinhFunc := makeComplexFunc(math.Sinh, cmplx.Sinh)
	return ApplyUnary(arr, sinhFunc)
}

// Cosh computes the hyperbolic cosine of array elements
func Cosh(arr *array.Array) (*array.Array, error) {
	coshFunc := makeComplexFunc(math.Cosh, cmplx.Cosh)
	return ApplyUnary(arr, coshFunc)
}

// Tanh computes the hyperbolic tangent of array elements
func Tanh(arr *array.Array) (*array.Array, error) {
	tanhFunc := makeComplexFunc(math.Tanh, cmplx.Tanh)
	return ApplyUnary(arr, tanhFunc)
}

// Asinh computes the inverse hyperbolic sine of array elements
func Asinh(arr *array.Array) (*array.Array, error) {
	asinhFunc := makeComplexFunc(math.Asinh, cmplx.Asinh)
	return ApplyUnary(arr, asinhFunc)
}

// Acosh computes the inverse hyperbolic cosine of array elements
func Acosh(arr *array.Array) (*array.Array, error) {
	acoshFunc := makeComplexFunc(math.Acosh, cmplx.Acosh)
	return ApplyUnary(arr, acoshFunc)
}

// Atanh computes the inverse hyperbolic tangent of array elements
func Atanh(arr *array.Array) (*array.Array, error) {
	atanhFunc := makeComplexFunc(math.Atanh, cmplx.Atanh)
	return ApplyUnary(arr, atanhFunc)
}

// In-place versions

// SinhInPlace computes the hyperbolic sine of array elements in-place
func SinhInPlace(arr *array.Array) error {
	sinhFunc := makeComplexFunc(math.Sinh, cmplx.Sinh)
	return ApplyUnaryInPlace(arr, sinhFunc)
}

// CoshInPlace computes the hyperbolic cosine of array elements in-place
func CoshInPlace(arr *array.Array) error {
	coshFunc := makeComplexFunc(math.Cosh, cmplx.Cosh)
	return ApplyUnaryInPlace(arr, coshFunc)
}

// TanhInPlace computes the hyperbolic tangent of array elements in-place
func TanhInPlace(arr *array.Array) error {
	tanhFunc := makeComplexFunc(math.Tanh, cmplx.Tanh)
	return ApplyUnaryInPlace(arr, tanhFunc)
}

// AsinhInPlace computes the inverse hyperbolic sine of array elements in-place
func AsinhInPlace(arr *array.Array) error {
	asinhFunc := makeComplexFunc(math.Asinh, cmplx.Asinh)
	return ApplyUnaryInPlace(arr, asinhFunc)
}

// AcoshInPlace computes the inverse hyperbolic cosine of array elements in-place
func AcoshInPlace(arr *array.Array) error {
	acoshFunc := makeComplexFunc(math.Acosh, cmplx.Acosh)
	return ApplyUnaryInPlace(arr, acoshFunc)
}

// AtanhInPlace computes the inverse hyperbolic tangent of array elements in-place
func AtanhInPlace(arr *array.Array) error {
	atanhFunc := makeComplexFunc(math.Atanh, cmplx.Atanh)
	return ApplyUnaryInPlace(arr, atanhFunc)
}
