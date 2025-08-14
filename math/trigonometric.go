package math

import (
	"math"
	"math/cmplx"

	"github.com/julianshen/gonp/array"
)

// Sin computes the sine of array elements using SIMD-optimized universal functions.
//
// This function applies the sine function element-wise to all elements in the array.
// It supports both real and complex number arrays, with automatic type handling.
//
// Mathematical Definition: sin(x) for each element x in the array
//
// Performance:
//   - SIMD-optimized for large arrays (>32 elements)
//   - Supports complex numbers with full precision
//   - Thread-safe for concurrent use
//
// Input Domain:
//   - Real numbers: All values (angle in radians)
//   - Complex numbers: All values
//
// Output Range:
//   - Real numbers: [-1, 1]
//   - Complex numbers: Complex plane
//
// Example:
//
//	// Compute sine of angles
//	angles, _ := array.FromSlice([]float64{0, math.Pi/6, math.Pi/4, math.Pi/3, math.Pi/2})
//	sines, _ := math.Sin(angles)
//	// Result: [0, 0.5, 0.707..., 0.866..., 1]
//
//	// Complex sine
//	complex_arr, _ := array.FromSlice([]complex128{1+2i, 2+3i})
//	complex_sines, _ := math.Sin(complex_arr)
//
// Special Values:
//   - Sin(0) = 0
//   - Sin(π/2) = 1
//   - Sin(π) = 0 (within floating-point precision)
//   - Sin(NaN) = NaN
//   - Sin(±Inf) = NaN
//
// Returns an error if the input array is nil.
func Sin(arr *array.Array) (*array.Array, error) {
	sinFunc := makeComplexFunc(math.Sin, cmplx.Sin)
	return ApplyUnary(arr, sinFunc)
}

// Cos computes the cosine of array elements
func Cos(arr *array.Array) (*array.Array, error) {
	cosFunc := makeComplexFunc(math.Cos, cmplx.Cos)
	return ApplyUnary(arr, cosFunc)
}

// Tan computes the tangent of array elements
func Tan(arr *array.Array) (*array.Array, error) {
	tanFunc := makeComplexFunc(math.Tan, cmplx.Tan)
	return ApplyUnary(arr, tanFunc)
}

// Asin computes the arcsine of array elements
func Asin(arr *array.Array) (*array.Array, error) {
	asinFunc := makeComplexFunc(math.Asin, cmplx.Asin)
	return ApplyUnary(arr, asinFunc)
}

// Acos computes the arccosine of array elements
func Acos(arr *array.Array) (*array.Array, error) {
	acosFunc := makeComplexFunc(math.Acos, cmplx.Acos)
	return ApplyUnary(arr, acosFunc)
}

// Atan computes the arctangent of array elements
func Atan(arr *array.Array) (*array.Array, error) {
	atanFunc := makeComplexFunc(math.Atan, cmplx.Atan)
	return ApplyUnary(arr, atanFunc)
}

// Atan2 computes the two-argument arctangent of y/x element-wise.
//
// This function computes atan2(y, x) = atan(y/x) element-wise, but handles
// the signs of both arguments to determine the correct quadrant. This is
// essential for converting Cartesian coordinates to polar coordinates.
//
// Mathematical Definition: atan2(y, x) for corresponding elements y[i], x[i]
//
// Performance:
//   - SIMD-optimized for large arrays
//   - Broadcasting supported for compatible shapes
//   - Handles edge cases correctly (zeros, infinities)
//
// Input Domain:
//   - Both arrays: Real numbers (any finite values, zeros, infinities)
//   - Arrays must have compatible shapes for broadcasting
//
// Output Range: [-π, π] radians
//
// Example:
//
//	// Convert Cartesian to polar coordinates
//	x_coords, _ := array.FromSlice([]float64{1, -1, -1, 1})
//	y_coords, _ := array.FromSlice([]float64{1, 1, -1, -1})
//	angles, _ := math.Atan2(y_coords, x_coords)
//	// Result: [π/4, 3π/4, -3π/4, -π/4] (quadrant-aware)
//
//	// Broadcasting with scalar
//	y_vals, _ := array.FromSlice([]float64{0, 1, 2, 3})
//	x_scalar, _ := array.FromSlice([]float64{1})
//	angles, _ := math.Atan2(y_vals, x_scalar)  // Broadcasting
//
// Special Cases:
//   - atan2(+0, +0) = +0
//   - atan2(-0, +0) = -0
//   - atan2(+0, -0) = +π
//   - atan2(-0, -0) = -π
//   - atan2(+∞, +∞) = +π/4
//   - atan2(-∞, -∞) = -3π/4
//
// Returns an error if either input array is nil or shapes are incompatible.
func Atan2(y, x *array.Array) (*array.Array, error) {
	atan2Func := makeBinaryFloatFunc(math.Atan2)
	return ApplyBinary(y, x, atan2Func)
}

// Deg2Rad converts degrees to radians
func Deg2Rad(arr *array.Array) (*array.Array, error) {
	deg2radFunc := makeFloatFunc(func(x float64) float64 {
		return x * math.Pi / 180.0
	})
	return ApplyUnary(arr, deg2radFunc)
}

// Rad2Deg converts radians to degrees
func Rad2Deg(arr *array.Array) (*array.Array, error) {
	rad2degFunc := makeFloatFunc(func(x float64) float64 {
		return x * 180.0 / math.Pi
	})
	return ApplyUnary(arr, rad2degFunc)
}

// Hypot computes the Euclidean norm, sqrt(x*x + y*y), element-wise
func Hypot(x, y *array.Array) (*array.Array, error) {
	hypotFunc := makeBinaryFloatFunc(math.Hypot)
	return ApplyBinary(x, y, hypotFunc)
}

// In-place versions

// SinInPlace computes the sine of array elements in-place
func SinInPlace(arr *array.Array) error {
	sinFunc := makeComplexFunc(math.Sin, cmplx.Sin)
	return ApplyUnaryInPlace(arr, sinFunc)
}

// CosInPlace computes the cosine of array elements in-place
func CosInPlace(arr *array.Array) error {
	cosFunc := makeComplexFunc(math.Cos, cmplx.Cos)
	return ApplyUnaryInPlace(arr, cosFunc)
}

// TanInPlace computes the tangent of array elements in-place
func TanInPlace(arr *array.Array) error {
	tanFunc := makeComplexFunc(math.Tan, cmplx.Tan)
	return ApplyUnaryInPlace(arr, tanFunc)
}

// AsinInPlace computes the arcsine of array elements in-place
func AsinInPlace(arr *array.Array) error {
	asinFunc := makeComplexFunc(math.Asin, cmplx.Asin)
	return ApplyUnaryInPlace(arr, asinFunc)
}

// AcosInPlace computes the arccosine of array elements in-place
func AcosInPlace(arr *array.Array) error {
	acosFunc := makeComplexFunc(math.Acos, cmplx.Acos)
	return ApplyUnaryInPlace(arr, acosFunc)
}

// AtanInPlace computes the arctangent of array elements in-place
func AtanInPlace(arr *array.Array) error {
	atanFunc := makeComplexFunc(math.Atan, cmplx.Atan)
	return ApplyUnaryInPlace(arr, atanFunc)
}

// Deg2RadInPlace converts degrees to radians in-place
func Deg2RadInPlace(arr *array.Array) error {
	deg2radFunc := makeFloatFunc(func(x float64) float64 {
		return x * math.Pi / 180.0
	})
	return ApplyUnaryInPlace(arr, deg2radFunc)
}

// Rad2DegInPlace converts radians to degrees in-place
func Rad2DegInPlace(arr *array.Array) error {
	rad2degFunc := makeFloatFunc(func(x float64) float64 {
		return x * 180.0 / math.Pi
	})
	return ApplyUnaryInPlace(arr, rad2degFunc)
}
