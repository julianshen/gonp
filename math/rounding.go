package math

import (
	"math"

	"github.com/julianshen/gonp/array"
)

// Round rounds array elements to the nearest integer
func Round(arr *array.Array) (*array.Array, error) {
	roundFunc := makeFloatFunc(math.Round)
	return ApplyUnary(arr, roundFunc)
}

// Floor returns the floor of array elements (largest integer <= x)
func Floor(arr *array.Array) (*array.Array, error) {
	floorFunc := makeFloatFunc(math.Floor)
	return ApplyUnary(arr, floorFunc)
}

// Ceil returns the ceiling of array elements (smallest integer >= x)
func Ceil(arr *array.Array) (*array.Array, error) {
	ceilFunc := makeFloatFunc(math.Ceil)
	return ApplyUnary(arr, ceilFunc)
}

// Trunc truncates array elements to integer (towards zero)
func Trunc(arr *array.Array) (*array.Array, error) {
	truncFunc := makeFloatFunc(math.Trunc)
	return ApplyUnary(arr, truncFunc)
}

// RoundN rounds array elements to n decimal places
func RoundN(arr *array.Array, n int) (*array.Array, error) {
	factor := math.Pow(10, float64(n))
	roundNFunc := makeFloatFunc(func(x float64) float64 {
		return math.Round(x*factor) / factor
	})
	return ApplyUnary(arr, roundNFunc)
}

// Fix rounds array elements towards zero (same as Trunc)
func Fix(arr *array.Array) (*array.Array, error) {
	return Trunc(arr)
}

// Rint rounds array elements to nearest integer (using current rounding mode)
func Rint(arr *array.Array) (*array.Array, error) {
	rintFunc := makeFloatFunc(math.RoundToEven)
	return ApplyUnary(arr, rintFunc)
}

// In-place versions

// RoundInPlace rounds array elements to the nearest integer in-place
func RoundInPlace(arr *array.Array) error {
	roundFunc := makeFloatFunc(math.Round)
	return ApplyUnaryInPlace(arr, roundFunc)
}

// FloorInPlace returns the floor of array elements in-place
func FloorInPlace(arr *array.Array) error {
	floorFunc := makeFloatFunc(math.Floor)
	return ApplyUnaryInPlace(arr, floorFunc)
}

// CeilInPlace returns the ceiling of array elements in-place
func CeilInPlace(arr *array.Array) error {
	ceilFunc := makeFloatFunc(math.Ceil)
	return ApplyUnaryInPlace(arr, ceilFunc)
}

// TruncInPlace truncates array elements to integer in-place
func TruncInPlace(arr *array.Array) error {
	truncFunc := makeFloatFunc(math.Trunc)
	return ApplyUnaryInPlace(arr, truncFunc)
}

// RoundNInPlace rounds array elements to n decimal places in-place
func RoundNInPlace(arr *array.Array, n int) error {
	factor := math.Pow(10, float64(n))
	roundNFunc := makeFloatFunc(func(x float64) float64 {
		return math.Round(x*factor) / factor
	})
	return ApplyUnaryInPlace(arr, roundNFunc)
}

// FixInPlace rounds array elements towards zero in-place
func FixInPlace(arr *array.Array) error {
	return TruncInPlace(arr)
}

// RintInPlace rounds array elements to nearest integer in-place
func RintInPlace(arr *array.Array) error {
	rintFunc := makeFloatFunc(math.RoundToEven)
	return ApplyUnaryInPlace(arr, rintFunc)
}
