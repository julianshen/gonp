package array

import (
	"math"
)

// AllClose returns true if two arrays are element-wise equal within a tolerance
// The tolerance values are: rtol * abs(b) + atol
// This function follows NumPy's allclose behavior
func AllClose(a, b *Array, rtol, atol float64) bool {
	// Check if shapes are compatible
	if !a.shape.Equal(b.shape) {
		return false
	}

	size := a.Size()

	// Compare each element
	for i := 0; i < size; i++ {
		indices := a.unflattenIndex(i)
		val1 := a.convertToFloat64(a.At(indices...))
		val2 := a.convertToFloat64(b.At(indices...))

		// Handle NaN: if either value is NaN, return false
		if math.IsNaN(val1) || math.IsNaN(val2) {
			return false
		}

		// Handle infinity
		if math.IsInf(val1, 0) || math.IsInf(val2, 0) {
			if val1 != val2 {
				return false
			}
			continue
		}

		// Calculate tolerance: atol + rtol * abs(val2)
		// This matches NumPy's allclose formula
		tolerance := atol + rtol*math.Abs(val2)

		// Check if values are close: |a - b| <= atol + rtol * |b|
		if math.Abs(val1-val2) > tolerance {
			return false
		}
	}

	return true
}

// ArrayEqual returns true if two arrays have the same shape and elements
// This is an exact equality check (no tolerance)
func ArrayEqual(a, b *Array) bool {
	// Check if shapes are compatible
	if !a.shape.Equal(b.shape) {
		return false
	}

	size := a.Size()

	// Compare each element exactly
	for i := 0; i < size; i++ {
		indices := a.unflattenIndex(i)
		val1 := a.convertToFloat64(a.At(indices...))
		val2 := a.convertToFloat64(b.At(indices...))

		// Handle NaN: NaN != NaN, so return false if either is NaN
		if math.IsNaN(val1) || math.IsNaN(val2) {
			return false
		}

		// Exact equality check
		if val1 != val2 {
			return false
		}
	}

	return true
}
