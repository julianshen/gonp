package math

import (
	"fmt"
	gmath "math"

	"github.com/julianshen/gonp/array"
)

// Kahan summation algorithm for compensated summation with reduced floating-point errors
// This algorithm significantly improves precision when summing many floating-point numbers
// by keeping track of and correcting for the accumulated errors.

// KahanSum performs Kahan summation algorithm on an array
// This provides much better numerical stability than naive summation for large arrays
func KahanSum(a *array.Array) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("array cannot be nil")
	}

	if a.Size() == 0 {
		return 0, nil
	}

	sum := 0.0
	c := 0.0 // A running compensation for lost low-order bits

	for i := 0; i < a.Size(); i++ {
		val, err := convertValueToFloat64(a.At(i))
		if err != nil {
			return 0, fmt.Errorf("failed to convert value at index %d: %v", i, err)
		}

		// Skip NaN and infinite values
		if gmath.IsNaN(val) || gmath.IsInf(val, 0) {
			continue
		}

		y := val - c      // So far, so good: c is zero
		t := sum + y      // Alas, sum is big, y small, so low-order digits of y are lost
		c = (t - sum) - y // (t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
		sum = t           // Algebraically, c should always be zero. Beware overly-clever compilers!
	}

	return sum, nil
}

// KahanMean computes the mean using Kahan summation for better precision
func KahanMean(a *array.Array) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("array cannot be nil")
	}

	if a.Size() == 0 {
		return gmath.NaN(), nil
	}

	sum, err := KahanSum(a)
	if err != nil {
		return 0, err
	}

	validCount := 0
	for i := 0; i < a.Size(); i++ {
		val, err := convertValueToFloat64(a.At(i))
		if err != nil {
			continue
		}
		if !gmath.IsNaN(val) && !gmath.IsInf(val, 0) {
			validCount++
		}
	}

	if validCount == 0 {
		return gmath.NaN(), nil
	}

	return sum / float64(validCount), nil
}

// KahanVariance computes variance using Kahan summation for numerical stability
// Uses the two-pass algorithm with compensated summation
func KahanVariance(a *array.Array, ddof int) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("array cannot be nil")
	}

	if a.Size() == 0 {
		return gmath.NaN(), nil
	}

	// First pass: compute mean using Kahan summation
	mean, err := KahanMean(a)
	if err != nil {
		return 0, err
	}

	if gmath.IsNaN(mean) {
		return gmath.NaN(), nil
	}

	// Second pass: compute sum of squared deviations using Kahan summation
	sum := 0.0
	c := 0.0 // Compensation term
	validCount := 0

	for i := 0; i < a.Size(); i++ {
		val, err := convertValueToFloat64(a.At(i))
		if err != nil {
			continue
		}

		if gmath.IsNaN(val) || gmath.IsInf(val, 0) {
			continue
		}

		deviation := val - mean
		squaredDev := deviation * deviation

		// Kahan summation
		y := squaredDev - c
		t := sum + y
		c = (t - sum) - y
		sum = t
		validCount++
	}

	if validCount <= ddof {
		return gmath.NaN(), nil
	}

	return sum / float64(validCount-ddof), nil
}

// KahanStandardDeviation computes standard deviation using Kahan-based variance
func KahanStandardDeviation(a *array.Array, ddof int) (float64, error) {
	variance, err := KahanVariance(a, ddof)
	if err != nil {
		return 0, err
	}

	if gmath.IsNaN(variance) || variance < 0 {
		return gmath.NaN(), nil
	}

	return gmath.Sqrt(variance), nil
}

// NeumaierSum implements Neumaier's improved Kahan summation algorithm
// This is even more accurate than standard Kahan summation for certain cases
func NeumaierSum(a *array.Array) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("array cannot be nil")
	}

	if a.Size() == 0 {
		return 0, nil
	}

	sum := 0.0
	c := 0.0 // Compensation term

	for i := 0; i < a.Size(); i++ {
		val, err := convertValueToFloat64(a.At(i))
		if err != nil {
			return 0, fmt.Errorf("failed to convert value at index %d: %v", i, err)
		}

		// Skip NaN and infinite values
		if gmath.IsNaN(val) || gmath.IsInf(val, 0) {
			continue
		}

		t := sum + val
		if gmath.Abs(sum) >= gmath.Abs(val) {
			c += (sum - t) + val // If sum is bigger, low-order digits of val are lost
		} else {
			c += (val - t) + sum // Else low-order digits of sum are lost
		}
		sum = t
	}

	return sum + c, nil
}

// PairwiseSum implements pairwise summation for improved numerical stability
// This algorithm has O(log n) error accumulation instead of O(n) for naive summation
func PairwiseSum(a *array.Array) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("array cannot be nil")
	}

	if a.Size() == 0 {
		return 0, nil
	}

	// Convert array to float64 slice, skipping invalid values
	values := make([]float64, 0, a.Size())
	for i := 0; i < a.Size(); i++ {
		val, err := convertValueToFloat64(a.At(i))
		if err != nil {
			continue
		}
		if !gmath.IsNaN(val) && !gmath.IsInf(val, 0) {
			values = append(values, val)
		}
	}

	if len(values) == 0 {
		return 0, nil
	}

	return pairwiseSumRecursive(values), nil
}

// pairwiseSumRecursive recursively computes pairwise sum
func pairwiseSumRecursive(values []float64) float64 {
	n := len(values)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return values[0]
	}
	if n == 2 {
		return values[0] + values[1]
	}

	// Divide and conquer
	mid := n / 2
	left := pairwiseSumRecursive(values[:mid])
	right := pairwiseSumRecursive(values[mid:])
	return left + right
}

// CompensatedDotProduct computes dot product with Kahan summation for better precision
func CompensatedDotProduct(a, b *array.Array) (float64, error) {
	if a == nil || b == nil {
		return 0, fmt.Errorf("arrays cannot be nil")
	}

	if a.Size() != b.Size() {
		return 0, fmt.Errorf("arrays must have the same size: %d vs %d", a.Size(), b.Size())
	}

	if a.Size() == 0 {
		return 0, nil
	}

	sum := 0.0
	c := 0.0 // Compensation term

	for i := 0; i < a.Size(); i++ {
		valA, err := convertValueToFloat64(a.At(i))
		if err != nil {
			return 0, fmt.Errorf("failed to convert value from array a at index %d: %v", i, err)
		}

		valB, err := convertValueToFloat64(b.At(i))
		if err != nil {
			return 0, fmt.Errorf("failed to convert value from array b at index %d: %v", i, err)
		}

		// Skip if either value is NaN or infinite
		if gmath.IsNaN(valA) || gmath.IsInf(valA, 0) || gmath.IsNaN(valB) || gmath.IsInf(valB, 0) {
			continue
		}

		product := valA * valB

		// Kahan summation
		y := product - c
		t := sum + y
		c = (t - sum) - y
		sum = t
	}

	return sum, nil
}

// CompensatedNorm computes the L2 norm with enhanced numerical stability
func CompensatedNorm(a *array.Array) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("array cannot be nil")
	}

	if a.Size() == 0 {
		return 0, nil
	}

	// Use Kahan summation for the sum of squares
	sum := 0.0
	c := 0.0

	for i := 0; i < a.Size(); i++ {
		val, err := convertValueToFloat64(a.At(i))
		if err != nil {
			return 0, fmt.Errorf("failed to convert value at index %d: %v", i, err)
		}

		if gmath.IsNaN(val) || gmath.IsInf(val, 0) {
			continue
		}

		square := val * val

		// Kahan summation
		y := square - c
		t := sum + y
		c = (t - sum) - y
		sum = t
	}

	return gmath.Sqrt(sum), nil
}

// Helper function to convert interface{} to float64 with comprehensive type support
func convertValueToFloat64(value interface{}) (float64, error) {
	switch v := value.(type) {
	case float64:
		return v, nil
	case float32:
		return float64(v), nil
	case int:
		return float64(v), nil
	case int64:
		return float64(v), nil
	case int32:
		return float64(v), nil
	case int16:
		return float64(v), nil
	case int8:
		return float64(v), nil
	case uint:
		return float64(v), nil
	case uint64:
		return float64(v), nil
	case uint32:
		return float64(v), nil
	case uint16:
		return float64(v), nil
	case uint8:
		return float64(v), nil
	case bool:
		if v {
			return 1.0, nil
		}
		return 0.0, nil
	case complex64:
		// Use the magnitude for complex numbers
		return gmath.Sqrt(float64(real(v)*real(v) + imag(v)*imag(v))), nil
	case complex128:
		// Use the magnitude for complex numbers
		return gmath.Sqrt(real(v)*real(v) + imag(v)*imag(v)), nil
	default:
		return 0, fmt.Errorf("cannot convert %T to float64", value)
	}
}

// ExtendedPrecisionSum uses extended precision arithmetic for critical summations
// This uses a form of double-double arithmetic for even better precision
func ExtendedPrecisionSum(a *array.Array) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("array cannot be nil")
	}

	if a.Size() == 0 {
		return 0, nil
	}

	// Double-double arithmetic: represent each number as sum of two floats
	sumHi := 0.0
	sumLo := 0.0

	for i := 0; i < a.Size(); i++ {
		val, err := convertValueToFloat64(a.At(i))
		if err != nil {
			return 0, fmt.Errorf("failed to convert value at index %d: %v", i, err)
		}

		if gmath.IsNaN(val) || gmath.IsInf(val, 0) {
			continue
		}

		// Add val to (sumHi, sumLo) using exact two-sum
		s := sumHi + val
		v := s - sumHi
		e := (sumHi - (s - v)) + (val - v)
		sumHi = s
		sumLo = sumLo + e
	}

	// Final normalization
	s := sumHi + sumLo
	_ = sumLo - (s - sumHi) // Error term (not used in this simplified version)

	// Return the high-order part (s contains most significant bits)
	// In a full implementation, we'd return both s and the error term
	return s, nil
}
