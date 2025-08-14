package stats

import (
	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// tryFastSum attempts to use SIMD for sum calculation
func tryFastSum(arr *array.Array) (float64, bool) {
	// Check if we can use SIMD optimization
	if arr.DType() != internal.Float64 {
		return 0, false
	}

	// Must be contiguous for SIMD
	if !isArrayContiguous(arr) {
		return 0, false
	}

	// Check size threshold
	if arr.Size() < internal.SIMDThreshold {
		return 0, false
	}

	// Try to get raw data
	if data, ok := extractFloat64Data(arr); ok {
		sum := internal.SIMDSumFloat64(data)
		internal.DebugVerbose("Used SIMD for statistics sum, size=%d", len(data))
		return sum, true
	}

	return 0, false
}

// tryFastMean attempts to use SIMD for mean calculation
func tryFastMean(arr *array.Array) (float64, bool) {
	if sum, ok := tryFastSum(arr); ok {
		return sum / float64(arr.Size()), true
	}
	return 0, false
}

// Helper functions

// isArrayContiguous checks if array is stored contiguously
func isArrayContiguous(arr *array.Array) bool {
	// Simplified check - assumes most arrays are contiguous
	// A full implementation would check stride patterns
	return arr.Shape().Ndim() <= 2 // Simple heuristic
}

// extractFloat64Data attempts to extract raw float64 data from array
func extractFloat64Data(arr *array.Array) ([]float64, bool) {
	if arr.DType() != internal.Float64 {
		return nil, false
	}

	// For simplicity, iterate and collect values
	// In a production implementation, this would directly access storage
	size := arr.Size()
	data := make([]float64, size)

	flatArr := arr.Flatten()
	for i := 0; i < size; i++ {
		val := flatArr.At(i)
		if floatVal, ok := val.(float64); ok {
			data[i] = floatVal
		} else {
			// Type conversion failed, can't use SIMD
			return nil, false
		}
	}

	return data, true
}

// SIMD-optimized versions of statistical functions

// FastSum calculates sum using SIMD optimization when possible
func FastSum(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "FastSum", "array"); err != nil {
		return 0, err
	}

	// Try SIMD optimization first
	if sum, ok := tryFastSum(arr); ok {
		return sum, nil
	}

	// Fall back to regular Sum
	return Sum(arr)
}

// FastMean calculates mean using SIMD optimization when possible
func FastMean(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "FastMean", "array"); err != nil {
		return 0, err
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "FastMean", "array"); err != nil {
		return 0, err
	}

	// Try SIMD optimization first
	if mean, ok := tryFastMean(arr); ok {
		return mean, nil
	}

	// Fall back to regular Mean
	return Mean(arr)
}
