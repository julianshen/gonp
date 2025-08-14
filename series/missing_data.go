package series

import (
	"fmt"
	"math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Missing data handling methods

// IsNull returns a boolean Series indicating null/NaN values
func (s *Series) IsNull() *Series {
	result := array.Empty(internal.Shape{s.Len()}, internal.Bool)

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		isNull := isNaN(val)
		result.Set(isNull, i)
	}

	resultSeries, _ := NewSeries(result, s.index.Copy(), s.name)
	return resultSeries
}

// NotNull returns a boolean Series indicating non-null values
func (s *Series) NotNull() *Series {
	result := array.Empty(internal.Shape{s.Len()}, internal.Bool)

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		isNotNull := !isNaN(val)
		result.Set(isNotNull, i)
	}

	resultSeries, _ := NewSeries(result, s.index.Copy(), s.name)
	return resultSeries
}

// DropNa returns a new Series with null/NaN values removed
func (s *Series) DropNa() *Series {
	nonNullPositions := make([]int, 0)

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		if !isNaN(val) {
			nonNullPositions = append(nonNullPositions, i)
		}
	}

	if len(nonNullPositions) == 0 {
		return Empty(s.DType(), s.name)
	}

	result, _ := s.ILoc(nonNullPositions)
	return result
}

// FillNa returns a new Series with null/NaN values filled with the specified value
func (s *Series) FillNa(value interface{}) (*Series, error) {
	result := array.Empty(s.data.Shape(), s.data.DType())

	// Convert fill value to appropriate type
	fillValue, err := convertValue(value, s.DType())
	if err != nil {
		return nil, fmt.Errorf("failed to convert fill value: %v", err)
	}

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)

		var resultVal interface{}
		if isNaN(val) {
			resultVal = fillValue
		} else {
			resultVal = val
		}

		err := result.Set(resultVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set value at index %d: %v", i, err)
		}
	}

	return NewSeries(result, s.index.Copy(), s.name)
}

// FillNaMethod fills null/NaN values using the specified method
type FillMethod string

const (
	FillMethodPad      FillMethod = "pad"      // Forward fill (use previous value)
	FillMethodFfill    FillMethod = "ffill"    // Forward fill (same as pad)
	FillMethodBfill    FillMethod = "bfill"    // Backward fill (use next value)
	FillMethodBackfill FillMethod = "backfill" // Backward fill (same as bfill)
)

// FillNaMethod returns a new Series with null/NaN values filled using the specified method
func (s *Series) FillNaMethod(method FillMethod) (*Series, error) {
	result := array.Empty(s.data.Shape(), s.data.DType())

	// Copy all values first
	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		result.Set(val, i)
	}

	switch method {
	case FillMethodPad, FillMethodFfill:
		// Forward fill
		var lastValidValue interface{}
		hasValidValue := false

		for i := 0; i < s.Len(); i++ {
			val := s.At(i)

			if !isNaN(val) {
				lastValidValue = val
				hasValidValue = true
			} else if hasValidValue {
				result.Set(lastValidValue, i)
			}
		}

	case FillMethodBfill, FillMethodBackfill:
		// Backward fill
		var nextValidValue interface{}
		hasValidValue := false

		// First pass: find next valid values
		for i := s.Len() - 1; i >= 0; i-- {
			val := s.At(i)

			if !isNaN(val) {
				nextValidValue = val
				hasValidValue = true
			} else if hasValidValue {
				result.Set(nextValidValue, i)
			}
		}

	default:
		return nil, fmt.Errorf("unknown fill method: %s", method)
	}

	return NewSeries(result, s.index.Copy(), s.name)
}

// Interpolate fills null/NaN values using linear interpolation (for numeric data)
func (s *Series) Interpolate() (*Series, error) {
	if !isNumericDType(s.DType()) {
		return nil, fmt.Errorf("interpolation is only supported for numeric data types")
	}

	result := array.Empty(s.data.Shape(), internal.Float64) // Always use float64 for interpolation

	// Convert all values to float64 and identify NaN positions
	values := make([]float64, s.Len())
	nanPositions := make([]bool, s.Len())

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		if isNaN(val) {
			values[i] = math.NaN()
			nanPositions[i] = true
		} else {
			floatVal, err := convertToFloat64(val)
			if err != nil {
				return nil, fmt.Errorf("failed to convert value to float64 at index %d: %v", i, err)
			}
			values[i] = floatVal
			nanPositions[i] = false
		}
	}

	// Perform linear interpolation
	for i := 0; i < len(values); i++ {
		if nanPositions[i] {
			// Find previous and next non-NaN values
			prevIdx := -1
			nextIdx := -1

			// Look backward for previous valid value
			for j := i - 1; j >= 0; j-- {
				if !nanPositions[j] {
					prevIdx = j
					break
				}
			}

			// Look forward for next valid value
			for j := i + 1; j < len(values); j++ {
				if !nanPositions[j] {
					nextIdx = j
					break
				}
			}

			// Interpolate if both boundaries exist
			if prevIdx != -1 && nextIdx != -1 {
				prevVal := values[prevIdx]
				nextVal := values[nextIdx]

				// Linear interpolation
				progress := float64(i-prevIdx) / float64(nextIdx-prevIdx)
				values[i] = prevVal + progress*(nextVal-prevVal)
			} else if prevIdx != -1 {
				// Only previous value exists - forward fill
				values[i] = values[prevIdx]
			} else if nextIdx != -1 {
				// Only next value exists - backward fill
				values[i] = values[nextIdx]
			}
			// If neither exists, leave as NaN
		}
	}

	// Set interpolated values
	for i := 0; i < len(values); i++ {
		result.Set(values[i], i)
	}

	return NewSeries(result, s.index.Copy(), s.name)
}

// NullCount returns the number of null/NaN values
func (s *Series) NullCount() int {
	count := 0
	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		if isNaN(val) {
			count++
		}
	}
	return count
}

// NotNullCount returns the number of non-null values
func (s *Series) NotNullCount() int {
	return s.Len() - s.NullCount()
}

// HasNulls returns true if the Series contains any null/NaN values
func (s *Series) HasNulls() bool {
	return s.NullCount() > 0
}

// isNumericDType checks if a data type is numeric
func isNumericDType(dtype internal.DType) bool {
	switch dtype {
	case internal.Int8, internal.Int16, internal.Int32, internal.Int64,
		internal.Uint8, internal.Uint16, internal.Uint32, internal.Uint64,
		internal.Float32, internal.Float64,
		internal.Complex64, internal.Complex128:
		return true
	default:
		return false
	}
}

// Advanced missing data operations

// FillNaWithSeries fills null/NaN values using values from another Series (aligned by index)
func (s *Series) FillNaWithSeries(other *Series) (*Series, error) {
	if other == nil {
		return nil, fmt.Errorf("fill Series cannot be nil")
	}

	result := array.Empty(s.data.Shape(), s.data.DType())

	// Create index lookup for other series
	otherIndexMap := make(map[interface{}]int)
	otherIndices := other.index.Values()
	for i, idx := range otherIndices {
		otherIndexMap[idx] = i
	}

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)

		var resultVal interface{}
		if isNaN(val) {
			// Try to get value from other series at same index
			indexLabel := s.index.Get(i)
			if otherPos, exists := otherIndexMap[indexLabel]; exists {
				otherVal := other.At(otherPos)
				if !isNaN(otherVal) {
					// Convert to appropriate type
					convertedVal, err := convertValue(otherVal, s.DType())
					if err != nil {
						return nil, fmt.Errorf("failed to convert fill value at index %v: %v", indexLabel, err)
					}
					resultVal = convertedVal
				} else {
					resultVal = val // Keep NaN if other value is also NaN
				}
			} else {
				resultVal = val // Keep NaN if index not found in other series
			}
		} else {
			resultVal = val
		}

		err := result.Set(resultVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set value at index %d: %v", i, err)
		}
	}

	return NewSeries(result, s.index.Copy(), s.name)
}

// DropNaThreshold drops rows where the number of non-null values is below threshold
func (s *Series) DropNaThreshold(threshold int) *Series {
	if threshold <= 0 {
		return s.Copy()
	}

	nonNullCount := s.NotNullCount()
	if nonNullCount >= threshold {
		return s.DropNa() // Remove all NaN values
	} else {
		return Empty(s.DType(), s.name) // Return empty series if below threshold
	}
}

// FillNaLimit fills null/NaN values with limit on consecutive fills
func (s *Series) FillNaLimit(value interface{}, limit int) (*Series, error) {
	if limit <= 0 {
		return s.Copy(), nil
	}

	result := array.Empty(s.data.Shape(), s.data.DType())

	// Convert fill value to appropriate type
	fillValue, err := convertValue(value, s.DType())
	if err != nil {
		return nil, fmt.Errorf("failed to convert fill value: %v", err)
	}

	consecutiveFills := 0

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)

		var resultVal interface{}
		if isNaN(val) {
			if consecutiveFills < limit {
				resultVal = fillValue
				consecutiveFills++
			} else {
				resultVal = val // Keep NaN if limit reached
			}
		} else {
			resultVal = val
			consecutiveFills = 0 // Reset counter on non-NaN value
		}

		err := result.Set(resultVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set value at index %d: %v", i, err)
		}
	}

	return NewSeries(result, s.index.Copy(), s.name)
}

// ReplaceNaN replaces specific values with NaN
func (s *Series) ReplaceNaN(values []interface{}) (*Series, error) {
	result := array.Empty(s.data.Shape(), s.data.DType())

	// Create set of values to replace
	replaceSet := make(map[string]bool)
	for _, val := range values {
		key := fmt.Sprintf("%v", val)
		replaceSet[key] = true
	}

	nanValue := getNaNValue(s.DType())

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		key := fmt.Sprintf("%v", val)

		var resultVal interface{}
		if replaceSet[key] {
			resultVal = nanValue
		} else {
			resultVal = val
		}

		err := result.Set(resultVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set value at index %d: %v", i, err)
		}
	}

	return NewSeries(result, s.index.Copy(), s.name)
}
