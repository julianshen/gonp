package array

import (
	"fmt"
	"reflect"

	"github.com/julianshen/gonp/internal"
)

// BooleanIndex performs boolean indexing on the array
// mask: boolean array of the same length as the array
// Returns a new array containing only elements where mask is true
func (a *Array) BooleanIndex(mask *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("array cannot be nil")
	}
	if mask == nil {
		return nil, fmt.Errorf("mask cannot be nil")
	}

	// Check that mask is boolean type
	if mask.dtype != internal.Bool {
		return nil, fmt.Errorf("mask must be boolean array, got %v", mask.dtype)
	}

	// Check that dimensions match
	if a.Size() != mask.Size() {
		return nil, fmt.Errorf("array and mask must have same length: %d vs %d", a.Size(), mask.Size())
	}

	// Count true values to determine result size
	trueCount := 0
	for i := 0; i < mask.Size(); i++ {
		if mask.At(i).(bool) {
			trueCount++
		}
	}

	// If no true values, return empty array
	if trueCount == 0 {
		return createEmptyArray(a.dtype)
	}

	// Create result array
	resultData := make([]interface{}, trueCount)
	resultIdx := 0

	// Copy values where mask is true
	for i := 0; i < a.Size(); i++ {
		if mask.At(i).(bool) {
			resultData[resultIdx] = a.At(i)
			resultIdx++
		}
	}

	// Convert to typed slice and create array
	typedSlice, err := convertToTypedSlice(resultData, a.dtype)
	if err != nil {
		return nil, fmt.Errorf("failed to create result array: %v", err)
	}

	return NewArray(typedSlice)
}

// FancyIndex performs fancy indexing on the array
// indices: integer array containing indices to select
// Returns a new array with selected elements
func (a *Array) FancyIndex(indices *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("array cannot be nil")
	}
	if indices == nil {
		return nil, fmt.Errorf("indices cannot be nil")
	}

	// Check that indices is integer type
	if indices.dtype != internal.Int64 {
		return nil, fmt.Errorf("indices must be integer array, got %v", indices.dtype)
	}

	// Validate all indices are in bounds
	for i := 0; i < indices.Size(); i++ {
		idx := indices.At(i).(int64)
		if idx < 0 || idx >= int64(a.Size()) {
			return nil, fmt.Errorf("index %d out of bounds for array length %d", idx, a.Size())
		}
	}

	// Create result array
	resultData := make([]interface{}, indices.Size())
	for i := 0; i < indices.Size(); i++ {
		idx := int(indices.At(i).(int64))
		resultData[i] = a.At(idx)
	}

	// Convert to typed slice and create array
	typedSlice, err := convertToTypedSlice(resultData, a.dtype)
	if err != nil {
		return nil, fmt.Errorf("failed to create result array: %v", err)
	}

	return NewArray(typedSlice)
}

// Where returns elements that satisfy the given condition
// condition: function that takes a value and returns bool
// Returns a new array containing only elements where condition is true
func (a *Array) Where(condition func(interface{}) bool) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("array cannot be nil")
	}
	if condition == nil {
		return nil, fmt.Errorf("condition cannot be nil")
	}

	// Find all matching elements
	var resultData []interface{}
	for i := 0; i < a.Size(); i++ {
		val := a.At(i)
		if condition(val) {
			resultData = append(resultData, val)
		}
	}

	// If no matches, return empty array
	if len(resultData) == 0 {
		return createEmptyArray(a.dtype)
	}

	// Convert to typed slice and create array
	typedSlice, err := convertToTypedSlice(resultData, a.dtype)
	if err != nil {
		return nil, fmt.Errorf("failed to create result array: %v", err)
	}

	return NewArray(typedSlice)
}

// Take returns elements at given indices
// indices: slice of int indices
// Returns a new array with selected elements
func (a *Array) Take(indices []int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("array cannot be nil")
	}
	if indices == nil {
		return nil, fmt.Errorf("indices cannot be nil")
	}

	// Validate indices
	for _, idx := range indices {
		if idx < 0 || idx >= a.Size() {
			return nil, fmt.Errorf("index %d out of bounds for array length %d", idx, a.Size())
		}
	}

	// Create result array
	resultData := make([]interface{}, len(indices))
	for i, idx := range indices {
		resultData[i] = a.At(idx)
	}

	// Convert to typed slice and create array
	typedSlice, err := convertToTypedSlice(resultData, a.dtype)
	if err != nil {
		return nil, fmt.Errorf("failed to create result array: %v", err)
	}

	return NewArray(typedSlice)
}

// Put places values at given indices in the array (modifies in-place)
// indices: slice of int indices where to place values
// values: slice of values to place
func (a *Array) Put(indices []int, values interface{}) error {
	if a == nil {
		return fmt.Errorf("array cannot be nil")
	}
	if indices == nil {
		return fmt.Errorf("indices cannot be nil")
	}
	if values == nil {
		return fmt.Errorf("values cannot be nil")
	}

	// Convert values to slice
	valSlice := reflect.ValueOf(values)
	if valSlice.Kind() != reflect.Slice {
		return fmt.Errorf("values must be a slice, got %T", values)
	}

	// Check lengths match
	if len(indices) != valSlice.Len() {
		return fmt.Errorf("indices and values must have same length: %d vs %d", len(indices), valSlice.Len())
	}

	// Validate indices and set values
	for i, idx := range indices {
		if idx < 0 || idx >= a.Size() {
			return fmt.Errorf("index %d out of bounds for array length %d", idx, a.Size())
		}

		val := valSlice.Index(i).Interface()
		err := a.Set(val, idx)
		if err != nil {
			return fmt.Errorf("failed to set value at index %d: %v", idx, err)
		}
	}

	return nil
}

// Helper functions

// createEmptyArray creates an empty array of the given dtype
func createEmptyArray(dtype internal.DType) (*Array, error) {
	// Create a zero-size array using the Zeros function
	emptyShape := internal.Shape{0}
	return &Array{
		storage: internal.NewTypedStorage(allocateSliceForDType(0, dtype), dtype),
		shape:   emptyShape,
		stride:  calculateStride(emptyShape),
		dtype:   dtype,
		offset:  0,
	}, nil
}

// convertToTypedSlice converts []interface{} to appropriate typed slice
func convertToTypedSlice(data []interface{}, dtype internal.DType) (interface{}, error) {
	switch dtype {
	case internal.Float64:
		result := make([]float64, len(data))
		for i, val := range data {
			if v, ok := val.(float64); ok {
				result[i] = v
			} else {
				return nil, fmt.Errorf("expected float64 at index %d, got %T", i, val)
			}
		}
		return result, nil

	case internal.Int64:
		result := make([]int64, len(data))
		for i, val := range data {
			if v, ok := val.(int64); ok {
				result[i] = v
			} else {
				return nil, fmt.Errorf("expected int64 at index %d, got %T", i, val)
			}
		}
		return result, nil

	case internal.Bool:
		result := make([]bool, len(data))
		for i, val := range data {
			if v, ok := val.(bool); ok {
				result[i] = v
			} else {
				return nil, fmt.Errorf("expected bool at index %d, got %T", i, val)
			}
		}
		return result, nil

	default:
		return nil, fmt.Errorf("unsupported dtype: %v", dtype)
	}
}
