package series

import (
	"fmt"
	"reflect"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Advanced indexing and selection methods

// LocMultiple returns values at multiple labels
func (s *Series) LocMultiple(labels []interface{}) (*Series, error) {
	if len(labels) == 0 {
		return Empty(s.DType(), s.name), nil
	}

	positions := make([]int, 0, len(labels))
	foundLabels := make([]interface{}, 0, len(labels))

	for _, label := range labels {
		if pos, found := s.index.Loc(label); found {
			positions = append(positions, pos)
			foundLabels = append(foundLabels, label)
		}
	}

	if len(positions) == 0 {
		return Empty(s.DType(), s.name), nil
	}

	return s.ILoc(positions)
}

// LocSlice returns values between two labels (inclusive)
func (s *Series) LocSlice(start, end interface{}) (*Series, error) {
	startPos, startFound := s.index.Loc(start)
	endPos, endFound := s.index.Loc(end)

	if !startFound {
		return nil, fmt.Errorf("start label %v not found in index", start)
	}
	if !endFound {
		return nil, fmt.Errorf("end label %v not found in index", end)
	}

	// Ensure start <= end
	if startPos > endPos {
		startPos, endPos = endPos, startPos
	}

	return s.Slice(startPos, endPos+1) // +1 because Slice is exclusive on end
}

// BooleanIndexing returns elements where the boolean Series is True
func (s *Series) BooleanIndexing(mask *Series) (*Series, error) {
	if mask == nil {
		return nil, fmt.Errorf("mask cannot be nil")
	}

	if mask.DType() != internal.Bool {
		return nil, fmt.Errorf("mask must be a boolean Series")
	}

	if mask.Len() != s.Len() {
		return nil, fmt.Errorf("mask length (%d) must match Series length (%d)", mask.Len(), s.Len())
	}

	// Find positions where mask is True
	truePositions := make([]int, 0)
	for i := 0; i < mask.Len(); i++ {
		if val, ok := mask.At(i).(bool); ok && val {
			truePositions = append(truePositions, i)
		}
	}

	return s.ILoc(truePositions)
}

// Where applies a condition and returns elements where condition is True, otherwise other
func (s *Series) Where(condition *Series, other interface{}) (*Series, error) {
	if condition == nil {
		return nil, fmt.Errorf("condition cannot be nil")
	}

	if condition.DType() != internal.Bool {
		return nil, fmt.Errorf("condition must be a boolean Series")
	}

	if condition.Len() != s.Len() {
		return nil, fmt.Errorf("condition length (%d) must match Series length (%d)", condition.Len(), s.Len())
	}

	result := array.Empty(s.data.Shape(), s.data.DType())

	for i := 0; i < s.Len(); i++ {
		var resultVal interface{}

		if condVal, ok := condition.At(i).(bool); ok && condVal {
			resultVal = s.At(i)
		} else {
			// Convert other to appropriate type
			convertedOther, err := convertValue(other, s.DType())
			if err != nil {
				return nil, fmt.Errorf("failed to convert 'other' value: %v", err)
			}
			resultVal = convertedOther
		}

		err := result.Set(resultVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set result at index %d: %v", i, err)
		}
	}

	return NewSeries(result, s.index.Copy(), s.name)
}

// Query applies a string query expression (simplified version)
func (s *Series) Query(expr string) (*Series, error) {
	// This is a simplified implementation. A full implementation would parse
	// the expression and apply it. For now, we'll support basic operations.

	// Example: "value > 5" or "value == 'text'"
	// For this implementation, we'll just return an error with a message
	return nil, fmt.Errorf("query functionality not yet implemented: %s", expr)
}

// Filter applies a function to each element and returns elements where function returns True
func (s *Series) Filter(predicate func(interface{}) bool) (*Series, error) {
	if predicate == nil {
		return nil, fmt.Errorf("predicate function cannot be nil")
	}

	truePositions := make([]int, 0)

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		if predicate(val) {
			truePositions = append(truePositions, i)
		}
	}

	return s.ILoc(truePositions)
}

// Map applies a function to each element and returns a new Series
func (s *Series) Map(mapper func(interface{}) interface{}) (*Series, error) {
	if mapper == nil {
		return nil, fmt.Errorf("mapper function cannot be nil")
	}

	// Apply mapper to first element to infer result type
	if s.Len() == 0 {
		return Empty(s.DType(), s.name), nil
	}

	firstResult := mapper(s.At(0))
	resultType := reflect.TypeOf(firstResult)

	// Determine appropriate dtype for result
	var resultDType internal.DType
	switch resultType.Kind() {
	case reflect.Bool:
		resultDType = internal.Bool
	case reflect.Int, reflect.Int64:
		resultDType = internal.Int64
	case reflect.Int32:
		resultDType = internal.Int32
	case reflect.Float32:
		resultDType = internal.Float32
	case reflect.Float64:
		resultDType = internal.Float64
	case reflect.Complex64:
		resultDType = internal.Complex64
	case reflect.Complex128:
		resultDType = internal.Complex128
	default:
		// Default to original dtype if we can't determine
		resultDType = s.DType()
	}

	result := array.Empty(s.data.Shape(), resultDType)

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		mappedVal := mapper(val)

		// Convert to appropriate type if needed
		convertedVal, err := convertValue(mappedVal, resultDType)
		if err != nil {
			return nil, fmt.Errorf("failed to convert mapped value at index %d: %v", i, err)
		}

		err = result.Set(convertedVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set mapped value at index %d: %v", i, err)
		}
	}

	return NewSeries(result, s.index.Copy(), s.name)
}

// Apply applies a function to the entire Series (useful for reductions)
func (s *Series) Apply(fn func(*Series) interface{}) interface{} {
	if fn == nil {
		return nil
	}
	return fn(s)
}

// GroupBy groups the Series by the given key function
type GroupBy struct {
	series *Series
	groups map[interface{}][]int
	keys   []interface{}
}

// GroupBy creates a GroupBy object for aggregation operations
func (s *Series) GroupBy(key func(interface{}) interface{}) *GroupBy {
	if key == nil {
		return &GroupBy{series: s, groups: make(map[interface{}][]int), keys: []interface{}{}}
	}

	groups := make(map[interface{}][]int)
	keyOrder := make([]interface{}, 0)
	keysSeen := make(map[interface{}]bool)

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		k := key(val)

		if !keysSeen[k] {
			keyOrder = append(keyOrder, k)
			keysSeen[k] = true
		}

		groups[k] = append(groups[k], i)
	}

	return &GroupBy{
		series: s,
		groups: groups,
		keys:   keyOrder,
	}
}

// Sum aggregates each group using sum
func (gb *GroupBy) Sum() (*Series, error) {
	return gb.aggregateNumeric(func(values []interface{}) interface{} {
		var sum float64
		for _, v := range values {
			if f, err := convertToFloat64(v); err == nil {
				sum += f
			}
		}
		return sum
	})
}

// Mean aggregates each group using mean
func (gb *GroupBy) Mean() (*Series, error) {
	return gb.aggregateNumeric(func(values []interface{}) interface{} {
		var sum float64
		count := 0
		for _, v := range values {
			if f, err := convertToFloat64(v); err == nil {
				sum += f
				count++
			}
		}
		if count == 0 {
			return 0.0
		}
		return sum / float64(count)
	})
}

// Count counts non-null values in each group
func (gb *GroupBy) Count() (*Series, error) {
	return gb.aggregateNumeric(func(values []interface{}) interface{} {
		count := 0
		for _, v := range values {
			if !isNaN(v) {
				count++
			}
		}
		return int64(count)
	})
}

// Max finds the maximum value in each group
func (gb *GroupBy) Max() (*Series, error) {
	return gb.aggregateNumeric(func(values []interface{}) interface{} {
		if len(values) == 0 {
			return nil
		}

		max := values[0]
		for _, v := range values[1:] {
			if compareValues(v, max) > 0 {
				max = v
			}
		}
		return max
	})
}

// Min finds the minimum value in each group
func (gb *GroupBy) Min() (*Series, error) {
	return gb.aggregateNumeric(func(values []interface{}) interface{} {
		if len(values) == 0 {
			return nil
		}

		min := values[0]
		for _, v := range values[1:] {
			if compareValues(v, min) < 0 {
				min = v
			}
		}
		return min
	})
}

// aggregateNumeric performs aggregation with a function
func (gb *GroupBy) aggregateNumeric(fn func([]interface{}) interface{}) (*Series, error) {
	if len(gb.keys) == 0 {
		return Empty(internal.Float64, gb.series.name), nil
	}

	results := make([]interface{}, len(gb.keys))
	indexLabels := make([]interface{}, len(gb.keys))

	for i, key := range gb.keys {
		positions := gb.groups[key]
		values := make([]interface{}, len(positions))

		for j, pos := range positions {
			values[j] = gb.series.At(pos)
		}

		results[i] = fn(values)
		indexLabels[i] = key
	}

	return FromValues(results, NewIndex(indexLabels), gb.series.name)
}

// Unique returns unique values in the Series
func (s *Series) Unique() *Series {
	seen := make(map[interface{}]bool)
	uniqueValues := make([]interface{}, 0)

	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		key := fmt.Sprintf("%v", val) // Use string representation as key

		if !seen[key] {
			seen[key] = true
			uniqueValues = append(uniqueValues, val)
		}
	}

	result, _ := FromValues(uniqueValues, nil, s.name)
	return result
}

// ValueCounts returns a Series with counts of unique values
func (s *Series) ValueCounts() *Series {
	counts := make(map[string]int)
	values := make([]interface{}, 0)

	// Count occurrences
	for i := 0; i < s.Len(); i++ {
		val := s.At(i)
		key := fmt.Sprintf("%v", val)

		if counts[key] == 0 {
			values = append(values, val)
		}
		counts[key]++
	}

	// Create result
	countValues := make([]interface{}, len(values))
	for i, val := range values {
		key := fmt.Sprintf("%v", val)
		countValues[i] = int64(counts[key])
	}

	result, _ := FromValues(countValues, NewIndex(values), s.name)
	return result
}

// Sort returns a new sorted Series
func (s *Series) Sort(ascending bool) *Series {
	// Create index array for sorting
	indices := make([]int, s.Len())
	for i := range indices {
		indices[i] = i
	}

	// Sort indices based on values
	for i := 0; i < len(indices)-1; i++ {
		for j := i + 1; j < len(indices); j++ {
			val1 := s.At(indices[i])
			val2 := s.At(indices[j])

			comparison := compareValues(val1, val2)

			shouldSwap := false
			if ascending {
				shouldSwap = comparison > 0
			} else {
				shouldSwap = comparison < 0
			}

			if shouldSwap {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}

	// Create sorted series
	result, _ := s.ILoc(indices)
	return result
}

// SortByIndex returns a new Series sorted by index
func (s *Series) SortByIndex(ascending bool) *Series {
	sortedIndex, permutation := s.index.Sort()

	if !ascending {
		// Reverse the permutation
		for i := 0; i < len(permutation)/2; i++ {
			permutation[i], permutation[len(permutation)-1-i] = permutation[len(permutation)-1-i], permutation[i]
		}

		// Create reversed index
		indexValues := sortedIndex.Values()
		for i := 0; i < len(indexValues)/2; i++ {
			indexValues[i], indexValues[len(indexValues)-1-i] = indexValues[len(indexValues)-1-i], indexValues[i]
		}
		sortedIndex = NewIndex(indexValues)
	}

	result, _ := s.ILoc(permutation)
	newSeries, _ := NewSeries(result.data, sortedIndex, result.name)
	return newSeries
}

// isNaN checks if a value represents NaN/missing data
func isNaN(value interface{}) bool {
	switch v := value.(type) {
	case float64:
		return isNaNFloat64(v)
	case float32:
		return isNaNFloat32(v)
	case complex64:
		return isNaNFloat32(real(v)) || isNaNFloat32(imag(v))
	case complex128:
		return isNaNFloat64(real(v)) || isNaNFloat64(imag(v))
	default:
		return value == nil
	}
}

// Helper functions for NaN detection
func isNaNFloat64(f float64) bool {
	return f != f
}

func isNaNFloat32(f float32) bool {
	return f != f
}
