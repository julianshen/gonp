package series

import (
	"fmt"
	"math"
	"reflect"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Series represents a one-dimensional labeled array capable of holding data of any type
type Series struct {
	data  *array.Array
	index Index
	name  string
}

// NewSeries creates a new Series with the given data, index, and name
func NewSeries(data *array.Array, index Index, name string) (*Series, error) {
	if data == nil {
		return nil, fmt.Errorf("data cannot be nil")
	}

	if index == nil {
		index = NewDefaultRangeIndex(data.Size())
	}

	if data.Size() != index.Len() {
		return nil, fmt.Errorf("data length (%d) does not match index length (%d)", data.Size(), index.Len())
	}

	return &Series{
		data:  data,
		index: index,
		name:  name,
	}, nil
}

// FromSlice creates a Series from a Go slice with optional index
func FromSlice(data interface{}, index Index, name string) (*Series, error) {
	arr, err := array.FromSlice(data)
	if err != nil {
		return nil, fmt.Errorf("failed to create array from slice: %v", err)
	}

	return NewSeries(arr, index, name)
}

// FromValues creates a Series from individual values
func FromValues(values []interface{}, index Index, name string) (*Series, error) {
	if len(values) == 0 {
		// Create empty series
		arr := array.Empty(internal.Shape{0}, internal.Float64)
		if index == nil {
			index = NewDefaultRangeIndex(0)
		}
		return NewSeries(arr, index, name)
	}

	// Determine the best dtype for the values
	dtype := inferDType(values)
	arr := array.Empty(internal.Shape{len(values)}, dtype)

	// Fill the array with values
	for i, val := range values {
		convertedVal, err := convertValue(val, dtype)
		if err != nil {
			return nil, fmt.Errorf("failed to convert value at index %d: %v", i, err)
		}
		err = arr.Set(convertedVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set value at index %d: %v", i, err)
		}
	}

	return NewSeries(arr, index, name)
}

// Empty creates an empty Series
func Empty(dtype internal.DType, name string) *Series {
	arr := array.Empty(internal.Shape{0}, dtype)
	index := NewDefaultRangeIndex(0)
	series, _ := NewSeries(arr, index, name)
	return series
}

// Zeros creates a Series filled with zeros
func Zeros(length int, dtype internal.DType, index Index, name string) *Series {
	arr := array.Zeros(internal.Shape{length}, dtype)
	if index == nil {
		index = NewDefaultRangeIndex(length)
	}
	series, _ := NewSeries(arr, index, name)
	return series
}

// Ones creates a Series filled with ones
func Ones(length int, dtype internal.DType, index Index, name string) *Series {
	arr := array.Ones(internal.Shape{length}, dtype)
	if index == nil {
		index = NewDefaultRangeIndex(length)
	}
	series, _ := NewSeries(arr, index, name)
	return series
}

// Full creates a Series filled with a specific value
func Full(length int, value interface{}, dtype internal.DType, index Index, name string) *Series {
	arr := array.Full(internal.Shape{length}, value, dtype)
	if index == nil {
		index = NewDefaultRangeIndex(length)
	}
	series, _ := NewSeries(arr, index, name)
	return series
}

// Basic accessors

// Len returns the length of the Series
func (s *Series) Len() int {
	return s.data.Size()
}

// Data returns the underlying array
func (s *Series) Data() *array.Array {
	return s.data
}

// Index returns the index
func (s *Series) Index() Index {
	return s.index
}

// Name returns the name of the Series
func (s *Series) Name() string {
	return s.name
}

// SetName sets the name of the Series and returns a new Series
func (s *Series) SetName(name string) *Series {
	newSeries, _ := NewSeries(s.data, s.index, name)
	return newSeries
}

// DType returns the data type of the Series
func (s *Series) DType() internal.DType {
	return s.data.DType()
}

// Shape returns the shape of the underlying array
func (s *Series) Shape() internal.Shape {
	return s.data.Shape()
}

// Data access methods

// At returns the value at the given integer position
func (s *Series) At(i int) interface{} {
	return s.data.At(i)
}

// Loc returns the value at the given label
func (s *Series) Loc(label interface{}) (interface{}, error) {
	pos, found := s.index.Loc(label)
	if !found {
		return nil, fmt.Errorf("label %v not found in index", label)
	}
	return s.data.At(pos), nil
}

// Set sets the value at the given integer position
func (s *Series) Set(i int, value interface{}) error {
	return s.data.Set(value, i)
}

// SetLoc sets the value at the given label
func (s *Series) SetLoc(label interface{}, value interface{}) error {
	pos, found := s.index.Loc(label)
	if !found {
		return fmt.Errorf("label %v not found in index", label)
	}
	return s.data.Set(value, pos)
}

// Indexing and slicing

// ILoc returns a new Series with elements at the given integer positions
func (s *Series) ILoc(positions []int) (*Series, error) {
	if len(positions) == 0 {
		return Empty(s.DType(), s.name), nil
	}

	// Create new array with selected elements
	newArr := array.Empty(internal.Shape{len(positions)}, s.DType())
	newIndexValues := make([]interface{}, len(positions))

	for i, pos := range positions {
		if pos < 0 || pos >= s.Len() {
			return nil, fmt.Errorf("position %d out of bounds", pos)
		}

		val := s.data.At(pos)
		err := newArr.Set(val, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set value at position %d: %v", i, err)
		}

		newIndexValues[i] = s.index.Get(pos)
	}

	newIndex := NewIndex(newIndexValues)
	return NewSeries(newArr, newIndex, s.name)
}

// Slice returns a new Series with elements from start to end (exclusive)
func (s *Series) Slice(start, end int) (*Series, error) {
	if start < 0 {
		start = 0
	}
	if end > s.Len() {
		end = s.Len()
	}
	if start >= end {
		return Empty(s.DType(), s.name), nil
	}

	positions := make([]int, end-start)
	for i := range positions {
		positions[i] = start + i
	}

	return s.ILoc(positions)
}

// Head returns the first n elements (default 5)
func (s *Series) Head(n ...int) *Series {
	count := 5
	if len(n) > 0 && n[0] > 0 {
		count = n[0]
	}
	if count > s.Len() {
		count = s.Len()
	}

	result, _ := s.Slice(0, count)
	return result
}

// Tail returns the last n elements (default 5)
func (s *Series) Tail(n ...int) *Series {
	count := 5
	if len(n) > 0 && n[0] > 0 {
		count = n[0]
	}
	if count > s.Len() {
		count = s.Len()
	}

	start := s.Len() - count
	result, _ := s.Slice(start, s.Len())
	return result
}

// Copy returns a deep copy of the Series
func (s *Series) Copy() *Series {
	// Create a copy of the underlying array
	newArr := array.Empty(s.data.Shape(), s.data.DType())
	for i := 0; i < s.Len(); i++ {
		val := s.data.At(i)
		newArr.Set(val, i)
	}

	// Copy the index
	newIndex := s.index.Copy()

	newSeries, _ := NewSeries(newArr, newIndex, s.name)
	return newSeries
}

// String representation
func (s *Series) String() string {
	if s.Len() == 0 {
		return fmt.Sprintf("Series([], Name: %s, dtype: %v)", s.name, s.DType())
	}

	// Show first few and last few elements if series is long
	maxDisplay := 10
	var result string

	if s.Len() <= maxDisplay {
		for i := 0; i < s.Len(); i++ {
			indexLabel := s.index.Get(i)
			value := s.data.At(i)
			result += fmt.Sprintf("%v    %v\n", indexLabel, value)
		}
	} else {
		// Show first 5
		for i := 0; i < 5; i++ {
			indexLabel := s.index.Get(i)
			value := s.data.At(i)
			result += fmt.Sprintf("%v    %v\n", indexLabel, value)
		}
		result += "...\n"
		// Show last 5
		for i := s.Len() - 5; i < s.Len(); i++ {
			indexLabel := s.index.Get(i)
			value := s.data.At(i)
			result += fmt.Sprintf("%v    %v\n", indexLabel, value)
		}
	}

	result += fmt.Sprintf("Name: %s, Length: %d, dtype: %v", s.name, s.Len(), s.DType())
	return result
}

// Values returns all values as a slice of interface{}
func (s *Series) Values() []interface{} {
	values := make([]interface{}, s.Len())
	for i := 0; i < s.Len(); i++ {
		values[i] = s.data.At(i)
	}
	return values
}

// ToSlice returns the values as a typed slice if possible
func (s *Series) ToSlice() interface{} {
	switch s.DType() {
	case internal.Float64:
		slice := make([]float64, s.Len())
		for i := 0; i < s.Len(); i++ {
			if val, ok := s.data.At(i).(float64); ok {
				slice[i] = val
			}
		}
		return slice
	case internal.Float32:
		slice := make([]float32, s.Len())
		for i := 0; i < s.Len(); i++ {
			if val, ok := s.data.At(i).(float32); ok {
				slice[i] = val
			}
		}
		return slice
	case internal.Int64:
		slice := make([]int64, s.Len())
		for i := 0; i < s.Len(); i++ {
			if val, ok := s.data.At(i).(int64); ok {
				slice[i] = val
			}
		}
		return slice
	case internal.Int32:
		slice := make([]int32, s.Len())
		for i := 0; i < s.Len(); i++ {
			if val, ok := s.data.At(i).(int32); ok {
				slice[i] = val
			}
		}
		return slice
	case internal.Bool:
		slice := make([]bool, s.Len())
		for i := 0; i < s.Len(); i++ {
			if val, ok := s.data.At(i).(bool); ok {
				slice[i] = val
			}
		}
		return slice
	default:
		return s.Values() // Return as []interface{}
	}
}

// Helper functions

// inferDType determines the best dtype for a slice of interface{} values
func inferDType(values []interface{}) internal.DType {
	if len(values) == 0 {
		return internal.Float64 // Default
	}

	hasFloat := false
	hasInt := false
	hasBool := false
	hasComplex := false
	hasString := false

	for _, val := range values {
		if val == nil {
			continue // Skip nil values for type inference
		}

		switch val.(type) {
		case string:
			hasString = true
		case bool:
			hasBool = true
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			hasInt = true
		case float32, float64:
			hasFloat = true
		case complex64, complex128:
			hasComplex = true
		default:
			// For unknown types, default to float64 unless we have strings
			hasString = true
		}
	}

	// Priority: string > complex > float > int > bool
	// Note: For now, we'll use Float64 to store strings as interface{}
	// since the array package doesn't support string dtype yet
	if hasString {
		return internal.Float64 // Will store as interface{} in practice
	}
	if hasComplex {
		return internal.Complex128
	}
	if hasFloat {
		return internal.Float64
	}
	if hasInt {
		return internal.Int64
	}
	if hasBool {
		return internal.Bool
	}

	return internal.Float64 // Default fallback
}

// convertValue converts a value to the specified dtype
func convertValue(value interface{}, dtype internal.DType) (interface{}, error) {
	if value == nil {
		// Handle nil values by returning appropriate zero/NaN values
		switch dtype {
		case internal.Float64:
			return math.NaN(), nil
		case internal.Float32:
			return float32(math.NaN()), nil
		case internal.Int64:
			return int64(0), nil // Could use a special NaN-like value
		case internal.Int32:
			return int32(0), nil
		case internal.Int16:
			return int16(0), nil
		case internal.Int8:
			return int8(0), nil
		case internal.Uint64:
			return uint64(0), nil
		case internal.Uint32:
			return uint32(0), nil
		case internal.Uint16:
			return uint16(0), nil
		case internal.Uint8:
			return uint8(0), nil
		case internal.Bool:
			return false, nil
		case internal.Complex64:
			return complex64(complex(float32(math.NaN()), 0)), nil
		case internal.Complex128:
			return complex(math.NaN(), 0), nil
		default:
			return nil, fmt.Errorf("unsupported dtype: %v", dtype)
		}
	}

	// Convert based on target dtype
	switch dtype {
	case internal.Float64:
		// Special case: if value is string and we're using Float64 as a container, keep as string
		if _, ok := value.(string); ok {
			return value, nil
		}
		return convertToFloat64(value)
	case internal.Float32:
		f64, err := convertToFloat64(value)
		if err != nil {
			return nil, err
		}
		return float32(f64), nil
	case internal.Int64:
		return convertToInt64(value)
	case internal.Int32:
		i64, err := convertToInt64(value)
		if err != nil {
			return nil, err
		}
		return int32(i64), nil
	case internal.Bool:
		return convertToBool(value)
	case internal.Complex128:
		return convertToComplex128(value)
	case internal.Complex64:
		c128, err := convertToComplex128(value)
		if err != nil {
			return nil, err
		}
		return complex64(c128), nil
	default:
		return nil, fmt.Errorf("unsupported dtype: %v", dtype)
	}
}

// Helper conversion functions
func convertToFloat64(value interface{}) (float64, error) {
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
	default:
		return 0, fmt.Errorf("cannot convert %T to float64", value)
	}
}

func convertToInt64(value interface{}) (int64, error) {
	switch v := value.(type) {
	case int64:
		return v, nil
	case int:
		return int64(v), nil
	case int32:
		return int64(v), nil
	case int16:
		return int64(v), nil
	case int8:
		return int64(v), nil
	case uint64:
		return int64(v), nil
	case uint32:
		return int64(v), nil
	case uint16:
		return int64(v), nil
	case uint8:
		return int64(v), nil
	case float64:
		return int64(v), nil
	case float32:
		return int64(v), nil
	case bool:
		if v {
			return 1, nil
		}
		return 0, nil
	default:
		return 0, fmt.Errorf("cannot convert %T to int64", value)
	}
}

func convertToBool(value interface{}) (bool, error) {
	switch v := value.(type) {
	case bool:
		return v, nil
	case int, int64, int32, int16, int8:
		return reflect.ValueOf(v).Int() != 0, nil
	case uint, uint64, uint32, uint16, uint8:
		return reflect.ValueOf(v).Uint() != 0, nil
	case float64:
		return v != 0.0 && !math.IsNaN(v), nil
	case float32:
		return v != 0.0 && !math.IsNaN(float64(v)), nil
	default:
		return false, fmt.Errorf("cannot convert %T to bool", value)
	}
}

func convertToComplex128(value interface{}) (complex128, error) {
	switch v := value.(type) {
	case complex128:
		return v, nil
	case complex64:
		return complex128(v), nil
	case float64:
		return complex(v, 0), nil
	case float32:
		return complex(float64(v), 0), nil
	case int, int64, int32, int16, int8:
		return complex(float64(reflect.ValueOf(v).Int()), 0), nil
	case uint, uint64, uint32, uint16, uint8:
		return complex(float64(reflect.ValueOf(v).Uint()), 0), nil
	default:
		return 0, fmt.Errorf("cannot convert %T to complex128", value)
	}
}
