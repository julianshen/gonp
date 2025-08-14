package io

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
)

// SeriesData represents the JSON structure for a Series
type SeriesData struct {
	Name   string        `json:"name"`
	Length int           `json:"length"`
	Data   []interface{} `json:"data"`
	Index  []interface{} `json:"index"`
	DType  string        `json:"dtype"`
}

// ArrayData represents the JSON structure for an Array
type ArrayData struct {
	Data  []interface{} `json:"data"`
	Shape []int         `json:"shape"`
	DType string        `json:"dtype"`
}

// SeriesJSON serializes a Series to JSON bytes
func SeriesJSON(s *series.Series) ([]byte, error) {
	if s == nil {
		return nil, fmt.Errorf("series cannot be nil")
	}

	data := SeriesData{
		Name:   s.Name(),
		Length: s.Len(),
		Data:   s.Values(),
		Index:  make([]interface{}, s.Len()),
		DType:  dtypeToString(s.DType()),
	}

	// Extract index values
	for i := 0; i < s.Len(); i++ {
		data.Index[i] = s.Index().Get(i)
	}

	return json.Marshal(data)
}

// SeriesFromJSON deserializes JSON bytes to a Series
func SeriesFromJSON(jsonData []byte) (*series.Series, error) {
	var data SeriesData
	err := json.Unmarshal(jsonData, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %v", err)
	}

	// Create series from values
	s, err := series.FromValues(data.Data, nil, data.Name)
	if err != nil {
		return nil, fmt.Errorf("failed to create series from data: %v", err)
	}

	return s, nil
}

// ArrayJSON serializes an Array to JSON bytes
func ArrayJSON(arr *array.Array) ([]byte, error) {
	if arr == nil {
		return nil, fmt.Errorf("array cannot be nil")
	}

	// Flatten array and get all values
	flat := arr.Flatten()
	values := make([]interface{}, flat.Size())
	for i := 0; i < flat.Size(); i++ {
		values[i] = flat.At(i)
	}

	data := ArrayData{
		Data:  values,
		Shape: []int(arr.Shape()),
		DType: dtypeToString(arr.DType()),
	}

	return json.Marshal(data)
}

// ArrayFromJSON deserializes JSON bytes to an Array
func ArrayFromJSON(jsonData []byte) (*array.Array, error) {
	var data ArrayData
	err := json.Unmarshal(jsonData, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %v", err)
	}

	shape := internal.Shape(data.Shape)

	// Fill the array with data
	if len(data.Data) != shape.Size() {
		return nil, fmt.Errorf("data length %d does not match shape size %d", len(data.Data), shape.Size())
	}

	// Fill array in row-major order by creating a flat array first then reshaping
	flatArr, err := array.FromSlice(data.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to create flat array: %v", err)
	}

	// Reshape to the target shape
	result := flatArr.Reshape(shape)
	return result, nil
}

// WriteSeriesJSON writes a Series to a JSON file
func WriteSeriesJSON(s *series.Series, filename string) error {
	jsonData, err := SeriesJSON(s)
	if err != nil {
		return err
	}

	return os.WriteFile(filename, jsonData, 0644)
}

// ReadSeriesJSON reads a Series from a JSON file
func ReadSeriesJSON(filename string) (*series.Series, error) {
	jsonData, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %v", err)
	}

	return SeriesFromJSON(jsonData)
}

// WriteArrayJSON writes an Array to a JSON file
func WriteArrayJSON(arr *array.Array, filename string) error {
	jsonData, err := ArrayJSON(arr)
	if err != nil {
		return err
	}

	return os.WriteFile(filename, jsonData, 0644)
}

// ReadArrayJSON reads an Array from a JSON file
func ReadArrayJSON(filename string) (*array.Array, error) {
	jsonData, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %v", err)
	}

	return ArrayFromJSON(jsonData)
}

// Helper functions

// dtypeToString converts internal.DType to string
func dtypeToString(dtype internal.DType) string {
	switch dtype {
	case internal.Float64:
		return "float64"
	case internal.Float32:
		return "float32"
	case internal.Int64:
		return "int64"
	case internal.Int32:
		return "int32"
	case internal.Int16:
		return "int16"
	case internal.Int8:
		return "int8"
	case internal.Uint64:
		return "uint64"
	case internal.Uint32:
		return "uint32"
	case internal.Uint16:
		return "uint16"
	case internal.Uint8:
		return "uint8"
	case internal.Bool:
		return "bool"
	case internal.Complex64:
		return "complex64"
	case internal.Complex128:
		return "complex128"
	default:
		return "float64"
	}
}

// stringToDtype converts string to internal.DType
func stringToDtype(s string) internal.DType {
	switch s {
	case "float64":
		return internal.Float64
	case "float32":
		return internal.Float32
	case "int64":
		return internal.Int64
	case "int32":
		return internal.Int32
	case "int16":
		return internal.Int16
	case "int8":
		return internal.Int8
	case "uint64":
		return internal.Uint64
	case "uint32":
		return internal.Uint32
	case "uint16":
		return internal.Uint16
	case "uint8":
		return internal.Uint8
	case "bool":
		return internal.Bool
	case "complex64":
		return internal.Complex64
	case "complex128":
		return internal.Complex128
	default:
		return internal.Float64
	}
}
