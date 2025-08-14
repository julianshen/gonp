package io

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
)

// Test JSON serialization for Series
func TestSeriesJSON(t *testing.T) {
	t.Run("SerializeSeriesJSON", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0}
		s, err := series.FromSlice(data, nil, "test_series")
		if err != nil {
			t.Fatalf("Failed to create series: %v", err)
		}

		jsonBytes, err := SeriesJSON(s)
		if err != nil {
			t.Fatalf("SeriesJSON failed: %v", err)
		}

		// Verify JSON structure
		var result map[string]interface{}
		err = json.Unmarshal(jsonBytes, &result)
		if err != nil {
			t.Fatalf("Failed to parse JSON: %v", err)
		}

		// Check structure
		if result["name"] != "test_series" {
			t.Errorf("Expected name 'test_series', got %v", result["name"])
		}

		if result["length"] != float64(4) {
			t.Errorf("Expected length 4, got %v", result["length"])
		}

		dataArray, ok := result["data"].([]interface{})
		if !ok {
			t.Fatal("Expected data to be array")
		}

		if len(dataArray) != 4 {
			t.Errorf("Expected 4 data elements, got %d", len(dataArray))
		}

		if dataArray[0] != float64(1.0) {
			t.Errorf("Expected first element to be 1.0, got %v", dataArray[0])
		}
	})

	t.Run("DeserializeSeriesJSON", func(t *testing.T) {
		jsonStr := `{
			"name": "test_series",
			"length": 3,
			"dtype": "float64",
			"data": [1.0, 2.0, 3.0],
			"index": [0, 1, 2]
		}`

		s, err := SeriesFromJSON([]byte(jsonStr))
		if err != nil {
			t.Fatalf("SeriesFromJSON failed: %v", err)
		}

		if s.Name() != "test_series" {
			t.Errorf("Expected name 'test_series', got %s", s.Name())
		}

		if s.Len() != 3 {
			t.Errorf("Expected length 3, got %d", s.Len())
		}

		if s.At(0) != 1.0 {
			t.Errorf("Expected first element 1.0, got %v", s.At(0))
		}
	})
}

// Test JSON serialization for Arrays
func TestArrayJSON(t *testing.T) {
	t.Run("SerializeArrayJSON", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0}
		arr, err := array.FromSlice(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		jsonBytes, err := ArrayJSON(arr)
		if err != nil {
			t.Fatalf("ArrayJSON failed: %v", err)
		}

		var result map[string]interface{}
		err = json.Unmarshal(jsonBytes, &result)
		if err != nil {
			t.Fatalf("Failed to parse JSON: %v", err)
		}

		// Check structure
		shapeArray, ok := result["shape"].([]interface{})
		if !ok {
			t.Fatal("Expected shape to be array")
		}

		if len(shapeArray) != 1 || shapeArray[0] != float64(4) {
			t.Errorf("Expected shape [4], got %v", shapeArray)
		}

		dataArray, ok := result["data"].([]interface{})
		if !ok {
			t.Fatal("Expected data to be array")
		}

		if len(dataArray) != 4 {
			t.Errorf("Expected 4 data elements, got %d", len(dataArray))
		}
	})

	t.Run("Deserialize2DArrayJSON", func(t *testing.T) {
		jsonStr := `{
			"shape": [2, 2],
			"dtype": "float64",
			"data": [1.0, 2.0, 3.0, 4.0]
		}`

		arr, err := ArrayFromJSON([]byte(jsonStr))
		if err != nil {
			t.Fatalf("ArrayFromJSON failed: %v", err)
		}

		expectedShape := []int{2, 2}
		if !arr.Shape().Equal(expectedShape) {
			t.Errorf("Expected shape %v, got %v", expectedShape, arr.Shape())
		}

		if arr.At(0, 0) != 1.0 {
			t.Errorf("Expected element [0,0] to be 1.0, got %v", arr.At(0, 0))
		}

		if arr.At(1, 1) != 4.0 {
			t.Errorf("Expected element [1,1] to be 4.0, got %v", arr.At(1, 1))
		}
	})
}

// Test file I/O for JSON
func TestJSONFileIO(t *testing.T) {
	t.Run("WriteAndReadSeriesJSON", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0}
		s, err := series.FromSlice(data, nil, "file_test")
		if err != nil {
			t.Fatalf("Failed to create series: %v", err)
		}

		tmpDir := t.TempDir()
		jsonFile := filepath.Join(tmpDir, "series.json")

		err = WriteSeriesJSON(s, jsonFile)
		if err != nil {
			t.Fatalf("WriteSeriesJSON failed: %v", err)
		}

		// Verify file exists
		if _, err := os.Stat(jsonFile); os.IsNotExist(err) {
			t.Fatal("JSON file was not created")
		}

		// Read back
		s2, err := ReadSeriesJSON(jsonFile)
		if err != nil {
			t.Fatalf("ReadSeriesJSON failed: %v", err)
		}

		if s2.Name() != "file_test" {
			t.Errorf("Expected name 'file_test', got %s", s2.Name())
		}

		if s2.Len() != 3 {
			t.Errorf("Expected length 3, got %d", s2.Len())
		}

		for i := 0; i < s.Len(); i++ {
			if s.At(i) != s2.At(i) {
				t.Errorf("Mismatch at position %d: expected %v, got %v", i, s.At(i), s2.At(i))
			}
		}
	})

	t.Run("WriteAndReadArrayJSON", func(t *testing.T) {
		// Create a 2x2 array using flat data with proper shape
		data := []float64{1.0, 2.0, 3.0, 4.0}
		shape := internal.Shape{2, 2}
		arr, err := array.NewArrayWithShape(data, shape)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		tmpDir := t.TempDir()
		jsonFile := filepath.Join(tmpDir, "array.json")

		err = WriteArrayJSON(arr, jsonFile)
		if err != nil {
			t.Fatalf("WriteArrayJSON failed: %v", err)
		}

		arr2, err := ReadArrayJSON(jsonFile)
		if err != nil {
			t.Fatalf("ReadArrayJSON failed: %v", err)
		}

		if !arr.Shape().Equal(arr2.Shape()) {
			t.Errorf("Shape mismatch: expected %v, got %v", arr.Shape(), arr2.Shape())
		}

		if arr.At(0, 0) != arr2.At(0, 0) {
			t.Errorf("Data mismatch at [0,0]: expected %v, got %v", arr.At(0, 0), arr2.At(0, 0))
		}
	})
}

// Test JSON error conditions
func TestJSONErrors(t *testing.T) {
	t.Run("InvalidJSON", func(t *testing.T) {
		invalidJSON := `{"invalid": json}`

		_, err := SeriesFromJSON([]byte(invalidJSON))
		if err == nil {
			t.Error("Expected error for invalid JSON")
		}
	})

	t.Run("ReadNonexistentJSONFile", func(t *testing.T) {
		_, err := ReadSeriesJSON("/nonexistent/file.json")
		if err == nil {
			t.Error("Expected error when reading nonexistent file")
		}
	})

	t.Run("WriteToInvalidPath", func(t *testing.T) {
		data := []float64{1.0, 2.0}
		s, _ := series.FromSlice(data, nil, "test")

		err := WriteSeriesJSON(s, "/invalid/path/file.json")
		if err == nil {
			t.Error("Expected error when writing to invalid path")
		}
	})
}
