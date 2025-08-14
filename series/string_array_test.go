package series

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestStringMethodsWithArrayBackedSeries(t *testing.T) {
	// Create a string array directly
	stringData := []string{"hello world", "GOLANG", "test string", "123", "   padded   "}
	arr, err := array.FromSlice(stringData)
	if err != nil {
		t.Fatalf("Failed to create string array: %v", err)
	}

	// Create Series from the string array
	series, err := NewSeries(arr, nil, "test_strings")
	if err != nil {
		t.Fatalf("Failed to create series: %v", err)
	}

	// Verify the series uses String dtype
	if series.DType() != internal.String {
		t.Errorf("Expected series dtype %v, got %v", internal.String, series.DType())
	}

	// Test basic string methods
	t.Run("Lower", func(t *testing.T) {
		lower, err := series.Str().Lower()
		if err != nil {
			t.Errorf("Lower() failed: %v", err)
		}

		expected := []string{"hello world", "golang", "test string", "123", "   padded   "}
		for i, exp := range expected {
			val := lower.data.At(i)
			if val != exp {
				t.Errorf("Lower()[%d]: expected %s, got %v", i, exp, val)
			}
		}
	})

	t.Run("Upper", func(t *testing.T) {
		upper, err := series.Str().Upper()
		if err != nil {
			t.Errorf("Upper() failed: %v", err)
		}

		expected := []string{"HELLO WORLD", "GOLANG", "TEST STRING", "123", "   PADDED   "}
		for i, exp := range expected {
			val := upper.data.At(i)
			if val != exp {
				t.Errorf("Upper()[%d]: expected %s, got %v", i, exp, val)
			}
		}
	})

	t.Run("Strip", func(t *testing.T) {
		strip, err := series.Str().Strip()
		if err != nil {
			t.Errorf("Strip() failed: %v", err)
		}

		expected := []string{"hello world", "GOLANG", "test string", "123", "padded"}
		for i, exp := range expected {
			val := strip.data.At(i)
			if val != exp {
				t.Errorf("Strip()[%d]: expected %s, got %v", i, exp, val)
			}
		}
	})

	t.Run("Contains", func(t *testing.T) {
		contains, err := series.Str().Contains("test", false)
		if err != nil {
			t.Errorf("Contains() failed: %v", err)
		}

		expected := []bool{false, false, true, false, false}
		for i, exp := range expected {
			val := contains.data.At(i)
			if val != exp {
				t.Errorf("Contains()[%d]: expected %v, got %v", i, exp, val)
			}
		}
	})

	t.Run("StartsWith", func(t *testing.T) {
		startsWith, err := series.Str().StartsWith("hello")
		if err != nil {
			t.Errorf("StartsWith() failed: %v", err)
		}

		expected := []bool{true, false, false, false, false}
		for i, exp := range expected {
			val := startsWith.data.At(i)
			if val != exp {
				t.Errorf("StartsWith()[%d]: expected %v, got %v", i, exp, val)
			}
		}
	})

	t.Run("Len", func(t *testing.T) {
		lengths, err := series.Str().Len()
		if err != nil {
			t.Errorf("Len() failed: %v", err)
		}

		expected := []int64{11, 6, 11, 3, 12} // Length of each string
		for i, exp := range expected {
			val := lengths.data.At(i)
			if val != exp {
				t.Errorf("Len()[%d]: expected %d, got %v", i, exp, val)
			}
		}
	})

	t.Run("Replace", func(t *testing.T) {
		replaced, err := series.Str().Replace("test", "demo", -1)
		if err != nil {
			t.Errorf("Replace() failed: %v", err)
		}

		expected := []string{"hello world", "GOLANG", "demo string", "123", "   padded   "}
		for i, exp := range expected {
			val := replaced.data.At(i)
			if val != exp {
				t.Errorf("Replace()[%d]: expected %s, got %v", i, exp, val)
			}
		}
	})
}

func TestStringMethodsWithInterfaceArrayBackedSeries(t *testing.T) {
	// Create an interface array with mixed string and non-string data
	interfaceData := []interface{}{"hello", 123, "world", true, "test"}
	arr, err := array.FromSlice(interfaceData)
	if err != nil {
		t.Fatalf("Failed to create interface array: %v", err)
	}

	// Create Series from the interface array
	series, err := NewSeries(arr, nil, "mixed_data")
	if err != nil {
		t.Fatalf("Failed to create series: %v", err)
	}

	// Verify the series uses Interface dtype
	if series.DType() != internal.Interface {
		t.Errorf("Expected series dtype %v, got %v", internal.Interface, series.DType())
	}

	// Test string methods on mixed data - they should handle non-strings gracefully
	t.Run("Lower with mixed data", func(t *testing.T) {
		lower, err := series.Str().Lower()
		if err != nil {
			t.Errorf("Lower() failed: %v", err)
		}

		// String values should be lowercased, non-string values should be unchanged
		expected := []interface{}{"hello", 123, "world", true, "test"}
		for i, exp := range expected {
			val := lower.data.At(i)
			if val != exp {
				t.Errorf("Lower()[%d]: expected %v, got %v", i, exp, val)
			}
		}
	})

	t.Run("IsAlpha with mixed data", func(t *testing.T) {
		isAlpha, err := series.Str().IsAlpha()
		if err != nil {
			t.Errorf("IsAlpha() failed: %v", err)
		}

		// Only string values should be tested, non-strings return false
		expected := []bool{true, false, true, false, true}
		for i, exp := range expected {
			val := isAlpha.data.At(i)
			if val != exp {
				t.Errorf("IsAlpha()[%d]: expected %v, got %v", i, exp, val)
			}
		}
	})

	t.Run("Len with mixed data", func(t *testing.T) {
		lengths, err := series.Str().Len()
		if err != nil {
			t.Errorf("Len() failed: %v", err)
		}

		// String values return their length, non-strings return 0
		expected := []int64{5, 0, 5, 0, 4}
		for i, exp := range expected {
			val := lengths.data.At(i)
			if val != exp {
				t.Errorf("Len()[%d]: expected %d, got %v", i, exp, val)
			}
		}
	})
}

func TestStringArrayCreationFromSeries(t *testing.T) {
	// Test creating a Series directly from string slice using FromSlice
	stringData := []string{"apple", "banana", "cherry"}
	series, err := FromSlice(stringData, nil, "fruits")
	if err != nil {
		t.Fatalf("Failed to create series from string slice: %v", err)
	}

	// Verify dtype is String
	if series.DType() != internal.String {
		t.Errorf("Expected series dtype %v, got %v", internal.String, series.DType())
	}

	// Verify data integrity
	for i, expected := range stringData {
		val := series.data.At(i)
		if val != expected {
			t.Errorf("At index %d: expected %s, got %v", i, expected, val)
		}
	}

	// Test string methods work correctly
	capitalize, err := series.Str().Capitalize()
	if err != nil {
		t.Errorf("Capitalize() failed: %v", err)
	}

	expectedCapitalized := []string{"Apple", "Banana", "Cherry"}
	for i, expected := range expectedCapitalized {
		val := capitalize.data.At(i)
		if val != expected {
			t.Errorf("Capitalize()[%d]: expected %s, got %v", i, expected, val)
		}
	}
}

func TestStringConversionMethods(t *testing.T) {
	// Test string to numeric conversion (float)
	numericStrings := []string{"123", "45.67", "-89", "0", "3.14159"}
	series, err := FromSlice(numericStrings, nil, "numbers_as_strings")
	if err != nil {
		t.Fatalf("Failed to create series: %v", err)
	}

	// Convert to numeric (float64)
	numeric, err := series.Str().ToNumeric()
	if err != nil {
		t.Errorf("ToNumeric() failed: %v", err)
	}

	// Verify conversion
	expected := []float64{123.0, 45.67, -89.0, 0.0, 3.14159}
	for i, exp := range expected {
		val := numeric.data.At(i)
		if val != exp {
			t.Errorf("ToNumeric()[%d]: expected %f, got %v", i, exp, val)
		}
	}

	// Test AsType to int64 with integer strings only
	intStrings := []string{"123", "45", "-89", "0", "100"}
	intSeries, err := FromSlice(intStrings, nil, "integers_as_strings")
	if err != nil {
		t.Fatalf("Failed to create integer series: %v", err)
	}

	asInt, err := intSeries.Str().AsType(internal.Int64)
	if err != nil {
		t.Errorf("AsType(Int64) failed: %v", err)
	}

	expectedInt := []int64{123, 45, -89, 0, 100}
	for i, exp := range expectedInt {
		val := asInt.data.At(i)
		if val != exp {
			t.Errorf("AsType(Int64)[%d]: expected %d, got %v", i, exp, val)
		}
	}

	// Test conversion errors - non-numeric strings should get NaN/0 values
	invalidStrings := []string{"abc", "12.x", "not_a_number"}
	invalidSeries, err := FromSlice(invalidStrings, nil, "invalid_numbers")
	if err != nil {
		t.Fatalf("Failed to create invalid series: %v", err)
	}

	invalidNumeric, err := invalidSeries.Str().ToNumeric()
	if err != nil {
		t.Errorf("ToNumeric() with invalid strings failed: %v", err)
	}

	// Should contain NaN values (represented as math.NaN())
	for i := 0; i < invalidNumeric.Len(); i++ {
		val := invalidNumeric.data.At(i)
		if floatVal, ok := val.(float64); ok {
			if !math.IsNaN(floatVal) {
				t.Errorf("Expected NaN for invalid string conversion at index %d, got %v", i, val)
			}
		} else {
			t.Errorf("Expected float64 NaN value at index %d, got %T", i, val)
		}
	}
}
