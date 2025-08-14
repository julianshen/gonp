package array

import (
	"testing"

	"github.com/julianshen/gonp/internal"
)

func TestStringArrayCreation(t *testing.T) {
	// Test creating string array from slice
	stringData := []string{"hello", "world", "test"}
	arr, err := FromSlice(stringData)
	if err != nil {
		t.Errorf("Failed to create string array: %v", err)
	}

	if arr.DType() != internal.String {
		t.Errorf("Expected dtype %v, got %v", internal.String, arr.DType())
	}

	if !arr.Shape().Equal(internal.Shape{3}) {
		t.Errorf("Expected shape [3], got %v", arr.Shape())
	}

	// Test getting values
	for i, expected := range stringData {
		val := arr.At(i)
		if val != expected {
			t.Errorf("At index %d: expected %s, got %v", i, expected, val)
		}
	}
}

func TestStringArrayFull(t *testing.T) {
	// Test creating full string array
	arr := Full(internal.Shape{3}, "hello", internal.String)

	if arr.DType() != internal.String {
		t.Errorf("Expected dtype %v, got %v", internal.String, arr.DType())
	}

	for i := 0; i < 3; i++ {
		val := arr.At(i)
		if val != "hello" {
			t.Errorf("At index %d: expected 'hello', got %v", i, val)
		}
	}
}

func TestInterfaceArrayCreation(t *testing.T) {
	// Test creating interface{} array from slice
	interfaceData := []interface{}{1, "hello", 3.14, true}
	arr, err := FromSlice(interfaceData)
	if err != nil {
		t.Errorf("Failed to create interface array: %v", err)
	}

	if arr.DType() != internal.Interface {
		t.Errorf("Expected dtype %v, got %v", internal.Interface, arr.DType())
	}

	if !arr.Shape().Equal(internal.Shape{4}) {
		t.Errorf("Expected shape [4], got %v", arr.Shape())
	}

	// Test getting values
	for i, expected := range interfaceData {
		val := arr.At(i)
		if val != expected {
			t.Errorf("At index %d: expected %v, got %v", i, expected, val)
		}
	}
}

func TestInterfaceArrayFull(t *testing.T) {
	// Test creating full interface array
	arr := Full(internal.Shape{3}, "test_value", internal.Interface)

	if arr.DType() != internal.Interface {
		t.Errorf("Expected dtype %v, got %v", internal.Interface, arr.DType())
	}

	for i := 0; i < 3; i++ {
		val := arr.At(i)
		if val != "test_value" {
			t.Errorf("At index %d: expected 'test_value', got %v", i, val)
		}
	}
}

func TestDTypeStringMethods(t *testing.T) {
	tests := []struct {
		dtype    internal.DType
		expected string
	}{
		{internal.String, "string"},
		{internal.Interface, "interface{}"},
		{internal.Float64, "float64"},
		{internal.Int32, "int32"},
	}

	for _, tt := range tests {
		result := tt.dtype.String()
		if result != tt.expected {
			t.Errorf("DType.String() for %v: expected %s, got %s", tt.dtype, tt.expected, result)
		}
	}
}

func TestDTypeSize(t *testing.T) {
	// Test that String and Interface have reasonable sizes
	stringSize := internal.String.Size()
	interfaceSize := internal.Interface.Size()

	if stringSize <= 0 {
		t.Errorf("String dtype size should be positive, got %d", stringSize)
	}

	if interfaceSize <= 0 {
		t.Errorf("Interface dtype size should be positive, got %d", interfaceSize)
	}

	// String and interface headers should be reasonable sizes
	if stringSize < 8 || stringSize > 32 {
		t.Errorf("String dtype size %d seems unreasonable", stringSize)
	}

	if interfaceSize < 8 || interfaceSize > 32 {
		t.Errorf("Interface dtype size %d seems unreasonable", interfaceSize)
	}
}

func TestInferDType(t *testing.T) {
	tests := []struct {
		value    interface{}
		expected internal.DType
	}{
		{"hello", internal.String},
		{[]string{"a", "b"}, internal.Interface}, // slice falls back to interface
		{42, internal.Int64},                     // int maps to int64
		{3.14, internal.Float64},
		{true, internal.Bool},
		{complex64(1 + 2i), internal.Complex64},
		{struct{}{}, internal.Interface}, // unknown types go to interface
	}

	for _, tt := range tests {
		result := internal.InferDType(tt.value)
		if result != tt.expected {
			t.Errorf("InferDType(%T): expected %v, got %v", tt.value, tt.expected, result)
		}
	}
}

func TestStringArrayOperations(t *testing.T) {
	// Test string array operations beyond basic creation
	arr1, err := FromSlice([]string{"hello", "world", "test"})
	if err != nil {
		t.Fatalf("Failed to create string array: %v", err)
	}

	// Test setting values
	err = arr1.Set("golang", 1)
	if err != nil {
		t.Errorf("Failed to set string value: %v", err)
	}

	val := arr1.At(1)
	if val != "golang" {
		t.Errorf("Expected 'golang', got %v", val)
	}

	// Test reshaping string arrays
	reshaped := arr1.Reshape(internal.Shape{1, 3})
	if !reshaped.Shape().Equal(internal.Shape{1, 3}) {
		t.Errorf("Expected shape [1, 3], got %v", reshaped.Shape())
	}

	// Test flattening
	flattened := reshaped.Flatten()
	if !flattened.Shape().Equal(internal.Shape{3}) {
		t.Errorf("Expected shape [3], got %v", flattened.Shape())
	}

	// Test copying
	copied := arr1.Copy()
	if copied.At(0) != arr1.At(0) {
		t.Errorf("Copy failed for string array")
	}

	// Test converting to slice
	slice := copied.ToSlice()
	stringSlice, ok := slice.([]string)
	if !ok {
		t.Errorf("Expected []string, got %T", slice)
	}

	expected := []string{"hello", "golang", "test"}
	for i, v := range stringSlice {
		if v != expected[i] {
			t.Errorf("At index %d: expected %s, got %s", i, expected[i], v)
		}
	}
}

func TestInterfaceArrayOperations(t *testing.T) {
	// Test heterogeneous interface array operations
	arr1, err := FromSlice([]interface{}{1, "hello", 3.14, true})
	if err != nil {
		t.Fatalf("Failed to create interface array: %v", err)
	}

	// Test setting values
	err = arr1.Set(false, 3)
	if err != nil {
		t.Errorf("Failed to set interface value: %v", err)
	}

	val := arr1.At(3)
	if val != false {
		t.Errorf("Expected false, got %v", val)
	}

	// Test converting different values to interface
	arr1.Set(42, 0)
	arr1.Set("world", 1)
	arr1.Set(2.71, 2)

	// Verify the values
	expected := []interface{}{42, "world", 2.71, false}
	for i, exp := range expected {
		val := arr1.At(i)
		if val != exp {
			t.Errorf("At index %d: expected %v, got %v", i, exp, val)
		}
	}

	// Test converting to slice
	slice := arr1.ToSlice()
	interfaceSlice, ok := slice.([]interface{})
	if !ok {
		t.Errorf("Expected []interface{}, got %T", slice)
	}

	for i, v := range interfaceSlice {
		if v != expected[i] {
			t.Errorf("At index %d: expected %v, got %v", i, expected[i], v)
		}
	}
}

func TestStringArrayEdgeCases(t *testing.T) {
	// Test empty string array - use Empty function instead of FromSlice for empty arrays
	arr := Empty(internal.Shape{0}, internal.String)

	if arr.Size() != 0 {
		t.Errorf("Expected length 0, got %d", arr.Size())
	}

	// Test array with empty strings
	arr2, err := FromSlice([]string{"", "hello", ""})
	if err != nil {
		t.Fatalf("Failed to create string array with empty strings: %v", err)
	}

	if arr2.At(0) != "" || arr2.At(2) != "" {
		t.Errorf("Empty strings not handled correctly")
	}

	// Test very long strings
	longString := make([]rune, 10000)
	for i := range longString {
		longString[i] = 'a'
	}
	longStr := string(longString)

	arr3, err := FromSlice([]string{longStr})
	if err != nil {
		t.Fatalf("Failed to create array with long string: %v", err)
	}

	if arr3.At(0) != longStr {
		t.Errorf("Long string not stored correctly")
	}
}

func TestConvertToStringFunction(t *testing.T) {
	// Test our convertToString helper function through Full
	tests := []struct {
		value    interface{}
		expected string
	}{
		{"hello", "hello"},
		{42, "42"},
		{3.14, "3.14"},
		{true, "true"},
		{false, "false"},
	}

	for _, tt := range tests {
		arr := Full(internal.Shape{1}, tt.value, internal.String)
		result := arr.At(0)
		if result != tt.expected {
			t.Errorf("convertToString(%v): expected %s, got %v", tt.value, tt.expected, result)
		}
	}
}
