package array

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/internal"
)

// ===== ELEMENT-WISE COMPARISON TESTS =====

func TestArrayEqual_BasicFloat64(t *testing.T) {
	data1 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	data2 := []float64{1.0, 2.0, 0.0, 4.0, 6.0}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr1.Equal(arr2)

	// Verify result has same shape as input
	if !result.Shape().Equal(arr1.Shape()) {
		t.Errorf("Equal() result shape = %v, want %v", result.Shape(), arr1.Shape())
	}

	// Verify result dtype is Bool
	if result.DType() != internal.Bool {
		t.Errorf("Equal() result dtype = %v, want %v", result.DType(), internal.Bool)
	}

	// Expected result: [true, true, false, true, false]
	expected := []bool{true, true, false, true, false}
	for i, exp := range expected {
		actual := result.At(i).(bool)
		if actual != exp {
			t.Errorf("Equal()[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayEqual_WithScalar(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 2.0, 5.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Equal(2.0)

	// Verify result has same shape as input
	if !result.Shape().Equal(arr.Shape()) {
		t.Errorf("Equal() result shape = %v, want %v", result.Shape(), arr.Shape())
	}

	// Expected result: [false, true, false, true, false]
	expected := []bool{false, true, false, true, false}
	for i, exp := range expected {
		actual := result.At(i).(bool)
		if actual != exp {
			t.Errorf("Equal(scalar)[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayEqual_2D(t *testing.T) {
	data1 := []float64{1, 2, 3, 4, 5, 6}
	data2 := []float64{1, 0, 3, 4, 0, 6}
	arr1, err := NewArrayWithShape(data1, internal.Shape{2, 3})
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}
	arr2, err := NewArrayWithShape(data2, internal.Shape{2, 3})
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr1.Equal(arr2)

	// Should compare element-wise: [[1==1, 2==0, 3==3], [4==4, 5==0, 6==6]]
	// Expected: [[true, false, true], [true, false, true]]
	expected := [][]bool{{true, false, true}, {true, false, true}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			actual := result.At(i, j).(bool)
			if actual != expected[i][j] {
				t.Errorf("Equal()[%d,%d] = %v, want %v", i, j, actual, expected[i][j])
			}
		}
	}
}

func TestArrayNotEqual_BasicFloat64(t *testing.T) {
	data1 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	data2 := []float64{1.0, 2.0, 0.0, 4.0, 6.0}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr1.NotEqual(arr2)

	// Verify result dtype is Bool
	if result.DType() != internal.Bool {
		t.Errorf("NotEqual() result dtype = %v, want %v", result.DType(), internal.Bool)
	}

	// Expected result: [false, false, true, false, true] (opposite of Equal)
	expected := []bool{false, false, true, false, true}
	for i, exp := range expected {
		actual := result.At(i).(bool)
		if actual != exp {
			t.Errorf("NotEqual()[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayGreater_BasicFloat64(t *testing.T) {
	data1 := []float64{1.0, 3.0, 2.0, 5.0, 4.0}
	data2 := []float64{2.0, 2.0, 2.0, 4.0, 4.0}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr1.Greater(arr2)

	// Expected result: [false, true, false, true, false]
	expected := []bool{false, true, false, true, false}
	for i, exp := range expected {
		actual := result.At(i).(bool)
		if actual != exp {
			t.Errorf("Greater()[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayGreater_WithScalar(t *testing.T) {
	data := []float64{1.0, 3.0, 2.0, 5.0, 4.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Greater(3.0)

	// Expected result: [false, false, false, true, true]
	expected := []bool{false, false, false, true, true}
	for i, exp := range expected {
		actual := result.At(i).(bool)
		if actual != exp {
			t.Errorf("Greater(scalar)[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayGreaterEqual_BasicFloat64(t *testing.T) {
	data1 := []float64{1.0, 3.0, 2.0, 5.0, 4.0}
	data2 := []float64{2.0, 3.0, 2.0, 4.0, 5.0}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr1.GreaterEqual(arr2)

	// Expected result: [false, true, true, true, false]
	expected := []bool{false, true, true, true, false}
	for i, exp := range expected {
		actual := result.At(i).(bool)
		if actual != exp {
			t.Errorf("GreaterEqual()[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayLess_BasicFloat64(t *testing.T) {
	data1 := []float64{1.0, 3.0, 2.0, 5.0, 4.0}
	data2 := []float64{2.0, 2.0, 2.0, 4.0, 4.0}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr1.Less(arr2)

	// Expected result: [true, false, false, false, false]
	expected := []bool{true, false, false, false, false}
	for i, exp := range expected {
		actual := result.At(i).(bool)
		if actual != exp {
			t.Errorf("Less()[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayLessEqual_BasicFloat64(t *testing.T) {
	data1 := []float64{1.0, 3.0, 2.0, 5.0, 4.0}
	data2 := []float64{2.0, 3.0, 2.0, 4.0, 5.0}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr1.LessEqual(arr2)

	// Expected result: [true, true, true, false, true]
	expected := []bool{true, true, true, false, true}
	for i, exp := range expected {
		actual := result.At(i).(bool)
		if actual != exp {
			t.Errorf("LessEqual()[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayComparisons_DifferentTypes(t *testing.T) {
	dataFloat := []float64{1.0, 2.0, 3.0}
	dataInt := []int64{1, 3, 2}
	arrFloat, err := NewArray(dataFloat)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arrInt, err := NewArray(dataInt)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arrFloat.Greater(arrInt)

	// Expected result: [false, false, true] (1.0 > 1, 2.0 > 3, 3.0 > 2)
	expected := []bool{false, false, true}
	for i, exp := range expected {
		actual := result.At(i).(bool)
		if actual != exp {
			t.Errorf("Greater(different types)[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayComparisons_NaN(t *testing.T) {
	data1 := []float64{1.0, math.NaN(), 3.0}
	data2 := []float64{1.0, 2.0, math.NaN()}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr1.Equal(arr2)

	// NaN != NaN in IEEE 754, so expected: [true, false, false]
	expected := []bool{true, false, false}
	for i, exp := range expected {
		actual := result.At(i).(bool)
		if actual != exp {
			t.Errorf("Equal(with NaN)[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayComparisons_IncompatibleShapes(t *testing.T) {
	data1 := []float64{1.0, 2.0, 3.0}
	data2 := []float64{1.0, 2.0}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test with incompatible shapes (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Equal() with incompatible shapes should panic")
		}
	}()

	arr1.Equal(arr2)
}

// ===== ARRAY-WISE COMPARISON TESTS =====

func TestAllClose_BasicFloat64(t *testing.T) {
	data1 := []float64{1.0, 2.0, 3.0}
	data2 := []float64{1.0001, 1.9999, 3.0001}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Should be close with lenient tolerances
	result := AllClose(arr1, arr2, 1e-03, 1e-06)
	if !result {
		t.Errorf("AllClose() = %v, want %v", result, true)
	}

	// Should not be close with very strict tolerances
	result = AllClose(arr1, arr2, 1e-10, 1e-10)
	if result {
		t.Errorf("AllClose(strict) = %v, want %v", result, false)
	}
}

func TestAllClose_WithNaN(t *testing.T) {
	data1 := []float64{1.0, math.NaN(), 3.0}
	data2 := []float64{1.0, math.NaN(), 3.0}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// NaN should make AllClose return false
	result := AllClose(arr1, arr2, 1e-03, 1e-06)
	if result {
		t.Errorf("AllClose(with NaN) = %v, want %v", result, false)
	}
}

func TestAllClose_DifferentShapes(t *testing.T) {
	data1 := []float64{1.0, 2.0, 3.0}
	data2 := []float64{1.0, 2.0}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Should return false for different shapes
	result := AllClose(arr1, arr2, 1e-03, 1e-06)
	if result {
		t.Errorf("AllClose(different shapes) = %v, want %v", result, false)
	}
}

func TestArrayEqual_Exact(t *testing.T) {
	data1 := []float64{1.0, 2.0, 3.0}
	data2 := []float64{1.0, 2.0, 3.0}
	data3 := []float64{1.0, 2.0, 3.0001}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr3, err := NewArray(data3)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Exact equality
	result := ArrayEqual(arr1, arr2)
	if !result {
		t.Errorf("ArrayEqual(exact) = %v, want %v", result, true)
	}

	// Not exactly equal
	result = ArrayEqual(arr1, arr3)
	if result {
		t.Errorf("ArrayEqual(not exact) = %v, want %v", result, false)
	}
}

func TestArrayEqual_DifferentShapes(t *testing.T) {
	data1 := []float64{1.0, 2.0, 3.0}
	data2 := []float64{1.0, 2.0}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Should return false for different shapes
	result := ArrayEqual(arr1, arr2)
	if result {
		t.Errorf("ArrayEqual(different shapes) = %v, want %v", result, false)
	}
}

func TestArrayEqual_WithNaN(t *testing.T) {
	data1 := []float64{1.0, math.NaN(), 3.0}
	data2 := []float64{1.0, math.NaN(), 3.0}
	arr1, err := NewArray(data1)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	arr2, err := NewArray(data2)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// NaN != NaN, so should return false
	result := ArrayEqual(arr1, arr2)
	if result {
		t.Errorf("ArrayEqual(with NaN) = %v, want %v", result, false)
	}
}
