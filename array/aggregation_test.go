package array

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/internal"
)

// TDD Phase: RED - Write failing tests for Sum() method
func TestArraySum_BasicFloat64(t *testing.T) {
	// Test basic sum without axis (should sum all elements)
	data := []float64{1.0, 2.0, 3.0, 4.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Sum()
	if result == nil {
		t.Fatal("Sum() returned nil")
	}

	// Sum should return a scalar (0-dimensional array)
	if result.Ndim() != 0 {
		t.Errorf("Sum() result dimensions = %d, want 0 (scalar)", result.Ndim())
	}

	// Check the actual sum value
	expected := 10.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Sum() = %v, want %v", actual, expected)
	}
}

func TestArraySum_BasicInt64(t *testing.T) {
	data := []int64{5, 10, 15}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Sum()
	if result == nil {
		t.Fatal("Sum() returned nil")
	}

	expected := int64(30)
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("Sum() = %v, want %v", actual, expected)
	}
}

func TestArraySum_EmptyArray(t *testing.T) {
	data := []float64{}
	arr, err := NewArray(data)
	// This might fail during array creation - that's expected for now
	if err != nil {
		t.Skip("Empty array creation not supported yet")
		return
	}

	result := arr.Sum()
	expected := 0.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Sum() of empty array = %v, want %v", actual, expected)
	}
}

func TestArraySum_2D_NoAxis(t *testing.T) {
	// Test 2D array sum without axis (should sum all elements)
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3} // 2x3 matrix
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Sum()
	if result == nil {
		t.Fatal("Sum() returned nil")
	}

	// Should sum all elements: 1+2+3+4+5+6 = 21
	expected := 21.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Sum() = %v, want %v", actual, expected)
	}
}

func TestArraySum_WithAxis0(t *testing.T) {
	// Test 2D array sum along axis 0 (sum columns)
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[1,2,3], [4,5,6]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Sum(0) // Sum along axis 0
	if result == nil {
		t.Fatal("Sum(0) returned nil")
	}

	// Should return [5, 7, 9] (column sums)
	expectedShape := internal.Shape{3}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("Sum(0) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []float64{5, 7, 9} // [1+4, 2+5, 3+6]
	for i, expected := range expectedValues {
		actual := result.At(i).(float64)
		if actual != expected {
			t.Errorf("Sum(0)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArraySum_WithAxis1(t *testing.T) {
	// Test 2D array sum along axis 1 (sum rows)
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[1,2,3], [4,5,6]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Sum(1) // Sum along axis 1
	if result == nil {
		t.Fatal("Sum(1) returned nil")
	}

	// Should return [6, 15] (row sums)
	expectedShape := internal.Shape{2}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("Sum(1) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []float64{6, 15} // [1+2+3, 4+5+6]
	for i, expected := range expectedValues {
		actual := result.At(i).(float64)
		if actual != expected {
			t.Errorf("Sum(1)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArraySum_3D_WithAxis(t *testing.T) {
	// Test 3D array sum along different axes
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	shape := internal.Shape{2, 2, 2} // 2x2x2 cube
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	// Test sum along axis 0
	result0 := arr.Sum(0)
	expectedShape0 := internal.Shape{2, 2}
	if !result0.Shape().Equal(expectedShape0) {
		t.Errorf("Sum(0) result shape = %v, want %v", result0.Shape(), expectedShape0)
	}

	// Test sum along axis 1
	result1 := arr.Sum(1)
	expectedShape1 := internal.Shape{2, 2}
	if !result1.Shape().Equal(expectedShape1) {
		t.Errorf("Sum(1) result shape = %v, want %v", result1.Shape(), expectedShape1)
	}

	// Test sum along axis 2
	result2 := arr.Sum(2)
	expectedShape2 := internal.Shape{2, 2}
	if !result2.Shape().Equal(expectedShape2) {
		t.Errorf("Sum(2) result shape = %v, want %v", result2.Shape(), expectedShape2)
	}
}

func TestArraySum_InvalidAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test with invalid axis (should panic or return error)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Sum() with invalid axis should panic")
		}
	}()

	arr.Sum(5) // Invalid axis for 1D array
}

func TestArraySum_MultipleAxes(t *testing.T) {
	// Test sum along multiple axes at once
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	shape := internal.Shape{2, 2, 2}
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	// For now, skip multiple axes as it's not implemented yet
	defer func() {
		if r := recover(); r != nil {
			if str, ok := r.(string); ok && str == "multiple axes not implemented yet" {
				t.Skip("Multiple axes sum not implemented yet")
				return
			}
			panic(r) // Re-panic if it's a different error
		}
	}()

	result := arr.Sum(0, 2) // Sum along axes 0 and 2
	expectedShape := internal.Shape{2}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("Sum(0,2) result shape = %v, want %v", result.Shape(), expectedShape)
	}
}

// TDD Phase: RED - Write failing tests for Mean() method
func TestArrayMean_BasicFloat64(t *testing.T) {
	// Test basic mean without axis (should average all elements)
	data := []float64{2.0, 4.0, 6.0, 8.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Mean()
	if result == nil {
		t.Fatal("Mean() returned nil")
	}

	// Mean should return a scalar (0-dimensional array)
	if result.Ndim() != 0 {
		t.Errorf("Mean() result dimensions = %d, want 0 (scalar)", result.Ndim())
	}

	// Check the actual mean value: (2+4+6+8)/4 = 5.0
	expected := 5.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Mean() = %v, want %v", actual, expected)
	}
}

func TestArrayMean_BasicInt64(t *testing.T) {
	data := []int64{10, 20, 30}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Mean()
	if result == nil {
		t.Fatal("Mean() returned nil")
	}

	// Mean of integers should return float64: (10+20+30)/3 = 20.0
	expected := 20.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Mean() = %v, want %v", actual, expected)
	}

	// Result should be Float64 type even for integer input
	if result.DType() != internal.Float64 {
		t.Errorf("Mean() result dtype = %v, want %v", result.DType(), internal.Float64)
	}
}

func TestArrayMean_2D_NoAxis(t *testing.T) {
	// Test 2D array mean without axis (should average all elements)
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[1,2,3], [4,5,6]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Mean()
	if result == nil {
		t.Fatal("Mean() returned nil")
	}

	// Should average all elements: (1+2+3+4+5+6)/6 = 3.5
	expected := 3.5
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Mean() = %v, want %v", actual, expected)
	}
}

func TestArrayMean_WithAxis0(t *testing.T) {
	// Test 2D array mean along axis 0 (mean of columns)
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[1,2,3], [4,5,6]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Mean(0) // Mean along axis 0
	if result == nil {
		t.Fatal("Mean(0) returned nil")
	}

	// Should return [2.5, 3.5, 4.5] (column means: [(1+4)/2, (2+5)/2, (3+6)/2])
	expectedShape := internal.Shape{3}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("Mean(0) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []float64{2.5, 3.5, 4.5}
	for i, expected := range expectedValues {
		actual := result.At(i).(float64)
		if actual != expected {
			t.Errorf("Mean(0)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayMean_WithAxis1(t *testing.T) {
	// Test 2D array mean along axis 1 (mean of rows)
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[1,2,3], [4,5,6]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Mean(1) // Mean along axis 1
	if result == nil {
		t.Fatal("Mean(1) returned nil")
	}

	// Should return [2.0, 5.0] (row means: [(1+2+3)/3, (4+5+6)/3])
	expectedShape := internal.Shape{2}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("Mean(1) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []float64{2.0, 5.0}
	for i, expected := range expectedValues {
		actual := result.At(i).(float64)
		if actual != expected {
			t.Errorf("Mean(1)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayMean_EmptyArray(t *testing.T) {
	data := []float64{}
	arr, err := NewArray(data)
	// This might fail during array creation - that's expected for now
	if err != nil {
		t.Skip("Empty array creation not supported yet")
		return
	}

	defer func() {
		if r := recover(); r != nil {
			// Expected: division by zero or similar error
			return
		}
		t.Error("Mean() of empty array should panic or handle gracefully")
	}()

	arr.Mean()
}

func TestArrayMean_InvalidAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test with invalid axis (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Mean() with invalid axis should panic")
		}
	}()

	arr.Mean(5) // Invalid axis for 1D array
}

// TDD Phase: RED - Write failing tests for Min() method
func TestArrayMin_BasicFloat64(t *testing.T) {
	// Test basic min without axis (should find minimum of all elements)
	data := []float64{3.0, 1.0, 4.0, 1.0, 5.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Min()
	if result == nil {
		t.Fatal("Min() returned nil")
	}

	// Min should return a scalar (0-dimensional array)
	if result.Ndim() != 0 {
		t.Errorf("Min() result dimensions = %d, want 0 (scalar)", result.Ndim())
	}

	// Check the actual min value
	expected := 1.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Min() = %v, want %v", actual, expected)
	}

	// Result should maintain input dtype
	if result.DType() != internal.Float64 {
		t.Errorf("Min() result dtype = %v, want %v", result.DType(), internal.Float64)
	}
}

func TestArrayMin_BasicInt64(t *testing.T) {
	data := []int64{5, 2, 8, 1, 3}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Min()
	if result == nil {
		t.Fatal("Min() returned nil")
	}

	expected := int64(1)
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("Min() = %v, want %v", actual, expected)
	}

	// Result should maintain Int64 type (unlike Mean)
	if result.DType() != internal.Int64 {
		t.Errorf("Min() result dtype = %v, want %v", result.DType(), internal.Int64)
	}
}

func TestArrayMin_2D_NoAxis(t *testing.T) {
	// Test 2D array min without axis (should find minimum of all elements)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Min()
	if result == nil {
		t.Fatal("Min() returned nil")
	}

	// Should find minimum of all elements: min(3,1,4,6,2,8) = 1
	expected := 1.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Min() = %v, want %v", actual, expected)
	}
}

func TestArrayMin_WithAxis0(t *testing.T) {
	// Test 2D array min along axis 0 (min of columns)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Min(0) // Min along axis 0
	if result == nil {
		t.Fatal("Min(0) returned nil")
	}

	// Should return [3, 1, 4] (column mins: [min(3,6), min(1,2), min(4,8)])
	expectedShape := internal.Shape{3}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("Min(0) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []float64{3, 1, 4}
	for i, expected := range expectedValues {
		actual := result.At(i).(float64)
		if actual != expected {
			t.Errorf("Min(0)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayMin_WithAxis1(t *testing.T) {
	// Test 2D array min along axis 1 (min of rows)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Min(1) // Min along axis 1
	if result == nil {
		t.Fatal("Min(1) returned nil")
	}

	// Should return [1, 2] (row mins: [min(3,1,4), min(6,2,8)])
	expectedShape := internal.Shape{2}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("Min(1) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []float64{1, 2}
	for i, expected := range expectedValues {
		actual := result.At(i).(float64)
		if actual != expected {
			t.Errorf("Min(1)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayMin_NegativeValues(t *testing.T) {
	// Test with negative values
	data := []float64{-3.0, -1.0, -4.0, -1.5}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Min()
	expected := -4.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Min() with negatives = %v, want %v", actual, expected)
	}
}

func TestArrayMin_SingleElement(t *testing.T) {
	// Test with single element
	data := []float64{42.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Min()
	expected := 42.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Min() single element = %v, want %v", actual, expected)
	}
}

func TestArrayMin_InvalidAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test with invalid axis (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Min() with invalid axis should panic")
		}
	}()

	arr.Min(5) // Invalid axis for 1D array
}

// TDD Phase: RED - Write failing tests for Max() method
func TestArrayMax_BasicFloat64(t *testing.T) {
	// Test basic max without axis (should find maximum of all elements)
	data := []float64{3.0, 1.0, 4.0, 1.0, 5.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Max()
	if result == nil {
		t.Fatal("Max() returned nil")
	}

	// Max should return a scalar (0-dimensional array)
	if result.Ndim() != 0 {
		t.Errorf("Max() result dimensions = %d, want 0 (scalar)", result.Ndim())
	}

	// Check the actual max value
	expected := 5.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Max() = %v, want %v", actual, expected)
	}

	// Result should maintain input dtype
	if result.DType() != internal.Float64 {
		t.Errorf("Max() result dtype = %v, want %v", result.DType(), internal.Float64)
	}
}

func TestArrayMax_BasicInt64(t *testing.T) {
	data := []int64{5, 2, 8, 1, 3}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Max()
	if result == nil {
		t.Fatal("Max() returned nil")
	}

	expected := int64(8)
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("Max() = %v, want %v", actual, expected)
	}

	// Result should maintain Int64 type
	if result.DType() != internal.Int64 {
		t.Errorf("Max() result dtype = %v, want %v", result.DType(), internal.Int64)
	}
}

func TestArrayMax_2D_NoAxis(t *testing.T) {
	// Test 2D array max without axis (should find maximum of all elements)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Max()
	if result == nil {
		t.Fatal("Max() returned nil")
	}

	// Should find maximum of all elements: max(3,1,4,6,2,8) = 8
	expected := 8.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Max() = %v, want %v", actual, expected)
	}
}

func TestArrayMax_WithAxis0(t *testing.T) {
	// Test 2D array max along axis 0 (max of columns)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Max(0) // Max along axis 0
	if result == nil {
		t.Fatal("Max(0) returned nil")
	}

	// Should return [6, 2, 8] (column maxs: [max(3,6), max(1,2), max(4,8)])
	expectedShape := internal.Shape{3}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("Max(0) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []float64{6, 2, 8}
	for i, expected := range expectedValues {
		actual := result.At(i).(float64)
		if actual != expected {
			t.Errorf("Max(0)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayMax_WithAxis1(t *testing.T) {
	// Test 2D array max along axis 1 (max of rows)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Max(1) // Max along axis 1
	if result == nil {
		t.Fatal("Max(1) returned nil")
	}

	// Should return [4, 8] (row maxs: [max(3,1,4), max(6,2,8)])
	expectedShape := internal.Shape{2}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("Max(1) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []float64{4, 8}
	for i, expected := range expectedValues {
		actual := result.At(i).(float64)
		if actual != expected {
			t.Errorf("Max(1)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayMax_NegativeValues(t *testing.T) {
	// Test with negative values
	data := []float64{-3.0, -1.0, -4.0, -1.5}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Max()
	expected := -1.0
	actual := result.At().(float64)
	if actual != expected {
		t.Errorf("Max() with negatives = %v, want %v", actual, expected)
	}
}

func TestArrayMax_InvalidAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test with invalid axis (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Max() with invalid axis should panic")
		}
	}()

	arr.Max(5) // Invalid axis for 1D array
}

// TDD Phase: RED - Write failing tests for ArgMin() method
func TestArrayArgMin_BasicFloat64(t *testing.T) {
	// Test basic argmin without axis (should return index of minimum element)
	data := []float64{3.0, 1.0, 4.0, 1.0, 5.0} // min at indices 1 and 3, should return 1 (first occurrence)
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.ArgMin()
	if result == nil {
		t.Fatal("ArgMin() returned nil")
	}

	// ArgMin should return a scalar (0-dimensional array) containing index
	if result.Ndim() != 0 {
		t.Errorf("ArgMin() result dimensions = %d, want 0 (scalar)", result.Ndim())
	}

	// Check the actual index (should be 1 - first occurrence of minimum)
	expected := int64(1)
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("ArgMin() = %v, want %v", actual, expected)
	}

	// Result should always be Int64 regardless of input dtype
	if result.DType() != internal.Int64 {
		t.Errorf("ArgMin() result dtype = %v, want %v", result.DType(), internal.Int64)
	}
}

func TestArrayArgMin_BasicInt64(t *testing.T) {
	data := []int64{5, 2, 8, 1, 3} // min at index 3
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.ArgMin()
	if result == nil {
		t.Fatal("ArgMin() returned nil")
	}

	expected := int64(3)
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("ArgMin() = %v, want %v", actual, expected)
	}

	// Result should always be Int64 type
	if result.DType() != internal.Int64 {
		t.Errorf("ArgMin() result dtype = %v, want %v", result.DType(), internal.Int64)
	}
}

func TestArrayArgMin_2D_NoAxis(t *testing.T) {
	// Test 2D array argmin without axis (should return flat index of minimum)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.ArgMin()
	if result == nil {
		t.Fatal("ArgMin() returned nil")
	}

	// Should return flat index of minimum element (1 is at flat index 1)
	expected := int64(1)
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("ArgMin() = %v, want %v", actual, expected)
	}
}

func TestArrayArgMin_WithAxis0(t *testing.T) {
	// Test 2D array argmin along axis 0 (indices of column minimums)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.ArgMin(0) // ArgMin along axis 0
	if result == nil {
		t.Fatal("ArgMin(0) returned nil")
	}

	// Should return [0, 0, 0] (indices within each column where min occurs)
	// Column 0: min(3,6) = 3 at index 0
	// Column 1: min(1,2) = 1 at index 0
	// Column 2: min(4,8) = 4 at index 0
	expectedShape := internal.Shape{3}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("ArgMin(0) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []int64{0, 0, 0}
	for i, expected := range expectedValues {
		actual := result.At(i).(int64)
		if actual != expected {
			t.Errorf("ArgMin(0)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayArgMin_WithAxis1(t *testing.T) {
	// Test 2D array argmin along axis 1 (indices of row minimums)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.ArgMin(1) // ArgMin along axis 1
	if result == nil {
		t.Fatal("ArgMin(1) returned nil")
	}

	// Should return [1, 1] (indices within each row where min occurs)
	// Row 0: min(3,1,4) = 1 at index 1
	// Row 1: min(6,2,8) = 2 at index 1
	expectedShape := internal.Shape{2}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("ArgMin(1) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []int64{1, 1}
	for i, expected := range expectedValues {
		actual := result.At(i).(int64)
		if actual != expected {
			t.Errorf("ArgMin(1)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayArgMin_Ties(t *testing.T) {
	// Test behavior with tied minimum values (should return first occurrence)
	data := []float64{2.0, 1.0, 3.0, 1.0, 2.0} // min value 1.0 at indices 1 and 3
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.ArgMin()
	expected := int64(1) // Should return first occurrence
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("ArgMin() with ties = %v, want %v (first occurrence)", actual, expected)
	}
}

func TestArrayArgMin_SingleElement(t *testing.T) {
	// Test with single element
	data := []float64{42.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.ArgMin()
	expected := int64(0)
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("ArgMin() single element = %v, want %v", actual, expected)
	}
}

func TestArrayArgMin_InvalidAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test with invalid axis (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("ArgMin() with invalid axis should panic")
		}
	}()

	arr.ArgMin(5) // Invalid axis for 1D array
}

// TDD Phase: RED - Write failing tests for ArgMax() method
func TestArrayArgMax_BasicFloat64(t *testing.T) {
	// Test basic argmax without axis (should return index of maximum element)
	data := []float64{3.0, 1.0, 4.0, 1.0, 5.0} // max at index 4
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.ArgMax()
	if result == nil {
		t.Fatal("ArgMax() returned nil")
	}

	// ArgMax should return a scalar (0-dimensional array) containing index
	if result.Ndim() != 0 {
		t.Errorf("ArgMax() result dimensions = %d, want 0 (scalar)", result.Ndim())
	}

	// Check the actual index
	expected := int64(4)
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("ArgMax() = %v, want %v", actual, expected)
	}

	// Result should always be Int64 regardless of input dtype
	if result.DType() != internal.Int64 {
		t.Errorf("ArgMax() result dtype = %v, want %v", result.DType(), internal.Int64)
	}
}

func TestArrayArgMax_BasicInt64(t *testing.T) {
	data := []int64{5, 2, 8, 1, 3} // max at index 2
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.ArgMax()
	if result == nil {
		t.Fatal("ArgMax() returned nil")
	}

	expected := int64(2)
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("ArgMax() = %v, want %v", actual, expected)
	}

	// Result should always be Int64 type
	if result.DType() != internal.Int64 {
		t.Errorf("ArgMax() result dtype = %v, want %v", result.DType(), internal.Int64)
	}
}

func TestArrayArgMax_2D_NoAxis(t *testing.T) {
	// Test 2D array argmax without axis (should return flat index of maximum)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.ArgMax()
	if result == nil {
		t.Fatal("ArgMax() returned nil")
	}

	// Should return flat index of maximum element (8 is at flat index 5)
	expected := int64(5)
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("ArgMax() = %v, want %v", actual, expected)
	}
}

func TestArrayArgMax_WithAxis0(t *testing.T) {
	// Test 2D array argmax along axis 0 (indices of column maximums)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.ArgMax(0) // ArgMax along axis 0
	if result == nil {
		t.Fatal("ArgMax(0) returned nil")
	}

	// Should return [1, 1, 1] (indices within each column where max occurs)
	// Column 0: max(3,6) = 6 at index 1
	// Column 1: max(1,2) = 2 at index 1
	// Column 2: max(4,8) = 8 at index 1
	expectedShape := internal.Shape{3}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("ArgMax(0) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []int64{1, 1, 1}
	for i, expected := range expectedValues {
		actual := result.At(i).(int64)
		if actual != expected {
			t.Errorf("ArgMax(0)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayArgMax_WithAxis1(t *testing.T) {
	// Test 2D array argmax along axis 1 (indices of row maximums)
	data := []float64{3, 1, 4, 6, 2, 8}
	shape := internal.Shape{2, 3} // 2x3 matrix: [[3,1,4], [6,2,8]]
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.ArgMax(1) // ArgMax along axis 1
	if result == nil {
		t.Fatal("ArgMax(1) returned nil")
	}

	// Should return [2, 2] (indices within each row where max occurs)
	// Row 0: max(3,1,4) = 4 at index 2
	// Row 1: max(6,2,8) = 8 at index 2
	expectedShape := internal.Shape{2}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("ArgMax(1) result shape = %v, want %v", result.Shape(), expectedShape)
	}

	expectedValues := []int64{2, 2}
	for i, expected := range expectedValues {
		actual := result.At(i).(int64)
		if actual != expected {
			t.Errorf("ArgMax(1)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayArgMax_Ties(t *testing.T) {
	// Test behavior with tied maximum values (should return first occurrence)
	data := []float64{2.0, 3.0, 1.0, 3.0, 2.0} // max value 3.0 at indices 1 and 3
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.ArgMax()
	expected := int64(1) // Should return first occurrence
	actual := result.At().(int64)
	if actual != expected {
		t.Errorf("ArgMax() with ties = %v, want %v (first occurrence)", actual, expected)
	}
}

func TestArrayArgMax_InvalidAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test with invalid axis (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("ArgMax() with invalid axis should panic")
		}
	}()

	arr.ArgMax(5) // Invalid axis for 1D array
}

// ===== VAR() TESTS =====

func TestArrayVar_BasicFloat64(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Var()

	// Verify result is scalar (0-dimensional)
	if result.Ndim() != 0 {
		t.Errorf("Var() result should be scalar, got %d dimensions", result.Ndim())
	}

	// Verify result dtype is Float64
	if result.DType() != internal.Float64 {
		t.Errorf("Var() result dtype = %v, want %v", result.DType(), internal.Float64)
	}

	// Expected variance: mean = 3.0, variance = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5 = (4+1+0+1+4)/5 = 2.0
	expected := 2.0
	actual := result.At().(float64)
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("Var() = %v, want %v", actual, expected)
	}
}

func TestArrayVar_BasicInt64(t *testing.T) {
	data := []int64{1, 2, 3, 4, 5}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Var()

	// Verify result dtype is Float64 (always converts to float for variance)
	if result.DType() != internal.Float64 {
		t.Errorf("Var() result dtype = %v, want %v", result.DType(), internal.Float64)
	}

	// Expected variance: same as above = 2.0
	expected := 2.0
	actual := result.At().(float64)
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("Var() = %v, want %v", actual, expected)
	}
}

func TestArrayVar_2D_NoAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArrayWithShape(data, internal.Shape{2, 3})
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.Var()

	// Should compute variance over all elements: mean = 3.5, var = ((1-3.5)² + ... + (6-3.5)²) / 6
	// = (6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25) / 6 = 17.5 / 6 ≈ 2.9167
	expected := 17.5 / 6.0
	actual := result.At().(float64)
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("Var() = %v, want %v", actual, expected)
	}
}

func TestArrayVar_WithAxis0(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArrayWithShape(data, internal.Shape{2, 3}) // [[1,2,3], [4,5,6]]
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Var(0)

	// Should compute variance along axis 0: for each column
	// Column 0: [1,4] -> mean=2.5, var=((1-2.5)² + (4-2.5)²)/2 = (2.25 + 2.25)/2 = 2.25
	// Column 1: [2,5] -> mean=3.5, var=((2-3.5)² + (5-3.5)²)/2 = (2.25 + 2.25)/2 = 2.25
	// Column 2: [3,6] -> mean=4.5, var=((3-4.5)² + (6-4.5)²)/2 = (2.25 + 2.25)/2 = 2.25

	if result.Ndim() != 1 || result.Size() != 3 {
		t.Errorf("Var(0) should return 1D array with 3 elements, got shape %v", result.Shape())
	}

	for i := 0; i < 3; i++ {
		expected := 2.25
		actual := result.At(i).(float64)
		if math.Abs(actual-expected) > 1e-10 {
			t.Errorf("Var(0)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayVar_WithAxis1(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArrayWithShape(data, internal.Shape{2, 3}) // [[1,2,3], [4,5,6]]
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Var(1)

	// Should compute variance along axis 1: for each row
	// Row 0: [1,2,3] -> mean=2, var=((1-2)² + (2-2)² + (3-2)²)/3 = (1 + 0 + 1)/3 = 2/3
	// Row 1: [4,5,6] -> mean=5, var=((4-5)² + (5-5)² + (6-5)²)/3 = (1 + 0 + 1)/3 = 2/3

	if result.Ndim() != 1 || result.Size() != 2 {
		t.Errorf("Var(1) should return 1D array with 2 elements, got shape %v", result.Shape())
	}

	for i := 0; i < 2; i++ {
		expected := 2.0 / 3.0
		actual := result.At(i).(float64)
		if math.Abs(actual-expected) > 1e-10 {
			t.Errorf("Var(1)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayVar_SingleElement(t *testing.T) {
	data := []float64{42.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Var()

	// Variance of single element should be 0
	expected := 0.0
	actual := result.At().(float64)
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("Var() of single element = %v, want %v", actual, expected)
	}
}

func TestArrayVar_InvalidAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test with invalid axis (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Var() with invalid axis should panic")
		}
	}()

	arr.Var(5) // Invalid axis for 1D array
}

// ===== STD() TESTS =====

func TestArrayStd_BasicFloat64(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Std()

	// Verify result is scalar (0-dimensional)
	if result.Ndim() != 0 {
		t.Errorf("Std() result should be scalar, got %d dimensions", result.Ndim())
	}

	// Verify result dtype is Float64
	if result.DType() != internal.Float64 {
		t.Errorf("Std() result dtype = %v, want %v", result.DType(), internal.Float64)
	}

	// Expected std: sqrt(variance) = sqrt(2.0) ≈ 1.414213562373095
	expected := math.Sqrt(2.0)
	actual := result.At().(float64)
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("Std() = %v, want %v", actual, expected)
	}
}

func TestArrayStd_BasicInt64(t *testing.T) {
	data := []int64{1, 2, 3, 4, 5}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Std()

	// Verify result dtype is Float64 (always converts to float for std)
	if result.DType() != internal.Float64 {
		t.Errorf("Std() result dtype = %v, want %v", result.DType(), internal.Float64)
	}

	// Expected std: sqrt(2.0) ≈ 1.414213562373095
	expected := math.Sqrt(2.0)
	actual := result.At().(float64)
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("Std() = %v, want %v", actual, expected)
	}
}

func TestArrayStd_2D_NoAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArrayWithShape(data, internal.Shape{2, 3})
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Std()

	// Should compute std over all elements: sqrt(17.5/6)
	expected := math.Sqrt(17.5 / 6.0)
	actual := result.At().(float64)
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("Std() = %v, want %v", actual, expected)
	}
}

func TestArrayStd_WithAxis0(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArrayWithShape(data, internal.Shape{2, 3}) // [[1,2,3], [4,5,6]]
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Std(0)

	// Should compute std along axis 0: sqrt(2.25) = 1.5 for each column
	if result.Ndim() != 1 || result.Size() != 3 {
		t.Errorf("Std(0) should return 1D array with 3 elements, got shape %v", result.Shape())
	}

	for i := 0; i < 3; i++ {
		expected := math.Sqrt(2.25)
		actual := result.At(i).(float64)
		if math.Abs(actual-expected) > 1e-10 {
			t.Errorf("Std(0)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayStd_WithAxis1(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArrayWithShape(data, internal.Shape{2, 3}) // [[1,2,3], [4,5,6]]
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Std(1)

	// Should compute std along axis 1: sqrt(2/3) for each row
	if result.Ndim() != 1 || result.Size() != 2 {
		t.Errorf("Std(1) should return 1D array with 2 elements, got shape %v", result.Shape())
	}

	for i := 0; i < 2; i++ {
		expected := math.Sqrt(2.0 / 3.0)
		actual := result.At(i).(float64)
		if math.Abs(actual-expected) > 1e-10 {
			t.Errorf("Std(1)[%d] = %v, want %v", i, actual, expected)
		}
	}
}

func TestArrayStd_SingleElement(t *testing.T) {
	data := []float64{42.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.Std()

	// Standard deviation of single element should be 0
	expected := 0.0
	actual := result.At().(float64)
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("Std() of single element = %v, want %v", actual, expected)
	}
}

func TestArrayStd_InvalidAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test with invalid axis (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Std() with invalid axis should panic")
		}
	}()

	arr.Std(5) // Invalid axis for 1D array
}

// ===== CUMSUM() TESTS =====

func TestArrayCumSum_BasicFloat64(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.CumSum(0)

	// Verify result has same shape as input
	if !result.Shape().Equal(arr.Shape()) {
		t.Errorf("CumSum() result shape = %v, want %v", result.Shape(), arr.Shape())
	}

	// Verify result dtype is same as input
	if result.DType() != arr.DType() {
		t.Errorf("CumSum() result dtype = %v, want %v", result.DType(), arr.DType())
	}

	// Expected cumulative sum: [1, 1+2, 1+2+3, 1+2+3+4, 1+2+3+4+5] = [1, 3, 6, 10, 15]
	expected := []float64{1.0, 3.0, 6.0, 10.0, 15.0}
	for i, exp := range expected {
		actual := result.At(i).(float64)
		if math.Abs(actual-exp) > 1e-10 {
			t.Errorf("CumSum()[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayCumSum_BasicInt64(t *testing.T) {
	data := []int64{1, 2, 3, 4, 5}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.CumSum(0)

	// Verify result dtype is same as input
	if result.DType() != internal.Int64 {
		t.Errorf("CumSum() result dtype = %v, want %v", result.DType(), internal.Int64)
	}

	// Expected cumulative sum: [1, 3, 6, 10, 15]
	expected := []int64{1, 3, 6, 10, 15}
	for i, exp := range expected {
		actual := result.At(i).(int64)
		if actual != exp {
			t.Errorf("CumSum()[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayCumSum_2D_Axis0(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArrayWithShape(data, internal.Shape{2, 3}) // [[1,2,3], [4,5,6]]
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.CumSum(0)

	// Should compute cumulative sum along axis 0 (down rows)
	// [[1,2,3], [1+4,2+5,3+6]] = [[1,2,3], [5,7,9]]

	if !result.Shape().Equal(arr.Shape()) {
		t.Errorf("CumSum(0) result shape = %v, want %v", result.Shape(), arr.Shape())
	}

	expected := [][]float64{{1, 2, 3}, {5, 7, 9}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			actual := result.At(i, j).(float64)
			if math.Abs(actual-expected[i][j]) > 1e-10 {
				t.Errorf("CumSum(0)[%d,%d] = %v, want %v", i, j, actual, expected[i][j])
			}
		}
	}
}

func TestArrayCumSum_2D_Axis1(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArrayWithShape(data, internal.Shape{2, 3}) // [[1,2,3], [4,5,6]]
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.CumSum(1)

	// Should compute cumulative sum along axis 1 (across columns)
	// [[1,1+2,1+2+3], [4,4+5,4+5+6]] = [[1,3,6], [4,9,15]]

	if !result.Shape().Equal(arr.Shape()) {
		t.Errorf("CumSum(1) result shape = %v, want %v", result.Shape(), arr.Shape())
	}

	expected := [][]float64{{1, 3, 6}, {4, 9, 15}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			actual := result.At(i, j).(float64)
			if math.Abs(actual-expected[i][j]) > 1e-10 {
				t.Errorf("CumSum(1)[%d,%d] = %v, want %v", i, j, actual, expected[i][j])
			}
		}
	}
}

func TestArrayCumSum_SingleElement(t *testing.T) {
	data := []float64{42.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.CumSum(0)

	// Cumulative sum of single element should be the element itself
	expected := 42.0
	actual := result.At(0).(float64)
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("CumSum() of single element = %v, want %v", actual, expected)
	}
}

func TestArrayCumSum_InvalidAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test with invalid axis (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("CumSum() with invalid axis should panic")
		}
	}()

	arr.CumSum(5) // Invalid axis for 1D array
}

// ===== CUMPROD() TESTS =====

func TestArrayCumProd_BasicFloat64(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.CumProd(0)

	// Verify result has same shape as input
	if !result.Shape().Equal(arr.Shape()) {
		t.Errorf("CumProd() result shape = %v, want %v", result.Shape(), arr.Shape())
	}

	// Verify result dtype is same as input
	if result.DType() != arr.DType() {
		t.Errorf("CumProd() result dtype = %v, want %v", result.DType(), arr.DType())
	}

	// Expected cumulative product: [1, 1*2, 1*2*3, 1*2*3*4, 1*2*3*4*5] = [1, 2, 6, 24, 120]
	expected := []float64{1.0, 2.0, 6.0, 24.0, 120.0}
	for i, exp := range expected {
		actual := result.At(i).(float64)
		if math.Abs(actual-exp) > 1e-10 {
			t.Errorf("CumProd()[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayCumProd_BasicInt64(t *testing.T) {
	data := []int64{1, 2, 3, 4, 5}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.CumProd(0)

	// Verify result dtype is same as input
	if result.DType() != internal.Int64 {
		t.Errorf("CumProd() result dtype = %v, want %v", result.DType(), internal.Int64)
	}

	// Expected cumulative product: [1, 2, 6, 24, 120]
	expected := []int64{1, 2, 6, 24, 120}
	for i, exp := range expected {
		actual := result.At(i).(int64)
		if actual != exp {
			t.Errorf("CumProd()[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayCumProd_2D_Axis0(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArrayWithShape(data, internal.Shape{2, 3}) // [[1,2,3], [4,5,6]]
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.CumProd(0)

	// Should compute cumulative product along axis 0 (down rows)
	// [[1,2,3], [1*4,2*5,3*6]] = [[1,2,3], [4,10,18]]

	if !result.Shape().Equal(arr.Shape()) {
		t.Errorf("CumProd(0) result shape = %v, want %v", result.Shape(), arr.Shape())
	}

	expected := [][]float64{{1, 2, 3}, {4, 10, 18}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			actual := result.At(i, j).(float64)
			if math.Abs(actual-expected[i][j]) > 1e-10 {
				t.Errorf("CumProd(0)[%d,%d] = %v, want %v", i, j, actual, expected[i][j])
			}
		}
	}
}

func TestArrayCumProd_2D_Axis1(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArrayWithShape(data, internal.Shape{2, 3}) // [[1,2,3], [4,5,6]]
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	result := arr.CumProd(1)

	// Should compute cumulative product along axis 1 (across columns)
	// [[1,1*2,1*2*3], [4,4*5,4*5*6]] = [[1,2,6], [4,20,120]]

	if !result.Shape().Equal(arr.Shape()) {
		t.Errorf("CumProd(1) result shape = %v, want %v", result.Shape(), arr.Shape())
	}

	expected := [][]float64{{1, 2, 6}, {4, 20, 120}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			actual := result.At(i, j).(float64)
			if math.Abs(actual-expected[i][j]) > 1e-10 {
				t.Errorf("CumProd(1)[%d,%d] = %v, want %v", i, j, actual, expected[i][j])
			}
		}
	}
}

func TestArrayCumProd_WithZero(t *testing.T) {
	data := []float64{1.0, 2.0, 0.0, 4.0, 5.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.CumProd(0)

	// Expected cumulative product: [1, 2, 0, 0, 0] (zero propagates)
	expected := []float64{1.0, 2.0, 0.0, 0.0, 0.0}
	for i, exp := range expected {
		actual := result.At(i).(float64)
		if math.Abs(actual-exp) > 1e-10 {
			t.Errorf("CumProd()[%d] = %v, want %v", i, actual, exp)
		}
	}
}

func TestArrayCumProd_SingleElement(t *testing.T) {
	data := []float64{42.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	result := arr.CumProd(0)

	// Cumulative product of single element should be the element itself
	expected := 42.0
	actual := result.At(0).(float64)
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("CumProd() of single element = %v, want %v", actual, expected)
	}
}

func TestArrayCumProd_InvalidAxis(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test with invalid axis (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Error("CumProd() with invalid axis should panic")
		}
	}()

	arr.CumProd(5) // Invalid axis for 1D array
}
