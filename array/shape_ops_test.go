package array

import (
	"testing"

	"github.com/julianshen/gonp/internal"
)

func TestSqueeze(t *testing.T) {
	// Test squeezing all dimensions of size 1
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6}, internal.Shape{1, 2, 1, 3, 1})

	squeezed, err := arr.Squeeze()
	if err != nil {
		t.Fatalf("Squeeze() error = %v", err)
	}

	expectedShape := internal.Shape{2, 3}
	if !squeezed.Shape().Equal(expectedShape) {
		t.Errorf("Squeeze() shape = %v, want %v", squeezed.Shape(), expectedShape)
	}

	// Check values are preserved
	if squeezed.At(0, 0) != 1.0 || squeezed.At(1, 2) != 6.0 {
		t.Error("Squeeze() did not preserve values correctly")
	}
}

func TestSqueezeSpecificAxis(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4}, internal.Shape{1, 2, 1, 2})

	// Squeeze only axis 0
	squeezed, err := arr.Squeeze(0)
	if err != nil {
		t.Fatalf("Squeeze(0) error = %v", err)
	}

	expectedShape := internal.Shape{2, 1, 2}
	if !squeezed.Shape().Equal(expectedShape) {
		t.Errorf("Squeeze(0) shape = %v, want %v", squeezed.Shape(), expectedShape)
	}
}

func TestSqueezeInvalidAxis(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4}, internal.Shape{2, 2})

	// Try to squeeze axis with size > 1
	_, err := arr.Squeeze(0)
	if err == nil {
		t.Error("Squeeze() should return error when trying to squeeze axis with size > 1")
	}
}

func TestExpandDims(t *testing.T) {
	arr, _ := FromSlice([]float64{1, 2, 3})

	// Expand at axis 0
	expanded, err := arr.ExpandDims(0)
	if err != nil {
		t.Fatalf("ExpandDims(0) error = %v", err)
	}

	expectedShape := internal.Shape{1, 3}
	if !expanded.Shape().Equal(expectedShape) {
		t.Errorf("ExpandDims(0) shape = %v, want %v", expanded.Shape(), expectedShape)
	}

	// Check values are preserved
	if expanded.At(0, 0) != 1.0 || expanded.At(0, 2) != 3.0 {
		t.Error("ExpandDims() did not preserve values correctly")
	}
}

func TestExpandDimsMultiple(t *testing.T) {
	arr, _ := FromSlice([]float64{1, 2, 3})

	// Expand at multiple axes
	expanded, err := arr.ExpandDims(0, 2)
	if err != nil {
		t.Fatalf("ExpandDims(0, 2) error = %v", err)
	}

	expectedShape := internal.Shape{1, 3, 1}
	if !expanded.Shape().Equal(expectedShape) {
		t.Errorf("ExpandDims(0, 2) shape = %v, want %v", expanded.Shape(), expectedShape)
	}
}

func TestSwapaxes(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6}, internal.Shape{2, 3})

	swapped, err := arr.Swapaxes(0, 1)
	if err != nil {
		t.Fatalf("Swapaxes(0, 1) error = %v", err)
	}

	expectedShape := internal.Shape{3, 2}
	if !swapped.Shape().Equal(expectedShape) {
		t.Errorf("Swapaxes(0, 1) shape = %v, want %v", swapped.Shape(), expectedShape)
	}

	// Check that values are correctly transposed
	// Original: [[1,2,3], [4,5,6]] -> Swapped: [[1,4], [2,5], [3,6]]
	if swapped.At(0, 0) != 1.0 || swapped.At(0, 1) != 4.0 {
		t.Error("Swapaxes() did not transpose values correctly")
	}
	if swapped.At(1, 0) != 2.0 || swapped.At(2, 1) != 6.0 {
		t.Error("Swapaxes() did not transpose values correctly")
	}
}

func TestSwapaxesSameAxis(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4}, internal.Shape{2, 2})

	swapped, err := arr.Swapaxes(0, 0)
	if err != nil {
		t.Fatalf("Swapaxes(0, 0) error = %v", err)
	}

	// Should be a copy with same shape
	if !swapped.Shape().Equal(arr.Shape()) {
		t.Errorf("Swapaxes(0, 0) should preserve shape")
	}

	// Check values are preserved
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if swapped.At(i, j) != arr.At(i, j) {
				t.Error("Swapaxes(0, 0) should preserve values")
			}
		}
	}
}

func TestTransposeWithAxes(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6, 7, 8}, internal.Shape{2, 2, 2})

	// Transpose with custom axes permutation
	transposed, err := arr.TransposeWithAxes(2, 0, 1)
	if err != nil {
		t.Fatalf("TransposeWithAxes(2, 0, 1) error = %v", err)
	}

	expectedShape := internal.Shape{2, 2, 2}
	if !transposed.Shape().Equal(expectedShape) {
		t.Errorf("TransposeWithAxes(2, 0, 1) shape = %v, want %v", transposed.Shape(), expectedShape)
	}
}

func TestTransposeWithAxesDefault(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6}, internal.Shape{2, 3})

	// Default transpose (no axes specified)
	transposed, err := arr.TransposeWithAxes()
	if err != nil {
		t.Fatalf("TransposeWithAxes() error = %v", err)
	}

	expectedShape := internal.Shape{3, 2}
	if !transposed.Shape().Equal(expectedShape) {
		t.Errorf("TransposeWithAxes() shape = %v, want %v", transposed.Shape(), expectedShape)
	}
}

func TestTransposeWithAxesInvalidLength(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4}, internal.Shape{2, 2})

	// Wrong number of axes
	_, err := arr.TransposeWithAxes(0)
	if err == nil {
		t.Error("TransposeWithAxes() should return error for wrong number of axes")
	}
}

func TestTransposeWithAxesRepeatedAxis(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4}, internal.Shape{2, 2})

	// Repeated axis
	_, err := arr.TransposeWithAxes(0, 0)
	if err == nil {
		t.Error("TransposeWithAxes() should return error for repeated axis")
	}
}

func TestMoveaxis(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6, 7, 8}, internal.Shape{2, 2, 2})

	// Move axis 0 to position 2
	moved, err := arr.Moveaxis([]int{0}, []int{2})
	if err != nil {
		t.Fatalf("Moveaxis([0], [2]) error = %v", err)
	}

	expectedShape := internal.Shape{2, 2, 2}
	if !moved.Shape().Equal(expectedShape) {
		t.Errorf("Moveaxis([0], [2]) shape = %v, want %v", moved.Shape(), expectedShape)
	}
}

func TestMoveaxisMultiple(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, internal.Shape{2, 2, 3})

	// Move axes 0,2 to positions 1,0
	moved, err := arr.Moveaxis([]int{0, 2}, []int{1, 0})
	if err != nil {
		t.Fatalf("Moveaxis([0,2], [1,0]) error = %v", err)
	}

	expectedShape := internal.Shape{3, 2, 2}
	if !moved.Shape().Equal(expectedShape) {
		t.Errorf("Moveaxis([0,2], [1,0]) shape = %v, want %v", moved.Shape(), expectedShape)
	}
}

func TestMoveaxisInvalidLength(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4}, internal.Shape{2, 2})

	// Mismatched source and destination lengths
	_, err := arr.Moveaxis([]int{0}, []int{0, 1})
	if err == nil {
		t.Error("Moveaxis() should return error for mismatched source and destination lengths")
	}
}

func TestRollaxis(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6}, internal.Shape{2, 3})

	// Roll axis 1 to position 0
	rolled, err := arr.Rollaxis(1, 0)
	if err != nil {
		t.Fatalf("Rollaxis(1, 0) error = %v", err)
	}

	expectedShape := internal.Shape{3, 2}
	if !rolled.Shape().Equal(expectedShape) {
		t.Errorf("Rollaxis(1, 0) shape = %v, want %v", rolled.Shape(), expectedShape)
	}
}

func TestRollaxisSamePosition(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4}, internal.Shape{2, 2})

	// Roll axis to same position
	rolled, err := arr.Rollaxis(0, 0)
	if err != nil {
		t.Fatalf("Rollaxis(0, 0) error = %v", err)
	}

	// Should be a copy with same shape
	if !rolled.Shape().Equal(arr.Shape()) {
		t.Errorf("Rollaxis(0, 0) should preserve shape")
	}
}

func TestRollaxisInvalidAxis(t *testing.T) {
	arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4}, internal.Shape{2, 2})

	// Invalid axis
	_, err := arr.Rollaxis(5, 0)
	if err == nil {
		t.Error("Rollaxis() should return error for invalid axis")
	}
}
