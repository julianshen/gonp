package array

import (
	"testing"

	"github.com/julianshen/gonp/internal"
)

func TestInPlaceAdd(t *testing.T) {
	a, _ := FromSlice([]float64{1, 2, 3})
	b, _ := FromSlice([]float64{4, 5, 6})

	// Make a copy to verify in-place modification
	original, _ := FromSlice([]float64{1, 2, 3})

	err := a.AddInPlace(b)
	if err != nil {
		t.Fatalf("AddInPlace() error = %v", err)
	}

	// Check that a was modified
	expected := []float64{5, 7, 9}
	for i, exp := range expected {
		val := a.At(i)
		if val != exp {
			t.Errorf("AddInPlace result[%d] = %v, want %v", i, val, exp)
		}
	}

	// Check that b was not modified
	for i := 0; i < 3; i++ {
		originalVal := original.At(i)
		bVal := b.At(i)
		if bVal != float64(4+i) {
			t.Errorf("b was modified: b[%d] = %v, want %v", i, bVal, float64(4+i))
		}
		_ = originalVal // Use originalVal to avoid unused variable warning
	}
}

func TestInPlaceAddBroadcasting(t *testing.T) {
	// Test (2,3) += (3,) broadcasting
	a, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6}, internal.Shape{2, 3})
	b, _ := FromSlice([]float64{10, 20, 30})

	err := a.AddInPlace(b)
	if err != nil {
		t.Fatalf("AddInPlace() with broadcasting error = %v", err)
	}

	// Check values
	expected := [][]float64{{11, 22, 33}, {14, 25, 36}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			val := a.At(i, j)
			if val != expected[i][j] {
				t.Errorf("AddInPlace result[%d,%d] = %v, want %v", i, j, val, expected[i][j])
			}
		}
	}
}

func TestInPlaceSubtraction(t *testing.T) {
	a, _ := FromSlice([]float64{10, 8, 6})
	b, _ := FromSlice([]float64{3, 2, 1})

	err := a.SubInPlace(b)
	if err != nil {
		t.Fatalf("SubInPlace() error = %v", err)
	}

	expected := []float64{7, 6, 5}
	for i, exp := range expected {
		val := a.At(i)
		if val != exp {
			t.Errorf("SubInPlace result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestInPlaceMultiplication(t *testing.T) {
	a, _ := FromSlice([]float64{2, 3, 4})
	b, _ := FromSlice([]float64{5, 6, 7})

	err := a.MulInPlace(b)
	if err != nil {
		t.Fatalf("MulInPlace() error = %v", err)
	}

	expected := []float64{10, 18, 28}
	for i, exp := range expected {
		val := a.At(i)
		if val != exp {
			t.Errorf("MulInPlace result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestInPlaceDivision(t *testing.T) {
	a, _ := FromSlice([]float64{12, 15, 20})
	b, _ := FromSlice([]float64{3, 5, 4})

	err := a.DivInPlace(b)
	if err != nil {
		t.Fatalf("DivInPlace() error = %v", err)
	}

	expected := []float64{4, 3, 5}
	for i, exp := range expected {
		val := a.At(i)
		if val != exp {
			t.Errorf("DivInPlace result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestInPlaceScalarOperations(t *testing.T) {
	a, _ := FromSlice([]float64{1, 2, 3})

	// Test scalar addition in-place
	err := a.AddScalarInPlace(5.0)
	if err != nil {
		t.Fatalf("AddScalarInPlace() error = %v", err)
	}

	expected := []float64{6, 7, 8}
	for i, exp := range expected {
		val := a.At(i)
		if val != exp {
			t.Errorf("AddScalarInPlace result[%d] = %v, want %v", i, val, exp)
		}
	}

	// Test scalar multiplication in-place
	err = a.MulScalarInPlace(2.0)
	if err != nil {
		t.Fatalf("MulScalarInPlace() error = %v", err)
	}

	expected = []float64{12, 14, 16}
	for i, exp := range expected {
		val := a.At(i)
		if val != exp {
			t.Errorf("MulScalarInPlace result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestInPlaceIncompatibleShapes(t *testing.T) {
	a := Ones(internal.Shape{3, 2}, internal.Float64)
	b := Ones(internal.Shape{3, 4}, internal.Float64)

	err := a.AddInPlace(b)
	if err == nil {
		t.Error("AddInPlace() should return error for incompatible shapes")
	}
}

func TestInPlaceMixedDataTypes(t *testing.T) {
	a, _ := FromSlice([]float64{1.5, 2.5, 3.5})
	b, _ := FromSlice([]int64{1, 2, 3})

	err := a.AddInPlace(b)
	if err != nil {
		t.Fatalf("Mixed type AddInPlace() error = %v", err)
	}

	expected := []float64{2.5, 4.5, 6.5}
	for i, exp := range expected {
		val := a.At(i)
		if val != exp {
			t.Errorf("Mixed type AddInPlace result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestInPlaceChaining(t *testing.T) {
	a, _ := FromSlice([]float64{1, 2, 3})
	b, _ := FromSlice([]float64{2, 3, 4})
	c, _ := FromSlice([]float64{1, 1, 1})

	// Chain operations: a += b; a *= c
	err := a.AddInPlace(b)
	if err != nil {
		t.Fatalf("First AddInPlace() error = %v", err)
	}

	err = a.MulInPlace(c)
	if err != nil {
		t.Fatalf("MulInPlace() error = %v", err)
	}

	// Should be (1+2)*1, (2+3)*1, (3+4)*1 = 3, 5, 7
	expected := []float64{3, 5, 7}
	for i, exp := range expected {
		val := a.At(i)
		if val != exp {
			t.Errorf("Chained operations result[%d] = %v, want %v", i, val, exp)
		}
	}
}
