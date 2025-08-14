package array

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/internal"
)

func TestElementwiseAdd(t *testing.T) {
	// Test same shape arrays
	a, _ := FromSlice([]float64{1, 2, 3})
	b, _ := FromSlice([]float64{4, 5, 6})

	result, err := a.Add(b)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}

	expected := []float64{5, 7, 9}
	for i, exp := range expected {
		val := result.At(i)
		if val != exp {
			t.Errorf("Add result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestElementwiseAddBroadcasting(t *testing.T) {
	// Test broadcasting: (2,3) + (3,) -> (2,3)
	a, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6}, internal.Shape{2, 3})
	b, _ := FromSlice([]float64{10, 20, 30})

	result, err := a.Add(b)
	if err != nil {
		t.Fatalf("Add() with broadcasting error = %v", err)
	}

	expectedShape := internal.Shape{2, 3}
	if !result.Shape().Equal(expectedShape) {
		t.Errorf("Add result shape = %v, want %v", result.Shape(), expectedShape)
	}

	// Check specific values
	expected := [][]float64{{11, 22, 33}, {14, 25, 36}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			val := result.At(i, j)
			if val != expected[i][j] {
				t.Errorf("Add result[%d,%d] = %v, want %v", i, j, val, expected[i][j])
			}
		}
	}
}

func TestElementwiseSubtraction(t *testing.T) {
	a, _ := FromSlice([]float64{10, 8, 6})
	b, _ := FromSlice([]float64{3, 2, 1})

	result, err := a.Sub(b)
	if err != nil {
		t.Fatalf("Sub() error = %v", err)
	}

	expected := []float64{7, 6, 5}
	for i, exp := range expected {
		val := result.At(i)
		if val != exp {
			t.Errorf("Sub result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestElementwiseMultiplication(t *testing.T) {
	a, _ := FromSlice([]float64{2, 3, 4})
	b, _ := FromSlice([]float64{5, 6, 7})

	result, err := a.Mul(b)
	if err != nil {
		t.Fatalf("Mul() error = %v", err)
	}

	expected := []float64{10, 18, 28}
	for i, exp := range expected {
		val := result.At(i)
		if val != exp {
			t.Errorf("Mul result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestElementwiseDivision(t *testing.T) {
	a, _ := FromSlice([]float64{12, 15, 20})
	b, _ := FromSlice([]float64{3, 5, 4})

	result, err := a.Div(b)
	if err != nil {
		t.Fatalf("Div() error = %v", err)
	}

	expected := []float64{4, 3, 5}
	for i, exp := range expected {
		val := result.At(i)
		if val != exp {
			t.Errorf("Div result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestElementwiseDivisionByZero(t *testing.T) {
	a, _ := FromSlice([]float64{1, 2, 3})
	b, _ := FromSlice([]float64{1, 0, 3})

	result, err := a.Div(b)
	if err != nil {
		t.Fatalf("Div() error = %v", err)
	}

	// Check that division by zero produces infinity
	val := result.At(1)
	if !math.IsInf(val.(float64), 1) {
		t.Errorf("Div by zero result = %v, want +Inf", val)
	}
}

func TestElementwisePower(t *testing.T) {
	a, _ := FromSlice([]float64{2, 3, 4})
	b, _ := FromSlice([]float64{3, 2, 2})

	result, err := a.Pow(b)
	if err != nil {
		t.Fatalf("Pow() error = %v", err)
	}

	expected := []float64{8, 9, 16}
	for i, exp := range expected {
		val := result.At(i)
		if val != exp {
			t.Errorf("Pow result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestElementwiseModulo(t *testing.T) {
	a, _ := FromSlice([]int64{10, 15, 7})
	b, _ := FromSlice([]int64{3, 4, 2})

	result, err := a.Mod(b)
	if err != nil {
		t.Fatalf("Mod() error = %v", err)
	}

	expected := []int64{1, 3, 1}
	for i, exp := range expected {
		val := result.At(i)
		if val != exp {
			t.Errorf("Mod result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestScalarOperations(t *testing.T) {
	a, _ := FromSlice([]float64{1, 2, 3})

	// Test scalar addition
	result, err := a.AddScalar(5.0)
	if err != nil {
		t.Fatalf("AddScalar() error = %v", err)
	}

	expected := []float64{6, 7, 8}
	for i, exp := range expected {
		val := result.At(i)
		if val != exp {
			t.Errorf("AddScalar result[%d] = %v, want %v", i, val, exp)
		}
	}

	// Test scalar multiplication
	result, err = a.MulScalar(3.0)
	if err != nil {
		t.Fatalf("MulScalar() error = %v", err)
	}

	expected = []float64{3, 6, 9}
	for i, exp := range expected {
		val := result.At(i)
		if val != exp {
			t.Errorf("MulScalar result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestComplexArithmetic(t *testing.T) {
	a, _ := FromSlice([]complex128{1 + 2i, 3 + 4i})
	b, _ := FromSlice([]complex128{2 + 1i, 1 + 1i})

	// Test complex addition
	result, err := a.Add(b)
	if err != nil {
		t.Fatalf("Complex Add() error = %v", err)
	}

	expected := []complex128{3 + 3i, 4 + 5i}
	for i, exp := range expected {
		val := result.At(i)
		if val != exp {
			t.Errorf("Complex Add result[%d] = %v, want %v", i, val, exp)
		}
	}

	// Test complex multiplication
	result, err = a.Mul(b)
	if err != nil {
		t.Fatalf("Complex Mul() error = %v", err)
	}

	// (1+2i)*(2+1i) = 2+i+4i+2i² = 2+5i-2 = 0+5i
	// (3+4i)*(1+1i) = 3+3i+4i+4i² = 3+7i-4 = -1+7i
	expected = []complex128{0 + 5i, -1 + 7i}
	for i, exp := range expected {
		val := result.At(i)
		if val != exp {
			t.Errorf("Complex Mul result[%d] = %v, want %v", i, val, exp)
		}
	}
}

func TestIncompatibleShapes(t *testing.T) {
	a := Ones(internal.Shape{3, 2}, internal.Float64)
	b := Ones(internal.Shape{3, 4}, internal.Float64)

	_, err := a.Add(b)
	if err == nil {
		t.Error("Add() should return error for incompatible shapes")
	}
}

func TestMixedDataTypes(t *testing.T) {
	a, _ := FromSlice([]float64{1.5, 2.5, 3.5})
	b, _ := FromSlice([]int64{1, 2, 3})

	result, err := a.Add(b)
	if err != nil {
		t.Fatalf("Mixed type Add() error = %v", err)
	}

	expected := []float64{2.5, 4.5, 6.5}
	for i, exp := range expected {
		val := result.At(i)
		if val != exp {
			t.Errorf("Mixed type Add result[%d] = %v, want %v", i, val, exp)
		}
	}
}
