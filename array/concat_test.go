package array

import (
	"testing"

	"github.com/julianshen/gonp/internal"
)

func TestArrayConcatenate(t *testing.T) {
	t.Run("Concatenate 1D Arrays", func(t *testing.T) {
		// Create arrays [1, 2] and [3, 4]
		arr1, _ := NewArray([]float64{1, 2})
		arr2, _ := NewArray([]float64{3, 4})

		result, err := Concatenate([]*Array{arr1, arr2}, 0)
		if err != nil {
			t.Fatalf("Concatenate failed: %v", err)
		}

		// Should return [1, 2, 3, 4]
		if result.Size() != 4 {
			t.Errorf("Expected size 4, got %d", result.Size())
		}

		expected := []float64{1, 2, 3, 4}
		for i, exp := range expected {
			val := result.At(i).(float64)
			if val != exp {
				t.Errorf("Expected %v at index %d, got %v", exp, i, val)
			}
		}
	})

	t.Run("Concatenate Multiple 1D Arrays", func(t *testing.T) {
		arr1, _ := NewArray([]int64{1, 2})
		arr2, _ := NewArray([]int64{3})
		arr3, _ := NewArray([]int64{4, 5, 6})

		result, err := Concatenate([]*Array{arr1, arr2, arr3}, 0)
		if err != nil {
			t.Fatalf("Concatenate failed: %v", err)
		}

		// Should return [1, 2, 3, 4, 5, 6]
		if result.Size() != 6 {
			t.Errorf("Expected size 6, got %d", result.Size())
		}

		expected := []int64{1, 2, 3, 4, 5, 6}
		for i, exp := range expected {
			val := result.At(i).(int64)
			if val != exp {
				t.Errorf("Expected %v at index %d, got %v", exp, i, val)
			}
		}
	})

	t.Run("Concatenate 2D Arrays Along Axis 0", func(t *testing.T) {
		// Create 2x2 arrays
		arr1, _ := NewArrayWithShape([]float64{1, 2, 3, 4}, internal.Shape{2, 2})
		arr2, _ := NewArrayWithShape([]float64{5, 6, 7, 8}, internal.Shape{2, 2})

		result, err := Concatenate([]*Array{arr1, arr2}, 0)
		if err != nil {
			t.Fatalf("Concatenate failed: %v", err)
		}

		// Should be 4x2 array
		expectedShape := internal.Shape{4, 2}
		if !result.Shape().Equal(expectedShape) {
			t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Check values
		expected := [][]float64{
			{1, 2},
			{3, 4},
			{5, 6},
			{7, 8},
		}
		for i := 0; i < 4; i++ {
			for j := 0; j < 2; j++ {
				val := result.At(i, j).(float64)
				if val != expected[i][j] {
					t.Errorf("Expected %v at (%d,%d), got %v", expected[i][j], i, j, val)
				}
			}
		}
	})

	t.Run("Concatenate 2D Arrays Along Axis 1", func(t *testing.T) {
		// Create 2x2 arrays
		arr1, _ := NewArrayWithShape([]float64{1, 2, 3, 4}, internal.Shape{2, 2})
		arr2, _ := NewArrayWithShape([]float64{5, 6, 7, 8}, internal.Shape{2, 2})

		result, err := Concatenate([]*Array{arr1, arr2}, 1)
		if err != nil {
			t.Fatalf("Concatenate failed: %v", err)
		}

		// Should be 2x4 array
		expectedShape := internal.Shape{2, 4}
		if !result.Shape().Equal(expectedShape) {
			t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Check values
		expected := [][]float64{
			{1, 2, 5, 6},
			{3, 4, 7, 8},
		}
		for i := 0; i < 2; i++ {
			for j := 0; j < 4; j++ {
				val := result.At(i, j).(float64)
				if val != expected[i][j] {
					t.Errorf("Expected %v at (%d,%d), got %v", expected[i][j], i, j, val)
				}
			}
		}
	})
}

func TestArrayStack(t *testing.T) {
	t.Run("VStack 1D Arrays", func(t *testing.T) {
		arr1, _ := NewArray([]float64{1, 2, 3})
		arr2, _ := NewArray([]float64{4, 5, 6})

		result, err := VStack([]*Array{arr1, arr2})
		if err != nil {
			t.Fatalf("VStack failed: %v", err)
		}

		// Should be 2x3 array
		expectedShape := internal.Shape{2, 3}
		if !result.Shape().Equal(expectedShape) {
			t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape())
		}

		expected := [][]float64{
			{1, 2, 3},
			{4, 5, 6},
		}
		for i := 0; i < 2; i++ {
			for j := 0; j < 3; j++ {
				val := result.At(i, j).(float64)
				if val != expected[i][j] {
					t.Errorf("Expected %v at (%d,%d), got %v", expected[i][j], i, j, val)
				}
			}
		}
	})

	t.Run("HStack 1D Arrays", func(t *testing.T) {
		arr1, _ := NewArray([]float64{1, 2})
		arr2, _ := NewArray([]float64{3, 4})

		result, err := HStack([]*Array{arr1, arr2})
		if err != nil {
			t.Fatalf("HStack failed: %v", err)
		}

		// Should be 1D array [1, 2, 3, 4]
		expectedShape := internal.Shape{4}
		if !result.Shape().Equal(expectedShape) {
			t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape())
		}

		expected := []float64{1, 2, 3, 4}
		for i, exp := range expected {
			val := result.At(i).(float64)
			if val != exp {
				t.Errorf("Expected %v at index %d, got %v", exp, i, val)
			}
		}
	})

	t.Run("DStack 2D Arrays", func(t *testing.T) {
		// Create 2x2 arrays
		arr1, _ := NewArrayWithShape([]float64{1, 2, 3, 4}, internal.Shape{2, 2})
		arr2, _ := NewArrayWithShape([]float64{5, 6, 7, 8}, internal.Shape{2, 2})

		result, err := DStack([]*Array{arr1, arr2})
		if err != nil {
			t.Fatalf("DStack failed: %v", err)
		}

		// Should be 2x2x2 array
		expectedShape := internal.Shape{2, 2, 2}
		if !result.Shape().Equal(expectedShape) {
			t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Check a few values
		if result.At(0, 0, 0).(float64) != 1 {
			t.Errorf("Expected 1 at (0,0,0), got %v", result.At(0, 0, 0))
		}
		if result.At(0, 0, 1).(float64) != 5 {
			t.Errorf("Expected 5 at (0,0,1), got %v", result.At(0, 0, 1))
		}
	})

	t.Run("Stack Multiple Arrays", func(t *testing.T) {
		arr1, _ := NewArray([]int64{1})
		arr2, _ := NewArray([]int64{2})
		arr3, _ := NewArray([]int64{3})

		result, err := VStack([]*Array{arr1, arr2, arr3})
		if err != nil {
			t.Fatalf("VStack failed: %v", err)
		}

		// Should be 3x1 array
		expectedShape := internal.Shape{3, 1}
		if !result.Shape().Equal(expectedShape) {
			t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape())
		}

		for i := 0; i < 3; i++ {
			val := result.At(i, 0).(int64)
			if val != int64(i+1) {
				t.Errorf("Expected %d at (%d,0), got %v", i+1, i, val)
			}
		}
	})
}

func TestArraySplit(t *testing.T) {
	t.Run("Split 1D Array", func(t *testing.T) {
		arr, _ := NewArray([]float64{1, 2, 3, 4, 5, 6})

		// Split into 3 parts
		result, err := Split(arr, 3, 0)
		if err != nil {
			t.Fatalf("Split failed: %v", err)
		}

		if len(result) != 3 {
			t.Errorf("Expected 3 arrays, got %d", len(result))
		}

		// Each should have 2 elements
		for i, subArr := range result {
			if subArr.Size() != 2 {
				t.Errorf("Sub-array %d should have size 2, got %d", i, subArr.Size())
			}
		}

		// Check values
		expected := [][]float64{
			{1, 2},
			{3, 4},
			{5, 6},
		}
		for i, subArr := range result {
			for j := 0; j < 2; j++ {
				val := subArr.At(j).(float64)
				if val != expected[i][j] {
					t.Errorf("Expected %v at sub-array %d, index %d, got %v", expected[i][j], i, j, val)
				}
			}
		}
	})

	t.Run("Split 2D Array Along Axis 0", func(t *testing.T) {
		arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6, 7, 8}, internal.Shape{4, 2})

		result, err := Split(arr, 2, 0)
		if err != nil {
			t.Fatalf("Split failed: %v", err)
		}

		if len(result) != 2 {
			t.Errorf("Expected 2 arrays, got %d", len(result))
		}

		// Each should be 2x2
		expectedShape := internal.Shape{2, 2}
		for i, subArr := range result {
			if !subArr.Shape().Equal(expectedShape) {
				t.Errorf("Sub-array %d should have shape %v, got %v", i, expectedShape, subArr.Shape())
			}
		}
	})

	t.Run("HSplit 2D Array", func(t *testing.T) {
		arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6, 7, 8}, internal.Shape{2, 4})

		result, err := HSplit(arr, 2)
		if err != nil {
			t.Fatalf("HSplit failed: %v", err)
		}

		if len(result) != 2 {
			t.Errorf("Expected 2 arrays, got %d", len(result))
		}

		// Each should be 2x2
		expectedShape := internal.Shape{2, 2}
		for i, subArr := range result {
			if !subArr.Shape().Equal(expectedShape) {
				t.Errorf("Sub-array %d should have shape %v, got %v", i, expectedShape, subArr.Shape())
			}
		}
	})

	t.Run("VSplit 2D Array", func(t *testing.T) {
		arr, _ := NewArrayWithShape([]float64{1, 2, 3, 4, 5, 6, 7, 8}, internal.Shape{4, 2})

		result, err := VSplit(arr, 2)
		if err != nil {
			t.Fatalf("VSplit failed: %v", err)
		}

		if len(result) != 2 {
			t.Errorf("Expected 2 arrays, got %d", len(result))
		}

		// Each should be 2x2
		expectedShape := internal.Shape{2, 2}
		for i, subArr := range result {
			if !subArr.Shape().Equal(expectedShape) {
				t.Errorf("Sub-array %d should have shape %v, got %v", i, expectedShape, subArr.Shape())
			}
		}
	})
}

func TestArrayConcatErrors(t *testing.T) {
	t.Run("Empty Array List", func(t *testing.T) {
		_, err := Concatenate([]*Array{}, 0)
		if err == nil {
			t.Error("Expected error for empty array list")
		}
	})

	t.Run("Type Mismatch", func(t *testing.T) {
		arr1, _ := NewArray([]float64{1, 2})
		arr2, _ := NewArray([]int64{3, 4})

		_, err := Concatenate([]*Array{arr1, arr2}, 0)
		if err == nil {
			t.Error("Expected error for type mismatch")
		}
	})

	t.Run("Shape Mismatch", func(t *testing.T) {
		arr1, _ := NewArrayWithShape([]float64{1, 2}, internal.Shape{2, 1})
		arr2, _ := NewArrayWithShape([]float64{3, 4, 5, 6}, internal.Shape{2, 2})

		_, err := Concatenate([]*Array{arr1, arr2}, 0)
		if err == nil {
			t.Error("Expected error for incompatible shapes")
		}
	})

	t.Run("Invalid Axis", func(t *testing.T) {
		arr, _ := NewArray([]float64{1, 2, 3})

		_, err := Concatenate([]*Array{arr}, 5)
		if err == nil {
			t.Error("Expected error for invalid axis")
		}
	})

	t.Run("Split Non-Divisible", func(t *testing.T) {
		arr, _ := NewArray([]float64{1, 2, 3, 4, 5})

		_, err := Split(arr, 3, 0) // 5 is not divisible by 3
		if err == nil {
			t.Error("Expected error for non-divisible split")
		}
	})
}
