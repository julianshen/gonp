package array

import (
	"testing"
)

func TestArrayBooleanIndexing(t *testing.T) {
	t.Run("Basic Boolean Indexing", func(t *testing.T) {
		// Create array [1, 2, 3, 4, 5]
		data := []float64{1, 2, 3, 4, 5}
		arr, err := NewArray(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		// Create boolean mask [true, false, true, false, true]
		mask := []bool{true, false, true, false, true}
		maskArr, err := NewArray(mask)
		if err != nil {
			t.Fatalf("Failed to create mask array: %v", err)
		}

		// Apply boolean indexing
		result, err := arr.BooleanIndex(maskArr)
		if err != nil {
			t.Fatalf("BooleanIndex failed: %v", err)
		}

		// Should return [1, 3, 5]
		if result.Size() != 3 {
			t.Errorf("Expected length 3, got %d", result.Size())
		}

		expected := []float64{1, 3, 5}
		for i, exp := range expected {
			val := result.At(i).(float64)
			if val != exp {
				t.Errorf("Expected %v at index %d, got %v", exp, i, val)
			}
		}
	})

	t.Run("Boolean Indexing with Different Types", func(t *testing.T) {
		// Test with integer array
		data := []int64{10, 20, 30, 40}
		arr, _ := NewArray(data)

		mask := []bool{false, true, true, false}
		maskArr, _ := NewArray(mask)

		result, err := arr.BooleanIndex(maskArr)
		if err != nil {
			t.Fatalf("BooleanIndex failed: %v", err)
		}

		expected := []int64{20, 30}
		if result.Size() != len(expected) {
			t.Errorf("Expected length %d, got %d", len(expected), result.Size())
		}

		for i, exp := range expected {
			val := result.At(i).(int64)
			if val != exp {
				t.Errorf("Expected %v at index %d, got %v", exp, i, val)
			}
		}
	})

	t.Run("Boolean Indexing All False", func(t *testing.T) {
		data := []float64{1, 2, 3}
		arr, _ := NewArray(data)

		mask := []bool{false, false, false}
		maskArr, _ := NewArray(mask)

		result, err := arr.BooleanIndex(maskArr)
		if err != nil {
			t.Fatalf("BooleanIndex failed: %v", err)
		}

		if result.Size() != 0 {
			t.Errorf("Expected empty result, got length %d", result.Size())
		}
	})

	t.Run("Boolean Indexing All True", func(t *testing.T) {
		data := []float64{1, 2, 3}
		arr, _ := NewArray(data)

		mask := []bool{true, true, true}
		maskArr, _ := NewArray(mask)

		result, err := arr.BooleanIndex(maskArr)
		if err != nil {
			t.Fatalf("BooleanIndex failed: %v", err)
		}

		if result.Size() != 3 {
			t.Errorf("Expected length 3, got %d", result.Size())
		}

		for i := 0; i < 3; i++ {
			if result.At(i).(float64) != data[i] {
				t.Errorf("Expected %v at index %d, got %v", data[i], i, result.At(i))
			}
		}
	})
}

func TestArrayFancyIndexing(t *testing.T) {
	t.Run("Basic Fancy Indexing", func(t *testing.T) {
		// Create array [10, 20, 30, 40, 50]
		data := []float64{10, 20, 30, 40, 50}
		arr, err := NewArray(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		// Create index array [4, 1, 3, 0]
		indices := []int64{4, 1, 3, 0}
		idxArr, err := NewArray(indices)
		if err != nil {
			t.Fatalf("Failed to create index array: %v", err)
		}

		// Apply fancy indexing
		result, err := arr.FancyIndex(idxArr)
		if err != nil {
			t.Fatalf("FancyIndex failed: %v", err)
		}

		// Should return [50, 20, 40, 10]
		expected := []float64{50, 20, 40, 10}
		if result.Size() != len(expected) {
			t.Errorf("Expected length %d, got %d", len(expected), result.Size())
		}

		for i, exp := range expected {
			val := result.At(i).(float64)
			if val != exp {
				t.Errorf("Expected %v at index %d, got %v", exp, i, val)
			}
		}
	})

	t.Run("Fancy Indexing with Duplicates", func(t *testing.T) {
		data := []int64{100, 200, 300}
		arr, _ := NewArray(data)

		// Index array with duplicates [0, 2, 0, 1, 2]
		indices := []int64{0, 2, 0, 1, 2}
		idxArr, _ := NewArray(indices)

		result, err := arr.FancyIndex(idxArr)
		if err != nil {
			t.Fatalf("FancyIndex failed: %v", err)
		}

		expected := []int64{100, 300, 100, 200, 300}
		if result.Size() != len(expected) {
			t.Errorf("Expected length %d, got %d", len(expected), result.Size())
		}

		for i, exp := range expected {
			val := result.At(i).(int64)
			if val != exp {
				t.Errorf("Expected %v at index %d, got %v", exp, i, val)
			}
		}
	})

	t.Run("Fancy Indexing Single Element", func(t *testing.T) {
		data := []float64{1.1, 2.2, 3.3, 4.4}
		arr, _ := NewArray(data)

		indices := []int64{2}
		idxArr, _ := NewArray(indices)

		result, err := arr.FancyIndex(idxArr)
		if err != nil {
			t.Fatalf("FancyIndex failed: %v", err)
		}

		if result.Size() != 1 {
			t.Errorf("Expected length 1, got %d", result.Size())
		}

		if result.At(0).(float64) != 3.3 {
			t.Errorf("Expected 3.3, got %v", result.At(0))
		}
	})
}

func TestArrayWhereCondition(t *testing.T) {
	t.Run("Where with Condition", func(t *testing.T) {
		// Create array [1, 2, 3, 4, 5, 6]
		data := []float64{1, 2, 3, 4, 5, 6}
		arr, _ := NewArray(data)

		// Where condition: values > 3
		condition := func(val interface{}) bool {
			return val.(float64) > 3.0
		}

		result, err := arr.Where(condition)
		if err != nil {
			t.Fatalf("Where failed: %v", err)
		}

		// Should return [4, 5, 6]
		expected := []float64{4, 5, 6}
		if result.Size() != len(expected) {
			t.Errorf("Expected length %d, got %d", len(expected), result.Size())
		}

		for i, exp := range expected {
			val := result.At(i).(float64)
			if val != exp {
				t.Errorf("Expected %v at index %d, got %v", exp, i, val)
			}
		}
	})

	t.Run("Where with No Matches", func(t *testing.T) {
		data := []int64{1, 2, 3}
		arr, _ := NewArray(data)

		condition := func(val interface{}) bool {
			return val.(int64) > 10
		}

		result, err := arr.Where(condition)
		if err != nil {
			t.Fatalf("Where failed: %v", err)
		}

		if result.Size() != 0 {
			t.Errorf("Expected empty result, got length %d", result.Size())
		}
	})

	t.Run("Where with All Matches", func(t *testing.T) {
		data := []float64{5.0, 6.0, 7.0}
		arr, _ := NewArray(data)

		condition := func(val interface{}) bool {
			return val.(float64) >= 5.0
		}

		result, err := arr.Where(condition)
		if err != nil {
			t.Fatalf("Where failed: %v", err)
		}

		if result.Size() != 3 {
			t.Errorf("Expected length 3, got %d", result.Size())
		}

		for i := 0; i < 3; i++ {
			if result.At(i).(float64) != data[i] {
				t.Errorf("Expected %v at index %d, got %v", data[i], i, result.At(i))
			}
		}
	})
}

func TestArrayIndexingErrors(t *testing.T) {
	t.Run("Boolean Index Size Mismatch", func(t *testing.T) {
		data := []float64{1, 2, 3}
		arr, _ := NewArray(data)

		// Wrong size mask
		mask := []bool{true, false}
		maskArr, _ := NewArray(mask)

		_, err := arr.BooleanIndex(maskArr)
		if err == nil {
			t.Error("Expected error for size mismatch")
		}
	})

	t.Run("Boolean Index Non-Boolean Mask", func(t *testing.T) {
		data := []float64{1, 2, 3}
		arr, _ := NewArray(data)

		// Non-boolean mask
		mask := []int64{1, 0, 1}
		maskArr, _ := NewArray(mask)

		_, err := arr.BooleanIndex(maskArr)
		if err == nil {
			t.Error("Expected error for non-boolean mask")
		}
	})

	t.Run("Fancy Index Out of Bounds", func(t *testing.T) {
		data := []float64{1, 2, 3}
		arr, _ := NewArray(data)

		// Out of bounds index
		indices := []int64{0, 1, 5}
		idxArr, _ := NewArray(indices)

		_, err := arr.FancyIndex(idxArr)
		if err == nil {
			t.Error("Expected error for out of bounds index")
		}
	})

	t.Run("Fancy Index Negative Index", func(t *testing.T) {
		data := []float64{1, 2, 3}
		arr, _ := NewArray(data)

		// Negative index
		indices := []int64{0, -1, 2}
		idxArr, _ := NewArray(indices)

		_, err := arr.FancyIndex(idxArr)
		if err == nil {
			t.Error("Expected error for negative index")
		}
	})

	t.Run("Fancy Index Non-Integer", func(t *testing.T) {
		data := []float64{1, 2, 3}
		arr, _ := NewArray(data)

		// Non-integer indices
		indices := []float64{0.0, 1.5, 2.0}
		idxArr, _ := NewArray(indices)

		_, err := arr.FancyIndex(idxArr)
		if err == nil {
			t.Error("Expected error for non-integer indices")
		}
	})
}

func TestArrayTakeAndPut(t *testing.T) {
	t.Run("Take with Indices", func(t *testing.T) {
		data := []float64{10, 20, 30, 40, 50}
		arr, _ := NewArray(data)

		indices := []int{4, 0, 2}
		result, err := arr.Take(indices)
		if err != nil {
			t.Fatalf("Take failed: %v", err)
		}

		expected := []float64{50, 10, 30}
		if result.Size() != len(expected) {
			t.Errorf("Expected length %d, got %d", len(expected), result.Size())
		}

		for i, exp := range expected {
			val := result.At(i).(float64)
			if val != exp {
				t.Errorf("Expected %v at index %d, got %v", exp, i, val)
			}
		}
	})

	t.Run("Put with Values", func(t *testing.T) {
		data := []float64{1, 2, 3, 4, 5}
		arr, _ := NewArray(data)

		indices := []int{0, 2, 4}
		values := []float64{100, 300, 500}

		err := arr.Put(indices, values)
		if err != nil {
			t.Fatalf("Put failed: %v", err)
		}

		// Check updated values
		if arr.At(0).(float64) != 100 {
			t.Errorf("Expected 100 at index 0, got %v", arr.At(0))
		}
		if arr.At(2).(float64) != 300 {
			t.Errorf("Expected 300 at index 2, got %v", arr.At(2))
		}
		if arr.At(4).(float64) != 500 {
			t.Errorf("Expected 500 at index 4, got %v", arr.At(4))
		}

		// Check unchanged values
		if arr.At(1).(float64) != 2 {
			t.Errorf("Expected 2 at index 1, got %v", arr.At(1))
		}
		if arr.At(3).(float64) != 4 {
			t.Errorf("Expected 4 at index 3, got %v", arr.At(3))
		}
	})
}
