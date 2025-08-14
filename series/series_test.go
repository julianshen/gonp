package series

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Test helper functions
func createTestSeries(data []float64, index Index) *Series {
	arr, err := array.FromSlice(data)
	if err != nil {
		panic(fmt.Sprintf("Failed to create array: %v", err))
	}
	s, err := NewSeries(arr, index, "test_series")
	if err != nil {
		panic(fmt.Sprintf("Failed to create series: %v", err))
	}
	return s
}

func createTestSeriesInt(data []int64, index Index) *Series {
	arr, err := array.FromSlice(data)
	if err != nil {
		panic(fmt.Sprintf("Failed to create array: %v", err))
	}
	s, err := NewSeries(arr, index, "test_series")
	if err != nil {
		panic(fmt.Sprintf("Failed to create series: %v", err))
	}
	return s
}

func createTestSeriesString(data []string, index Index) *Series {
	// Convert string slice to interface{} slice since array doesn't support strings directly
	values := make([]interface{}, len(data))
	for i, v := range data {
		values[i] = v
	}
	s, err := FromValues(values, index, "test_series")
	if err != nil {
		panic(fmt.Sprintf("Failed to create series: %v", err))
	}
	return s
}

func assertFloatEqual(t *testing.T, actual, expected float64, tolerance float64) {
	if math.IsNaN(expected) && math.IsNaN(actual) {
		return // Both NaN is considered equal
	}
	if math.Abs(actual-expected) > tolerance {
		t.Errorf("Expected %f, got %f", expected, actual)
	}
}

func assertSeriesEqual(t *testing.T, actual, expected *Series, tolerance float64) {
	if actual.Len() != expected.Len() {
		t.Errorf("Length mismatch: expected %d, got %d", expected.Len(), actual.Len())
		return
	}

	for i := 0; i < actual.Len(); i++ {
		actualVal := actual.At(i)
		expectedVal := expected.At(i)

		if actualFloat, ok := actualVal.(float64); ok {
			if expectedFloat, ok := expectedVal.(float64); ok {
				assertFloatEqual(t, actualFloat, expectedFloat, tolerance)
				continue
			}
		}

		if actualVal != expectedVal {
			t.Errorf("Value mismatch at index %d: expected %v, got %v", i, expectedVal, actualVal)
		}
	}
}

// Test Index implementations
func TestRangeIndex(t *testing.T) {
	t.Run("BasicRange", func(t *testing.T) {
		idx := NewRangeIndex(0, 5, 1)

		if idx.Len() != 5 {
			t.Errorf("Expected length 5, got %d", idx.Len())
		}

		for i := 0; i < 5; i++ {
			val := idx.Get(i)
			if val != i {
				t.Errorf("Expected %d at position %d, got %v", i, i, val)
			}
		}
	})

	t.Run("StepRange", func(t *testing.T) {
		idx := NewRangeIndex(0, 10, 2)

		if idx.Len() != 5 {
			t.Errorf("Expected length 5, got %d", idx.Len())
		}

		expected := []int{0, 2, 4, 6, 8}
		for i, exp := range expected {
			val := idx.Get(i)
			if val != exp {
				t.Errorf("Expected %d at position %d, got %v", exp, i, val)
			}
		}
	})

	t.Run("Loc", func(t *testing.T) {
		idx := NewRangeIndex(10, 20, 2)

		pos, found := idx.Loc(14)
		if !found || pos != 2 {
			t.Errorf("Expected position 2 for value 14, got %d (found: %v)", pos, found)
		}

		_, found = idx.Loc(15)
		if found {
			t.Errorf("Should not find 15 in range with step 2")
		}
	})
}

func TestStringIndex(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
		values := []string{"a", "b", "c", "d"}
		idx := NewStringIndex(values)

		if idx.Len() != 4 {
			t.Errorf("Expected length 4, got %d", idx.Len())
		}

		pos, found := idx.Loc("c")
		if !found || pos != 2 {
			t.Errorf("Expected position 2 for 'c', got %d (found: %v)", pos, found)
		}
	})
}

func TestDateTimeIndex(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
		times := []time.Time{
			time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC),
		}
		idx := NewDateTimeIndex(times)

		if idx.Len() != 3 {
			t.Errorf("Expected length 3, got %d", idx.Len())
		}

		pos, found := idx.Loc(times[1])
		if !found || pos != 1 {
			t.Errorf("Expected position 1 for second time, got %d (found: %v)", pos, found)
		}
	})
}

// Test Series creation
func TestSeriesCreation(t *testing.T) {
	t.Run("FromSlice", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0}
		s, err := FromSlice(data, nil, "test")

		if err != nil {
			t.Fatalf("Failed to create series: %v", err)
		}

		if s.Len() != 4 {
			t.Errorf("Expected length 4, got %d", s.Len())
		}

		if s.Name() != "test" {
			t.Errorf("Expected name 'test', got '%s'", s.Name())
		}

		for i, expected := range data {
			if s.At(i) != expected {
				t.Errorf("Expected %f at position %d, got %v", expected, i, s.At(i))
			}
		}
	})

	t.Run("FromValues", func(t *testing.T) {
		values := []interface{}{1, 2.5, 3, 4.0}
		s, err := FromValues(values, nil, "mixed")

		if err != nil {
			t.Fatalf("Failed to create series: %v", err)
		}

		if s.Len() != 4 {
			t.Errorf("Expected length 4, got %d", s.Len())
		}

		if s.DType() != internal.Float64 {
			t.Errorf("Expected Float64 dtype, got %v", s.DType())
		}
	})

	t.Run("WithCustomIndex", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0}
		index := NewStringIndex([]string{"a", "b", "c"})
		s, err := FromSlice(data, index, "indexed")

		if err != nil {
			t.Fatalf("Failed to create series: %v", err)
		}

		val, err := s.Loc("b")
		if err != nil {
			t.Fatalf("Failed to get value by label: %v", err)
		}

		if val != 2.0 {
			t.Errorf("Expected 2.0 for label 'b', got %v", val)
		}
	})
}

// Test Series operations
func TestSeriesArithmetic(t *testing.T) {
	t.Run("Add", func(t *testing.T) {
		s1 := createTestSeries([]float64{1, 2, 3}, NewDefaultRangeIndex(3))
		s2 := createTestSeries([]float64{4, 5, 6}, NewDefaultRangeIndex(3))

		result, err := s1.Add(s2)
		if err != nil {
			t.Fatalf("Failed to add series: %v", err)
		}

		expected := []float64{5, 7, 9}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}
	})

	t.Run("AddWithAlignment", func(t *testing.T) {
		s1 := createTestSeries([]float64{1, 2, 3}, NewStringIndex([]string{"a", "b", "c"}))
		s2 := createTestSeries([]float64{4, 5, 6}, NewStringIndex([]string{"b", "c", "d"}))

		result, err := s1.Add(s2)
		if err != nil {
			t.Fatalf("Failed to add series with alignment: %v", err)
		}

		// Should have 4 elements (union of indices)
		if result.Len() != 4 {
			t.Errorf("Expected length 4, got %d", result.Len())
		}

		// Check aligned values
		val, err := result.Loc("b")
		if err != nil {
			t.Fatalf("Failed to get value for 'b': %v", err)
		}
		if val != 6.0 { // 2 + 4
			t.Errorf("Expected 6.0 for 'b', got %v", val)
		}
	})

	t.Run("ScalarOperations", func(t *testing.T) {
		s := createTestSeries([]float64{1, 2, 3}, nil)

		result, err := s.AddScalar(10.0)
		if err != nil {
			t.Fatalf("Failed to add scalar: %v", err)
		}

		expected := []float64{11, 12, 13}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}
	})
}

func TestSeriesComparison(t *testing.T) {
	t.Run("GreaterThan", func(t *testing.T) {
		s1 := createTestSeries([]float64{1, 3, 5}, nil)
		s2 := createTestSeries([]float64{2, 2, 4}, nil)

		result, err := s1.Gt(s2)
		if err != nil {
			t.Fatalf("Failed to compare series: %v", err)
		}

		expected := []bool{false, true, true}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}
	})

	t.Run("ScalarComparison", func(t *testing.T) {
		s := createTestSeries([]float64{1, 2, 3, 4, 5}, nil)

		result, err := s.GtScalar(3.0)
		if err != nil {
			t.Fatalf("Failed to compare with scalar: %v", err)
		}

		expected := []bool{false, false, false, true, true}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}
	})
}

// Test indexing and selection
func TestSeriesIndexing(t *testing.T) {
	t.Run("ILoc", func(t *testing.T) {
		s := createTestSeries([]float64{10, 20, 30, 40, 50}, nil)

		result, err := s.ILoc([]int{0, 2, 4})
		if err != nil {
			t.Fatalf("Failed to select by integer location: %v", err)
		}

		expected := []float64{10, 30, 50}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}
	})

	t.Run("Slice", func(t *testing.T) {
		s := createTestSeries([]float64{1, 2, 3, 4, 5}, nil)

		result, err := s.Slice(1, 4)
		if err != nil {
			t.Fatalf("Failed to slice series: %v", err)
		}

		expected := []float64{2, 3, 4}
		if result.Len() != len(expected) {
			t.Errorf("Expected length %d, got %d", len(expected), result.Len())
		}

		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}
	})

	t.Run("BooleanIndexing", func(t *testing.T) {
		s := createTestSeries([]float64{1, 2, 3, 4, 5}, nil)
		mask, _ := FromSlice([]bool{true, false, true, false, true}, nil, "mask")

		result, err := s.BooleanIndexing(mask)
		if err != nil {
			t.Fatalf("Failed to apply boolean indexing: %v", err)
		}

		expected := []float64{1, 3, 5}
		if result.Len() != len(expected) {
			t.Errorf("Expected length %d, got %d", len(expected), result.Len())
		}

		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}
	})

	t.Run("HeadTail", func(t *testing.T) {
		s := createTestSeries([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, nil)

		head := s.Head(3)
		if head.Len() != 3 {
			t.Errorf("Expected head length 3, got %d", head.Len())
		}
		if head.At(0) != 1.0 || head.At(2) != 3.0 {
			t.Errorf("Head values incorrect")
		}

		tail := s.Tail(3)
		if tail.Len() != 3 {
			t.Errorf("Expected tail length 3, got %d", tail.Len())
		}
		if tail.At(0) != 8.0 || tail.At(2) != 10.0 {
			t.Errorf("Tail values incorrect")
		}
	})
}

// Test missing data handling
func TestSeriesMissingData(t *testing.T) {
	t.Run("IsNull", func(t *testing.T) {
		data := []float64{1.0, math.NaN(), 3.0, math.NaN(), 5.0}
		s := createTestSeries(data, nil)

		nullMask := s.IsNull()
		expected := []bool{false, true, false, true, false}

		for i, exp := range expected {
			if nullMask.At(i) != exp {
				t.Errorf("Expected %v at position %d, got %v", exp, i, nullMask.At(i))
			}
		}
	})

	t.Run("DropNa", func(t *testing.T) {
		data := []float64{1.0, math.NaN(), 3.0, math.NaN(), 5.0}
		s := createTestSeries(data, nil)

		result := s.DropNa()
		expected := []float64{1.0, 3.0, 5.0}

		if result.Len() != len(expected) {
			t.Errorf("Expected length %d, got %d", len(expected), result.Len())
		}

		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}
	})

	t.Run("FillNa", func(t *testing.T) {
		data := []float64{1.0, math.NaN(), 3.0, math.NaN(), 5.0}
		s := createTestSeries(data, nil)

		result, err := s.FillNa(999.0)
		if err != nil {
			t.Fatalf("Failed to fill NaN values: %v", err)
		}

		expected := []float64{1.0, 999.0, 3.0, 999.0, 5.0}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}
	})

	t.Run("ForwardFill", func(t *testing.T) {
		data := []float64{1.0, math.NaN(), math.NaN(), 4.0, math.NaN()}
		s := createTestSeries(data, nil)

		result, err := s.FillNaMethod(FillMethodFfill)
		if err != nil {
			t.Fatalf("Failed to forward fill: %v", err)
		}

		expected := []float64{1.0, 1.0, 1.0, 4.0, 4.0}
		for i, exp := range expected {
			actualVal := result.At(i).(float64)
			if actualVal != exp {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})
}

// Test string methods - commented out until array package supports strings
/*
func TestSeriesStringMethods(t *testing.T) {
	t.Run("BasicStringOps", func(t *testing.T) {
		data := []string{"hello", "WORLD", "Test"}
		s := createTestSeriesString(data, nil)

		upper, err := s.Str().Upper()
		if err != nil {
			t.Fatalf("Failed to convert to uppercase: %v", err)
		}

		expected := []string{"HELLO", "WORLD", "TEST"}
		for i, exp := range expected {
			if upper.At(i) != exp {
				t.Errorf("Expected %s at position %d, got %v", exp, i, upper.At(i))
			}
		}
	})

	t.Run("StringLength", func(t *testing.T) {
		data := []string{"a", "bb", "ccc"}
		s := createTestSeriesString(data, nil)

		lengths, err := s.Str().Len()
		if err != nil {
			t.Fatalf("Failed to get string lengths: %v", err)
		}

		expected := []int64{1, 2, 3}
		for i, exp := range expected {
			if lengths.At(i) != exp {
				t.Errorf("Expected %d at position %d, got %v", exp, i, lengths.At(i))
			}
		}
	})

	t.Run("Contains", func(t *testing.T) {
		data := []string{"apple", "banana", "cherry"}
		s := createTestSeriesString(data, nil)

		result, err := s.Str().Contains("an", false)
		if err != nil {
			t.Fatalf("Failed to check contains: %v", err)
		}

		expected := []bool{false, true, false}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}
	})
}
*/

// Test aggregation methods
func TestSeriesAggregation(t *testing.T) {
	t.Run("Sum", func(t *testing.T) {
		s := createTestSeries([]float64{1, 2, 3, 4, 5}, nil)

		result := s.Apply(func(series *Series) interface{} {
			sum := 0.0
			for i := 0; i < series.Len(); i++ {
				if val, ok := series.At(i).(float64); ok {
					sum += val
				}
			}
			return sum
		})

		if result != 15.0 {
			t.Errorf("Expected sum 15.0, got %v", result)
		}
	})

	t.Run("GroupBy", func(t *testing.T) {
		data := []float64{1, 2, 3, 4, 5, 6}
		s := createTestSeries(data, nil)

		// Group by even/odd
		gb := s.GroupBy(func(val interface{}) interface{} {
			if v, ok := val.(float64); ok {
				if int(v)%2 == 0 {
					return "even"
				}
				return "odd"
			}
			return "unknown"
		})

		sumResult, err := gb.Sum()
		if err != nil {
			t.Fatalf("Failed to compute group sum: %v", err)
		}

		// Odd numbers: 1, 3, 5 = 9
		// Even numbers: 2, 4, 6 = 12
		if sumResult.Len() != 2 {
			t.Errorf("Expected 2 groups, got %d", sumResult.Len())
		}
	})
}

// Test sorting
func TestSeriesSorting(t *testing.T) {
	t.Run("Sort", func(t *testing.T) {
		s := createTestSeries([]float64{3, 1, 4, 1, 5}, nil)

		sorted := s.Sort(true)
		expected := []float64{1, 1, 3, 4, 5}

		for i, exp := range expected {
			if sorted.At(i) != exp {
				t.Errorf("Expected %f at position %d, got %v", exp, i, sorted.At(i))
			}
		}
	})

	t.Run("SortByIndex", func(t *testing.T) {
		data := []float64{10, 20, 30}
		index := NewStringIndex([]string{"c", "a", "b"})
		s := createTestSeries(data, index)

		sorted := s.SortByIndex(true)

		// Should be ordered by index: a, b, c
		expectedValues := []float64{20, 30, 10}
		for i, exp := range expectedValues {
			if sorted.At(i) != exp {
				t.Errorf("Expected %f at position %d, got %v", exp, i, sorted.At(i))
			}
		}
	})
}

// Test unique and value counts
func TestSeriesUnique(t *testing.T) {
	t.Run("Unique", func(t *testing.T) {
		s := createTestSeries([]float64{1, 2, 2, 3, 3, 3}, nil)

		unique := s.Unique()
		expected := []float64{1, 2, 3}

		if unique.Len() != len(expected) {
			t.Errorf("Expected %d unique values, got %d", len(expected), unique.Len())
		}
	})

	t.Run("ValueCounts", func(t *testing.T) {
		s := createTestSeries([]float64{1, 2, 2, 3, 3, 3}, nil)

		counts := s.ValueCounts()

		// Should have 3 unique values
		if counts.Len() != 3 {
			t.Errorf("Expected 3 unique values, got %d", counts.Len())
		}

		// Check that we have the right counts (order may vary)
		totalCount := int64(0)
		for i := 0; i < counts.Len(); i++ {
			if count, ok := counts.At(i).(int64); ok {
				totalCount += count
			}
		}

		if totalCount != 6 {
			t.Errorf("Expected total count 6, got %d", totalCount)
		}
	})
}

// Benchmark tests
func BenchmarkSeriesCreation(b *testing.B) {
	data := make([]float64, 1000)
	for i := range data {
		data[i] = float64(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = FromSlice(data, nil, "benchmark")
	}
}

func BenchmarkSeriesArithmetic(b *testing.B) {
	data1 := make([]float64, 1000)
	data2 := make([]float64, 1000)
	for i := range data1 {
		data1[i] = float64(i)
		data2[i] = float64(i + 1)
	}

	s1 := createTestSeries(data1, nil)
	s2 := createTestSeries(data2, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = s1.Add(s2)
	}
}

func BenchmarkSeriesIndexing(b *testing.B) {
	data := make([]float64, 1000)
	for i := range data {
		data[i] = float64(i)
	}

	s := createTestSeries(data, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.At(i % 1000)
	}
}
