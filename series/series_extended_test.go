package series

import (
	"math"
	"testing"
	"time"
)

// Extended tests to increase coverage for series package

// Test Index implementations that weren't covered
func TestIndexExtended(t *testing.T) {
	t.Run("RangeIndexExtended", func(t *testing.T) {
		idx := NewRangeIndex(0, 5, 1)

		// Test Slice
		sliced := idx.Slice(1, 4)
		if sliced.Len() != 3 {
			t.Errorf("Expected sliced length 3, got %d", sliced.Len())
		}

		// Test Equal
		idx2 := NewRangeIndex(0, 5, 1)
		if !idx.Equal(idx2) {
			t.Errorf("Expected equal indices")
		}

		idx3 := NewRangeIndex(0, 6, 1)
		if idx.Equal(idx3) {
			t.Errorf("Expected unequal indices")
		}

		// Test String
		str := idx.String()
		if str == "" {
			t.Errorf("Expected non-empty string representation")
		}

		// Test Type
		typ := idx.Type()
		if typ == nil {
			t.Errorf("Expected non-nil type")
		}

		// Test Append
		appended := idx.Append(5, 6)
		if appended.Len() != 7 {
			t.Errorf("Expected appended length 7, got %d", appended.Len())
		}

		// Test Delete
		deleted := idx.Delete(1, 3)
		if deleted.Len() != 3 {
			t.Errorf("Expected deleted length 3, got %d", deleted.Len())
		}

		// Test IsSorted
		if !idx.IsSorted() {
			t.Errorf("Expected RangeIndex to be sorted")
		}

		// Test Sort
		sorted, indices := idx.Sort()
		if sorted.Len() != idx.Len() {
			t.Errorf("Expected sorted length to match original")
		}
		if len(indices) != idx.Len() {
			t.Errorf("Expected indices length to match original")
		}
	})

	t.Run("StringIndexExtended", func(t *testing.T) {
		values := []string{"c", "a", "b", "d"}
		idx := NewStringIndex(values)

		// Test Slice
		sliced := idx.Slice(1, 3)
		if sliced.Len() != 2 {
			t.Errorf("Expected sliced length 2, got %d", sliced.Len())
		}

		// Test Equal
		idx2 := NewStringIndex(values)
		if !idx.Equal(idx2) {
			t.Errorf("Expected equal indices")
		}

		// Test String
		str := idx.String()
		if str == "" {
			t.Errorf("Expected non-empty string representation")
		}

		// Test Type
		typ := idx.Type()
		if typ == nil {
			t.Errorf("Expected non-nil type")
		}

		// Test Append
		appended := idx.Append("e", "f")
		if appended.Len() != 6 {
			t.Errorf("Expected appended length 6, got %d", appended.Len())
		}

		// Test Delete
		deleted := idx.Delete(0, 2)
		if deleted.Len() != 2 {
			t.Errorf("Expected deleted length 2, got %d", deleted.Len())
		}

		// Test IsSorted
		if idx.IsSorted() {
			t.Errorf("Expected unsorted StringIndex to report not sorted")
		}

		// Test Sort
		sorted, indices := idx.Sort()
		if sorted.Len() != idx.Len() {
			t.Errorf("Expected sorted length to match original")
		}
		if len(indices) != idx.Len() {
			t.Errorf("Expected indices length to match original")
		}

		// Check that sorted index is actually sorted
		if !sorted.IsSorted() {
			t.Errorf("Expected sorted index to be sorted")
		}
	})

	t.Run("DateTimeIndexExtended", func(t *testing.T) {
		times := []time.Time{
			time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
		}
		idx := NewDateTimeIndex(times)

		// Test all methods similar to StringIndex
		sliced := idx.Slice(0, 2)
		if sliced.Len() != 2 {
			t.Errorf("Expected sliced length 2, got %d", sliced.Len())
		}

		idx2 := NewDateTimeIndex(times)
		if !idx.Equal(idx2) {
			t.Errorf("Expected equal indices")
		}

		str := idx.String()
		if str == "" {
			t.Errorf("Expected non-empty string representation")
		}

		typ := idx.Type()
		if typ == nil {
			t.Errorf("Expected non-nil type")
		}

		newTime := time.Date(2023, 1, 4, 0, 0, 0, 0, time.UTC)
		appended := idx.Append(newTime)
		if appended.Len() != 4 {
			t.Errorf("Expected appended length 4, got %d", appended.Len())
		}

		deleted := idx.Delete(1)
		if deleted.Len() != 2 {
			t.Errorf("Expected deleted length 2, got %d", deleted.Len())
		}

		if idx.IsSorted() {
			t.Errorf("Expected unsorted DateTimeIndex to report not sorted")
		}

		sorted, indices := idx.Sort()
		if !sorted.IsSorted() {
			t.Errorf("Expected sorted index to be sorted")
		}
		if len(indices) != idx.Len() {
			t.Errorf("Expected indices length to match original")
		}
	})

	t.Run("Int64IndexExtended", func(t *testing.T) {
		values := []int64{3, 1, 4, 2}
		idx := NewInt64Index(values)

		// Test all methods
		sliced := idx.Slice(1, 3)
		if sliced.Len() != 2 {
			t.Errorf("Expected sliced length 2, got %d", sliced.Len())
		}

		idx2 := NewInt64Index(values)
		if !idx.Equal(idx2) {
			t.Errorf("Expected equal indices")
		}

		str := idx.String()
		if str == "" {
			t.Errorf("Expected non-empty string representation")
		}

		typ := idx.Type()
		if typ == nil {
			t.Errorf("Expected non-nil type")
		}

		appended := idx.Append(int64(5), int64(6))
		if appended.Len() != 6 {
			t.Errorf("Expected appended length 6, got %d", appended.Len())
		}

		deleted := idx.Delete(0, 2)
		if deleted.Len() != 2 {
			t.Errorf("Expected deleted length 2, got %d", deleted.Len())
		}

		if idx.IsSorted() {
			t.Errorf("Expected unsorted Int64Index to report not sorted")
		}

		sorted, indices := idx.Sort()
		if !sorted.IsSorted() {
			t.Errorf("Expected sorted index to be sorted")
		}
		if len(indices) != idx.Len() {
			t.Errorf("Expected indices length to match original")
		}

		// Test Loc
		pos, found := idx.Loc(int64(4))
		if !found || pos != 2 {
			t.Errorf("Expected to find 4 at position 2, got %d (found: %v)", pos, found)
		}
	})
}

// Test Series operations that weren't covered
func TestSeriesExtended(t *testing.T) {
	t.Run("SeriesOperationsExtended", func(t *testing.T) {
		s1 := createTestSeries([]float64{1, 2, 3, 4, 5}, nil)
		s2 := createTestSeries([]float64{2, 4, 6, 8, 10}, nil)

		// Test Sub
		result, err := s1.Sub(s2)
		if err != nil {
			t.Fatalf("Sub failed: %v", err)
		}
		expected := []float64{-1, -2, -3, -4, -5}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Sub: Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}

		// Test Mul
		result, err = s1.Mul(s2)
		if err != nil {
			t.Fatalf("Mul failed: %v", err)
		}
		expected = []float64{2, 8, 18, 32, 50}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Mul: Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}

		// Test Div
		result, err = s1.Div(s2)
		if err != nil {
			t.Fatalf("Div failed: %v", err)
		}
		expected = []float64{0.5, 0.5, 0.5, 0.5, 0.5}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Div: Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}

		// Test SubScalar
		result, err = s1.SubScalar(1.0)
		if err != nil {
			t.Fatalf("SubScalar failed: %v", err)
		}
		expected = []float64{0, 1, 2, 3, 4}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("SubScalar: Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}

		// Test MulScalar
		result, err = s1.MulScalar(2.0)
		if err != nil {
			t.Fatalf("MulScalar failed: %v", err)
		}
		expected = []float64{2, 4, 6, 8, 10}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("MulScalar: Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}

		// Test DivScalar
		result, err = s1.DivScalar(2.0)
		if err != nil {
			t.Fatalf("DivScalar failed: %v", err)
		}
		expected = []float64{0.5, 1, 1.5, 2, 2.5}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("DivScalar: Expected %f at position %d, got %v", exp, i, result.At(i))
			}
		}
	})

	t.Run("ComparisonExtended", func(t *testing.T) {
		s1 := createTestSeries([]float64{1, 3, 5}, nil)
		s2 := createTestSeries([]float64{2, 2, 4}, nil)

		// Test Lt
		result, err := s1.Lt(s2)
		if err != nil {
			t.Fatalf("Lt failed: %v", err)
		}
		expected := []bool{true, false, false}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Lt: Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}

		// Test Le
		result, err = s1.Le(s2)
		if err != nil {
			t.Fatalf("Le failed: %v", err)
		}
		expected = []bool{true, false, false}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Le: Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}

		// Test Ge
		result, err = s1.Ge(s2)
		if err != nil {
			t.Fatalf("Ge failed: %v", err)
		}
		expected = []bool{false, true, true}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Ge: Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}

		// Test Eq
		result, err = s1.Eq(s2)
		if err != nil {
			t.Fatalf("Eq failed: %v", err)
		}
		expected = []bool{false, false, false}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Eq: Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}

		// Test Ne
		result, err = s1.Ne(s2)
		if err != nil {
			t.Fatalf("Ne failed: %v", err)
		}
		expected = []bool{true, true, true}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("Ne: Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}

		// Test scalar comparisons
		result, err = s1.LtScalar(3.0)
		if err != nil {
			t.Fatalf("LtScalar failed: %v", err)
		}
		expected = []bool{true, false, false}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("LtScalar: Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}

		result, err = s1.LeScalar(3.0)
		if err != nil {
			t.Fatalf("LeScalar failed: %v", err)
		}
		expected = []bool{true, true, false}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("LeScalar: Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}

		result, err = s1.GeScalar(3.0)
		if err != nil {
			t.Fatalf("GeScalar failed: %v", err)
		}
		expected = []bool{false, true, true}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("GeScalar: Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}

		result, err = s1.EqScalar(3.0)
		if err != nil {
			t.Fatalf("EqScalar failed: %v", err)
		}
		expected = []bool{false, true, false}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("EqScalar: Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}

		result, err = s1.NeScalar(3.0)
		if err != nil {
			t.Fatalf("NeScalar failed: %v", err)
		}
		expected = []bool{true, false, true}
		for i, exp := range expected {
			if result.At(i) != exp {
				t.Errorf("NeScalar: Expected %v at position %d, got %v", exp, i, result.At(i))
			}
		}
	})

	t.Run("MissingDataExtended", func(t *testing.T) {
		data := []float64{1.0, math.NaN(), 3.0, math.NaN(), 5.0}
		s := createTestSeries(data, nil)

		// Test NotNull (opposite of IsNull)
		notNullMask := s.NotNull()
		expected := []bool{true, false, true, false, true}
		for i, exp := range expected {
			if notNullMask.At(i) != exp {
				t.Errorf("NotNull: Expected %v at position %d, got %v", exp, i, notNullMask.At(i))
			}
		}

		// Test FillNaMethod with BackwardFill
		result, err := s.FillNaMethod(FillMethodBfill)
		if err != nil {
			t.Fatalf("BackwardFill failed: %v", err)
		}

		expectedValues := []float64{1.0, 3.0, 3.0, 5.0, 5.0}
		for i, exp := range expectedValues {
			actualVal := result.At(i).(float64)
			if actualVal != exp {
				t.Errorf("BackwardFill: Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}

		// Test that we have correct count of NaN values
		nanCount := 0
		for i := 0; i < s.Len(); i++ {
			if val, ok := s.At(i).(float64); ok && math.IsNaN(val) {
				nanCount++
			}
		}
		if nanCount != 2 {
			t.Errorf("Expected 2 NaN values, got %d", nanCount)
		}
	})

	t.Run("IndexingExtended", func(t *testing.T) {
		s := createTestSeries([]float64{10, 20, 30, 40, 50}, NewStringIndex([]string{"a", "b", "c", "d", "e"}))

		// Test individual Loc calls
		val, err := s.Loc("b")
		if err != nil {
			t.Fatalf("Loc failed: %v", err)
		}
		if val != 20.0 {
			t.Errorf("Loc: Expected 20.0 for 'b', got %v", val)
		}

		val, err = s.Loc("d")
		if err != nil {
			t.Fatalf("Loc failed: %v", err)
		}
		if val != 40.0 {
			t.Errorf("Loc: Expected 40.0 for 'd', got %v", val)
		}

		// Test Copy
		copied := s.Copy()
		if copied.Len() != s.Len() {
			t.Errorf("Copy: Expected same length")
		}

		if copied.Name() != s.Name() {
			t.Errorf("Copy: Expected same name")
		}

		// Test that copy is independent
		if copied.At(0) != s.At(0) {
			t.Errorf("Copy: Expected same initial values")
		}
	})
}
