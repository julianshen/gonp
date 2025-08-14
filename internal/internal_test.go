package internal

import (
	"testing"
)

// Test Shape functions
func TestShape(t *testing.T) {
	t.Run("Ndim", func(t *testing.T) {
		s := Shape{2, 3, 4}
		if s.Ndim() != 3 {
			t.Errorf("Expected Ndim 3, got %d", s.Ndim())
		}

		empty := Shape{}
		if empty.Ndim() != 0 {
			t.Errorf("Expected empty Ndim 0, got %d", empty.Ndim())
		}
	})

	t.Run("Size", func(t *testing.T) {
		s := Shape{2, 3, 4}
		if s.Size() != 24 {
			t.Errorf("Expected Size 24, got %d", s.Size())
		}

		empty := Shape{}
		if empty.Size() != 1 {
			t.Errorf("Expected empty Size 1 (scalar), got %d", empty.Size())
		}

		single := Shape{5}
		if single.Size() != 5 {
			t.Errorf("Expected single Size 5, got %d", single.Size())
		}
	})

	t.Run("Equal", func(t *testing.T) {
		s1 := Shape{2, 3, 4}
		s2 := Shape{2, 3, 4}
		s3 := Shape{2, 3, 5}
		s4 := Shape{2, 3}

		if !s1.Equal(s2) {
			t.Errorf("Expected s1 and s2 to be equal")
		}

		if s1.Equal(s3) {
			t.Errorf("Expected s1 and s3 to be unequal")
		}

		if s1.Equal(s4) {
			t.Errorf("Expected s1 and s4 to be unequal (different lengths)")
		}
	})

	t.Run("Copy", func(t *testing.T) {
		s := Shape{2, 3, 4}
		copied := s.Copy()

		if !s.Equal(copied) {
			t.Errorf("Expected copied shape to be equal to original")
		}

		// Modify original
		s[0] = 10
		if s.Equal(copied) {
			t.Errorf("Expected copied shape to be independent of original")
		}

		if copied[0] != 2 {
			t.Errorf("Expected copied shape to retain original values")
		}
	})
}

// Test Stride functions
func TestStride(t *testing.T) {
	t.Run("Copy", func(t *testing.T) {
		s := Stride{1, 2, 8}
		copied := s.Copy()

		if len(s) != len(copied) {
			t.Errorf("Expected copied stride to have same length")
		}

		for i, v := range s {
			if copied[i] != v {
				t.Errorf("Expected copied stride to be equal at index %d", i)
			}
		}

		// Modify original
		s[0] = 10
		if copied[0] == 10 {
			t.Errorf("Expected copied stride to be independent of original")
		}
	})
}

// Test Range functions
func TestRange(t *testing.T) {
	t.Run("NewRange", func(t *testing.T) {
		r := NewRange(0, 5)
		if r.Start != 0 || r.Stop != 5 || r.Step != 1 {
			t.Errorf("Expected NewRange(0, 5) to have Start=0, Stop=5, Step=1")
		}
	})

	t.Run("Length", func(t *testing.T) {
		// Positive step
		r1 := Range{Start: 0, Stop: 5, Step: 1}
		if r1.Length() != 5 {
			t.Errorf("Expected Range(0, 5, 1) length 5, got %d", r1.Length())
		}

		r2 := Range{Start: 0, Stop: 5, Step: 2}
		if r2.Length() != 3 {
			t.Errorf("Expected Range(0, 5, 2) length 3, got %d", r2.Length())
		}

		// Negative step
		r3 := Range{Start: 5, Stop: 0, Step: -1}
		if r3.Length() != 5 {
			t.Errorf("Expected Range(5, 0, -1) length 5, got %d", r3.Length())
		}

		r4 := Range{Start: 5, Stop: 0, Step: -2}
		if r4.Length() != 3 {
			t.Errorf("Expected Range(5, 0, -2) length 3, got %d", r4.Length())
		}

		// Zero step
		r5 := Range{Start: 0, Stop: 5, Step: 0}
		if r5.Length() != 0 {
			t.Errorf("Expected Range with step 0 to have length 0, got %d", r5.Length())
		}

		// Invalid ranges
		r6 := Range{Start: 5, Stop: 0, Step: 1} // positive step but start > stop
		if r6.Length() != 0 {
			t.Errorf("Expected invalid range length 0, got %d", r6.Length())
		}

		r7 := Range{Start: 0, Stop: 5, Step: -1} // negative step but start < stop
		if r7.Length() != 0 {
			t.Errorf("Expected invalid range length 0, got %d", r7.Length())
		}
	})
}

// Test DType constants
func TestDType(t *testing.T) {
	t.Run("Constants", func(t *testing.T) {
		// Just verify that constants are defined and have different values
		dtypes := []DType{
			Float32, Float64, Int8, Int16, Int32, Int64,
			Uint8, Uint16, Uint32, Uint64, Bool, Complex64, Complex128,
		}

		// Check that all dtypes have different values
		seen := make(map[DType]bool)
		for _, dtype := range dtypes {
			if seen[dtype] {
				t.Errorf("Duplicate DType value: %v", dtype)
			}
			seen[dtype] = true
		}

		if len(seen) != len(dtypes) {
			t.Errorf("Expected %d unique DType values, got %d", len(dtypes), len(seen))
		}
	})
}
