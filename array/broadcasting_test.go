package array

import (
	"testing"

	"github.com/julianshen/gonp/internal"
)

func TestBroadcastShapes(t *testing.T) {
	tests := []struct {
		name     string
		shape1   internal.Shape
		shape2   internal.Shape
		expected internal.Shape
		wantErr  bool
	}{
		{
			name:     "identical shapes",
			shape1:   internal.Shape{3, 4},
			shape2:   internal.Shape{3, 4},
			expected: internal.Shape{3, 4},
			wantErr:  false,
		},
		{
			name:     "scalar with array",
			shape1:   internal.Shape{1},
			shape2:   internal.Shape{3, 4},
			expected: internal.Shape{3, 4},
			wantErr:  false,
		},
		{
			name:     "different length compatible",
			shape1:   internal.Shape{4},
			shape2:   internal.Shape{3, 4},
			expected: internal.Shape{3, 4},
			wantErr:  false,
		},
		{
			name:     "broadcasting with 1s",
			shape1:   internal.Shape{3, 1},
			shape2:   internal.Shape{1, 4},
			expected: internal.Shape{3, 4},
			wantErr:  false,
		},
		{
			name:     "incompatible shapes",
			shape1:   internal.Shape{3, 2},
			shape2:   internal.Shape{3, 4},
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "complex broadcasting",
			shape1:   internal.Shape{2, 1, 4},
			shape2:   internal.Shape{3, 1},
			expected: internal.Shape{2, 3, 4},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := BroadcastShapes(tt.shape1, tt.shape2)

			if (err != nil) != tt.wantErr {
				t.Errorf("BroadcastShapes() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && !result.Equal(tt.expected) {
				t.Errorf("BroadcastShapes() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBroadcastArrays(t *testing.T) {
	// Test basic broadcasting
	a := Ones(internal.Shape{2, 1}, internal.Float64)
	b := Ones(internal.Shape{1, 3}, internal.Float64)

	broadA, broadB, err := BroadcastArrays(a, b)
	if err != nil {
		t.Fatalf("BroadcastArrays() error = %v", err)
	}

	expectedShape := internal.Shape{2, 3}
	if !broadA.Shape().Equal(expectedShape) {
		t.Errorf("Broadcasted A shape = %v, want %v", broadA.Shape(), expectedShape)
	}
	if !broadB.Shape().Equal(expectedShape) {
		t.Errorf("Broadcasted B shape = %v, want %v", broadB.Shape(), expectedShape)
	}

	// Test values are correctly broadcasted
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			valA := broadA.At(i, j)
			valB := broadB.At(i, j)
			if valA != 1.0 || valB != 1.0 {
				t.Errorf("Broadcasted values at (%d,%d): A=%v, B=%v, want 1.0, 1.0", i, j, valA, valB)
			}
		}
	}
}

func TestBroadcastArraysIncompatible(t *testing.T) {
	a := Ones(internal.Shape{3, 2}, internal.Float64)
	b := Ones(internal.Shape{3, 4}, internal.Float64)

	_, _, err := BroadcastArrays(a, b)
	if err == nil {
		t.Error("BroadcastArrays() should return error for incompatible shapes")
	}
}

func TestCanBroadcast(t *testing.T) {
	tests := []struct {
		name     string
		shape1   internal.Shape
		shape2   internal.Shape
		expected bool
	}{
		{
			name:     "compatible shapes",
			shape1:   internal.Shape{3, 1},
			shape2:   internal.Shape{1, 4},
			expected: true,
		},
		{
			name:     "incompatible shapes",
			shape1:   internal.Shape{3, 2},
			shape2:   internal.Shape{4, 3},
			expected: false,
		},
		{
			name:     "scalar with array",
			shape1:   internal.Shape{1},
			shape2:   internal.Shape{5, 4},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CanBroadcast(tt.shape1, tt.shape2)
			if result != tt.expected {
				t.Errorf("CanBroadcast() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGetBroadcastShape(t *testing.T) {
	shape1 := internal.Shape{2, 1, 3}
	shape2 := internal.Shape{4, 1}

	result, err := GetBroadcastShape(shape1, shape2)
	if err != nil {
		t.Fatalf("GetBroadcastShape() error = %v", err)
	}

	expected := internal.Shape{2, 4, 3}
	if !result.Equal(expected) {
		t.Errorf("GetBroadcastShape() = %v, want %v", result, expected)
	}
}
