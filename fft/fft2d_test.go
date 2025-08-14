package fft

import (
	"math/cmplx"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestFFT2D tests 2D FFT functionality
func TestFFT2D(t *testing.T) {
	tests := []struct {
		name  string
		input [][]complex128
		tol   float64
	}{
		{
			name: "2x2 constant matrix",
			input: [][]complex128{
				{1 + 0i, 1 + 0i},
				{1 + 0i, 1 + 0i},
			},
			tol: 1e-10,
		},
		{
			name: "2x2 impulse matrix",
			input: [][]complex128{
				{1 + 0i, 0 + 0i},
				{0 + 0i, 0 + 0i},
			},
			tol: 1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Flatten 2D input to 1D for NewArrayWithShape
			rows, cols := len(tt.input), len(tt.input[0])
			flatInput := make([]complex128, rows*cols)
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					flatInput[i*cols+j] = tt.input[i][j]
				}
			}

			// Convert input to 2D array
			inputArr, err := array.NewArrayWithShape(flatInput, internal.Shape{rows, cols})
			if err != nil {
				t.Fatalf("Failed to create input array: %v", err)
			}

			// This should fail initially - we haven't implemented FFT2D yet
			result, err := FFT2D(inputArr)
			if err != nil {
				t.Fatalf("FFT2D failed: %v", err)
			}

			// Basic check - result should have same shape as input
			if !result.Shape().Equal(inputArr.Shape()) {
				t.Errorf("Shape mismatch: got %v, want %v", result.Shape(), inputArr.Shape())
			}
		})
	}
}

// TestIFFT2D tests 2D IFFT functionality
func TestIFFT2D(t *testing.T) {
	tests := []struct {
		name  string
		input [][]complex128
		tol   float64
	}{
		{
			name: "2x2 constant matrix",
			input: [][]complex128{
				{4 + 0i, 0 + 0i},
				{0 + 0i, 0 + 0i},
			},
			tol: 1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Flatten 2D input to 1D for NewArrayWithShape
			rows, cols := len(tt.input), len(tt.input[0])
			flatInput := make([]complex128, rows*cols)
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					flatInput[i*cols+j] = tt.input[i][j]
				}
			}

			// Convert input to 2D array
			inputArr, err := array.NewArrayWithShape(flatInput, internal.Shape{rows, cols})
			if err != nil {
				t.Fatalf("Failed to create input array: %v", err)
			}

			// This should fail initially - we haven't implemented IFFT2D yet
			result, err := IFFT2D(inputArr)
			if err != nil {
				t.Fatalf("IFFT2D failed: %v", err)
			}

			// Basic check - result should have same shape as input
			if !result.Shape().Equal(inputArr.Shape()) {
				t.Errorf("Shape mismatch: got %v, want %v", result.Shape(), inputArr.Shape())
			}
		})
	}
}

// TestFFT2DRoundTrip tests that FFT2D followed by IFFT2D returns original signal
func TestFFT2DRoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		input [][]complex128
		tol   float64
	}{
		{
			name: "2x2 random matrix",
			input: [][]complex128{
				{1 + 2i, 3 - 1i},
				{-1 + 0i, 2 + 3i},
			},
			tol: 1e-10,
		},
		{
			name: "4x4 real matrix",
			input: [][]complex128{
				{1, 2, 3, 4},
				{5, 6, 7, 8},
				{9, 10, 11, 12},
				{13, 14, 15, 16},
			},
			tol: 1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Flatten 2D input to 1D for NewArrayWithShape
			rows, cols := len(tt.input), len(tt.input[0])
			flatInput := make([]complex128, rows*cols)
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					flatInput[i*cols+j] = tt.input[i][j]
				}
			}

			// Create input array
			inputArr, err := array.NewArrayWithShape(flatInput, internal.Shape{rows, cols})
			if err != nil {
				t.Fatalf("Failed to create input array: %v", err)
			}

			// Forward 2D FFT
			fftResult, err := FFT2D(inputArr)
			if err != nil {
				t.Fatalf("FFT2D failed: %v", err)
			}

			// Inverse 2D FFT
			ifftResult, err := IFFT2D(fftResult)
			if err != nil {
				t.Fatalf("IFFT2D failed: %v", err)
			}

			// Check results match original input
			if !ifftResult.Shape().Equal(inputArr.Shape()) {
				t.Fatalf("Shape mismatch: got %v, want %v", ifftResult.Shape(), inputArr.Shape())
			}

			// Compare values
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					original := tt.input[i][j]
					result := ifftResult.At(i, j)

					resultComplex, ok := result.(complex128)
					if !ok {
						t.Fatalf("Failed to get complex128 at [%d,%d]: got %T", i, j, result)
					}

					if cmplx.Abs(resultComplex-original) > tt.tol {
						t.Errorf("Element [%d,%d]: got %v, want %v (diff: %v)",
							i, j, resultComplex, original, cmplx.Abs(resultComplex-original))
					}
				}
			}
		})
	}
}
