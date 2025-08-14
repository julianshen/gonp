package fft

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/julianshen/gonp/array"
)

// TestFFTBasic tests basic FFT functionality
func TestFFTBasic(t *testing.T) {
	tests := []struct {
		name     string
		input    []complex128
		expected []complex128
		tol      float64
	}{
		{
			name:     "single element",
			input:    []complex128{1 + 0i},
			expected: []complex128{1 + 0i},
			tol:      1e-10,
		},
		{
			name:     "two elements",
			input:    []complex128{1 + 0i, 1 + 0i},
			expected: []complex128{2 + 0i, 0 + 0i},
			tol:      1e-10,
		},
		{
			name:     "four elements - constant",
			input:    []complex128{1 + 0i, 1 + 0i, 1 + 0i, 1 + 0i},
			expected: []complex128{4 + 0i, 0 + 0i, 0 + 0i, 0 + 0i},
			tol:      1e-10,
		},
		{
			name:     "four elements - impulse",
			input:    []complex128{1 + 0i, 0 + 0i, 0 + 0i, 0 + 0i},
			expected: []complex128{1 + 0i, 1 + 0i, 1 + 0i, 1 + 0i},
			tol:      1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input array
			inputArr, err := array.FromSlice(tt.input)
			if err != nil {
				t.Fatalf("Failed to create input array: %v", err)
			}

			// This should fail initially - we haven't implemented FFT yet
			result, err := FFT(inputArr)
			if err != nil {
				t.Fatalf("FFT failed: %v", err)
			}

			// Convert result back to slice for comparison
			resultSlice := make([]complex128, result.Size())
			for i := 0; i < result.Size(); i++ {
				val := result.At(i)
				complexVal, ok := val.(complex128)
				if !ok {
					t.Fatalf("Failed to get complex128 at index %d: got %T", i, val)
				}
				resultSlice[i] = complexVal
			}

			// Check results
			if len(resultSlice) != len(tt.expected) {
				t.Fatalf("Length mismatch: got %d, want %d", len(resultSlice), len(tt.expected))
			}

			for i, expected := range tt.expected {
				if cmplx.Abs(resultSlice[i]-expected) > tt.tol {
					t.Errorf("Element %d: got %v, want %v (diff: %v)",
						i, resultSlice[i], expected, cmplx.Abs(resultSlice[i]-expected))
				}
			}
		})
	}
}

// TestIFFTBasic tests basic Inverse FFT functionality
func TestIFFTBasic(t *testing.T) {
	tests := []struct {
		name     string
		input    []complex128
		expected []complex128
		tol      float64
	}{
		{
			name:     "single element",
			input:    []complex128{1 + 0i},
			expected: []complex128{1 + 0i},
			tol:      1e-10,
		},
		{
			name:     "two elements",
			input:    []complex128{2 + 0i, 0 + 0i},
			expected: []complex128{1 + 0i, 1 + 0i},
			tol:      1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input array
			inputArr, err := array.FromSlice(tt.input)
			if err != nil {
				t.Fatalf("Failed to create input array: %v", err)
			}

			// This should fail initially - we haven't implemented IFFT yet
			result, err := IFFT(inputArr)
			if err != nil {
				t.Fatalf("IFFT failed: %v", err)
			}

			// Convert result back to slice for comparison
			resultSlice := make([]complex128, result.Size())
			for i := 0; i < result.Size(); i++ {
				val := result.At(i)
				complexVal, ok := val.(complex128)
				if !ok {
					t.Fatalf("Failed to get complex128 at index %d: got %T", i, val)
				}
				resultSlice[i] = complexVal
			}

			// Check results
			if len(resultSlice) != len(tt.expected) {
				t.Fatalf("Length mismatch: got %d, want %d", len(resultSlice), len(tt.expected))
			}

			for i, expected := range tt.expected {
				if cmplx.Abs(resultSlice[i]-expected) > tt.tol {
					t.Errorf("Element %d: got %v, want %v (diff: %v)",
						i, resultSlice[i], expected, cmplx.Abs(resultSlice[i]-expected))
				}
			}
		})
	}
}

// TestFFTRoundTrip tests that FFT followed by IFFT returns original signal
func TestFFTRoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		input []complex128
		tol   float64
	}{
		{
			name:  "random signal - 4 elements",
			input: []complex128{1 + 2i, 3 - 1i, -1 + 0i, 2 + 3i},
			tol:   1e-10,
		},
		{
			name:  "real signal - 8 elements",
			input: []complex128{1, 2, 3, 4, 5, 6, 7, 8},
			tol:   1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input array
			inputArr, err := array.FromSlice(tt.input)
			if err != nil {
				t.Fatalf("Failed to create input array: %v", err)
			}

			// Forward FFT
			fftResult, err := FFT(inputArr)
			if err != nil {
				t.Fatalf("FFT failed: %v", err)
			}

			// Inverse FFT
			ifftResult, err := IFFT(fftResult)
			if err != nil {
				t.Fatalf("IFFT failed: %v", err)
			}

			// Convert result back to slice for comparison
			resultSlice := make([]complex128, ifftResult.Size())
			for i := 0; i < ifftResult.Size(); i++ {
				val := ifftResult.At(i)
				complexVal, ok := val.(complex128)
				if !ok {
					t.Fatalf("Failed to get complex128 at index %d: got %T", i, val)
				}
				resultSlice[i] = complexVal
			}

			// Check results match original input
			if len(resultSlice) != len(tt.input) {
				t.Fatalf("Length mismatch: got %d, want %d", len(resultSlice), len(tt.input))
			}

			for i, expected := range tt.input {
				if cmplx.Abs(resultSlice[i]-expected) > tt.tol {
					t.Errorf("Element %d: got %v, want %v (diff: %v)",
						i, resultSlice[i], expected, cmplx.Abs(resultSlice[i]-expected))
				}
			}
		})
	}
}

// TestFFTSineWave tests FFT on a known sine wave
func TestFFTSineWave(t *testing.T) {
	// Create a sine wave at frequency 1 with 8 samples
	n := 8
	input := make([]complex128, n)
	for i := 0; i < n; i++ {
		// Sin wave: sin(2*pi*k*f/N) where f=1, N=8
		angle := 2.0 * math.Pi * float64(i) * 1.0 / float64(n)
		input[i] = complex(math.Sin(angle), 0)
	}

	inputArr, err := array.FromSlice(input)
	if err != nil {
		t.Fatalf("Failed to create input array: %v", err)
	}

	result, err := FFT(inputArr)
	if err != nil {
		t.Fatalf("FFT failed: %v", err)
	}

	// For a sine wave at frequency 1, we expect:
	// - Most energy at bin 1 (and bin N-1 due to symmetry)
	// - Very small values elsewhere
	resultSlice := make([]complex128, result.Size())
	for i := 0; i < result.Size(); i++ {
		val := result.At(i)
		complexVal, ok := val.(complex128)
		if !ok {
			t.Fatalf("Failed to get complex128 at index %d: got %T", i, val)
		}
		resultSlice[i] = complexVal
	}

	// Check that bin 1 and bin 7 have the most energy
	tol := 1e-10
	for i, val := range resultSlice {
		magnitude := cmplx.Abs(val)
		if i == 1 || i == 7 {
			// These bins should have significant magnitude
			if magnitude < 1.0 {
				t.Errorf("Bin %d should have large magnitude, got %v", i, magnitude)
			}
		} else {
			// Other bins should be near zero
			if magnitude > tol {
				t.Errorf("Bin %d should be near zero, got %v", i, magnitude)
			}
		}
	}
}

// TestRFFT tests real FFT functionality
func TestRFFT(t *testing.T) {
	tests := []struct {
		name     string
		input    []float64
		expected []complex128
		tol      float64
	}{
		{
			name:     "single element",
			input:    []float64{1.0},
			expected: []complex128{1 + 0i},
			tol:      1e-10,
		},
		{
			name:     "two elements",
			input:    []float64{1.0, 1.0},
			expected: []complex128{2 + 0i, 0 + 0i},
			tol:      1e-10,
		},
		{
			name:     "four elements - constant",
			input:    []float64{1.0, 1.0, 1.0, 1.0},
			expected: []complex128{4 + 0i, 0 + 0i, 0 + 0i},
			tol:      1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input array
			inputArr, err := array.FromSlice(tt.input)
			if err != nil {
				t.Fatalf("Failed to create input array: %v", err)
			}

			// This should fail initially - we haven't implemented RFFT yet
			result, err := RFFT(inputArr)
			if err != nil {
				t.Fatalf("RFFT failed: %v", err)
			}

			// Convert result back to slice for comparison
			resultSlice := make([]complex128, result.Size())
			for i := 0; i < result.Size(); i++ {
				val := result.At(i)
				complexVal, ok := val.(complex128)
				if !ok {
					t.Fatalf("Failed to get complex128 at index %d: got %T", i, val)
				}
				resultSlice[i] = complexVal
			}

			// Check results
			if len(resultSlice) != len(tt.expected) {
				t.Fatalf("Length mismatch: got %d, want %d", len(resultSlice), len(tt.expected))
			}

			for i, expected := range tt.expected {
				if cmplx.Abs(resultSlice[i]-expected) > tt.tol {
					t.Errorf("Element %d: got %v, want %v (diff: %v)",
						i, resultSlice[i], expected, cmplx.Abs(resultSlice[i]-expected))
				}
			}
		})
	}
}

// TestIRFFT tests inverse real FFT functionality
func TestIRFFT(t *testing.T) {
	tests := []struct {
		name     string
		input    []complex128
		expected []float64
		tol      float64
	}{
		{
			name:     "single element",
			input:    []complex128{1 + 0i},
			expected: []float64{1.0},
			tol:      1e-10,
		},
		{
			name:     "two elements",
			input:    []complex128{2 + 0i, 0 + 0i},
			expected: []float64{1.0, 1.0},
			tol:      1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input array
			inputArr, err := array.FromSlice(tt.input)
			if err != nil {
				t.Fatalf("Failed to create input array: %v", err)
			}

			// This should fail initially - we haven't implemented IRFFT yet
			result, err := IRFFT(inputArr, len(tt.expected))
			if err != nil {
				t.Fatalf("IRFFT failed: %v", err)
			}

			// Convert result back to slice for comparison
			resultSlice := make([]float64, result.Size())
			for i := 0; i < result.Size(); i++ {
				val := result.At(i)
				realVal, ok := val.(float64)
				if !ok {
					t.Fatalf("Failed to get float64 at index %d: got %T", i, val)
				}
				resultSlice[i] = realVal
			}

			// Check results
			if len(resultSlice) != len(tt.expected) {
				t.Fatalf("Length mismatch: got %d, want %d", len(resultSlice), len(tt.expected))
			}

			for i, expected := range tt.expected {
				if math.Abs(resultSlice[i]-expected) > tt.tol {
					t.Errorf("Element %d: got %v, want %v (diff: %v)",
						i, resultSlice[i], expected, math.Abs(resultSlice[i]-expected))
				}
			}
		})
	}
}

// TestRFFTRoundTrip tests that RFFT followed by IRFFT returns original signal
func TestRFFTRoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		tol   float64
	}{
		{
			name:  "real signal - 4 elements",
			input: []float64{1.0, 2.0, 3.0, 4.0},
			tol:   1e-10,
		},
		{
			name:  "real signal - 8 elements",
			input: []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
			tol:   1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input array
			inputArr, err := array.FromSlice(tt.input)
			if err != nil {
				t.Fatalf("Failed to create input array: %v", err)
			}

			// Forward RFFT
			rfftResult, err := RFFT(inputArr)
			if err != nil {
				t.Fatalf("RFFT failed: %v", err)
			}

			// Inverse RFFT
			irfftResult, err := IRFFT(rfftResult, len(tt.input))
			if err != nil {
				t.Fatalf("IRFFT failed: %v", err)
			}

			// Convert result back to slice for comparison
			resultSlice := make([]float64, irfftResult.Size())
			for i := 0; i < irfftResult.Size(); i++ {
				val := irfftResult.At(i)
				realVal, ok := val.(float64)
				if !ok {
					t.Fatalf("Failed to get float64 at index %d: got %T", i, val)
				}
				resultSlice[i] = realVal
			}

			// Check results match original input
			if len(resultSlice) != len(tt.input) {
				t.Fatalf("Length mismatch: got %d, want %d", len(resultSlice), len(tt.input))
			}

			for i, expected := range tt.input {
				if math.Abs(resultSlice[i]-expected) > tt.tol {
					t.Errorf("Element %d: got %v, want %v (diff: %v)",
						i, resultSlice[i], expected, math.Abs(resultSlice[i]-expected))
				}
			}
		})
	}
}

// TestFFTErrorHandling tests error conditions
func TestFFTErrorHandling(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		wantErr bool
	}{
		{
			name:    "nil array",
			input:   nil,
			wantErr: true,
		},
		{
			name:    "empty array",
			input:   []complex128{},
			wantErr: true,
		},
		{
			name:    "non-power-of-2 length",
			input:   []complex128{1, 2, 3},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var inputArr *array.Array
			var err error

			if tt.input != nil {
				inputArr, err = array.FromSlice(tt.input)
				if err != nil && !tt.wantErr {
					t.Fatalf("Failed to create input array: %v", err)
				}
			}

			_, err = FFT(inputArr)
			if (err != nil) != tt.wantErr {
				t.Errorf("FFT() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
