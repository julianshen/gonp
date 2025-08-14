package fft

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// FFT computes the Discrete Fourier Transform using the Cooley-Tukey algorithm
func FFT(arr *array.Array) (*array.Array, error) {
	if arr == nil {
		return nil, fmt.Errorf("input array cannot be nil")
	}

	if arr.Size() == 0 {
		return nil, fmt.Errorf("input array cannot be empty")
	}

	// Check if length is power of 2
	n := arr.Size()
	if !isPowerOfTwo(n) {
		return nil, fmt.Errorf("array length must be a power of 2, got %d", n)
	}

	// Convert input to complex128 slice
	input := make([]complex128, n)
	for i := 0; i < n; i++ {
		val := arr.At(i)
		switch v := val.(type) {
		case complex128:
			input[i] = v
		case float64:
			input[i] = complex(v, 0)
		case float32:
			input[i] = complex(float64(v), 0)
		case int:
			input[i] = complex(float64(v), 0)
		case int64:
			input[i] = complex(float64(v), 0)
		default:
			return nil, fmt.Errorf("unsupported type for FFT: %T", v)
		}
	}

	// Perform FFT using Cooley-Tukey algorithm
	result := cooleyTukeyFFT(input)

	// Convert result back to array
	return array.FromSlice(result)
}

// IFFT computes the Inverse Discrete Fourier Transform
func IFFT(arr *array.Array) (*array.Array, error) {
	if arr == nil {
		return nil, fmt.Errorf("input array cannot be nil")
	}

	if arr.Size() == 0 {
		return nil, fmt.Errorf("input array cannot be empty")
	}

	// Check if length is power of 2
	n := arr.Size()
	if !isPowerOfTwo(n) {
		return nil, fmt.Errorf("array length must be a power of 2, got %d", n)
	}

	// Convert input to complex128 slice
	input := make([]complex128, n)
	for i := 0; i < n; i++ {
		val := arr.At(i)
		switch v := val.(type) {
		case complex128:
			input[i] = v
		case float64:
			input[i] = complex(v, 0)
		case float32:
			input[i] = complex(float64(v), 0)
		case int:
			input[i] = complex(float64(v), 0)
		case int64:
			input[i] = complex(float64(v), 0)
		default:
			return nil, fmt.Errorf("unsupported type for IFFT: %T", v)
		}
	}

	// Conjugate input
	for i := range input {
		input[i] = cmplx.Conj(input[i])
	}

	// Perform FFT
	result := cooleyTukeyFFT(input)

	// Conjugate result and scale by 1/N
	scale := 1.0 / float64(n)
	for i := range result {
		result[i] = cmplx.Conj(result[i]) * complex(scale, 0)
	}

	// Convert result back to array
	return array.FromSlice(result)
}

// cooleyTukeyFFT implements the Cooley-Tukey FFT algorithm
func cooleyTukeyFFT(x []complex128) []complex128 {
	n := len(x)

	// Base case
	if n == 1 {
		return []complex128{x[0]}
	}

	// Divide
	even := make([]complex128, n/2)
	odd := make([]complex128, n/2)

	for i := 0; i < n/2; i++ {
		even[i] = x[2*i]
		odd[i] = x[2*i+1]
	}

	// Conquer
	evenFFT := cooleyTukeyFFT(even)
	oddFFT := cooleyTukeyFFT(odd)

	// Combine
	result := make([]complex128, n)
	for k := 0; k < n/2; k++ {
		// Twiddle factor: e^(-2πik/n)
		angle := -2.0 * math.Pi * float64(k) / float64(n)
		twiddle := cmplx.Exp(complex(0, angle))

		// Butterfly operation
		t := twiddle * oddFFT[k]
		result[k] = evenFFT[k] + t
		result[k+n/2] = evenFFT[k] - t
	}

	return result
}

// RFFT computes the real FFT, which is optimized for real-valued input signals
// It only returns the positive frequency components due to Hermitian symmetry
func RFFT(arr *array.Array) (*array.Array, error) {
	if arr == nil {
		return nil, fmt.Errorf("input array cannot be nil")
	}

	if arr.Size() == 0 {
		return nil, fmt.Errorf("input array cannot be empty")
	}

	// Check if length is power of 2
	n := arr.Size()
	if !isPowerOfTwo(n) {
		return nil, fmt.Errorf("array length must be a power of 2, got %d", n)
	}

	// Convert input to complex128 slice (real values)
	input := make([]complex128, n)
	for i := 0; i < n; i++ {
		val := arr.At(i)
		switch v := val.(type) {
		case float64:
			input[i] = complex(v, 0)
		case float32:
			input[i] = complex(float64(v), 0)
		case int:
			input[i] = complex(float64(v), 0)
		case int64:
			input[i] = complex(float64(v), 0)
		case complex128:
			// Take only the real part for RFFT
			input[i] = complex(real(v), 0)
		default:
			return nil, fmt.Errorf("unsupported type for RFFT: %T", v)
		}
	}

	// Perform FFT using Cooley-Tukey algorithm
	result := cooleyTukeyFFT(input)

	// For real input, only return positive frequencies (due to Hermitian symmetry)
	// We return frequencies 0 to n/2 (inclusive)
	posFreqs := make([]complex128, n/2+1)
	for i := 0; i <= n/2; i++ {
		posFreqs[i] = result[i]
	}

	// Convert result back to array
	return array.FromSlice(posFreqs)
}

// IRFFT computes the inverse real FFT, reconstructing a real signal from frequency domain
// n specifies the desired length of the output signal
func IRFFT(arr *array.Array, n int) (*array.Array, error) {
	if arr == nil {
		return nil, fmt.Errorf("input array cannot be nil")
	}

	if arr.Size() == 0 {
		return nil, fmt.Errorf("input array cannot be empty")
	}

	// Check if output length is power of 2
	if !isPowerOfTwo(n) {
		return nil, fmt.Errorf("output length must be a power of 2, got %d", n)
	}

	// Validate input size (should be n/2 + 1)
	expectedSize := n/2 + 1
	if arr.Size() != expectedSize {
		return nil, fmt.Errorf("input size %d doesn't match expected size %d for output length %d",
			arr.Size(), expectedSize, n)
	}

	// Convert input to complex128 slice
	posFreqs := make([]complex128, arr.Size())
	for i := 0; i < arr.Size(); i++ {
		val := arr.At(i)
		switch v := val.(type) {
		case complex128:
			posFreqs[i] = v
		case float64:
			posFreqs[i] = complex(v, 0)
		case float32:
			posFreqs[i] = complex(float64(v), 0)
		case int:
			posFreqs[i] = complex(float64(v), 0)
		case int64:
			posFreqs[i] = complex(float64(v), 0)
		default:
			return nil, fmt.Errorf("unsupported type for IRFFT: %T", v)
		}
	}

	// Reconstruct full spectrum using Hermitian symmetry
	fullSpectrum := make([]complex128, n)

	// Copy positive frequencies
	for i := 0; i < len(posFreqs); i++ {
		fullSpectrum[i] = posFreqs[i]
	}

	// Add negative frequencies using Hermitian symmetry: X[-k] = conj(X[k])
	for i := 1; i < n/2; i++ {
		fullSpectrum[n-i] = cmplx.Conj(posFreqs[i])
	}

	// Conjugate input for IFFT
	for i := range fullSpectrum {
		fullSpectrum[i] = cmplx.Conj(fullSpectrum[i])
	}

	// Perform FFT
	result := cooleyTukeyFFT(fullSpectrum)

	// Conjugate result and scale by 1/N, then take real part
	scale := 1.0 / float64(n)
	realResult := make([]float64, n)
	for i := range result {
		complexResult := cmplx.Conj(result[i]) * complex(scale, 0)
		realResult[i] = real(complexResult)
	}

	// Convert result back to array
	return array.FromSlice(realResult)
}

// FFT2D computes the 2D Discrete Fourier Transform
func FFT2D(arr *array.Array) (*array.Array, error) {
	if arr == nil {
		return nil, fmt.Errorf("input array cannot be nil")
	}

	shape := arr.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("input must be 2D array, got %dD", len(shape))
	}

	rows, cols := shape[0], shape[1]

	// Check if dimensions are powers of 2
	if !isPowerOfTwo(rows) || !isPowerOfTwo(cols) {
		return nil, fmt.Errorf("array dimensions must be powers of 2, got %dx%d", rows, cols)
	}

	// Convert to 2D complex slice
	input := make([][]complex128, rows)
	for i := 0; i < rows; i++ {
		input[i] = make([]complex128, cols)
		for j := 0; j < cols; j++ {
			val := arr.At(i, j)
			switch v := val.(type) {
			case complex128:
				input[i][j] = v
			case float64:
				input[i][j] = complex(v, 0)
			case float32:
				input[i][j] = complex(float64(v), 0)
			case int:
				input[i][j] = complex(float64(v), 0)
			case int64:
				input[i][j] = complex(float64(v), 0)
			default:
				return nil, fmt.Errorf("unsupported type for FFT2D: %T", v)
			}
		}
	}

	// Apply 1D FFT to each row
	for i := 0; i < rows; i++ {
		input[i] = cooleyTukeyFFT(input[i])
	}

	// Apply 1D FFT to each column
	for j := 0; j < cols; j++ {
		// Extract column
		column := make([]complex128, rows)
		for i := 0; i < rows; i++ {
			column[i] = input[i][j]
		}

		// Apply FFT to column
		column = cooleyTukeyFFT(column)

		// Put column back
		for i := 0; i < rows; i++ {
			input[i][j] = column[i]
		}
	}

	// Flatten result for array creation
	flatResult := make([]complex128, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			flatResult[i*cols+j] = input[i][j]
		}
	}

	// Convert result back to array with proper shape
	return array.NewArrayWithShape(flatResult, internal.Shape{rows, cols})
}

// IFFT2D computes the 2D Inverse Discrete Fourier Transform
func IFFT2D(arr *array.Array) (*array.Array, error) {
	if arr == nil {
		return nil, fmt.Errorf("input array cannot be nil")
	}

	shape := arr.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("input must be 2D array, got %dD", len(shape))
	}

	rows, cols := shape[0], shape[1]

	// Check if dimensions are powers of 2
	if !isPowerOfTwo(rows) || !isPowerOfTwo(cols) {
		return nil, fmt.Errorf("array dimensions must be powers of 2, got %dx%d", rows, cols)
	}

	// Convert to 2D complex slice
	input := make([][]complex128, rows)
	for i := 0; i < rows; i++ {
		input[i] = make([]complex128, cols)
		for j := 0; j < cols; j++ {
			val := arr.At(i, j)
			switch v := val.(type) {
			case complex128:
				input[i][j] = v
			case float64:
				input[i][j] = complex(v, 0)
			case float32:
				input[i][j] = complex(float64(v), 0)
			case int:
				input[i][j] = complex(float64(v), 0)
			case int64:
				input[i][j] = complex(float64(v), 0)
			default:
				return nil, fmt.Errorf("unsupported type for IFFT2D: %T", v)
			}
		}
	}

	// Conjugate input
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			input[i][j] = cmplx.Conj(input[i][j])
		}
	}

	// Apply 1D FFT to each row
	for i := 0; i < rows; i++ {
		input[i] = cooleyTukeyFFT(input[i])
	}

	// Apply 1D FFT to each column
	for j := 0; j < cols; j++ {
		// Extract column
		column := make([]complex128, rows)
		for i := 0; i < rows; i++ {
			column[i] = input[i][j]
		}

		// Apply FFT to column
		column = cooleyTukeyFFT(column)

		// Put column back
		for i := 0; i < rows; i++ {
			input[i][j] = column[i]
		}
	}

	// Conjugate result and scale by 1/(rows*cols)
	scale := 1.0 / float64(rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			input[i][j] = cmplx.Conj(input[i][j]) * complex(scale, 0)
		}
	}

	// Flatten result for array creation
	flatResult := make([]complex128, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			flatResult[i*cols+j] = input[i][j]
		}
	}

	// Convert result back to array with proper shape
	return array.NewArrayWithShape(flatResult, internal.Shape{rows, cols})
}

// isPowerOfTwo checks if n is a power of 2
func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}
