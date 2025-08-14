package internal

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

// Tests for advanced SIMD mathematical functions following TDD

func TestAdvancedSIMDMathFunctions(t *testing.T) {
	size := 1000
	tolerance := 1e-10

	// Test data
	a := make([]float64, size)
	b := make([]float64, size)
	result := make([]float64, size)
	expected := make([]float64, size)

	// Initialize test data
	for i := 0; i < size; i++ {
		a[i] = float64(i+1) * 0.01  // Values from 0.01 to 10.0
		b[i] = float64(i+1) * 0.005 // Values from 0.005 to 5.0
	}

	tests := []struct {
		name        string
		simdFunc    func([]float64, []float64, int)
		scalarFunc  func(float64, float64) float64
		description string
	}{
		{
			name:        "Exp",
			simdFunc:    func(a, result []float64, n int) { SIMDExpFloat64(a, result, n) },
			scalarFunc:  func(x, _ float64) float64 { return math.Exp(x) },
			description: "Exponential function",
		},
		{
			name:        "Log",
			simdFunc:    func(a, result []float64, n int) { SIMDLogFloat64(a, result, n) },
			scalarFunc:  func(x, _ float64) float64 { return math.Log(x) },
			description: "Natural logarithm",
		},
		{
			name:        "Sqrt",
			simdFunc:    func(a, result []float64, n int) { SIMDSqrtFloat64(a, result, n) },
			scalarFunc:  func(x, _ float64) float64 { return math.Sqrt(x) },
			description: "Square root",
		},
		{
			name:        "Sin",
			simdFunc:    func(a, result []float64, n int) { SIMDSinFloat64(a, result, n) },
			scalarFunc:  func(x, _ float64) float64 { return math.Sin(x) },
			description: "Sine function",
		},
		{
			name:        "Cos",
			simdFunc:    func(a, result []float64, n int) { SIMDCosFloat64(a, result, n) },
			scalarFunc:  func(x, _ float64) float64 { return math.Cos(x) },
			description: "Cosine function",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Calculate expected results using scalar operations
			for i := 0; i < size; i++ {
				expected[i] = tt.scalarFunc(a[i], b[i])
			}

			// Test SIMD implementation
			tt.simdFunc(a, result, size)

			// Compare results
			maxError := 0.0
			for i := 0; i < size; i++ {
				error := math.Abs(result[i] - expected[i])
				if error > maxError {
					maxError = error
				}

				if error > tolerance {
					t.Errorf("%s: too large error at index %d: got %.12f, expected %.12f, error %.2e",
						tt.name, i, result[i], expected[i], error)
					break
				}
			}

			t.Logf("%s: max error = %.2e (tolerance = %.2e)", tt.name, maxError, tolerance)
		})
	}
}

func TestSIMDPowerFunction(t *testing.T) {
	size := 100
	tolerance := 1e-10

	// Test data - smaller range for pow to avoid overflow
	base := make([]float64, size)
	exponent := make([]float64, size)
	result := make([]float64, size)
	expected := make([]float64, size)

	for i := 0; i < size; i++ {
		base[i] = 1.0 + float64(i)*0.1      // 1.0 to 10.9
		exponent[i] = 0.5 + float64(i)*0.02 // 0.5 to 2.48
	}

	// Calculate expected results
	for i := 0; i < size; i++ {
		expected[i] = math.Pow(base[i], exponent[i])
	}

	// Test SIMD implementation
	SIMDPowFloat64(base, exponent, result, size)

	// Compare results
	maxError := 0.0
	for i := 0; i < size; i++ {
		error := math.Abs(result[i] - expected[i])
		if error > maxError {
			maxError = error
		}

		if error > tolerance {
			t.Errorf("Pow: too large error at index %d: got %.12f, expected %.12f, error %.2e",
				i, result[i], expected[i], error)
		}
	}

	t.Logf("Pow: max error = %.2e (tolerance = %.2e)", maxError, tolerance)
}

func TestSIMDStatisticalFunctions(t *testing.T) {
	size := 1000
	tolerance := 1e-10

	// Test data
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = float64(i) + 0.5 // Avoid exact integers for more realistic test
	}

	t.Run("Mean", func(t *testing.T) {
		// Calculate expected mean
		expected := 0.0
		for _, val := range data {
			expected += val
		}
		expected /= float64(size)

		// Test SIMD implementation
		result := SIMDMeanFloat64(data, size)

		error := math.Abs(result - expected)
		if error > tolerance {
			t.Errorf("Mean: got %.12f, expected %.12f, error %.2e", result, expected, error)
		}

		t.Logf("Mean: result = %.6f, error = %.2e", result, error)
	})

	t.Run("Variance", func(t *testing.T) {
		// Calculate expected variance
		mean := 0.0
		for _, val := range data {
			mean += val
		}
		mean /= float64(size)

		expected := 0.0
		for _, val := range data {
			diff := val - mean
			expected += diff * diff
		}
		expected /= float64(size - 1) // Sample variance

		// Test SIMD implementation
		result := SIMDVarianceFloat64(data, size)

		error := math.Abs(result - expected)
		if error > tolerance {
			t.Errorf("Variance: got %.12f, expected %.12f, error %.2e", result, expected, error)
		}

		t.Logf("Variance: result = %.6f, error = %.2e", result, error)
	})
}

func TestSIMDDotProduct(t *testing.T) {
	size := 1000
	tolerance := 1e-10

	a := make([]float64, size)
	b := make([]float64, size)

	for i := 0; i < size; i++ {
		a[i] = float64(i + 1)
		b[i] = float64(i+1) * 0.5
	}

	// Calculate expected result
	expected := 0.0
	for i := 0; i < size; i++ {
		expected += a[i] * b[i]
	}

	// Test SIMD implementation
	result := SIMDDotProductFloat64(a, b, size)

	error := math.Abs(result - expected)
	if error > tolerance {
		t.Errorf("DotProduct: got %.12f, expected %.12f, error %.2e", result, expected, error)
	}

	t.Logf("DotProduct: result = %.0f, error = %.2e", result, error)
}

func TestSIMDMemoryAlignment(t *testing.T) {
	provider := GetSIMDProvider()
	alignment := provider.AlignmentRequirement()

	t.Run("AlignmentDetection", func(t *testing.T) {
		// Test aligned memory
		alignedData := make([]float64, 100)
		alignedPtr := unsafe.Pointer(&alignedData[0])

		// Test unaligned memory (offset by 1 byte)
		unalignedData := make([]byte, 800+alignment)
		unalignedPtr := unsafe.Pointer(&unalignedData[1])

		if !provider.IsAligned(alignedPtr) {
			t.Errorf("Aligned pointer not detected as aligned")
		}

		// Note: unaligned test may pass depending on Go's memory allocation
		// This is mainly for testing the alignment detection logic
		t.Logf("Alignment requirement: %d bytes", alignment)
		t.Logf("Aligned pointer: %v", alignedPtr)
		t.Logf("Unaligned pointer: %v", unalignedPtr)
	})

	t.Run("ShouldUseSIMD", func(t *testing.T) {
		// Test with different sizes
		testSizes := []int{10, 32, 100, 1000, 10000}

		for _, size := range testSizes {
			shouldUse := ShouldUseSIMD(size)
			expectedUse := size >= SIMDThreshold && provider.Capability() != SIMDNone

			if shouldUse != expectedUse {
				t.Errorf("ShouldUseSIMD(%d): got %v, expected %v", size, shouldUse, expectedUse)
			}
		}

		t.Logf("SIMD threshold: %d", SIMDThreshold)
		t.Logf("Current capability: %s", provider.Capability().String())
	})
}

func TestSIMDFloat32Operations(t *testing.T) {
	size := 500
	tolerance := float32(1e-6)

	// Test data
	a := make([]float32, size)
	b := make([]float32, size)
	result := make([]float32, size)

	for i := 0; i < size; i++ {
		a[i] = float32(i+1) * 0.1
		b[i] = float32(i+1) * 0.05
	}

	tests := []struct {
		name     string
		simdFunc func([]float32, []float32, int)
		mathFunc func(float32, float32) float32
	}{
		{
			name:     "ExpFloat32",
			simdFunc: func(a, result []float32, n int) { SIMDExpFloat32(a, result, n) },
			mathFunc: func(x, _ float32) float32 { return float32(math.Exp(float64(x))) },
		},
		{
			name:     "SqrtFloat32",
			simdFunc: func(a, result []float32, n int) { SIMDSqrtFloat32(a, result, n) },
			mathFunc: func(x, _ float32) float32 { return float32(math.Sqrt(float64(x))) },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test SIMD implementation
			tt.simdFunc(a, result, size)

			// Compare with expected results
			maxError := float32(0.0)
			for i := 0; i < size; i++ {
				expected := tt.mathFunc(a[i], b[i])
				error := float32(math.Abs(float64(result[i] - expected)))
				if error > maxError {
					maxError = error
				}

				if error > tolerance {
					t.Errorf("%s: too large error at index %d: got %.6f, expected %.6f, error %.2e",
						tt.name, i, result[i], expected, error)
					break
				}
			}

			t.Logf("%s: max error = %.2e (tolerance = %.2e)", tt.name, maxError, tolerance)
		})
	}
}

// Benchmark tests
func BenchmarkSIMDvsScalar(b *testing.B) {
	sizes := []int{100, 1000, 10000, 100000}

	for _, size := range sizes {
		data := make([]float64, size)
		result := make([]float64, size)

		for i := 0; i < size; i++ {
			data[i] = float64(i+1) * 0.01
		}

		b.Run(fmt.Sprintf("Exp_Scalar_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for j := 0; j < size; j++ {
					result[j] = math.Exp(data[j])
				}
			}
		})

		b.Run(fmt.Sprintf("Exp_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SIMDExpFloat64(data, result, size)
			}
		})
	}
}
