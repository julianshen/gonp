package math

import (
	"fmt"
	"testing"

	"github.com/julianshen/gonp/array"
)

// Benchmark sizes for testing
var benchmarkSizes = []int{100, 1000, 10000}

// BenchmarkTrigonometricFunctions benchmarks all trigonometric functions
func BenchmarkTrigonometricFunctions(b *testing.B) {
	for _, size := range benchmarkSizes {
		data := make([]float64, size)
		for i := range data {
			data[i] = float64(i) * 0.001 // Small values to avoid overflow
		}
		arr := createTestArray(data)

		b.Run(fmt.Sprintf("Sin_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Sin(arr)
			}
		})

		b.Run(fmt.Sprintf("Cos_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Cos(arr)
			}
		})

		b.Run(fmt.Sprintf("Tan_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Tan(arr)
			}
		})
	}
}

// BenchmarkExponentialFunctions benchmarks exponential and logarithmic functions
func BenchmarkExponentialFunctions(b *testing.B) {
	for _, size := range benchmarkSizes {
		// For exp functions, use smaller values to avoid overflow
		expData := make([]float64, size)
		for i := range expData {
			expData[i] = float64(i%100) * 0.01 // Values from 0 to 1
		}
		expArr := createTestArray(expData)

		// For log functions, use positive values > 0
		logData := make([]float64, size)
		for i := range logData {
			logData[i] = float64(i+1) * 0.1 // Values from 0.1 upward
		}
		logArr := createTestArray(logData)

		b.Run(fmt.Sprintf("Exp_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Exp(expArr)
			}
		})

		b.Run(fmt.Sprintf("Log_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Log(logArr)
			}
		})

		b.Run(fmt.Sprintf("Log10_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Log10(logArr)
			}
		})
	}
}

// BenchmarkPowerFunctions benchmarks power and root functions
func BenchmarkPowerFunctions(b *testing.B) {
	for _, size := range benchmarkSizes {
		data := make([]float64, size)
		for i := range data {
			data[i] = float64(i + 1) // Positive values for sqrt
		}
		arr := createTestArray(data)

		b.Run(fmt.Sprintf("Sqrt_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Sqrt(arr)
			}
		})

		b.Run(fmt.Sprintf("Square_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Square(arr)
			}
		})

		b.Run(fmt.Sprintf("Cbrt_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Cbrt(arr)
			}
		})
	}
}

// BenchmarkUtilityFunctions benchmarks utility functions
func BenchmarkUtilityFunctions(b *testing.B) {
	for _, size := range benchmarkSizes {
		data := make([]float64, size)
		for i := range data {
			data[i] = float64(i - size/2) // Mix of positive and negative values
		}
		arr := createTestArray(data)

		b.Run(fmt.Sprintf("Abs_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Abs(arr)
			}
		})

		b.Run(fmt.Sprintf("Sign_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Sign(arr)
			}
		})

		b.Run(fmt.Sprintf("Round_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Round(arr)
			}
		})
	}
}

// BenchmarkInPlaceFunctions benchmarks in-place operations
func BenchmarkInPlaceFunctions(b *testing.B) {
	size := 1000

	b.Run("SinInPlace", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			data := make([]float64, size)
			for j := range data {
				data[j] = float64(j) * 0.001
			}
			arr := createTestArray(data)
			b.StartTimer()

			_ = SinInPlace(arr)
		}
	})

	b.Run("AbsInPlace", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			data := make([]float64, size)
			for j := range data {
				data[j] = float64(j - size/2)
			}
			arr := createTestArray(data)
			b.StartTimer()

			_ = AbsInPlace(arr)
		}
	})

	b.Run("SqrtInPlace", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			data := make([]float64, size)
			for j := range data {
				data[j] = float64(j + 1)
			}
			arr := createTestArray(data)
			b.StartTimer()

			_ = SqrtInPlace(arr)
		}
	})
}

// BenchmarkBinaryOperations benchmarks binary operations
func BenchmarkBinaryOperations(b *testing.B) {
	size := 1000
	data1 := make([]float64, size)
	data2 := make([]float64, size)
	for i := range data1 {
		data1[i] = float64(i + 1)
		data2[i] = float64(i + 2)
	}
	arr1 := createTestArray(data1)
	arr2 := createTestArray(data2)

	b.Run("Maximum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = Maximum(arr1, arr2)
		}
	})

	b.Run("Minimum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = Minimum(arr1, arr2)
		}
	})

	b.Run("Power", func(b *testing.B) {
		// Use smaller base values to avoid overflow
		baseData := make([]float64, size)
		expData := make([]float64, size)
		for i := range baseData {
			baseData[i] = 2.0 + float64(i%5)*0.1 // Values around 2
			expData[i] = 2.0 + float64(i%3)*0.5  // Values around 2-3
		}
		baseArr := createTestArray(baseData)
		expArr := createTestArray(expData)

		for i := 0; i < b.N; i++ {
			_, _ = Power(baseArr, expArr)
		}
	})
}

// BenchmarkComplexNumbers benchmarks complex number operations
func BenchmarkComplexNumbers(b *testing.B) {
	size := 1000
	complexData := make([]complex128, size)
	for i := range complexData {
		real := float64(i) * 0.01
		imag := float64(i) * 0.001
		complexData[i] = complex(real, imag)
	}
	arr, _ := array.FromSlice(complexData)

	b.Run("ComplexSin", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = Sin(arr)
		}
	})

	b.Run("ComplexExp", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = Exp(arr)
		}
	})

	b.Run("ComplexSqrt", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = Sqrt(arr)
		}
	})
}

// BenchmarkMemoryAllocation benchmarks memory allocation patterns
func BenchmarkMemoryAllocation(b *testing.B) {
	size := 1000
	data := make([]float64, size)
	for i := range data {
		data[i] = float64(i) * 0.001
	}
	arr := createTestArray(data)

	b.Run("Sin_MemAlloc", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = Sin(arr)
		}
	})

	b.Run("SinInPlace_MemAlloc", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			// Create a fresh array each iteration for in-place operation
			testData := make([]float64, size)
			copy(testData, data)
			testArr := createTestArray(testData)
			b.StartTimer()

			_ = SinInPlace(testArr)
		}
	})
}
