package internal

import (
	"fmt"
	"testing"
	"unsafe"
)

// Benchmark SIMD optimized functions vs scalar implementations
func BenchmarkSIMDSqrtFloat64(b *testing.B) {
	sizes := []int{100, 1000, 10000, 100000}

	for _, size := range sizes {
		data := make([]float64, size)
		result := make([]float64, size)

		for i := 0; i < size; i++ {
			data[i] = float64(i+1) * 0.01
		}

		b.Run(fmt.Sprintf("Scalar_%d", size), func(b *testing.B) {
			provider := NewScalarProvider()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				provider.SqrtFloat64(data, result, size)
			}
		})

		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			provider := GetSIMDProvider()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				provider.SqrtFloat64(data, result, size)
			}
		})
	}
}

func BenchmarkSIMDDotProductFloat64(b *testing.B) {
	sizes := []int{100, 1000, 10000, 100000}

	for _, size := range sizes {
		a := make([]float64, size)
		bData := make([]float64, size)

		for i := 0; i < size; i++ {
			a[i] = float64(i + 1)
			bData[i] = float64(i+1) * 0.5
		}

		b.Run(fmt.Sprintf("Scalar_%d", size), func(b *testing.B) {
			provider := NewScalarProvider()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = provider.DotProductFloat64(a, bData, size)
			}
		})

		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			provider := GetSIMDProvider()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = provider.DotProductFloat64(a, bData, size)
			}
		})
	}
}

func BenchmarkSIMDAddFloat64(b *testing.B) {
	sizes := []int{100, 1000, 10000, 100000}

	for _, size := range sizes {
		a := make([]float64, size)
		bData := make([]float64, size)
		result := make([]float64, size)

		for i := 0; i < size; i++ {
			a[i] = float64(i + 1)
			bData[i] = float64(i+1) * 0.5
		}

		b.Run(fmt.Sprintf("Scalar_%d", size), func(b *testing.B) {
			provider := NewScalarProvider()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				provider.AddFloat64(a, bData, result, size)
			}
		})

		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			provider := GetSIMDProvider()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				provider.AddFloat64(a, bData, result, size)
			}
		})
	}
}

// Memory alignment benchmarks
func BenchmarkAlignmentDetection(b *testing.B) {
	provider := GetSIMDProvider()
	data := make([]float64, 1000)
	ptr := unsafe.Pointer(&data[0])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = provider.IsAligned(ptr)
	}
}

func BenchmarkShouldUseSIMDDecision(b *testing.B) {
	data := make([]float64, 1000)
	ptr := unsafe.Pointer(&data[0])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ShouldUseSIMD(len(data), ptr)
	}
}

// SIMD capability detection benchmark
func BenchmarkCPUFeatureDetection(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GetBestSIMDCapability()
	}
}
