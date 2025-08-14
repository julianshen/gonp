package internal

import (
	"math"
	"testing"
	"unsafe"
)

func TestNEONProvider(t *testing.T) {
	// Skip NEON tests on non-ARM architectures
	if GetCPUFeatures().Architecture != "arm64" && GetCPUFeatures().Architecture != "arm" {
		t.Skip("Skipping NEON tests on non-ARM architecture")
	}

	provider := NewNEONProvider()

	// Test basic properties
	if provider.VectorWidth() != 16 {
		t.Errorf("Expected vector width 16, got %d", provider.VectorWidth())
	}

	if provider.AlignmentRequirement() != 16 {
		t.Errorf("Expected alignment requirement 16, got %d", provider.AlignmentRequirement())
	}

	// Test alignment check
	alignedPtr := make([]float64, 2)
	if !provider.IsAligned(unsafe.Pointer(&alignedPtr[0])) {
		t.Log("Note: Test data not aligned (expected for some test environments)")
	}
}

func TestNEONFloat64Operations(t *testing.T) {
	if GetCPUFeatures().Architecture != "arm64" && GetCPUFeatures().Architecture != "arm" {
		t.Skip("Skipping NEON tests on non-ARM architecture")
	}

	provider := NewNEONProvider()

	// Test data
	a := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	b := []float64{2.0, 3.0, 4.0, 5.0, 6.0, 7.0}
	result := make([]float64, len(a))

	// Test addition
	provider.AddFloat64(a, b, result, len(a))
	expected := []float64{3.0, 5.0, 7.0, 9.0, 11.0, 13.0}
	for i := range expected {
		if math.Abs(result[i]-expected[i]) > 1e-9 {
			t.Errorf("AddFloat64[%d]: expected %f, got %f", i, expected[i], result[i])
		}
	}

	// Test subtraction
	provider.SubFloat64(a, b, result, len(a))
	expected = []float64{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0}
	for i := range expected {
		if math.Abs(result[i]-expected[i]) > 1e-9 {
			t.Errorf("SubFloat64[%d]: expected %f, got %f", i, expected[i], result[i])
		}
	}

	// Test multiplication
	provider.MulFloat64(a, b, result, len(a))
	expected = []float64{2.0, 6.0, 12.0, 20.0, 30.0, 42.0}
	for i := range expected {
		if math.Abs(result[i]-expected[i]) > 1e-9 {
			t.Errorf("MulFloat64[%d]: expected %f, got %f", i, expected[i], result[i])
		}
	}

	// Test division
	provider.DivFloat64(a, b, result, len(a))
	expected = []float64{0.5, 2.0 / 3.0, 0.75, 0.8, 5.0 / 6.0, 6.0 / 7.0}
	for i := range expected {
		if math.Abs(result[i]-expected[i]) > 1e-9 {
			t.Errorf("DivFloat64[%d]: expected %f, got %f", i, expected[i], result[i])
		}
	}
}

func TestNEONFloat32Operations(t *testing.T) {
	if GetCPUFeatures().Architecture != "arm64" && GetCPUFeatures().Architecture != "arm" {
		t.Skip("Skipping NEON tests on non-ARM architecture")
	}

	provider := NewNEONProvider()

	// Test data (8 elements to test vectorization properly)
	a := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	b := []float32{2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	result := make([]float32, len(a))

	// Test addition
	provider.AddFloat32(a, b, result, len(a))
	expected := []float32{3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0}
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > 1e-6 {
			t.Errorf("AddFloat32[%d]: expected %f, got %f", i, expected[i], result[i])
		}
	}

	// Test multiplication
	provider.MulFloat32(a, b, result, len(a))
	expected = []float32{2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0}
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > 1e-6 {
			t.Errorf("MulFloat32[%d]: expected %f, got %f", i, expected[i], result[i])
		}
	}
}

func TestNEONScalarOperations(t *testing.T) {
	if GetCPUFeatures().Architecture != "arm64" && GetCPUFeatures().Architecture != "arm" {
		t.Skip("Skipping NEON tests on non-ARM architecture")
	}

	provider := NewNEONProvider()

	a := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	result := make([]float64, len(a))

	// Test scalar addition
	provider.AddScalarFloat64(a, 10.0, result, len(a))
	expected := []float64{11.0, 12.0, 13.0, 14.0, 15.0, 16.0}
	for i := range expected {
		if math.Abs(result[i]-expected[i]) > 1e-9 {
			t.Errorf("AddScalarFloat64[%d]: expected %f, got %f", i, expected[i], result[i])
		}
	}

	// Test scalar multiplication
	provider.MulScalarFloat64(a, 2.0, result, len(a))
	expected = []float64{2.0, 4.0, 6.0, 8.0, 10.0, 12.0}
	for i := range expected {
		if math.Abs(result[i]-expected[i]) > 1e-9 {
			t.Errorf("MulScalarFloat64[%d]: expected %f, got %f", i, expected[i], result[i])
		}
	}
}

func TestNEONStatisticalOperations(t *testing.T) {
	if GetCPUFeatures().Architecture != "arm64" && GetCPUFeatures().Architecture != "arm" {
		t.Skip("Skipping NEON tests on non-ARM architecture")
	}

	provider := NewNEONProvider()

	a := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	// Test sum
	sum := provider.SumFloat64(a, len(a))
	expectedSum := 21.0
	if math.Abs(sum-expectedSum) > 1e-9 {
		t.Errorf("SumFloat64: expected %f, got %f", expectedSum, sum)
	}

	// Test mean
	mean := provider.MeanFloat64(a, len(a))
	expectedMean := 3.5
	if math.Abs(mean-expectedMean) > 1e-9 {
		t.Errorf("MeanFloat64: expected %f, got %f", expectedMean, mean)
	}

	// Test dot product
	b := []float64{2.0, 3.0, 4.0, 5.0, 6.0, 7.0}
	dot := provider.DotProductFloat64(a, b, len(a))
	expectedDot := 112.0 // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 + 6*7 = 2+6+12+20+30+42 = 112
	if math.Abs(dot-expectedDot) > 1e-9 {
		t.Errorf("DotProductFloat64: expected %f, got %f", expectedDot, dot)
	}

	// Test variance
	variance := provider.VarianceFloat64(a, len(a))
	// Expected variance for [1,2,3,4,5,6] = 3.5 (sample variance)
	expectedVariance := 3.5
	if math.Abs(variance-expectedVariance) > 1e-9 {
		t.Errorf("VarianceFloat64: expected %f, got %f", expectedVariance, variance)
	}
}

func TestNEONSqrtOperations(t *testing.T) {
	if GetCPUFeatures().Architecture != "arm64" && GetCPUFeatures().Architecture != "arm" {
		t.Skip("Skipping NEON tests on non-ARM architecture")
	}

	provider := NewNEONProvider()

	a := []float64{1.0, 4.0, 9.0, 16.0, 25.0, 36.0}
	result := make([]float64, len(a))

	provider.SqrtFloat64(a, result, len(a))
	expected := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	for i := range expected {
		if math.Abs(result[i]-expected[i]) > 1e-9 {
			t.Errorf("SqrtFloat64[%d]: expected %f, got %f", i, expected[i], result[i])
		}
	}
}

// Benchmark tests to verify NEON performance improvements
func BenchmarkNEONAddFloat64(b *testing.B) {
	if GetCPUFeatures().Architecture != "arm64" && GetCPUFeatures().Architecture != "arm" {
		b.Skip("Skipping NEON benchmark on non-ARM architecture")
	}

	provider := NewNEONProvider()
	size := 1000
	a := make([]float64, size)
	arr_b := make([]float64, size)
	result := make([]float64, size)

	for i := range a {
		a[i] = float64(i)
		arr_b[i] = float64(i * 2)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.AddFloat64(a, arr_b, result, size)
	}
}

func BenchmarkScalarAddFloat64(b *testing.B) {
	size := 1000
	a := make([]float64, size)
	arr_b := make([]float64, size)
	result := make([]float64, size)

	for i := range a {
		a[i] = float64(i)
		arr_b[i] = float64(i * 2)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < size; j++ {
			result[j] = a[j] + arr_b[j]
		}
	}
}

func BenchmarkNEONDotProduct(b *testing.B) {
	if GetCPUFeatures().Architecture != "arm64" && GetCPUFeatures().Architecture != "arm" {
		b.Skip("Skipping NEON benchmark on non-ARM architecture")
	}

	provider := NewNEONProvider()
	size := 1000
	a := make([]float64, size)
	arr_b := make([]float64, size)

	for i := range a {
		a[i] = float64(i)
		arr_b[i] = float64(i * 2)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = provider.DotProductFloat64(a, arr_b, size)
	}
}
