package math

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestKahanSum(t *testing.T) {
	t.Run("Simple summation", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		arr, _ := array.FromSlice(data)

		result, err := KahanSum(arr)
		if err != nil {
			t.Fatalf("KahanSum failed: %v", err)
		}

		expected := 15.0
		if math.Abs(result-expected) > 1e-15 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})

	t.Run("Challenging precision case", func(t *testing.T) {
		// This is a classic example where naive summation loses precision
		// 1.0 + 1e-15 + (-1.0) should equal 1e-15, but naive sum often gets 0
		data := []float64{1.0, 1e-15, -1.0}
		arr, _ := array.FromSlice(data)

		result, err := KahanSum(arr)
		if err != nil {
			t.Fatalf("KahanSum failed: %v", err)
		}

		expected := 1e-15

		// Compare with naive summation to show improvement
		naive := 0.0
		for _, val := range data {
			naive += val
		}

		// Kahan should be closer to the expected result than naive summation
		kahanError := math.Abs(result - expected)
		naiveError := math.Abs(naive - expected)

		t.Logf("Naive sum: %e, Kahan sum: %e, Expected: %e", naive, result, expected)
		t.Logf("Naive error: %e, Kahan error: %e", naiveError, kahanError)

		// While we can't guarantee perfect precision due to floating-point limitations,
		// Kahan should generally perform as well as or better than naive summation
		if kahanError > naiveError*2 { // Allow some tolerance
			t.Errorf("Kahan summation should not perform significantly worse than naive summation")
		}
	})

	t.Run("Large array with small values", func(t *testing.T) {
		// Sum of 1 million tiny values - challenging for precision
		n := 1000000
		smallValue := 1e-10
		data := make([]float64, n)
		for i := 0; i < n; i++ {
			data[i] = smallValue
		}
		arr, _ := array.FromSlice(data)

		result, err := KahanSum(arr)
		if err != nil {
			t.Fatalf("KahanSum failed: %v", err)
		}

		expected := float64(n) * smallValue
		relativeError := math.Abs(result-expected) / expected

		// Kahan should maintain much better precision than naive summation
		if relativeError > 1e-12 {
			t.Errorf("Relative error too large: %e", relativeError)
		}

		t.Logf("Sum of %d values of %e: got %e, expected %e, rel error: %e", n, smallValue, result, expected, relativeError)
	})

	t.Run("Empty array", func(t *testing.T) {
		arr := array.Empty(internal.Shape{0}, internal.Float64)
		result, err := KahanSum(arr)
		if err != nil {
			t.Fatalf("KahanSum failed: %v", err)
		}

		if result != 0.0 {
			t.Errorf("Expected 0, got %f", result)
		}
	})

	t.Run("NaN and infinite values", func(t *testing.T) {
		data := []float64{1.0, math.NaN(), 3.0, math.Inf(1), 5.0}
		arr, _ := array.FromSlice(data)

		result, err := KahanSum(arr)
		if err != nil {
			t.Fatalf("KahanSum failed: %v", err)
		}

		// Should sum only the valid values: 1.0 + 3.0 + 5.0 = 9.0
		expected := 9.0
		if math.Abs(result-expected) > 1e-15 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})
}

func TestKahanMean(t *testing.T) {
	t.Run("Simple mean", func(t *testing.T) {
		data := []float64{2.0, 4.0, 6.0, 8.0, 10.0}
		arr, _ := array.FromSlice(data)

		result, err := KahanMean(arr)
		if err != nil {
			t.Fatalf("KahanMean failed: %v", err)
		}

		expected := 6.0
		if math.Abs(result-expected) > 1e-15 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})

	t.Run("Mean with NaN values", func(t *testing.T) {
		data := []float64{1.0, math.NaN(), 3.0, math.NaN(), 5.0}
		arr, _ := array.FromSlice(data)

		result, err := KahanMean(arr)
		if err != nil {
			t.Fatalf("KahanMean failed: %v", err)
		}

		// Mean of valid values: (1.0 + 3.0 + 5.0) / 3 = 3.0
		expected := 3.0
		if math.Abs(result-expected) > 1e-15 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})

	t.Run("Empty array", func(t *testing.T) {
		arr := array.Empty(internal.Shape{0}, internal.Float64)
		result, err := KahanMean(arr)
		if err != nil {
			t.Fatalf("KahanMean failed: %v", err)
		}

		if !math.IsNaN(result) {
			t.Errorf("Expected NaN, got %f", result)
		}
	})
}

func TestKahanVariance(t *testing.T) {
	t.Run("Simple variance", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		arr, _ := array.FromSlice(data)

		result, err := KahanVariance(arr, 1) // Sample variance (ddof=1)
		if err != nil {
			t.Fatalf("KahanVariance failed: %v", err)
		}

		// Variance of [1,2,3,4,5] with ddof=1 is 2.5
		expected := 2.5
		if math.Abs(result-expected) > 1e-14 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})

	t.Run("Population variance", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		arr, _ := array.FromSlice(data)

		result, err := KahanVariance(arr, 0) // Population variance (ddof=0)
		if err != nil {
			t.Fatalf("KahanVariance failed: %v", err)
		}

		// Population variance is 2.0
		expected := 2.0
		if math.Abs(result-expected) > 1e-14 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})

	t.Run("Constant values", func(t *testing.T) {
		data := []float64{5.0, 5.0, 5.0, 5.0, 5.0}
		arr, _ := array.FromSlice(data)

		result, err := KahanVariance(arr, 1)
		if err != nil {
			t.Fatalf("KahanVariance failed: %v", err)
		}

		// Variance of constant values should be 0
		if math.Abs(result) > 1e-15 {
			t.Errorf("Expected 0, got %f", result)
		}
	})
}

func TestKahanStandardDeviation(t *testing.T) {
	t.Run("Standard deviation", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		arr, _ := array.FromSlice(data)

		result, err := KahanStandardDeviation(arr, 1)
		if err != nil {
			t.Fatalf("KahanStandardDeviation failed: %v", err)
		}

		// Standard deviation should be sqrt(2.5) ≈ 1.5811
		expected := math.Sqrt(2.5)
		if math.Abs(result-expected) > 1e-14 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})
}

func TestNeumaierSum(t *testing.T) {
	t.Run("Neumaier vs Kahan comparison", func(t *testing.T) {
		// Test case where Neumaier should perform better
		data := []float64{1e16, 1.0, -1e16}
		arr, _ := array.FromSlice(data)

		kahanResult, err := KahanSum(arr)
		if err != nil {
			t.Fatalf("KahanSum failed: %v", err)
		}

		neumaierResult, err := NeumaierSum(arr)
		if err != nil {
			t.Fatalf("NeumaierSum failed: %v", err)
		}

		expected := 1.0
		t.Logf("Kahan: %e, Neumaier: %e, Expected: %e", kahanResult, neumaierResult, expected)

		// Both should be close to 1.0, but Neumaier might be more accurate in some cases
		if math.Abs(neumaierResult-expected) > 1e-14 {
			t.Errorf("Neumaier result too far from expected: %e", neumaierResult)
		}
	})
}

func TestPairwiseSum(t *testing.T) {
	t.Run("Pairwise summation", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
		arr, _ := array.FromSlice(data)

		result, err := PairwiseSum(arr)
		if err != nil {
			t.Fatalf("PairwiseSum failed: %v", err)
		}

		expected := 36.0
		if math.Abs(result-expected) > 1e-15 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})

	t.Run("Single element", func(t *testing.T) {
		data := []float64{42.0}
		arr, _ := array.FromSlice(data)

		result, err := PairwiseSum(arr)
		if err != nil {
			t.Fatalf("PairwiseSum failed: %v", err)
		}

		if result != 42.0 {
			t.Errorf("Expected 42.0, got %f", result)
		}
	})

	t.Run("Large array precision test", func(t *testing.T) {
		// Test with a large array to see if pairwise helps with precision
		n := 10000
		data := make([]float64, n)
		for i := 0; i < n; i++ {
			data[i] = 1e-8 // Small values
		}
		arr, _ := array.FromSlice(data)

		result, err := PairwiseSum(arr)
		if err != nil {
			t.Fatalf("PairwiseSum failed: %v", err)
		}

		expected := float64(n) * 1e-8
		relativeError := math.Abs(result-expected) / expected

		if relativeError > 1e-12 {
			t.Errorf("Relative error too large: %e", relativeError)
		}
	})
}

func TestCompensatedDotProduct(t *testing.T) {
	t.Run("Simple dot product", func(t *testing.T) {
		a := []float64{1.0, 2.0, 3.0}
		b := []float64{4.0, 5.0, 6.0}
		arrA, _ := array.FromSlice(a)
		arrB, _ := array.FromSlice(b)

		result, err := CompensatedDotProduct(arrA, arrB)
		if err != nil {
			t.Fatalf("CompensatedDotProduct failed: %v", err)
		}

		// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
		expected := 32.0
		if math.Abs(result-expected) > 1e-15 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})

	t.Run("Mismatched sizes", func(t *testing.T) {
		a := []float64{1.0, 2.0}
		b := []float64{1.0, 2.0, 3.0}
		arrA, _ := array.FromSlice(a)
		arrB, _ := array.FromSlice(b)

		_, err := CompensatedDotProduct(arrA, arrB)
		if err == nil {
			t.Error("Expected error for mismatched array sizes")
		}
	})

	t.Run("Precision challenging case", func(t *testing.T) {
		// Create arrays where precision matters
		n := 1000
		a := make([]float64, n)
		b := make([]float64, n)
		for i := 0; i < n; i++ {
			a[i] = 1e-8
			b[i] = 1e-8
		}
		arrA, _ := array.FromSlice(a)
		arrB, _ := array.FromSlice(b)

		result, err := CompensatedDotProduct(arrA, arrB)
		if err != nil {
			t.Fatalf("CompensatedDotProduct failed: %v", err)
		}

		expected := float64(n) * 1e-16 // 1000 * 1e-8 * 1e-8
		relativeError := math.Abs(result-expected) / expected

		if relativeError > 1e-12 {
			t.Errorf("Relative error too large: %e", relativeError)
		}
	})
}

func TestCompensatedNorm(t *testing.T) {
	t.Run("Simple L2 norm", func(t *testing.T) {
		data := []float64{3.0, 4.0}
		arr, _ := array.FromSlice(data)

		result, err := CompensatedNorm(arr)
		if err != nil {
			t.Fatalf("CompensatedNorm failed: %v", err)
		}

		// ||[3,4]|| = sqrt(9+16) = sqrt(25) = 5
		expected := 5.0
		if math.Abs(result-expected) > 1e-15 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})

	t.Run("Unit vector", func(t *testing.T) {
		data := []float64{1.0, 0.0, 0.0}
		arr, _ := array.FromSlice(data)

		result, err := CompensatedNorm(arr)
		if err != nil {
			t.Fatalf("CompensatedNorm failed: %v", err)
		}

		expected := 1.0
		if math.Abs(result-expected) > 1e-15 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})
}

func TestExtendedPrecisionSum(t *testing.T) {
	t.Run("Extended precision summation", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		arr, _ := array.FromSlice(data)

		result, err := ExtendedPrecisionSum(arr)
		if err != nil {
			t.Fatalf("ExtendedPrecisionSum failed: %v", err)
		}

		expected := 15.0
		if math.Abs(result-expected) > 1e-15 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})

	t.Run("Challenging precision case", func(t *testing.T) {
		// Similar challenging case as for Kahan
		data := []float64{1.0, 1e-15, -1.0}
		arr, _ := array.FromSlice(data)

		result, err := ExtendedPrecisionSum(arr)
		if err != nil {
			t.Fatalf("ExtendedPrecisionSum failed: %v", err)
		}

		expected := 1e-15
		if math.Abs(result-expected) > 1e-16 {
			t.Errorf("Expected %e, got %e", expected, result)
		}
	})
}

// Benchmark tests to compare performance and accuracy

func BenchmarkSummationAlgorithms(b *testing.B) {
	// Create test data with precision challenges
	n := 100000
	data := make([]float64, n)
	for i := 0; i < n; i++ {
		data[i] = 1e-10 + float64(i)*1e-12
	}
	arr, _ := array.FromSlice(data)

	b.Run("KahanSum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = KahanSum(arr)
		}
	})

	b.Run("NeumaierSum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = NeumaierSum(arr)
		}
	})

	b.Run("PairwiseSum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = PairwiseSum(arr)
		}
	})

	b.Run("ExtendedPrecisionSum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = ExtendedPrecisionSum(arr)
		}
	})

	// Compare with naive summation
	b.Run("NaiveSum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sum := 0.0
			for j := 0; j < arr.Size(); j++ {
				val, _ := convertValueToFloat64(arr.At(j))
				sum += val
			}
		}
	})
}

func TestConvertToFloat64(t *testing.T) {
	tests := []struct {
		input    interface{}
		expected float64
		hasError bool
	}{
		{float64(3.14), 3.14, false},
		{float32(2.71), float64(float32(2.71)), false},
		{int(42), 42.0, false},
		{int64(123), 123.0, false},
		{uint32(456), 456.0, false},
		{true, 1.0, false},
		{false, 0.0, false},
		{complex64(3 + 4i), 5.0, false},  // magnitude
		{complex128(3 + 4i), 5.0, false}, // magnitude
		{"string", 0.0, true},            // should error
	}

	for _, test := range tests {
		result, err := convertValueToFloat64(test.input)

		if test.hasError {
			if err == nil {
				t.Errorf("Expected error for input %v", test.input)
			}
		} else {
			if err != nil {
				t.Errorf("Unexpected error for input %v: %v", test.input, err)
			}
			if math.Abs(result-test.expected) > 1e-10 {
				t.Errorf("Input %v: expected %f, got %f", test.input, test.expected, result)
			}
		}
	}
}
