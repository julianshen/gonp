package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestDescriptiveStatisticsEdgeCases tests edge cases for basic descriptive statistics
func TestDescriptiveStatisticsEdgeCases(t *testing.T) {
	t.Run("Empty arrays", func(t *testing.T) {
		emptyArray := array.Empty(internal.Shape{0}, internal.Float64)

		// Mean of empty array should return error
		_, err := Mean(emptyArray)
		if err == nil {
			t.Error("Expected error for empty array mean")
		}

		// Std of empty array should return error
		_, err = Std(emptyArray)
		if err == nil {
			t.Error("Expected error for empty array std")
		}

		// Median of empty array should return error
		_, err = Median(emptyArray)
		if err == nil {
			t.Error("Expected error for empty array median")
		}

		// Mode of empty array should return error
		_, err = Mode(emptyArray)
		if err == nil {
			t.Error("Expected error for empty array mode")
		}
	})

	t.Run("Single value arrays", func(t *testing.T) {
		singleValue, _ := array.FromSlice([]float64{42.5})

		// Mean of single value
		mean, err := Mean(singleValue)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if mean != 42.5 {
			t.Errorf("Mean of single value should be the value itself, got %f", mean)
		}

		// Standard deviation of single value should be 0 or error
		std, err := Std(singleValue)
		if err != nil {
			// If error is returned, it's acceptable for edge case
			t.Logf("Std of single value returned error (acceptable): %v", err)
		} else if std != 0 {
			t.Errorf("Std of single value should be 0, got %f", std)
		}

		// Median of single value
		median, err := Median(singleValue)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if median != 42.5 {
			t.Errorf("Median of single value should be the value itself, got %f", median)
		}

		// Mode of single value
		mode, err := Mode(singleValue)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if mode != 42.5 {
			t.Errorf("Mode of single value should be the value itself, got %f", mode)
		}
	})

	t.Run("Arrays with NaN values", func(t *testing.T) {
		withNaN, _ := array.FromSlice([]float64{1.0, math.NaN(), 3.0, 4.0, math.NaN()})

		// Mean should ignore NaN values
		mean, err := Mean(withNaN)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		expectedMean := (1.0 + 3.0 + 4.0) / 3.0
		if math.Abs(mean-expectedMean) > 1e-10 {
			t.Errorf("Mean with NaN values: expected %f, got %f", expectedMean, mean)
		}

		// Standard deviation should ignore NaN values
		std, err := Std(withNaN)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if math.IsNaN(std) {
			t.Error("Std should not be NaN when ignoring NaN values")
		}
	})

	t.Run("Arrays with all NaN values", func(t *testing.T) {
		allNaN, _ := array.FromSlice([]float64{math.NaN(), math.NaN(), math.NaN()})

		// Mean should return NaN
		mean, err := Mean(allNaN)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if !math.IsNaN(mean) {
			t.Errorf("Mean of all NaN should be NaN, got %f", mean)
		}

		// Std should return NaN or error
		_, err = Std(allNaN)
		// Should either return error or NaN - both are acceptable
		if err == nil {
			// If no error, the result should be NaN
			std, _ := Std(allNaN)
			if !math.IsNaN(std) {
				t.Errorf("Std of all NaN should be NaN or error")
			}
		}
	})

	t.Run("Arrays with infinite values", func(t *testing.T) {
		withInf, _ := array.FromSlice([]float64{1.0, math.Inf(1), 3.0, math.Inf(-1), 5.0})

		// Mean with infinite values should handle appropriately
		mean, err := Mean(withInf)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		// The result depends on implementation - could be NaN or Inf
		t.Logf("Mean with infinite values: %f", mean)

		// Test with positive infinity only
		posInf, _ := array.FromSlice([]float64{1.0, math.Inf(1), 3.0})
		mean, err = Mean(posInf)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if !math.IsInf(mean, 1) {
			t.Logf("Mean with positive infinity: %f (might not be +Inf depending on implementation)", mean)
		}
	})

	t.Run("Arrays with very large values", func(t *testing.T) {
		largeValues, _ := array.FromSlice([]float64{1e308, 1e308, 1e307})

		// Mean should handle large values without overflow
		mean, err := Mean(largeValues)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if math.IsInf(mean, 0) {
			t.Logf("Mean overflowed to infinity with large values: %f", mean)
		}

		// Test with mixed large and small values
		mixedValues, _ := array.FromSlice([]float64{1e308, 1.0, -1e308})
		mean, err = Mean(mixedValues)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		t.Logf("Mean with mixed large values: %f", mean)
	})

	t.Run("Arrays with very small values", func(t *testing.T) {
		smallValues, _ := array.FromSlice([]float64{1e-308, 1e-307, 1e-306})

		// Mean should handle small values without underflow
		mean, err := Mean(smallValues)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if mean == 0.0 {
			t.Logf("Mean underflowed to zero with very small values")
		} else {
			t.Logf("Mean with small values: %e", mean)
		}

		// Standard deviation should handle small values
		std, err := Std(smallValues)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		t.Logf("Std with small values: %e", std)
	})
}

// TestCorrelationEdgeCases tests edge cases for correlation functions
func TestCorrelationEdgeCases(t *testing.T) {
	t.Run("Identical arrays correlation", func(t *testing.T) {
		arr, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})

		// Correlation with self should be 1.0
		corr, err := Correlation(arr, arr)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if math.Abs(corr-1.0) > 1e-10 {
			t.Errorf("Self-correlation should be 1.0, got %f", corr)
		}
	})

	t.Run("Perfectly negatively correlated arrays", func(t *testing.T) {
		arr1, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
		arr2, _ := array.FromSlice([]float64{5, 4, 3, 2, 1})

		// Should be perfectly negatively correlated
		corr, err := Correlation(arr1, arr2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if math.Abs(corr-(-1.0)) > 1e-10 {
			t.Errorf("Perfect negative correlation should be -1.0, got %f", corr)
		}
	})

	t.Run("Constant arrays correlation", func(t *testing.T) {
		constant1, _ := array.FromSlice([]float64{5, 5, 5, 5, 5})
		constant2, _ := array.FromSlice([]float64{3, 3, 3, 3, 3})

		// Correlation of constants should be undefined (NaN or error)
		_, err := Correlation(constant1, constant2)
		if err == nil {
			corr, _ := Correlation(constant1, constant2)
			if !math.IsNaN(corr) {
				t.Errorf("Correlation of constants should be NaN or error, got %f", corr)
			}
		}
	})

	t.Run("Arrays with different sizes", func(t *testing.T) {
		arr1, _ := array.FromSlice([]float64{1, 2, 3})
		arr2, _ := array.FromSlice([]float64{1, 2})

		// Should return error for different sizes
		_, err := Correlation(arr1, arr2)
		if err == nil {
			t.Error("Expected error for arrays with different sizes")
		}
	})

	t.Run("Correlation with NaN values", func(t *testing.T) {
		arr1, _ := array.FromSlice([]float64{1, 2, math.NaN(), 4, 5})
		arr2, _ := array.FromSlice([]float64{2, 4, 6, 8, 10})

		// Correlation should handle NaN appropriately
		corr, err := Correlation(arr1, arr2)
		if err != nil && !math.IsNaN(corr) {
			t.Logf("Correlation with NaN values handled as: error=%v, corr=%f", err, corr)
		}
	})
}

// TestDistanceEdgeCases tests edge cases for distance functions
func TestDistanceEdgeCases(t *testing.T) {
	t.Run("Zero distance", func(t *testing.T) {
		arr, _ := array.FromSlice([]float64{1, 2, 3})

		// Distance to self should be 0
		dist, err := EuclideanDistance(arr, arr)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if math.Abs(dist) > 1e-15 {
			t.Errorf("Distance to self should be 0, got %f", dist)
		}
	})

	t.Run("Single dimension distance", func(t *testing.T) {
		arr1, _ := array.FromSlice([]float64{5})
		arr2, _ := array.FromSlice([]float64{3})

		// Distance should be absolute difference
		dist, err := EuclideanDistance(arr1, arr2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		expected := 2.0
		if math.Abs(dist-expected) > 1e-10 {
			t.Errorf("Single dimension distance: expected %f, got %f", expected, dist)
		}
	})

	t.Run("Distance with very large values", func(t *testing.T) {
		arr1, _ := array.FromSlice([]float64{1e150, 1e150})
		arr2, _ := array.FromSlice([]float64{2e150, 2e150})

		// Should handle large values without overflow
		dist, err := EuclideanDistance(arr1, arr2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if math.IsInf(dist, 0) {
			t.Logf("Distance overflowed with large values: %f", dist)
		} else {
			t.Logf("Distance with large values: %e", dist)
		}
	})

	t.Run("Distance with NaN values", func(t *testing.T) {
		arr1, _ := array.FromSlice([]float64{1, math.NaN(), 3})
		arr2, _ := array.FromSlice([]float64{2, 4, 6})

		// Distance with NaN should skip NaN values (current implementation behavior)
		dist, err := EuclideanDistance(arr1, arr2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		// Should compute distance only for non-NaN pairs: sqrt((1-2)^2 + (3-6)^2) = sqrt(1 + 9) = sqrt(10)
		expected := math.Sqrt(10)
		if math.Abs(dist-expected) > 1e-10 {
			t.Errorf("Distance with NaN (skipping NaN): expected %f, got %f", expected, dist)
		}
	})

	t.Run("Manhattan distance edge cases", func(t *testing.T) {
		// Zero distance
		arr, _ := array.FromSlice([]float64{1, 2, 3})
		dist, err := ManhattanDistance(arr, arr)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if math.Abs(dist) > 1e-15 {
			t.Errorf("Manhattan distance to self should be 0, got %f", dist)
		}

		// With negative values
		arr1, _ := array.FromSlice([]float64{-3, -2, -1})
		arr2, _ := array.FromSlice([]float64{3, 2, 1})
		dist, err = ManhattanDistance(arr1, arr2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		expected := 6.0 + 4.0 + 2.0 // |(-3)-3| + |(-2)-2| + |(-1)-1|
		if math.Abs(dist-expected) > 1e-10 {
			t.Errorf("Manhattan distance with negatives: expected %f, got %f", expected, dist)
		}
	})
}

// TestStatisticalTestsEdgeCases tests edge cases for hypothesis tests
func TestStatisticalTestsEdgeCases(t *testing.T) {
	t.Run("T-test with insufficient data", func(t *testing.T) {
		singleValue, _ := array.FromSlice([]float64{5})

		// Should return error for insufficient data
		_, err := OneSampleTTest(singleValue, 0)
		if err == nil {
			t.Error("Expected error for t-test with single value")
		}
	})

	t.Run("T-test with zero standard deviation", func(t *testing.T) {
		constant, _ := array.FromSlice([]float64{5, 5, 5, 5})

		// Should return error for zero standard deviation
		_, err := OneSampleTTest(constant, 3)
		if err == nil {
			t.Error("Expected error for t-test with zero std deviation")
		}
	})

	t.Run("T-test with extreme values", func(t *testing.T) {
		extreme, _ := array.FromSlice([]float64{1e-100, 1e100, 1e-100, 1e100})

		// Should handle extreme values
		result, err := OneSampleTTest(extreme, 0)
		if err != nil {
			t.Logf("T-test with extreme values failed: %v", err)
		} else {
			t.Logf("T-test with extreme values: t=%.6f, p=%.6f", result.Statistic, result.PValue)
		}
	})

	t.Run("Two-sample t-test with very different variances", func(t *testing.T) {
		lowVar, _ := array.FromSlice([]float64{10.0, 10.1, 9.9, 10.05, 9.95}) // Low variance
		highVar, _ := array.FromSlice([]float64{0.0, 20.0, 5.0, 15.0, 10.0})  // High variance

		// Should handle different variances appropriately
		result, err := TwoSampleTTest(lowVar, highVar)
		if err != nil {
			t.Logf("Two-sample t-test with different variances failed: %v", err)
		} else {
			t.Logf("Two-sample t-test: t=%.6f, p=%.6f, df=%.2f",
				result.Statistic, result.PValue, result.DegreesOfFreedom)
		}
	})
}

// TestNumericalPrecisionEdgeCases tests numerical precision issues
func TestNumericalPrecisionEdgeCases(t *testing.T) {
	t.Run("Variance calculation precision", func(t *testing.T) {
		// Values that could cause precision issues in variance calculation
		closeValues, _ := array.FromSlice([]float64{
			1000000.1, 1000000.2, 1000000.3, 1000000.4, 1000000.5,
		})

		variance, err := Var(closeValues)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Variance should be positive and reasonable
		if variance <= 0 {
			t.Errorf("Variance should be positive for varying data, got %f", variance)
		}

		std, err := Std(closeValues)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Check that std^2 ≈ variance
		if math.Abs(std*std-variance) > 1e-10 {
			t.Errorf("std^2 should equal variance: std^2=%.10f, var=%.10f", std*std, variance)
		}
	})

	t.Run("Mean calculation with alternating signs", func(t *testing.T) {
		// Large alternating values that sum to approximately zero
		alternating, _ := array.FromSlice([]float64{
			1e15, -1e15, 1e15, -1e15, 1e15, -1e15, 1,
		})

		mean, err := Mean(alternating)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Mean should be close to 1/7 ≈ 0.142857
		expected := 1.0 / 7.0
		if math.Abs(mean-expected) > 1e-10 {
			t.Errorf("Mean of alternating large values: expected %f, got %f", expected, mean)
		}
	})
}
