package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
)

// Test descriptive statistics functions
func TestDescriptiveStats(t *testing.T) {
	t.Run("Mean", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		arr, err := array.FromSlice(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		mean, err := Mean(arr)
		if err != nil {
			t.Fatalf("Mean failed: %v", err)
		}

		expected := 3.0
		if math.Abs(mean-expected) > 1e-10 {
			t.Errorf("Expected mean %f, got %f", expected, mean)
		}
	})

	t.Run("Median", func(t *testing.T) {
		// Odd number of elements
		data1 := []float64{1.0, 3.0, 2.0, 5.0, 4.0}
		arr1, _ := array.FromSlice(data1)

		median1, err := Median(arr1)
		if err != nil {
			t.Fatalf("Median failed: %v", err)
		}

		if median1 != 3.0 {
			t.Errorf("Expected median 3.0, got %f", median1)
		}

		// Even number of elements
		data2 := []float64{1.0, 2.0, 3.0, 4.0}
		arr2, _ := array.FromSlice(data2)

		median2, err := Median(arr2)
		if err != nil {
			t.Fatalf("Median failed: %v", err)
		}

		expected2 := 2.5
		if math.Abs(median2-expected2) > 1e-10 {
			t.Errorf("Expected median %f, got %f", expected2, median2)
		}
	})

	t.Run("StandardDeviation", func(t *testing.T) {
		data := []float64{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}
		arr, err := array.FromSlice(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		std, err := Std(arr)
		if err != nil {
			t.Fatalf("Std failed: %v", err)
		}

		// Calculate expected std dev manually
		// Mean = (2+4+4+4+5+5+7+9)/8 = 40/8 = 5.0
		// Variance = [(2-5)^2 + (4-5)^2 + (4-5)^2 + (4-5)^2 + (5-5)^2 + (5-5)^2 + (7-5)^2 + (9-5)^2] / (8-1)
		//          = [9 + 1 + 1 + 1 + 0 + 0 + 4 + 16] / 7 = 32/7 ≈ 4.571429
		// Std = sqrt(4.571429) ≈ 2.138090
		expected := 2.138090
		if math.Abs(std-expected) > 1e-5 {
			t.Errorf("Expected std %f, got %f", expected, std)
		}
	})

	t.Run("Variance", func(t *testing.T) {
		data := []float64{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}
		arr, err := array.FromSlice(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		variance, err := Var(arr)
		if err != nil {
			t.Fatalf("Var failed: %v", err)
		}

		// Expected variance = 32/7 ≈ 4.571429 (using sample variance formula with n-1)
		expected := 32.0 / 7.0
		if math.Abs(variance-expected) > 1e-5 {
			t.Errorf("Expected variance %f, got %f", expected, variance)
		}
	})

	t.Run("MinMax", func(t *testing.T) {
		data := []float64{3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0}
		arr, err := array.FromSlice(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		min, err := Min(arr)
		if err != nil {
			t.Fatalf("Min failed: %v", err)
		}

		if min != 1.0 {
			t.Errorf("Expected min 1.0, got %f", min)
		}

		max, err := Max(arr)
		if err != nil {
			t.Fatalf("Max failed: %v", err)
		}

		if max != 9.0 {
			t.Errorf("Expected max 9.0, got %f", max)
		}
	})

	t.Run("Sum", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0}
		arr, err := array.FromSlice(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		sum, err := Sum(arr)
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}

		expected := 10.0
		if math.Abs(sum-expected) > 1e-10 {
			t.Errorf("Expected sum %f, got %f", expected, sum)
		}
	})
}

// Test statistics with missing data
func TestStatsWithMissingData(t *testing.T) {
	t.Run("MeanWithNaN", func(t *testing.T) {
		data := []float64{1.0, math.NaN(), 3.0, 4.0}
		arr, err := array.FromSlice(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		mean, err := MeanSkipNaN(arr)
		if err != nil {
			t.Fatalf("MeanSkipNaN failed: %v", err)
		}

		// Should compute mean of 1, 3, 4 = 8/3
		expected := 8.0 / 3.0
		if math.Abs(mean-expected) > 1e-10 {
			t.Errorf("Expected mean %f, got %f", expected, mean)
		}
	})

	t.Run("StdWithNaN", func(t *testing.T) {
		data := []float64{1.0, math.NaN(), 3.0, 5.0}
		arr, err := array.FromSlice(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		std, err := StdSkipNaN(arr)
		if err != nil {
			t.Fatalf("StdSkipNaN failed: %v", err)
		}

		// Should compute std of 1, 3, 5
		if math.IsNaN(std) {
			t.Error("Expected valid std, got NaN")
		}
	})
}

// Test quantile functions
func TestQuantiles(t *testing.T) {
	t.Run("Quantile", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		arr, err := array.FromSlice(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		// Test 50th percentile (median)
		q50, err := Quantile(arr, 0.5)
		if err != nil {
			t.Fatalf("Quantile failed: %v", err)
		}

		if q50 != 3.0 {
			t.Errorf("Expected 50th percentile 3.0, got %f", q50)
		}

		// Test 25th percentile
		q25, err := Quantile(arr, 0.25)
		if err != nil {
			t.Fatalf("Quantile failed: %v", err)
		}

		if q25 != 2.0 {
			t.Errorf("Expected 25th percentile 2.0, got %f", q25)
		}

		// Test 75th percentile
		q75, err := Quantile(arr, 0.75)
		if err != nil {
			t.Fatalf("Quantile failed: %v", err)
		}

		if q75 != 4.0 {
			t.Errorf("Expected 75th percentile 4.0, got %f", q75)
		}
	})

	t.Run("InterquartileRange", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		arr, err := array.FromSlice(data)
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		iqr, err := IQR(arr)
		if err != nil {
			t.Fatalf("IQR failed: %v", err)
		}

		// IQR = Q75 - Q25 = 4 - 2 = 2
		expected := 2.0
		if math.Abs(iqr-expected) > 1e-10 {
			t.Errorf("Expected IQR %f, got %f", expected, iqr)
		}
	})
}

// Test Series-specific statistics
func TestSeriesStats(t *testing.T) {
	t.Run("SeriesDescribe", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		s, err := series.FromSlice(data, nil, "test_series")
		if err != nil {
			t.Fatalf("Failed to create series: %v", err)
		}

		desc, err := Describe(s)
		if err != nil {
			t.Fatalf("Describe failed: %v", err)
		}

		// Verify description structure
		if desc.Count != 5 {
			t.Errorf("Expected count 5, got %d", desc.Count)
		}

		if math.Abs(desc.Mean-3.0) > 1e-10 {
			t.Errorf("Expected mean 3.0, got %f", desc.Mean)
		}

		if desc.Min != 1.0 {
			t.Errorf("Expected min 1.0, got %f", desc.Min)
		}

		if desc.Max != 5.0 {
			t.Errorf("Expected max 5.0, got %f", desc.Max)
		}

		if math.Abs(desc.Q25-2.0) > 1e-10 {
			t.Errorf("Expected Q25 2.0, got %f", desc.Q25)
		}

		if math.Abs(desc.Median-3.0) > 1e-10 {
			t.Errorf("Expected median 3.0, got %f", desc.Median)
		}

		if math.Abs(desc.Q75-4.0) > 1e-10 {
			t.Errorf("Expected Q75 4.0, got %f", desc.Q75)
		}
	})
}

// Test error conditions
func TestStatsErrors(t *testing.T) {
	t.Run("EmptyArray", func(t *testing.T) {
		// Create empty array using array.Empty for testing
		arr := array.Empty(internal.Shape{0}, internal.Float64)

		_, err := Mean(arr)
		if err == nil {
			t.Error("Expected error for empty array mean")
		}
	})

	t.Run("InvalidQuantile", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0}
		arr, _ := array.FromSlice(data)

		_, err := Quantile(arr, -0.1) // Invalid quantile < 0
		if err == nil {
			t.Error("Expected error for invalid quantile")
		}

		_, err = Quantile(arr, 1.1) // Invalid quantile > 1
		if err == nil {
			t.Error("Expected error for invalid quantile")
		}
	})
}
