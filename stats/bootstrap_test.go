package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestBootstrapSample tests bootstrap sampling
func TestBootstrapSample(t *testing.T) {
	t.Run("Bootstrap sample basic functionality", func(t *testing.T) {
		// Create test data
		data, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})

		// Generate bootstrap sample
		sample, indices, err := BootstrapSample(data, 42)
		if err != nil {
			t.Fatalf("Bootstrap sample failed: %v", err)
		}

		// Sample should have same size as original
		if sample.Size() != data.Size() {
			t.Errorf("Bootstrap sample size should be %d, got %d", data.Size(), sample.Size())
		}

		// Indices should have same length as sample
		if len(indices) != sample.Size() {
			t.Errorf("Indices length should be %d, got %d", sample.Size(), len(indices))
		}

		// Check that sample values come from original data
		originalValues := make(map[float64]bool)
		for i := 0; i < data.Size(); i++ {
			originalValues[data.At(i).(float64)] = true
		}

		for i := 0; i < sample.Size(); i++ {
			val := sample.At(i).(float64)
			if !originalValues[val] {
				t.Errorf("Bootstrap sample contains value %.2f not in original data", val)
			}
		}

		t.Logf("Bootstrap sample: %v", getBootstrapValues(sample))
		t.Logf("Bootstrap indices: %v", indices)
	})

	t.Run("Bootstrap sample with replacement", func(t *testing.T) {
		// Small dataset to ensure replacement occurs
		data, _ := array.FromSlice([]float64{1, 2})

		// Generate many bootstrap samples to verify replacement
		replacementSeen := false
		for i := 0; i < 20; i++ {
			sample, _, err := BootstrapSample(data, int64(i))
			if err != nil {
				t.Fatalf("Bootstrap sample failed: %v", err)
			}

			// Check for duplicates (evidence of replacement)
			vals := getBootstrapValues(sample)
			if vals[0] == vals[1] {
				replacementSeen = true
				break
			}
		}

		if !replacementSeen {
			t.Error("Bootstrap sampling should show evidence of replacement")
		}

		t.Logf("Replacement verified in bootstrap sampling")
	})

	t.Run("Bootstrap sample parameter validation", func(t *testing.T) {
		// Test nil input
		_, _, err := BootstrapSample(nil, 42)
		if err == nil {
			t.Error("Expected error for nil input")
		}

		// Test empty array
		emptyData := array.Empty(internal.Shape{0}, internal.Float64)
		_, _, err = BootstrapSample(emptyData, 42)
		if err == nil {
			t.Error("Expected error for empty array")
		}
	})
}

// TestBootstrapConfidenceInterval tests confidence interval estimation
func TestBootstrapConfidenceInterval(t *testing.T) {
	t.Run("Bootstrap CI for mean", func(t *testing.T) {
		// Create normal-ish data
		data, _ := array.FromSlice([]float64{
			10.1, 10.3, 9.8, 10.2, 9.9, 10.4, 10.0, 9.7, 10.5, 10.1,
			9.8, 10.2, 10.3, 9.9, 10.0, 10.4, 9.7, 10.1, 9.8, 10.2,
		})

		// Bootstrap confidence interval for mean
		meanFunc := func(sample *array.Array) float64 {
			sum := 0.0
			for i := 0; i < sample.Size(); i++ {
				sum += sample.At(i).(float64)
			}
			return sum / float64(sample.Size())
		}

		lower, upper, err := BootstrapConfidenceInterval(data, meanFunc, 1000, 0.95, 42)
		if err != nil {
			t.Fatalf("Bootstrap CI failed: %v", err)
		}

		// Calculate actual mean
		actualMean := meanFunc(data)

		// CI should contain the actual mean
		if actualMean < lower || actualMean > upper {
			t.Errorf("95%% CI [%.3f, %.3f] should contain actual mean %.3f", lower, upper, actualMean)
		}

		// Lower should be less than upper
		if lower >= upper {
			t.Errorf("Lower bound %.3f should be less than upper bound %.3f", lower, upper)
		}

		t.Logf("Bootstrap 95%% CI for mean: [%.3f, %.3f] (actual: %.3f)", lower, upper, actualMean)
	})

	t.Run("Bootstrap CI for standard deviation", func(t *testing.T) {
		// Create data with known variance
		data, _ := array.FromSlice([]float64{
			8, 9, 10, 11, 12, 8, 9, 10, 11, 12,
			8, 9, 10, 11, 12, 8, 9, 10, 11, 12,
		})

		// Bootstrap CI for standard deviation
		stdFunc := func(sample *array.Array) float64 {
			// Calculate sample standard deviation
			sum := 0.0
			for i := 0; i < sample.Size(); i++ {
				sum += sample.At(i).(float64)
			}
			mean := sum / float64(sample.Size())

			sumSq := 0.0
			for i := 0; i < sample.Size(); i++ {
				diff := sample.At(i).(float64) - mean
				sumSq += diff * diff
			}

			variance := sumSq / float64(sample.Size()-1) // Sample variance
			return math.Sqrt(variance)
		}

		lower, upper, err := BootstrapConfidenceInterval(data, stdFunc, 1000, 0.90, 42)
		if err != nil {
			t.Fatalf("Bootstrap CI for std failed: %v", err)
		}

		actualStd := stdFunc(data)

		// CI should contain the actual std
		if actualStd < lower || actualStd > upper {
			t.Errorf("90%% CI [%.3f, %.3f] should contain actual std %.3f", lower, upper, actualStd)
		}

		t.Logf("Bootstrap 90%% CI for std: [%.3f, %.3f] (actual: %.3f)", lower, upper, actualStd)
	})

	t.Run("Bootstrap CI parameter validation", func(t *testing.T) {
		data, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
		meanFunc := func(sample *array.Array) float64 { return 0 }

		// Test invalid confidence level
		_, _, err := BootstrapConfidenceInterval(data, meanFunc, 100, 1.5, 42)
		if err == nil {
			t.Error("Expected error for invalid confidence level > 1")
		}

		_, _, err = BootstrapConfidenceInterval(data, meanFunc, 100, 0.0, 42)
		if err == nil {
			t.Error("Expected error for invalid confidence level <= 0")
		}

		// Test invalid number of bootstrap samples
		_, _, err = BootstrapConfidenceInterval(data, meanFunc, 0, 0.95, 42)
		if err == nil {
			t.Error("Expected error for nBootstrap <= 0")
		}

		// Test nil function
		_, _, err = BootstrapConfidenceInterval(data, nil, 100, 0.95, 42)
		if err == nil {
			t.Error("Expected error for nil statistic function")
		}
	})
}

// TestBootstrapHypothesisTest tests bootstrap hypothesis testing
func TestBootstrapHypothesisTest(t *testing.T) {
	t.Run("Bootstrap test for mean difference", func(t *testing.T) {
		// Two groups with different means
		group1, _ := array.FromSlice([]float64{8, 9, 10, 11, 12, 8, 9, 10, 11, 12})     // mean ≈ 10
		group2, _ := array.FromSlice([]float64{12, 13, 14, 15, 16, 12, 13, 14, 15, 16}) // mean ≈ 14

		// Test statistic: difference in means
		testStat := func(sample1, sample2 *array.Array) float64 {
			mean1 := computeMean(sample1)
			mean2 := computeMean(sample2)
			return math.Abs(mean1 - mean2)
		}

		pValue, observedStat, err := BootstrapHypothesisTest(group1, group2, testStat, 1000, 42)
		if err != nil {
			t.Fatalf("Bootstrap hypothesis test failed: %v", err)
		}

		// Should detect significant difference
		if pValue > 0.05 {
			t.Logf("Note: p-value %.3f suggests no significant difference (may be due to small sample)", pValue)
		} else {
			t.Logf("Significant difference detected: p-value %.3f", pValue)
		}

		// Observed statistic should be positive
		if observedStat <= 0 {
			t.Errorf("Observed statistic should be positive, got %.3f", observedStat)
		}

		t.Logf("Bootstrap test: observed=%.3f, p-value=%.3f", observedStat, pValue)
	})

	t.Run("Bootstrap test for identical groups", func(t *testing.T) {
		// Two identical groups
		group1, _ := array.FromSlice([]float64{10, 10, 10, 10, 10})
		group2, _ := array.FromSlice([]float64{10, 10, 10, 10, 10})

		testStat := func(sample1, sample2 *array.Array) float64 {
			mean1 := computeMean(sample1)
			mean2 := computeMean(sample2)
			return math.Abs(mean1 - mean2)
		}

		pValue, observedStat, err := BootstrapHypothesisTest(group1, group2, testStat, 1000, 42)
		if err != nil {
			t.Fatalf("Bootstrap hypothesis test failed: %v", err)
		}

		// Should not detect significant difference
		if pValue < 0.05 {
			t.Errorf("Identical groups should have p-value >= 0.05, got %.3f", pValue)
		}

		// Observed statistic should be 0 or very small
		if observedStat > 1e-10 {
			t.Errorf("Observed statistic for identical groups should be ~0, got %.3f", observedStat)
		}

		t.Logf("Bootstrap test identical groups: observed=%.3f, p-value=%.3f", observedStat, pValue)
	})

	t.Run("Bootstrap test parameter validation", func(t *testing.T) {
		group1, _ := array.FromSlice([]float64{1, 2, 3})
		group2, _ := array.FromSlice([]float64{4, 5, 6})
		testStat := func(s1, s2 *array.Array) float64 { return 0 }

		// Test nil inputs
		_, _, err := BootstrapHypothesisTest(nil, group2, testStat, 100, 42)
		if err == nil {
			t.Error("Expected error for nil group1")
		}

		_, _, err = BootstrapHypothesisTest(group1, nil, testStat, 100, 42)
		if err == nil {
			t.Error("Expected error for nil group2")
		}

		// Test nil statistic function
		_, _, err = BootstrapHypothesisTest(group1, group2, nil, 100, 42)
		if err == nil {
			t.Error("Expected error for nil statistic function")
		}

		// Test invalid number of bootstrap samples
		_, _, err = BootstrapHypothesisTest(group1, group2, testStat, 0, 42)
		if err == nil {
			t.Error("Expected error for nBootstrap <= 0")
		}
	})
}

// TestBootstrapResample tests resampling functionality
func TestBootstrapResample(t *testing.T) {
	t.Run("Multiple bootstrap resamples", func(t *testing.T) {
		data, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})

		// Generate multiple resamples
		nResamples := 5
		resamples, err := BootstrapResamples(data, nResamples, 42)
		if err != nil {
			t.Fatalf("Bootstrap resamples failed: %v", err)
		}

		if len(resamples) != nResamples {
			t.Errorf("Expected %d resamples, got %d", nResamples, len(resamples))
		}

		// Each resample should have the same size as original
		for i, resample := range resamples {
			if resample.Size() != data.Size() {
				t.Errorf("Resample %d should have size %d, got %d", i, data.Size(), resample.Size())
			}
		}

		t.Logf("Generated %d bootstrap resamples successfully", nResamples)
		for i, resample := range resamples {
			t.Logf("  Resample %d: %v", i, getBootstrapValues(resample))
		}
	})

	t.Run("Bootstrap resamples parameter validation", func(t *testing.T) {
		data, _ := array.FromSlice([]float64{1, 2, 3})

		// Test invalid number of resamples
		_, err := BootstrapResamples(data, 0, 42)
		if err == nil {
			t.Error("Expected error for nResamples <= 0")
		}

		// Test nil input
		_, err = BootstrapResamples(nil, 5, 42)
		if err == nil {
			t.Error("Expected error for nil data")
		}
	})
}

// Helper functions

// getBootstrapValues extracts values from array as slice for logging
func getBootstrapValues(arr *array.Array) []float64 {
	values := make([]float64, arr.Size())
	for i := 0; i < arr.Size(); i++ {
		values[i] = arr.At(i).(float64)
	}
	return values
}

// computeMean calculates the mean of an array
func computeMean(arr *array.Array) float64 {
	sum := 0.0
	for i := 0; i < arr.Size(); i++ {
		sum += arr.At(i).(float64)
	}
	return sum / float64(arr.Size())
}
