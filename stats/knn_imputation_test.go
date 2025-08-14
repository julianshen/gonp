package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestKNNImputer tests the KNN imputation implementation using TDD
func TestKNNImputer(t *testing.T) {
	t.Run("Basic KNN imputation with k=3", func(t *testing.T) {
		// Red phase: Write a failing test first
		// Create simple 2D data where KNN imputation should be predictable
		data := array.Empty(internal.Shape{5, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(1.0, 0, 1) // [1, 1]
		data.Set(2.0, 1, 0)
		data.Set(2.0, 1, 1) // [2, 2]
		data.Set(math.NaN(), 2, 0)
		data.Set(3.0, 2, 1) // [?, 3] - missing value to impute
		data.Set(4.0, 3, 0)
		data.Set(4.0, 3, 1) // [4, 4]
		data.Set(5.0, 4, 0)
		data.Set(5.0, 4, 1) // [5, 5]

		// KNN imputation should find the 3 nearest neighbors for sample with missing value
		// For sample [?, 3], the nearest neighbors by Euclidean distance in known dimensions should be:
		// [2, 2] (distance=1), [1, 1] (distance=2), [4, 4] (distance=1)
		// Expected imputed value for first feature: mean of [2, 1, 4] = 2.33 (approximately)

		imputer := NewKNNImputer(3) // k=3 neighbors
		imputed, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("KNN imputation failed: %v", err)
		}

		// Check that the missing value was imputed
		imputedValue := imputed.At(2, 0).(float64)
		if math.IsNaN(imputedValue) {
			t.Error("Expected missing value to be imputed, but got NaN")
		}

		// The imputed value should be reasonable (between 2 and 4 given the neighbors)
		if imputedValue < 1.5 || imputedValue > 4.5 {
			t.Errorf("Imputed value %.3f seems unreasonable for neighbors", imputedValue)
		}

		// Check that non-missing values are preserved
		if imputed.At(0, 0).(float64) != 1.0 {
			t.Error("Non-missing values should be preserved")
		}

		t.Logf("KNN imputed missing value: %.3f", imputedValue)
	})

	t.Run("KNN imputation with different k values", func(t *testing.T) {
		// Create data where we can predict the expected behavior
		data := array.Empty(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(10.0, 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(20.0, 1, 1) // Missing value
		data.Set(3.0, 2, 0)
		data.Set(30.0, 2, 1)
		data.Set(5.0, 3, 0)
		data.Set(50.0, 3, 1)

		// Test k=1 (should use only nearest neighbor)
		imputer1 := NewKNNImputer(1)
		imputed1, err := imputer1.FitTransform(data)
		if err != nil {
			t.Fatalf("KNN imputation with k=1 failed: %v", err)
		}

		// Test k=2 (should use 2 nearest neighbors)
		imputer2 := NewKNNImputer(2)
		imputed2, err := imputer2.FitTransform(data)
		if err != nil {
			t.Fatalf("KNN imputation with k=2 failed: %v", err)
		}

		imputedK1 := imputed1.At(1, 0).(float64)
		imputedK2 := imputed2.At(1, 0).(float64)

		// Results should be different (k=1 vs k=2 should give different results)
		if math.Abs(imputedK1-imputedK2) < 1e-10 {
			t.Error("Expected different results for k=1 vs k=2")
		}

		t.Logf("KNN k=1: %.3f, k=2: %.3f", imputedK1, imputedK2)
	})

	t.Run("KNN imputation with multiple missing values", func(t *testing.T) {
		// Test case with multiple missing values in the same sample
		data := array.Empty(internal.Shape{4, 3}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.0, 0, 2)
		data.Set(math.NaN(), 1, 0)
		data.Set(math.NaN(), 1, 1)
		data.Set(6.0, 1, 2) // Two missing values
		data.Set(7.0, 2, 0)
		data.Set(8.0, 2, 1)
		data.Set(9.0, 2, 2)
		data.Set(4.0, 3, 0)
		data.Set(5.0, 3, 1)
		data.Set(12.0, 3, 2)

		imputer := NewKNNImputer(2)
		imputed, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("KNN imputation with multiple missing values failed: %v", err)
		}

		// Both missing values should be imputed
		imputed10 := imputed.At(1, 0).(float64)
		imputed11 := imputed.At(1, 1).(float64)

		if math.IsNaN(imputed10) || math.IsNaN(imputed11) {
			t.Error("Expected both missing values to be imputed")
		}

		t.Logf("Multiple missing values imputed: [%.3f, %.3f]", imputed10, imputed11)
	})

	t.Run("KNN imputation parameter validation", func(t *testing.T) {
		// Test invalid k values
		imputer := NewKNNImputer(0) // k=0 should be invalid
		data := array.Ones(internal.Shape{3, 2}, internal.Float64)
		_, err := imputer.FitTransform(data)
		if err == nil {
			t.Error("Expected error for k=0")
		}

		// Test k larger than available samples
		imputer = NewKNNImputer(10)
		smallData := array.Ones(internal.Shape{3, 2}, internal.Float64)
		smallData.Set(math.NaN(), 1, 0) // One missing value
		_, err = imputer.FitTransform(smallData)
		if err == nil {
			t.Error("Expected error for k > available samples")
		}

		// Test nil data
		imputer = NewKNNImputer(2)
		_, err = imputer.FitTransform(nil)
		if err == nil {
			t.Error("Expected error for nil data")
		}
	})

	t.Run("KNN imputation distance metrics", func(t *testing.T) {
		// Test different distance metrics (when implemented)
		data := array.Empty(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(1.0, 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(2.0, 1, 1)
		data.Set(3.0, 2, 0)
		data.Set(3.0, 2, 1)
		data.Set(4.0, 3, 0)
		data.Set(4.0, 3, 1)

		// Test Euclidean distance (default)
		euclideanImputer := NewKNNImputerWithDistance(2, "euclidean")
		euclideanResult, err := euclideanImputer.FitTransform(data)
		if err != nil {
			t.Fatalf("Euclidean KNN imputation failed: %v", err)
		}

		// Test Manhattan distance
		manhattanImputer := NewKNNImputerWithDistance(2, "manhattan")
		manhattanResult, err := manhattanImputer.FitTransform(data)
		if err != nil {
			t.Fatalf("Manhattan KNN imputation failed: %v", err)
		}

		euclideanValue := euclideanResult.At(1, 0).(float64)
		manhattanValue := manhattanResult.At(1, 0).(float64)

		// Results might be different depending on distance metric
		t.Logf("Euclidean: %.3f, Manhattan: %.3f", euclideanValue, manhattanValue)
	})
}

// TestKNNImputerEdgeCases tests edge cases and boundary conditions
func TestKNNImputerEdgeCases(t *testing.T) {
	t.Run("All values missing in a sample", func(t *testing.T) {
		// Sample with all values missing - should this be handled?
		data := array.Empty(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(math.NaN(), 1, 1) // All missing
		data.Set(3.0, 2, 0)
		data.Set(4.0, 2, 1)
		data.Set(5.0, 3, 0)
		data.Set(6.0, 3, 1)

		imputer := NewKNNImputer(2)
		imputed, err := imputer.FitTransform(data)

		// This might be an error case or might use global means
		// The behavior should be documented and tested
		if err != nil {
			t.Logf("All missing values in sample handled with error: %v", err)
		} else {
			imputed10 := imputed.At(1, 0).(float64)
			imputed11 := imputed.At(1, 1).(float64)
			t.Logf("All missing values imputed: [%.3f, %.3f]", imputed10, imputed11)
		}
	})

	t.Run("No missing values", func(t *testing.T) {
		// Data without any missing values - should return unchanged
		data := array.Empty(internal.Shape{3, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.0, 1, 0)
		data.Set(4.0, 1, 1)
		data.Set(5.0, 2, 0)
		data.Set(6.0, 2, 1)

		imputer := NewKNNImputer(2)
		imputed, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("KNN imputation on complete data failed: %v", err)
		}

		// Data should be unchanged
		for i := 0; i < data.Shape()[0]; i++ {
			for j := 0; j < data.Shape()[1]; j++ {
				original := data.At(i, j).(float64)
				result := imputed.At(i, j).(float64)
				if math.Abs(original-result) > 1e-15 {
					t.Errorf("Data changed at [%d,%d]: %.6f -> %.6f", i, j, original, result)
				}
			}
		}

		t.Logf("Complete data preserved correctly")
	})

	t.Run("Single feature with missing values", func(t *testing.T) {
		// 1D case - KNN should still work
		data := array.Empty(internal.Shape{4, 1}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(math.NaN(), 1, 0) // Missing
		data.Set(3.0, 2, 0)
		data.Set(5.0, 3, 0)

		imputer := NewKNNImputer(2)
		imputed, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("Single feature KNN imputation failed: %v", err)
		}

		imputedValue := imputed.At(1, 0).(float64)
		if math.IsNaN(imputedValue) {
			t.Error("Expected single feature missing value to be imputed")
		}

		// Should be somewhere between 1 and 3 (nearest neighbors)
		if imputedValue < 1.0 || imputedValue > 3.0 {
			t.Errorf("Single feature imputed value %.3f outside expected range", imputedValue)
		}

		t.Logf("Single feature imputed: %.3f", imputedValue)
	})
}
