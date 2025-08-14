package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestSimpleImputer tests the basic imputation functionality
func TestSimpleImputer(t *testing.T) {
	t.Run("Mean imputation", func(t *testing.T) {
		// Create data with missing values (NaN)
		data := array.Empty(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(4.0, 1, 1) // Missing value at [1,0]
		data.Set(3.0, 2, 0)
		data.Set(math.NaN(), 2, 1) // Missing value at [2,1]
		data.Set(5.0, 3, 0)
		data.Set(8.0, 3, 1)

		imputer := NewSimpleImputer() // Default is mean strategy
		imputed, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("Mean imputation failed: %v", err)
		}

		// Check shape is preserved
		if imputed.Shape()[0] != 4 || imputed.Shape()[1] != 2 {
			t.Errorf("Expected shape [4, 2], got %v", imputed.Shape())
		}

		// Feature 0: mean of [1, 3, 5] = 3.0
		// Feature 1: mean of [2, 4, 8] = 4.67 (approximately)
		expectedCol0Mean := (1.0 + 3.0 + 5.0) / 3.0 // = 3.0
		expectedCol1Mean := (2.0 + 4.0 + 8.0) / 3.0 // = 4.67

		// Check statistics
		stats := imputer.GetStatistics()
		if len(stats) != 2 {
			t.Errorf("Expected 2 statistics, got %d", len(stats))
		}
		if math.Abs(stats[0]-expectedCol0Mean) > 1e-10 {
			t.Errorf("Column 0 mean: expected %.6f, got %.6f", expectedCol0Mean, stats[0])
		}
		if math.Abs(stats[1]-expectedCol1Mean) > 1e-10 {
			t.Errorf("Column 1 mean: expected %.6f, got %.6f", expectedCol1Mean, stats[1])
		}

		// Check imputed values
		imputedValue00 := imputed.At(1, 0).(float64) // Should be mean of column 0
		imputedValue21 := imputed.At(2, 1).(float64) // Should be mean of column 1

		if math.Abs(imputedValue00-expectedCol0Mean) > 1e-10 {
			t.Errorf("Imputed [1,0]: expected %.6f, got %.6f", expectedCol0Mean, imputedValue00)
		}
		if math.Abs(imputedValue21-expectedCol1Mean) > 1e-10 {
			t.Errorf("Imputed [2,1]: expected %.6f, got %.6f", expectedCol1Mean, imputedValue21)
		}

		// Check non-missing values are preserved
		if imputed.At(0, 0).(float64) != 1.0 {
			t.Errorf("Non-missing value [0,0] should be preserved: got %.6f", imputed.At(0, 0).(float64))
		}

		t.Logf("Mean imputation: column means [%.3f, %.3f]", stats[0], stats[1])
	})

	t.Run("Median imputation", func(t *testing.T) {
		// Create data with missing values
		data := array.Empty(internal.Shape{5, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(10.0, 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(20.0, 1, 1) // Missing value
		data.Set(3.0, 2, 0)
		data.Set(30.0, 2, 1)
		data.Set(7.0, 3, 0)
		data.Set(math.NaN(), 3, 1) // Missing value
		data.Set(5.0, 4, 0)
		data.Set(50.0, 4, 1)

		imputer := NewSimpleImputerWithStrategy(ImputeMedian)
		_, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("Median imputation failed: %v", err)
		}

		// Feature 0: median of [1, 3, 7, 5] = median of [1, 3, 5, 7] = (3+5)/2 = 4.0
		// Feature 1: median of [10, 20, 30, 50] = (20+30)/2 = 25.0
		expectedCol0Median := 4.0  // median of [1, 3, 5, 7]
		expectedCol1Median := 25.0 // median of [10, 20, 30, 50]

		stats := imputer.GetStatistics()
		if math.Abs(stats[0]-expectedCol0Median) > 1e-10 {
			t.Errorf("Column 0 median: expected %.1f, got %.6f", expectedCol0Median, stats[0])
		}
		if math.Abs(stats[1]-expectedCol1Median) > 1e-10 {
			t.Errorf("Column 1 median: expected %.1f, got %.6f", expectedCol1Median, stats[1])
		}

		// Check strategy
		if imputer.GetStrategy() != ImputeMedian {
			t.Errorf("Expected median strategy, got %v", imputer.GetStrategy())
		}

		t.Logf("Median imputation: column medians [%.1f, %.1f]", stats[0], stats[1])
	})

	t.Run("Most frequent imputation", func(t *testing.T) {
		// Create data with repeated values and missing values
		data := array.Empty(internal.Shape{6, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(100.0, 0, 1)
		data.Set(2.0, 1, 0)
		data.Set(200.0, 1, 1)
		data.Set(1.0, 2, 0)
		data.Set(100.0, 2, 1) // 1.0 and 100.0 appear most frequently
		data.Set(math.NaN(), 3, 0)
		data.Set(math.NaN(), 3, 1) // Missing values
		data.Set(1.0, 4, 0)
		data.Set(300.0, 4, 1) // 1.0 appears 3 times (most frequent)
		data.Set(3.0, 5, 0)
		data.Set(100.0, 5, 1) // 100.0 appears 3 times (most frequent)

		imputer := NewSimpleImputerWithStrategy(ImputeMostFrequent)
		imputed, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("Most frequent imputation failed: %v", err)
		}

		// Feature 0: 1.0 appears 3 times (most frequent)
		// Feature 1: 100.0 appears 3 times (most frequent)
		expectedCol0MostFreq := 1.0
		expectedCol1MostFreq := 100.0

		stats := imputer.GetStatistics()
		if math.Abs(stats[0]-expectedCol0MostFreq) > 1e-10 {
			t.Errorf("Column 0 most frequent: expected %.1f, got %.6f", expectedCol0MostFreq, stats[0])
		}
		if math.Abs(stats[1]-expectedCol1MostFreq) > 1e-10 {
			t.Errorf("Column 1 most frequent: expected %.1f, got %.6f", expectedCol1MostFreq, stats[1])
		}

		// Check imputed values
		imputedValue30 := imputed.At(3, 0).(float64)
		imputedValue31 := imputed.At(3, 1).(float64)

		if math.Abs(imputedValue30-expectedCol0MostFreq) > 1e-10 {
			t.Errorf("Imputed [3,0]: expected %.1f, got %.6f", expectedCol0MostFreq, imputedValue30)
		}
		if math.Abs(imputedValue31-expectedCol1MostFreq) > 1e-10 {
			t.Errorf("Imputed [3,1]: expected %.1f, got %.6f", expectedCol1MostFreq, imputedValue31)
		}

		t.Logf("Most frequent imputation: most frequent values [%.1f, %.1f]", stats[0], stats[1])
	})

	t.Run("Constant imputation", func(t *testing.T) {
		// Create data with missing values
		data := array.Empty(internal.Shape{3, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(math.NaN(), 1, 1) // Missing values
		data.Set(5.0, 2, 0)
		data.Set(6.0, 2, 1)

		fillValue := -999.0
		imputer := NewSimpleImputerWithOptions(ImputeConstant, fillValue, math.NaN())
		imputed, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("Constant imputation failed: %v", err)
		}

		// All missing values should be replaced with fillValue
		imputedValue10 := imputed.At(1, 0).(float64)
		imputedValue11 := imputed.At(1, 1).(float64)

		if math.Abs(imputedValue10-fillValue) > 1e-10 {
			t.Errorf("Imputed [1,0]: expected %.1f, got %.6f", fillValue, imputedValue10)
		}
		if math.Abs(imputedValue11-fillValue) > 1e-10 {
			t.Errorf("Imputed [1,1]: expected %.1f, got %.6f", fillValue, imputedValue11)
		}

		// Check fill value
		if math.Abs(imputer.GetFillValue()-fillValue) > 1e-10 {
			t.Errorf("Fill value: expected %.1f, got %.6f", fillValue, imputer.GetFillValue())
		}

		t.Logf("Constant imputation: fill value %.1f", fillValue)
	})

	t.Run("Custom missing value marker", func(t *testing.T) {
		// Use -999 as missing value marker instead of NaN
		missingMarker := -999.0
		data := array.Empty(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(missingMarker, 1, 0)
		data.Set(4.0, 1, 1) // Missing value
		data.Set(3.0, 2, 0)
		data.Set(missingMarker, 2, 1) // Missing value
		data.Set(5.0, 3, 0)
		data.Set(8.0, 3, 1)

		imputer := NewSimpleImputerWithOptions(ImputeMean, 0.0, missingMarker)
		_, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("Custom missing value imputation failed: %v", err)
		}

		// Feature 0: mean of [1, 3, 5] = 3.0
		// Feature 1: mean of [2, 4, 8] = 4.67
		expectedCol0Mean := 3.0
		expectedCol1Mean := (2.0 + 4.0 + 8.0) / 3.0

		stats := imputer.GetStatistics()
		if math.Abs(stats[0]-expectedCol0Mean) > 1e-10 {
			t.Errorf("Column 0 mean: expected %.1f, got %.6f", expectedCol0Mean, stats[0])
		}
		if math.Abs(stats[1]-expectedCol1Mean) > 1e-10 {
			t.Errorf("Column 1 mean: expected %.6f, got %.6f", expectedCol1Mean, stats[1])
		}

		// Check missing value marker
		if math.Abs(imputer.GetMissingValues()-missingMarker) > 1e-10 {
			t.Errorf("Missing value marker: expected %.1f, got %.6f", missingMarker, imputer.GetMissingValues())
		}

		t.Logf("Custom missing marker (%.1f) imputation successful", missingMarker)
	})

	t.Run("Parameter validation", func(t *testing.T) {
		// Test nil data
		imputer := NewSimpleImputer()
		err := imputer.Fit(nil)
		if err == nil {
			t.Error("Expected error for nil data")
		}

		// Test wrong data dimensions
		data1D := array.Ones(internal.Shape{5}, internal.Float64)
		err = imputer.Fit(data1D)
		if err == nil {
			t.Error("Expected error for 1D data")
		}

		// Test transform before fit
		validData := array.Ones(internal.Shape{3, 2}, internal.Float64)
		_, err = imputer.Transform(validData)
		if err == nil {
			t.Error("Expected error for Transform before Fit")
		}

		// Test empty data
		emptyData := array.Empty(internal.Shape{0, 2}, internal.Float64)
		err = imputer.Fit(emptyData)
		if err == nil {
			t.Error("Expected error for empty data")
		}

		// Test mismatched features in transform
		trainData := array.Ones(internal.Shape{3, 2}, internal.Float64)
		testData := array.Ones(internal.Shape{3, 3}, internal.Float64)

		err = imputer.Fit(trainData)
		if err != nil {
			t.Fatalf("Unexpected fit error: %v", err)
		}

		_, err = imputer.Transform(testData)
		if err == nil {
			t.Error("Expected error for mismatched feature count")
		}
	})
}

// TestImputerEdgeCases tests edge cases and boundary conditions
func TestImputerEdgeCases(t *testing.T) {
	t.Run("All values missing in a column", func(t *testing.T) {
		data := array.Empty(internal.Shape{3, 2}, internal.Float64)
		data.Set(math.NaN(), 0, 0)
		data.Set(1.0, 0, 1) // All values missing in column 0
		data.Set(math.NaN(), 1, 0)
		data.Set(2.0, 1, 1)
		data.Set(math.NaN(), 2, 0)
		data.Set(3.0, 2, 1)

		imputer := NewSimpleImputer()
		_, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("All-missing column imputation failed: %v", err)
		}

		// Column with all missing values should be imputed with 0.0 (default)
		stats := imputer.GetStatistics()
		if stats[0] != 0.0 {
			t.Errorf("All-missing column statistic should be 0.0, got %.6f", stats[0])
		}

		// Column 1 should have correct mean
		expectedMean := (1.0 + 2.0 + 3.0) / 3.0
		if math.Abs(stats[1]-expectedMean) > 1e-10 {
			t.Errorf("Column 1 mean: expected %.6f, got %.6f", expectedMean, stats[1])
		}

		t.Logf("All-missing column handled: stats [%.1f, %.3f]", stats[0], stats[1])
	})

	t.Run("No missing values", func(t *testing.T) {
		// Data without any missing values
		data := array.Empty(internal.Shape{3, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.0, 1, 0)
		data.Set(4.0, 1, 1)
		data.Set(5.0, 2, 0)
		data.Set(6.0, 2, 1)

		imputer := NewSimpleImputer()
		imputed, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("No missing values imputation failed: %v", err)
		}

		// Data should be unchanged
		for i := 0; i < data.Shape()[0]; i++ {
			for j := 0; j < data.Shape()[1]; j++ {
				original := data.At(i, j).(float64)
				imputed_val := imputed.At(i, j).(float64)
				if math.Abs(original-imputed_val) > 1e-15 {
					t.Errorf("Data changed at [%d,%d]: %.6f -> %.6f", i, j, original, imputed_val)
				}
			}
		}

		t.Logf("No missing values: data preserved correctly")
	})

	t.Run("Single value per column", func(t *testing.T) {
		// Only one non-missing value per column
		data := array.Empty(internal.Shape{3, 2}, internal.Float64)
		data.Set(42.0, 0, 0)
		data.Set(math.NaN(), 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(99.0, 1, 1)
		data.Set(math.NaN(), 2, 0)
		data.Set(math.NaN(), 2, 1)

		imputer := NewSimpleImputer()
		imputed, err := imputer.FitTransform(data)
		if err != nil {
			t.Fatalf("Single value imputation failed: %v", err)
		}

		// Missing values should be replaced with the single available value
		stats := imputer.GetStatistics()
		if stats[0] != 42.0 {
			t.Errorf("Column 0: expected 42.0, got %.6f", stats[0])
		}
		if stats[1] != 99.0 {
			t.Errorf("Column 1: expected 99.0, got %.6f", stats[1])
		}

		// Check all values in column 0 are 42.0
		for i := 0; i < 3; i++ {
			val := imputed.At(i, 0).(float64)
			if val != 42.0 {
				t.Errorf("Column 0, row %d: expected 42.0, got %.6f", i, val)
			}
		}

		t.Logf("Single value per column handled correctly")
	})

	t.Run("Fit-transform equivalence", func(t *testing.T) {
		data := array.Empty(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(math.NaN(), 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(3.0, 1, 1)
		data.Set(2.0, 2, 0)
		data.Set(4.0, 2, 1)
		data.Set(4.0, 3, 0)
		data.Set(math.NaN(), 3, 1)

		// Method 1: Separate fit and transform
		imputer1 := NewSimpleImputer()
		err := imputer1.Fit(data)
		if err != nil {
			t.Fatalf("Fit failed: %v", err)
		}
		imputed1, err := imputer1.Transform(data)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		// Method 2: FitTransform
		imputer2 := NewSimpleImputer()
		imputed2, err := imputer2.FitTransform(data)
		if err != nil {
			t.Fatalf("FitTransform failed: %v", err)
		}

		// Results should be identical
		if imputed1.Shape()[0] != imputed2.Shape()[0] ||
			imputed1.Shape()[1] != imputed2.Shape()[1] {
			t.Errorf("Shape mismatch: separate [%v] vs fit_transform [%v]",
				imputed1.Shape(), imputed2.Shape())
		}

		for i := 0; i < imputed1.Shape()[0]; i++ {
			for j := 0; j < imputed1.Shape()[1]; j++ {
				val1 := imputed1.At(i, j).(float64)
				val2 := imputed2.At(i, j).(float64)
				if math.Abs(val1-val2) > 1e-15 {
					t.Errorf("Value mismatch at [%d,%d]: separate %.10f vs fit_transform %.10f",
						i, j, val1, val2)
				}
			}
		}

		t.Logf("Fit+Transform equivalent to FitTransform")
	})
}

// TestImputationUtilityFunctions tests utility functions
func TestImputationUtilityFunctions(t *testing.T) {
	t.Run("CountMissingValues", func(t *testing.T) {
		data := array.Empty(internal.Shape{4, 3}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(math.NaN(), 0, 1)
		data.Set(3.0, 0, 2) // 1 missing in col 1
		data.Set(math.NaN(), 1, 0)
		data.Set(2.0, 1, 1)
		data.Set(math.NaN(), 1, 2) // 1 missing in col 0, 1 in col 2
		data.Set(4.0, 2, 0)
		data.Set(math.NaN(), 2, 1)
		data.Set(5.0, 2, 2) // 1 missing in col 1
		data.Set(6.0, 3, 0)
		data.Set(7.0, 3, 1)
		data.Set(8.0, 3, 2) // No missing values

		counts, err := CountMissingValues(data, math.NaN())
		if err != nil {
			t.Fatalf("CountMissingValues failed: %v", err)
		}

		expected := []int{1, 2, 1} // Column 0: 1 missing, Column 1: 2 missing, Column 2: 1 missing
		if len(counts) != len(expected) {
			t.Errorf("Expected %d counts, got %d", len(expected), len(counts))
		}

		for i, expected_count := range expected {
			if counts[i] != expected_count {
				t.Errorf("Column %d: expected %d missing values, got %d", i, expected_count, counts[i])
			}
		}

		t.Logf("Missing value counts: %v", counts)
	})

	t.Run("HasMissingValues", func(t *testing.T) {
		// Data with missing values
		dataWithMissing := array.Empty(internal.Shape{3, 2}, internal.Float64)
		dataWithMissing.Set(1.0, 0, 0)
		dataWithMissing.Set(2.0, 0, 1)
		dataWithMissing.Set(math.NaN(), 1, 0)
		dataWithMissing.Set(4.0, 1, 1)
		dataWithMissing.Set(5.0, 2, 0)
		dataWithMissing.Set(6.0, 2, 1)

		hasMissing, err := HasMissingValues(dataWithMissing, math.NaN())
		if err != nil {
			t.Fatalf("HasMissingValues failed: %v", err)
		}
		if !hasMissing {
			t.Error("Expected to detect missing values")
		}

		// Data without missing values
		dataWithoutMissing := array.Empty(internal.Shape{2, 2}, internal.Float64)
		dataWithoutMissing.Set(1.0, 0, 0)
		dataWithoutMissing.Set(2.0, 0, 1)
		dataWithoutMissing.Set(3.0, 1, 0)
		dataWithoutMissing.Set(4.0, 1, 1)

		hasMissing, err = HasMissingValues(dataWithoutMissing, math.NaN())
		if err != nil {
			t.Fatalf("HasMissingValues failed: %v", err)
		}
		if hasMissing {
			t.Error("Expected no missing values detected")
		}

		// 1D array test
		data1D := array.Empty(internal.Shape{4}, internal.Float64)
		data1D.Set(1.0, 0)
		data1D.Set(math.NaN(), 1)
		data1D.Set(3.0, 2)
		data1D.Set(4.0, 3)

		hasMissing, err = HasMissingValues(data1D, math.NaN())
		if err != nil {
			t.Fatalf("HasMissingValues 1D failed: %v", err)
		}
		if !hasMissing {
			t.Error("Expected to detect missing values in 1D array")
		}

		t.Logf("HasMissingValues working correctly")
	})

	t.Run("Utility function parameter validation", func(t *testing.T) {
		// Test nil array for CountMissingValues
		_, err := CountMissingValues(nil, math.NaN())
		if err == nil {
			t.Error("Expected error for nil array in CountMissingValues")
		}

		// Test nil array for HasMissingValues
		_, err = HasMissingValues(nil, math.NaN())
		if err == nil {
			t.Error("Expected error for nil array in HasMissingValues")
		}

		// Test 1D array for CountMissingValues (should fail)
		data1D := array.Ones(internal.Shape{5}, internal.Float64)
		_, err = CountMissingValues(data1D, math.NaN())
		if err == nil {
			t.Error("Expected error for 1D array in CountMissingValues")
		}

		// Test 3D array for HasMissingValues (should fail)
		data3D := array.Ones(internal.Shape{2, 2, 2}, internal.Float64)
		_, err = HasMissingValues(data3D, math.NaN())
		if err == nil {
			t.Error("Expected error for 3D array in HasMissingValues")
		}
	})
}
