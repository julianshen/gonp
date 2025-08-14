package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestStandardScaler tests StandardScaler implementation
func TestStandardScaler(t *testing.T) {
	t.Run("StandardScaler basic functionality", func(t *testing.T) {
		// Create test data with known mean and std
		data := array.Zeros(internal.Shape{6, 2}, internal.Float64)
		// Feature 0: [1, 2, 3, 4, 5, 6] -> mean=3.5, std≈1.71
		// Feature 1: [10, 20, 30, 40, 50, 60] -> mean=35, std≈18.71
		for i := 0; i < 6; i++ {
			data.Set(float64(i+1), i, 0)
			data.Set(float64((i+1)*10), i, 1)
		}

		// Fit scaler
		scaler := NewStandardScaler()
		err := scaler.Fit(data)
		if err != nil {
			t.Fatalf("StandardScaler fit failed: %v", err)
		}

		// Check fitted parameters
		expectedMean0 := 3.5
		expectedMean1 := 35.0
		if math.Abs(scaler.Mean.At(0).(float64)-expectedMean0) > 1e-10 {
			t.Errorf("Expected mean[0] %.3f, got %.3f", expectedMean0, scaler.Mean.At(0).(float64))
		}
		if math.Abs(scaler.Mean.At(1).(float64)-expectedMean1) > 1e-10 {
			t.Errorf("Expected mean[1] %.3f, got %.3f", expectedMean1, scaler.Mean.At(1).(float64))
		}

		// Transform data
		scaled, err := scaler.Transform(data)
		if err != nil {
			t.Fatalf("StandardScaler transform failed: %v", err)
		}

		// Check that transformed data has zero mean and unit variance
		scaledMean0, scaledStd0 := computeMeanStd(scaled, 0)
		scaledMean1, scaledStd1 := computeMeanStd(scaled, 1)

		if math.Abs(scaledMean0) > 1e-10 {
			t.Errorf("Scaled feature 0 should have mean ≈ 0, got %.6f", scaledMean0)
		}
		if math.Abs(scaledMean1) > 1e-10 {
			t.Errorf("Scaled feature 1 should have mean ≈ 0, got %.6f", scaledMean1)
		}
		if math.Abs(scaledStd0-1.0) > 1e-10 {
			t.Errorf("Scaled feature 0 should have std ≈ 1, got %.6f", scaledStd0)
		}
		if math.Abs(scaledStd1-1.0) > 1e-10 {
			t.Errorf("Scaled feature 1 should have std ≈ 1, got %.6f", scaledStd1)
		}

		t.Logf("StandardScaler: original means [%.3f, %.3f], scaled means [%.6f, %.6f]",
			expectedMean0, expectedMean1, scaledMean0, scaledMean1)
		t.Logf("StandardScaler: scaled stds [%.6f, %.6f]", scaledStd0, scaledStd1)
	})

	t.Run("StandardScaler fit_transform", func(t *testing.T) {
		data := array.Zeros(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(10.0, 0, 1)
		data.Set(2.0, 1, 0)
		data.Set(20.0, 1, 1)
		data.Set(3.0, 2, 0)
		data.Set(30.0, 2, 1)
		data.Set(4.0, 3, 0)
		data.Set(40.0, 3, 1)

		scaler := NewStandardScaler()
		scaled, err := scaler.FitTransform(data)
		if err != nil {
			t.Fatalf("StandardScaler fit_transform failed: %v", err)
		}

		// Should be equivalent to separate fit and transform
		scaler2 := NewStandardScaler()
		scaler2.Fit(data)
		scaled2, _ := scaler2.Transform(data)

		for i := 0; i < scaled.Shape()[0]; i++ {
			for j := 0; j < scaled.Shape()[1]; j++ {
				val1 := scaled.At(i, j).(float64)
				val2 := scaled2.At(i, j).(float64)
				if math.Abs(val1-val2) > 1e-10 {
					t.Errorf("FitTransform and Fit+Transform should be equivalent")
				}
			}
		}

		t.Logf("FitTransform equivalent to Fit+Transform: ✓")
	})

	t.Run("StandardScaler inverse_transform", func(t *testing.T) {
		data := array.Zeros(internal.Shape{3, 2}, internal.Float64)
		data.Set(5.0, 0, 0)
		data.Set(50.0, 0, 1)
		data.Set(10.0, 1, 0)
		data.Set(100.0, 1, 1)
		data.Set(15.0, 2, 0)
		data.Set(150.0, 2, 1)

		scaler := NewStandardScaler()
		scaled, err := scaler.FitTransform(data)
		if err != nil {
			t.Fatalf("FitTransform failed: %v", err)
		}

		// Inverse transform should recover original data
		recovered, err := scaler.InverseTransform(scaled)
		if err != nil {
			t.Fatalf("InverseTransform failed: %v", err)
		}

		for i := 0; i < data.Shape()[0]; i++ {
			for j := 0; j < data.Shape()[1]; j++ {
				original := data.At(i, j).(float64)
				recovered_val := recovered.At(i, j).(float64)
				if math.Abs(original-recovered_val) > 1e-10 {
					t.Errorf("InverseTransform failed: original %.6f, recovered %.6f",
						original, recovered_val)
				}
			}
		}

		t.Logf("InverseTransform successfully recovered original data")
	})

	t.Run("StandardScaler parameter validation", func(t *testing.T) {
		scaler := NewStandardScaler()

		// Test fit with nil data
		err := scaler.Fit(nil)
		if err == nil {
			t.Error("Expected error for nil data in Fit")
		}

		// Test transform before fit
		data := array.Ones(internal.Shape{2, 2}, internal.Float64)
		_, err = scaler.Transform(data)
		if err == nil {
			t.Error("Expected error for Transform before Fit")
		}

		// Fit with valid data
		scaler.Fit(data)

		// Test transform with wrong shape
		wrongData := array.Ones(internal.Shape{2, 3}, internal.Float64)
		_, err = scaler.Transform(wrongData)
		if err == nil {
			t.Error("Expected error for Transform with wrong number of features")
		}
	})
}

// TestMinMaxScaler tests MinMaxScaler implementation
func TestMinMaxScaler(t *testing.T) {
	t.Run("MinMaxScaler basic functionality", func(t *testing.T) {
		// Create test data
		data := array.Zeros(internal.Shape{4, 2}, internal.Float64)
		// Feature 0: [1, 2, 3, 4] -> min=1, max=4, range=3
		// Feature 1: [10, 30, 50, 70] -> min=10, max=70, range=60
		data.Set(1.0, 0, 0)
		data.Set(10.0, 0, 1)
		data.Set(2.0, 1, 0)
		data.Set(30.0, 1, 1)
		data.Set(3.0, 2, 0)
		data.Set(50.0, 2, 1)
		data.Set(4.0, 3, 0)
		data.Set(70.0, 3, 1)

		// Default scaling [0, 1]
		scaler := NewMinMaxScaler(0.0, 1.0)
		scaled, err := scaler.FitTransform(data)
		if err != nil {
			t.Fatalf("MinMaxScaler fit_transform failed: %v", err)
		}

		// Check bounds
		for i := 0; i < scaled.Shape()[0]; i++ {
			for j := 0; j < scaled.Shape()[1]; j++ {
				val := scaled.At(i, j).(float64)
				if val < 0.0 || val > 1.0 {
					t.Errorf("Scaled value %.3f should be in [0, 1]", val)
				}
			}
		}

		// Check that min and max are achieved
		min0, max0 := computeMinMax(scaled, 0)
		min1, max1 := computeMinMax(scaled, 1)

		if math.Abs(min0-0.0) > 1e-10 || math.Abs(max0-1.0) > 1e-10 {
			t.Errorf("Feature 0: expected range [0, 1], got [%.6f, %.6f]", min0, max0)
		}
		if math.Abs(min1-0.0) > 1e-10 || math.Abs(max1-1.0) > 1e-10 {
			t.Errorf("Feature 1: expected range [0, 1], got [%.6f, %.6f]", min1, max1)
		}

		t.Logf("MinMaxScaler [0,1]: ranges [%.6f,%.6f], [%.6f,%.6f]", min0, max0, min1, max1)
	})

	t.Run("MinMaxScaler custom range", func(t *testing.T) {
		data := array.Zeros(internal.Shape{3, 2}, internal.Float64)
		data.Set(0.0, 0, 0)
		data.Set(100.0, 0, 1)
		data.Set(5.0, 1, 0)
		data.Set(200.0, 1, 1)
		data.Set(10.0, 2, 0)
		data.Set(300.0, 2, 1)

		// Scale to [-1, 1]
		scaler := NewMinMaxScaler(-1.0, 1.0)
		scaled, err := scaler.FitTransform(data)
		if err != nil {
			t.Fatalf("MinMaxScaler custom range failed: %v", err)
		}

		min0, max0 := computeMinMax(scaled, 0)
		min1, max1 := computeMinMax(scaled, 1)

		if math.Abs(min0-(-1.0)) > 1e-10 || math.Abs(max0-1.0) > 1e-10 {
			t.Errorf("Feature 0: expected range [-1, 1], got [%.6f, %.6f]", min0, max0)
		}
		if math.Abs(min1-(-1.0)) > 1e-10 || math.Abs(max1-1.0) > 1e-10 {
			t.Errorf("Feature 1: expected range [-1, 1], got [%.6f, %.6f]", min1, max1)
		}

		t.Logf("MinMaxScaler [-1,1]: ranges [%.6f,%.6f], [%.6f,%.6f]", min0, max0, min1, max1)
	})

	t.Run("MinMaxScaler parameter validation", func(t *testing.T) {
		// Test invalid range
		scaler := NewMinMaxScaler(1.0, 0.0) // min > max
		data := array.Ones(internal.Shape{2, 2}, internal.Float64)
		_, err := scaler.FitTransform(data)
		if err == nil {
			t.Error("Expected error for invalid range (min > max)")
		}
	})
}

// TestRobustScaler tests RobustScaler implementation
func TestRobustScaler(t *testing.T) {
	t.Run("RobustScaler basic functionality", func(t *testing.T) {
		// Create data with outliers
		data := array.Zeros(internal.Shape{7, 2}, internal.Float64)
		// Feature 0: [1, 2, 3, 4, 5, 100] -> median=3, Q1=2, Q3=4, IQR=2
		// Feature 1: [10, 20, 30, 40, 50, 1000] -> median=30, Q1=20, Q3=40, IQR=20
		data.Set(1.0, 0, 0)
		data.Set(10.0, 0, 1)
		data.Set(2.0, 1, 0)
		data.Set(20.0, 1, 1)
		data.Set(3.0, 2, 0)
		data.Set(30.0, 2, 1)
		data.Set(4.0, 3, 0)
		data.Set(40.0, 3, 1)
		data.Set(5.0, 4, 0)
		data.Set(50.0, 4, 1)
		data.Set(100.0, 5, 0)
		data.Set(1000.0, 5, 1) // Outliers
		data.Set(3.0, 6, 0)
		data.Set(30.0, 6, 1) // Additional median values

		scaler := NewRobustScaler()
		scaled, err := scaler.FitTransform(data)
		if err != nil {
			t.Fatalf("RobustScaler fit_transform failed: %v", err)
		}

		// Check that median becomes 0
		scaledMedian0 := computeMedian(scaled, 0)
		scaledMedian1 := computeMedian(scaled, 1)

		if math.Abs(scaledMedian0) > 1e-10 {
			t.Errorf("Scaled feature 0 should have median ≈ 0, got %.6f", scaledMedian0)
		}
		if math.Abs(scaledMedian1) > 1e-10 {
			t.Errorf("Scaled feature 1 should have median ≈ 0, got %.6f", scaledMedian1)
		}

		// Outliers should be less extreme than with StandardScaler
		outlierVal0 := scaled.At(5, 0).(float64) // The 100 outlier
		outlierVal1 := scaled.At(5, 1).(float64) // The 1000 outlier

		t.Logf("RobustScaler: scaled medians [%.6f, %.6f]", scaledMedian0, scaledMedian1)
		t.Logf("RobustScaler: outlier values [%.3f, %.3f]", outlierVal0, outlierVal1)

		// Outliers should be present but not as extreme as with standard scaling
		if math.Abs(outlierVal0) > 50 || math.Abs(outlierVal1) > 50 {
			t.Errorf("Outliers seem too extreme: [%.3f, %.3f]", outlierVal0, outlierVal1)
		}
	})

	t.Run("RobustScaler parameter validation", func(t *testing.T) {
		scaler := NewRobustScaler()

		// Test with nil data
		err := scaler.Fit(nil)
		if err == nil {
			t.Error("Expected error for nil data")
		}

		// Test with insufficient data (need at least 2 points for quartiles)
		singlePoint := array.Ones(internal.Shape{1, 2}, internal.Float64)
		err = scaler.Fit(singlePoint)
		if err == nil {
			t.Error("Expected error for insufficient data")
		}
	})
}

// Helper functions for testing

// computeMeanStd computes mean and standard deviation for a specific feature column
func computeMeanStd(data *array.Array, featureIdx int) (float64, float64) {
	n := data.Shape()[0]

	// Compute mean
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += data.At(i, featureIdx).(float64)
	}
	mean := sum / float64(n)

	// Compute standard deviation
	sumSq := 0.0
	for i := 0; i < n; i++ {
		diff := data.At(i, featureIdx).(float64) - mean
		sumSq += diff * diff
	}
	std := math.Sqrt(sumSq / float64(n-1)) // Sample standard deviation

	return mean, std
}

// computeMinMax computes min and max for a specific feature column
func computeMinMax(data *array.Array, featureIdx int) (float64, float64) {
	n := data.Shape()[0]
	min := data.At(0, featureIdx).(float64)
	max := min

	for i := 1; i < n; i++ {
		val := data.At(i, featureIdx).(float64)
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}

	return min, max
}

// computeMedian computes median for a specific feature column
func computeMedian(data *array.Array, featureIdx int) float64 {
	n := data.Shape()[0]
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		values[i] = data.At(i, featureIdx).(float64)
	}

	// Simple bubble sort for small arrays
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if values[j] > values[j+1] {
				values[j], values[j+1] = values[j+1], values[j]
			}
		}
	}

	if n%2 == 0 {
		return (values[n/2-1] + values[n/2]) / 2.0
	}
	return values[n/2]
}
