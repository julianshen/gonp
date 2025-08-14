package stats

import (
	"testing"

	"github.com/julianshen/gonp/array"
)

func TestZScoreOutlierDetection(t *testing.T) {
	t.Run("Simple outlier detection", func(t *testing.T) {
		// Create data with clear outliers
		data := []float64{1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10} // 100 is an obvious outlier
		arr, _ := array.FromSlice(data)

		outliers, err := ZScoreOutlierDetection(arr, 2.0) // Standard threshold
		if err != nil {
			t.Fatalf("ZScoreOutlierDetection failed: %v", err)
		}

		// Should detect index 5 (value=100) as outlier
		expectedOutliers := []int{5}
		if len(outliers) != len(expectedOutliers) {
			t.Errorf("Expected %d outliers, got %d", len(expectedOutliers), len(outliers))
		}

		if len(outliers) > 0 && outliers[0] != expectedOutliers[0] {
			t.Errorf("Expected outlier at index %d, got %d", expectedOutliers[0], outliers[0])
		}
	})

	t.Run("No outliers", func(t *testing.T) {
		// Normal distribution data - no outliers
		data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		arr, _ := array.FromSlice(data)

		outliers, err := ZScoreOutlierDetection(arr, 2.0)
		if err != nil {
			t.Fatalf("ZScoreOutlierDetection failed: %v", err)
		}

		if len(outliers) != 0 {
			t.Errorf("Expected no outliers, got %d", len(outliers))
		}
	})

	t.Run("Multiple outliers", func(t *testing.T) {
		// Data with multiple outliers - use threshold that catches both
		data := []float64{-50, 2, 3, 4, 5, 6, 7, 8, 9, 150} // -50 and 150 are outliers
		arr, _ := array.FromSlice(data)

		outliers, err := ZScoreOutlierDetection(arr, 1.0) // Lower threshold
		if err != nil {
			t.Fatalf("ZScoreOutlierDetection failed: %v", err)
		}

		if len(outliers) < 2 {
			t.Errorf("Expected at least 2 outliers, got %d", len(outliers))
		}

		// Should include indices 0 and 9
		hasIndex0 := false
		hasIndex9 := false
		for _, idx := range outliers {
			if idx == 0 {
				hasIndex0 = true
			}
			if idx == 9 {
				hasIndex9 = true
			}
		}

		if !hasIndex0 || !hasIndex9 {
			t.Errorf("Expected outliers at indices 0 and 9, got %v", outliers)
		}
	})
}

func TestIQROutlierDetection(t *testing.T) {
	t.Run("Simple IQR outlier detection", func(t *testing.T) {
		// Data with outliers beyond IQR * 1.5
		data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50} // 50 is an outlier
		arr, _ := array.FromSlice(data)

		outliers, err := IQROutlierDetection(arr, 1.5)
		if err != nil {
			t.Fatalf("IQROutlierDetection failed: %v", err)
		}

		// Should detect the last element (50) as outlier
		if len(outliers) == 0 {
			t.Error("Expected to find outliers")
		}

		if len(outliers) > 0 && outliers[0] != 10 { // Index of value 50
			t.Errorf("Expected outlier at index 10, got %d", outliers[0])
		}
	})

	t.Run("No outliers within IQR bounds", func(t *testing.T) {
		// Normal data within IQR bounds
		data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		arr, _ := array.FromSlice(data)

		outliers, err := IQROutlierDetection(arr, 1.5)
		if err != nil {
			t.Fatalf("IQROutlierDetection failed: %v", err)
		}

		if len(outliers) != 0 {
			t.Errorf("Expected no outliers, got %d", len(outliers))
		}
	})
}

func TestModifiedZScoreOutlierDetection(t *testing.T) {
	t.Run("Modified Z-score with MAD", func(t *testing.T) {
		// Data with outliers - modified Z-score should be more robust
		data := []float64{1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10}
		arr, _ := array.FromSlice(data)

		outliers, err := ModifiedZScoreOutlierDetection(arr, 3.5) // Common threshold for modified Z-score
		if err != nil {
			t.Fatalf("ModifiedZScoreOutlierDetection failed: %v", err)
		}

		// Should detect the outlier (100)
		if len(outliers) == 0 {
			t.Error("Expected to find outliers")
		}

		// Should find index 5 (value=100)
		hasOutlier := false
		for _, idx := range outliers {
			if idx == 5 {
				hasOutlier = true
				break
			}
		}

		if !hasOutlier {
			t.Errorf("Expected outlier at index 5, got %v", outliers)
		}
	})
}

func TestRemoveOutliers(t *testing.T) {
	t.Run("Remove outliers from array", func(t *testing.T) {
		data := []float64{1, 2, 3, 100, 4, 5}
		arr, _ := array.FromSlice(data)

		outlierIndices := []int{3} // Remove the value 100 at index 3
		cleaned, err := RemoveOutliers(arr, outlierIndices)
		if err != nil {
			t.Fatalf("RemoveOutliers failed: %v", err)
		}

		expectedSize := len(data) - len(outlierIndices)
		if cleaned.Size() != expectedSize {
			t.Errorf("Expected size %d, got %d", expectedSize, cleaned.Size())
		}

		// Check that 100 is not in the cleaned data
		for i := 0; i < cleaned.Size(); i++ {
			val := convertToFloat64(cleaned.At(i))
			if val == 100 {
				t.Error("Outlier value 100 was not removed")
			}
		}

		// Check that other values are preserved
		expected := []float64{1, 2, 3, 4, 5}
		for i := 0; i < len(expected); i++ {
			actual := convertToFloat64(cleaned.At(i))
			if actual != expected[i] {
				t.Errorf("At index %d: expected %f, got %f", i, expected[i], actual)
			}
		}
	})

	t.Run("Remove multiple outliers", func(t *testing.T) {
		data := []float64{100, 1, 2, 200, 3, 4, 300}
		arr, _ := array.FromSlice(data)

		outlierIndices := []int{0, 3, 6} // Remove 100, 200, 300
		cleaned, err := RemoveOutliers(arr, outlierIndices)
		if err != nil {
			t.Fatalf("RemoveOutliers failed: %v", err)
		}

		expectedSize := len(data) - len(outlierIndices)
		if cleaned.Size() != expectedSize {
			t.Errorf("Expected size %d, got %d", expectedSize, cleaned.Size())
		}

		// Should contain only [1, 2, 3, 4]
		expected := []float64{1, 2, 3, 4}
		for i := 0; i < len(expected); i++ {
			actual := convertToFloat64(cleaned.At(i))
			if actual != expected[i] {
				t.Errorf("At index %d: expected %f, got %f", i, expected[i], actual)
			}
		}
	})
}

func TestOutlierDetectionParameterValidation(t *testing.T) {
	t.Run("Nil array validation", func(t *testing.T) {
		_, err := ZScoreOutlierDetection(nil, 2.0)
		if err == nil {
			t.Error("Expected error for nil array")
		}

		_, err = IQROutlierDetection(nil, 1.5)
		if err == nil {
			t.Error("Expected error for nil array")
		}

		_, err = ModifiedZScoreOutlierDetection(nil, 3.5)
		if err == nil {
			t.Error("Expected error for nil array")
		}
	})

	t.Run("Invalid threshold validation", func(t *testing.T) {
		data := []float64{1, 2, 3, 4, 5}
		arr, _ := array.FromSlice(data)

		_, err := ZScoreOutlierDetection(arr, 0)
		if err == nil {
			t.Error("Expected error for zero threshold")
		}

		_, err = IQROutlierDetection(arr, -1)
		if err == nil {
			t.Error("Expected error for negative threshold")
		}

		_, err = ModifiedZScoreOutlierDetection(arr, 0)
		if err == nil {
			t.Error("Expected error for zero threshold")
		}
	})
}
