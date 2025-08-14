package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestPolynomialFeatures tests the polynomial feature generation implementation
func TestPolynomialFeatures(t *testing.T) {
	t.Run("Simple polynomial features degree 2", func(t *testing.T) {
		// Create simple 2D data
		data := array.Empty(internal.Shape{3, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1) // [1, 2]
		data.Set(2.0, 1, 0)
		data.Set(3.0, 1, 1) // [2, 3]
		data.Set(3.0, 2, 0)
		data.Set(4.0, 2, 1) // [3, 4]

		poly := NewPolynomialFeatures(2)
		transformed, err := poly.FitTransform(data)
		if err != nil {
			t.Fatalf("PolynomialFeatures failed: %v", err)
		}

		// Expected features: [1, x0, x1, x0^2, x0*x1, x1^2]
		// For sample [1, 2]: [1, 1, 2, 1, 2, 4]
		expectedShape := internal.Shape{3, 6}
		if transformed.Shape()[0] != expectedShape[0] || transformed.Shape()[1] != expectedShape[1] {
			t.Errorf("Expected shape %v, got %v", expectedShape, transformed.Shape())
		}

		// Check feature names
		featureNames := poly.GetFeatureNames()
		expectedNames := []string{"1", "x0", "x1", "x0^2", "x0 x1", "x1^2"}
		if len(featureNames) != len(expectedNames) {
			t.Errorf("Expected %d feature names, got %d", len(expectedNames), len(featureNames))
		}

		// Check first sample: [1, 2] -> [1, 1, 2, 1, 2, 4]
		expectedValues := []float64{1, 1, 2, 1, 2, 4}
		for j := 0; j < len(expectedValues); j++ {
			actual := transformed.At(0, j).(float64)
			if math.Abs(actual-expectedValues[j]) > 1e-10 {
				t.Errorf("Sample 0, feature %d: expected %.6f, got %.6f", j, expectedValues[j], actual)
			}
		}

		t.Logf("Feature names: %v", featureNames)
		t.Logf("First sample transformation: [1, 2] -> %v", extractSample(transformed, 0))
	})

	t.Run("Polynomial features without bias", func(t *testing.T) {
		data := array.Empty(internal.Shape{2, 2}, internal.Float64)
		data.Set(2.0, 0, 0)
		data.Set(3.0, 0, 1)
		data.Set(1.0, 1, 0)
		data.Set(4.0, 1, 1)

		poly := NewPolynomialFeaturesWithOptions(2, false, false) // No bias
		transformed, err := poly.FitTransform(data)
		if err != nil {
			t.Fatalf("PolynomialFeatures without bias failed: %v", err)
		}

		// Expected features: [x0, x1, x0^2, x0*x1, x1^2] - no bias term
		expectedShape := internal.Shape{2, 5}
		if transformed.Shape()[0] != expectedShape[0] || transformed.Shape()[1] != expectedShape[1] {
			t.Errorf("Expected shape %v, got %v", expectedShape, transformed.Shape())
		}

		// Check that there's no bias column (no column of all 1s)
		featureNames := poly.GetFeatureNames()
		for _, name := range featureNames {
			if name == "1" {
				t.Error("Expected no bias term, but found bias column")
			}
		}

		t.Logf("Feature names without bias: %v", featureNames)
	})

	t.Run("Interaction-only features", func(t *testing.T) {
		data := array.Empty(internal.Shape{2, 3}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.0, 0, 2)
		data.Set(2.0, 1, 0)
		data.Set(3.0, 1, 1)
		data.Set(4.0, 1, 2)

		poly := NewPolynomialFeaturesWithOptions(2, true, true) // Interaction-only
		transformed, err := poly.FitTransform(data)
		if err != nil {
			t.Fatalf("Interaction-only features failed: %v", err)
		}

		// Expected features: [1, x0, x1, x2, x0*x1, x0*x2, x1*x2] - no pure powers
		featureNames := poly.GetFeatureNames()

		// Check that no feature names contain '^' (pure powers)
		for _, name := range featureNames {
			if name != "1" && len(name) > 2 && name[2] == '^' {
				t.Errorf("Found pure power feature '%s' in interaction-only mode", name)
			}
		}

		// For first sample [1, 2, 3]:
		// Expected: [1, 1, 2, 3, 1*2, 1*3, 2*3] = [1, 1, 2, 3, 2, 3, 6]
		expectedValues := []float64{1, 1, 2, 3, 2, 3, 6}
		if transformed.Shape()[1] != len(expectedValues) {
			t.Errorf("Expected %d features, got %d", len(expectedValues), transformed.Shape()[1])
		}

		for j := 0; j < len(expectedValues); j++ {
			actual := transformed.At(0, j).(float64)
			if math.Abs(actual-expectedValues[j]) > 1e-10 {
				t.Errorf("Sample 0, feature %d: expected %.6f, got %.6f", j, expectedValues[j], actual)
			}
		}

		t.Logf("Interaction-only features: %v", featureNames)
	})

	t.Run("Higher degree polynomial features", func(t *testing.T) {
		data := array.Empty(internal.Shape{2, 2}, internal.Float64)
		data.Set(2.0, 0, 0)
		data.Set(3.0, 0, 1)
		data.Set(1.0, 1, 0)
		data.Set(2.0, 1, 1)

		poly := NewPolynomialFeatures(3) // Degree 3
		transformed, err := poly.FitTransform(data)
		if err != nil {
			t.Fatalf("Degree 3 polynomial features failed: %v", err)
		}

		// Should include terms up to degree 3
		featureNames := poly.GetFeatureNames()

		// Check that we have cubic terms
		hasCubicTerm := false
		for _, name := range featureNames {
			if len(name) >= 4 && name[2:4] == "^3" {
				hasCubicTerm = true
				break
			}
		}
		if !hasCubicTerm {
			t.Error("Expected cubic terms in degree 3 polynomial features")
		}

		// For first sample [2, 3], should include x0^3 = 8 and x1^3 = 27
		found_x0_cubed := false
		found_x1_cubed := false

		for j := 0; j < transformed.Shape()[1]; j++ {
			name := featureNames[j]
			value := transformed.At(0, j).(float64)

			if name == "x0^3" && math.Abs(value-8.0) < 1e-10 {
				found_x0_cubed = true
			}
			if name == "x1^3" && math.Abs(value-27.0) < 1e-10 {
				found_x1_cubed = true
			}
		}

		if !found_x0_cubed {
			t.Error("Expected x0^3 = 8 for sample [2, 3]")
		}
		if !found_x1_cubed {
			t.Error("Expected x1^3 = 27 for sample [2, 3]")
		}

		t.Logf("Degree 3 features: %v", featureNames)
		t.Logf("Sample [2, 3] -> %v", extractSample(transformed, 0))
	})

	t.Run("Parameter validation", func(t *testing.T) {
		// Test invalid degree
		poly := NewPolynomialFeatures(0)
		data := array.Ones(internal.Shape{3, 2}, internal.Float64)
		err := poly.Fit(data)
		if err == nil {
			t.Error("Expected error for degree = 0")
		}

		// Test nil data
		poly = NewPolynomialFeatures(2)
		err = poly.Fit(nil)
		if err == nil {
			t.Error("Expected error for nil data")
		}

		// Test wrong data dimensions
		data1D := array.Ones(internal.Shape{5}, internal.Float64)
		err = poly.Fit(data1D)
		if err == nil {
			t.Error("Expected error for 1D data")
		}

		// Test transform before fit
		poly = NewPolynomialFeatures(2)
		validData := array.Ones(internal.Shape{3, 2}, internal.Float64)
		_, err = poly.Transform(validData)
		if err == nil {
			t.Error("Expected error for Transform before Fit")
		}

		// Test different number of features in transform
		poly = NewPolynomialFeatures(2)
		trainData := array.Ones(internal.Shape{3, 2}, internal.Float64)
		testData := array.Ones(internal.Shape{3, 3}, internal.Float64)

		err = poly.Fit(trainData)
		if err != nil {
			t.Fatalf("Unexpected error in fit: %v", err)
		}

		_, err = poly.Transform(testData)
		if err == nil {
			t.Error("Expected error for mismatched feature count")
		}
	})

	t.Run("Single feature polynomial", func(t *testing.T) {
		// Test with single feature
		data := array.Empty(internal.Shape{3, 1}, internal.Float64)
		data.Set(2.0, 0, 0)
		data.Set(3.0, 1, 0)
		data.Set(4.0, 2, 0)

		poly := NewPolynomialFeatures(3)
		transformed, err := poly.FitTransform(data)
		if err != nil {
			t.Fatalf("Single feature polynomial failed: %v", err)
		}

		// Expected features: [1, x0, x0^2, x0^3]
		// For sample [2]: [1, 2, 4, 8]
		expectedValues := []float64{1, 2, 4, 8}
		if transformed.Shape()[1] != len(expectedValues) {
			t.Errorf("Expected %d features, got %d", len(expectedValues), transformed.Shape()[1])
		}

		for j := 0; j < len(expectedValues); j++ {
			actual := transformed.At(0, j).(float64)
			if math.Abs(actual-expectedValues[j]) > 1e-10 {
				t.Errorf("Feature %d: expected %.6f, got %.6f", j, expectedValues[j], actual)
			}
		}

		featureNames := poly.GetFeatureNames()
		expectedNames := []string{"1", "x0", "x0^2", "x0^3"}
		for i, expected := range expectedNames {
			if i < len(featureNames) && featureNames[i] != expected {
				t.Errorf("Feature name %d: expected '%s', got '%s'", i, expected, featureNames[i])
			}
		}

		t.Logf("Single feature names: %v", featureNames)
	})
}

// TestPolynomialFeaturesEdgeCases tests edge cases and boundary conditions
func TestPolynomialFeaturesEdgeCases(t *testing.T) {
	t.Run("Zero and negative values", func(t *testing.T) {
		data := array.Empty(internal.Shape{3, 2}, internal.Float64)
		data.Set(0.0, 0, 0)
		data.Set(-1.0, 0, 1) // [0, -1]
		data.Set(-2.0, 1, 0)
		data.Set(0.0, 1, 1) // [-2, 0]
		data.Set(-1.0, 2, 0)
		data.Set(-1.0, 2, 1) // [-1, -1]

		poly := NewPolynomialFeatures(2)
		transformed, err := poly.FitTransform(data)
		if err != nil {
			t.Fatalf("Polynomial with zero/negative values failed: %v", err)
		}

		// For sample [0, -1]: [1, 0, -1, 0, 0, 1]
		expectedValues := []float64{1, 0, -1, 0, 0, 1}
		for j := 0; j < len(expectedValues); j++ {
			actual := transformed.At(0, j).(float64)
			if math.Abs(actual-expectedValues[j]) > 1e-10 {
				t.Errorf("Sample [0, -1], feature %d: expected %.6f, got %.6f", j, expectedValues[j], actual)
			}
		}

		// For sample [-1, -1]: [1, -1, -1, 1, 1, 1]
		expectedValues2 := []float64{1, -1, -1, 1, 1, 1}
		for j := 0; j < len(expectedValues2); j++ {
			actual := transformed.At(2, j).(float64)
			if math.Abs(actual-expectedValues2[j]) > 1e-10 {
				t.Errorf("Sample [-1, -1], feature %d: expected %.6f, got %.6f", j, expectedValues2[j], actual)
			}
		}

		t.Logf("Zero/negative values handled correctly")
	})

	t.Run("Large feature values", func(t *testing.T) {
		data := array.Empty(internal.Shape{2, 2}, internal.Float64)
		data.Set(10.0, 0, 0)
		data.Set(10.0, 0, 1) // [10, 10]
		data.Set(5.0, 1, 0)
		data.Set(20.0, 1, 1) // [5, 20]

		poly := NewPolynomialFeatures(2)
		transformed, err := poly.FitTransform(data)
		if err != nil {
			t.Fatalf("Polynomial with large values failed: %v", err)
		}

		// For sample [10, 10]: [1, 10, 10, 100, 100, 100]
		expectedValues := []float64{1, 10, 10, 100, 100, 100}
		for j := 0; j < len(expectedValues); j++ {
			actual := transformed.At(0, j).(float64)
			if math.Abs(actual-expectedValues[j]) > 1e-8 {
				t.Errorf("Large values, feature %d: expected %.1f, got %.1f", j, expectedValues[j], actual)
			}
		}

		t.Logf("Large feature values handled correctly")
	})

	t.Run("Fit-transform equivalence", func(t *testing.T) {
		data := array.Empty(internal.Shape{3, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.0, 1, 0)
		data.Set(4.0, 1, 1)
		data.Set(5.0, 2, 0)
		data.Set(6.0, 2, 1)

		// Method 1: Separate fit and transform
		poly1 := NewPolynomialFeatures(2)
		err := poly1.Fit(data)
		if err != nil {
			t.Fatalf("Fit failed: %v", err)
		}
		transformed1, err := poly1.Transform(data)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		// Method 2: FitTransform
		poly2 := NewPolynomialFeatures(2)
		transformed2, err := poly2.FitTransform(data)
		if err != nil {
			t.Fatalf("FitTransform failed: %v", err)
		}

		// Results should be identical
		if transformed1.Shape()[0] != transformed2.Shape()[0] ||
			transformed1.Shape()[1] != transformed2.Shape()[1] {
			t.Errorf("Shape mismatch: separate [%v] vs fit_transform [%v]",
				transformed1.Shape(), transformed2.Shape())
		}

		for i := 0; i < transformed1.Shape()[0]; i++ {
			for j := 0; j < transformed1.Shape()[1]; j++ {
				val1 := transformed1.At(i, j).(float64)
				val2 := transformed2.At(i, j).(float64)
				if math.Abs(val1-val2) > 1e-12 {
					t.Errorf("Value mismatch at [%d,%d]: separate %.10f vs fit_transform %.10f",
						i, j, val1, val2)
				}
			}
		}

		t.Logf("Fit+Transform equivalent to FitTransform")
	})
}

// Helper function to extract a sample as slice for logging
func extractSample(arr *array.Array, sampleIdx int) []float64 {
	nFeatures := arr.Shape()[1]
	sample := make([]float64, nFeatures)
	for j := 0; j < nFeatures; j++ {
		sample[j] = arr.At(sampleIdx, j).(float64)
	}
	return sample
}
