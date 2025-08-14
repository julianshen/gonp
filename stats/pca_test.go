package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestPrincipalComponentAnalysis tests the class-based PCA implementation
func TestPrincipalComponentAnalysis(t *testing.T) {
	t.Run("PCA basic functionality", func(t *testing.T) {
		// Create test data with known structure
		// 2D data that lies mostly along a diagonal line
		data := array.Zeros(internal.Shape{6, 2}, internal.Float64)
		// Points along diagonal with some noise
		data.Set(1.0, 0, 0)
		data.Set(1.1, 0, 1)
		data.Set(2.0, 1, 0)
		data.Set(2.2, 1, 1)
		data.Set(3.0, 2, 0)
		data.Set(3.1, 2, 1)
		data.Set(4.0, 3, 0)
		data.Set(4.2, 3, 1)
		data.Set(5.0, 4, 0)
		data.Set(5.1, 4, 1)
		data.Set(6.0, 5, 0)
		data.Set(6.2, 5, 1)

		// Create PCA with 2 components
		pca := NewPCA(2)
		err := pca.Fit(data)
		if err != nil {
			t.Fatalf("PCA fit failed: %v", err)
		}

		// Check that PCA was fitted
		if !pca.fitted {
			t.Error("PCA should be marked as fitted")
		}

		// Check components shape
		if pca.Components.Shape()[0] != 2 || pca.Components.Shape()[1] != 2 {
			t.Errorf("Expected components shape [2, 2], got %v", pca.Components.Shape())
		}

		// Check explained variance
		if len(pca.ExplainedVariance) != 2 {
			t.Errorf("Expected 2 explained variance values, got %d", len(pca.ExplainedVariance))
		}

		// First component should capture most variance
		if pca.ExplainedVariance[0] <= pca.ExplainedVariance[1] {
			t.Errorf("First component should explain more variance than second: %.4f vs %.4f",
				pca.ExplainedVariance[0], pca.ExplainedVariance[1])
		}

		// Explained variance ratios should sum to <= 1
		totalRatio := 0.0
		for _, ratio := range pca.ExplainedVarianceRatio {
			totalRatio += ratio
		}
		if totalRatio > 1.0+1e-10 {
			t.Errorf("Explained variance ratios sum to %.6f, should be <= 1", totalRatio)
		}

		t.Logf("PCA fitted successfully")
		t.Logf("Explained variance: [%.4f, %.4f]", pca.ExplainedVariance[0], pca.ExplainedVariance[1])
		t.Logf("Explained variance ratio: [%.4f, %.4f]", pca.ExplainedVarianceRatio[0], pca.ExplainedVarianceRatio[1])
	})

	t.Run("PCA transform and inverse transform", func(t *testing.T) {
		// Create simple 2D data
		data := array.Zeros(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(1.0, 0, 1)
		data.Set(2.0, 1, 0)
		data.Set(2.0, 1, 1)
		data.Set(3.0, 2, 0)
		data.Set(3.0, 2, 1)
		data.Set(4.0, 3, 0)
		data.Set(4.0, 3, 1)

		pca := NewPCA(2)
		err := pca.Fit(data)
		if err != nil {
			t.Fatalf("PCA fit failed: %v", err)
		}

		// Transform data
		transformed, err := pca.Transform(data)
		if err != nil {
			t.Fatalf("PCA transform failed: %v", err)
		}

		// Check transformed data shape
		if transformed.Shape()[0] != 4 || transformed.Shape()[1] != 2 {
			t.Errorf("Expected transformed shape [4, 2], got %v", transformed.Shape())
		}

		// Inverse transform
		reconstructed, err := pca.InverseTransform(transformed)
		if err != nil {
			t.Fatalf("PCA inverse transform failed: %v", err)
		}

		// Check that reconstruction is close to original
		for i := 0; i < data.Shape()[0]; i++ {
			for j := 0; j < data.Shape()[1]; j++ {
				original := data.At(i, j).(float64)
				reconstructed_val := reconstructed.At(i, j).(float64)
				if math.Abs(original-reconstructed_val) > 1e-10 {
					t.Errorf("Reconstruction error at [%d,%d]: original %.6f, reconstructed %.6f",
						i, j, original, reconstructed_val)
				}
			}
		}

		t.Logf("Transform and inverse transform successful")
	})

	t.Run("PCA dimensionality reduction", func(t *testing.T) {
		// Create 3D data where third dimension is just noise
		data := array.Zeros(internal.Shape{5, 3}, internal.Float64)
		// First two dimensions correlated, third is random noise
		data.Set(1.0, 0, 0)
		data.Set(1.1, 0, 1)
		data.Set(0.01, 0, 2)
		data.Set(2.0, 1, 0)
		data.Set(2.2, 1, 1)
		data.Set(-0.01, 1, 2)
		data.Set(3.0, 2, 0)
		data.Set(3.1, 2, 1)
		data.Set(0.02, 2, 2)
		data.Set(4.0, 3, 0)
		data.Set(4.2, 3, 1)
		data.Set(-0.02, 3, 2)
		data.Set(5.0, 4, 0)
		data.Set(5.1, 4, 1)
		data.Set(0.01, 4, 2)

		// Reduce to 2 components
		pca := NewPCA(2)
		transformed, err := pca.FitTransform(data)
		if err != nil {
			t.Fatalf("PCA fit_transform failed: %v", err)
		}

		// Check reduced dimensionality
		if transformed.Shape()[0] != 5 || transformed.Shape()[1] != 2 {
			t.Errorf("Expected reduced shape [5, 2], got %v", transformed.Shape())
		}

		// First two components should capture most of the variance
		if len(pca.ExplainedVarianceRatio) != 2 {
			t.Errorf("Expected 2 explained variance ratios, got %d", len(pca.ExplainedVarianceRatio))
		}

		totalExplained := pca.ExplainedVarianceRatio[0] + pca.ExplainedVarianceRatio[1]
		if totalExplained < 0.9 { // Should capture at least 90% of variance
			t.Errorf("First 2 components should explain >= 90%% of variance, got %.2f%%", totalExplained*100)
		}

		t.Logf("Dimensionality reduction: 3D -> 2D")
		t.Logf("Total explained variance: %.2f%%", totalExplained*100)
	})

	t.Run("PCA parameter validation", func(t *testing.T) {
		// Test invalid n_components
		pca := NewPCA(0)
		data := array.Ones(internal.Shape{3, 2}, internal.Float64)
		err := pca.Fit(data)
		if err == nil {
			t.Error("Expected error for n_components = 0")
		}

		// Test n_components > n_features
		pca = NewPCA(5)
		err = pca.Fit(data)
		if err == nil {
			t.Error("Expected error for n_components > n_features")
		}

		// Test nil data
		pca = NewPCA(2)
		err = pca.Fit(nil)
		if err == nil {
			t.Error("Expected error for nil data")
		}

		// Test wrong data dimensions
		data1D := array.Ones(internal.Shape{5}, internal.Float64)
		err = pca.Fit(data1D)
		if err == nil {
			t.Error("Expected error for 1D data")
		}

		// Test transform before fit
		pca = NewPCA(2)
		validData := array.Ones(internal.Shape{3, 2}, internal.Float64)
		_, err = pca.Transform(validData)
		if err == nil {
			t.Error("Expected error for Transform before Fit")
		}

		// Test insufficient data samples
		insufficientData := array.Ones(internal.Shape{1, 2}, internal.Float64)
		err = pca.Fit(insufficientData)
		if err == nil {
			t.Error("Expected error for insufficient data samples")
		}
	})

	t.Run("PCA with real-world-like data", func(t *testing.T) {
		// Simulate measurements with correlated features
		// Feature 1: height, Feature 2: weight (correlated), Feature 3: age (less correlated)
		data := array.Zeros(internal.Shape{8, 3}, internal.Float64)
		// [height, weight, age] - height and weight are correlated
		data.Set(170.0, 0, 0)
		data.Set(65.0, 0, 1)
		data.Set(25.0, 0, 2)
		data.Set(175.0, 1, 0)
		data.Set(70.0, 1, 1)
		data.Set(30.0, 1, 2)
		data.Set(160.0, 2, 0)
		data.Set(55.0, 2, 1)
		data.Set(35.0, 2, 2)
		data.Set(180.0, 3, 0)
		data.Set(80.0, 3, 1)
		data.Set(28.0, 3, 2)
		data.Set(165.0, 4, 0)
		data.Set(60.0, 4, 1)
		data.Set(22.0, 4, 2)
		data.Set(185.0, 5, 0)
		data.Set(85.0, 5, 1)
		data.Set(40.0, 5, 2)
		data.Set(155.0, 6, 0)
		data.Set(50.0, 6, 1)
		data.Set(45.0, 6, 2)
		data.Set(172.0, 7, 0)
		data.Set(68.0, 7, 1)
		data.Set(32.0, 7, 2)

		// Fit PCA with all components
		pca := NewPCA(3)
		err := pca.Fit(data)
		if err != nil {
			t.Fatalf("PCA fit failed: %v", err)
		}

		// Transform data
		transformed, err := pca.Transform(data)
		if err != nil {
			t.Fatalf("PCA transform failed: %v", err)
		}

		// Check that we can recover original dimensions
		if transformed.Shape()[1] != 3 {
			t.Errorf("Expected 3 components, got %d", transformed.Shape()[1])
		}

		// First component should explain most variance (height/weight correlation)
		if pca.ExplainedVarianceRatio[0] < 0.5 {
			t.Errorf("First component should explain >50%% variance, got %.2f%%",
				pca.ExplainedVarianceRatio[0]*100)
		}

		// Check component orthogonality (approximately)
		comp1 := []float64{
			pca.Components.At(0, 0).(float64),
			pca.Components.At(0, 1).(float64),
			pca.Components.At(0, 2).(float64),
		}
		comp2 := []float64{
			pca.Components.At(1, 0).(float64),
			pca.Components.At(1, 1).(float64),
			pca.Components.At(1, 2).(float64),
		}

		// Compute dot product (should be close to 0 for orthogonal vectors)
		dotProduct := comp1[0]*comp2[0] + comp1[1]*comp2[1] + comp1[2]*comp2[2]
		if math.Abs(dotProduct) > 1e-10 {
			t.Errorf("Components should be orthogonal, dot product = %.6f", dotProduct)
		}

		t.Logf("Real-world-like PCA:")
		for i := 0; i < len(pca.ExplainedVarianceRatio); i++ {
			t.Logf("  Component %d: %.2f%% variance", i+1, pca.ExplainedVarianceRatio[i]*100)
		}
		t.Logf("  Components are orthogonal (dot product = %.6f)", dotProduct)
	})
}

// TestPCAEdgeCases tests edge cases and boundary conditions
func TestPCAEdgeCases(t *testing.T) {
	t.Run("PCA with constant features", func(t *testing.T) {
		// Create data where one feature is constant
		data := array.Zeros(internal.Shape{4, 3}, internal.Float64)
		// Feature 1: variable, Feature 2: constant, Feature 3: variable
		data.Set(1.0, 0, 0)
		data.Set(5.0, 0, 1)
		data.Set(2.0, 0, 2)
		data.Set(2.0, 1, 0)
		data.Set(5.0, 1, 1)
		data.Set(4.0, 1, 2)
		data.Set(3.0, 2, 0)
		data.Set(5.0, 2, 1)
		data.Set(6.0, 2, 2)
		data.Set(4.0, 3, 0)
		data.Set(5.0, 3, 1)
		data.Set(8.0, 3, 2)

		pca := NewPCA(2)
		err := pca.Fit(data)
		if err != nil {
			t.Fatalf("PCA with constant feature failed: %v", err)
		}

		// Should still work, constant feature contributes 0 variance
		if len(pca.ExplainedVariance) != 2 {
			t.Errorf("Expected 2 components, got %d", len(pca.ExplainedVariance))
		}

		t.Logf("PCA with constant feature handled successfully")
	})

	t.Run("PCA with perfect correlation", func(t *testing.T) {
		// Create data where features are perfectly correlated
		data := array.Zeros(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1) // y = 2x
		data.Set(2.0, 1, 0)
		data.Set(4.0, 1, 1) // y = 2x
		data.Set(3.0, 2, 0)
		data.Set(6.0, 2, 1) // y = 2x
		data.Set(4.0, 3, 0)
		data.Set(8.0, 3, 1) // y = 2x

		pca := NewPCA(1) // Single component should capture all variance
		err := pca.Fit(data)
		if err != nil {
			t.Fatalf("PCA with perfect correlation failed: %v", err)
		}

		// Single component should explain nearly all variance
		if pca.ExplainedVarianceRatio[0] < 0.99 {
			t.Errorf("Single component should explain >99%% variance with perfect correlation, got %.2f%%",
				pca.ExplainedVarianceRatio[0]*100)
		}

		t.Logf("PCA with perfect correlation: %.2f%% variance explained by first component",
			pca.ExplainedVarianceRatio[0]*100)
	})

	t.Run("PCA fit_transform equivalence", func(t *testing.T) {
		data := array.Zeros(internal.Shape{3, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.0, 1, 0)
		data.Set(4.0, 1, 1)
		data.Set(5.0, 2, 0)
		data.Set(6.0, 2, 1)

		// Method 1: Separate fit and transform
		pca1 := NewPCA(2)
		err := pca1.Fit(data)
		if err != nil {
			t.Fatalf("PCA fit failed: %v", err)
		}
		transformed1, err := pca1.Transform(data)
		if err != nil {
			t.Fatalf("PCA transform failed: %v", err)
		}

		// Method 2: fit_transform
		pca2 := NewPCA(2)
		transformed2, err := pca2.FitTransform(data)
		if err != nil {
			t.Fatalf("PCA fit_transform failed: %v", err)
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

// Keep the original functional PCA tests for backward compatibility
func TestPCA(t *testing.T) {
	t.Run("Simple 2D PCA", func(t *testing.T) {
		// Create simple 2D data with clear principal components
		// Data points form an ellipse along the diagonal
		data := [][]float64{
			{1, 1},
			{2, 2},
			{3, 3},
			{4, 4},
			{1.5, 1.2},
			{2.5, 2.3},
			{3.5, 3.1},
		}

		// Convert to array
		dataArray := array.Empty(internal.Shape{len(data), len(data[0])}, internal.Float64)
		for i, row := range data {
			for j, val := range row {
				dataArray.Set(val, i, j)
			}
		}

		result, err := PCA(dataArray, 2) // Keep all components
		if err != nil {
			t.Fatalf("PCA failed: %v", err)
		}

		// Should have 2 components
		if len(result.Components) != 2 {
			t.Errorf("Expected 2 components, got %d", len(result.Components))
		}

		// Should have explained variance
		if len(result.ExplainedVariance) != 2 {
			t.Errorf("Expected 2 explained variances, got %d", len(result.ExplainedVariance))
		}

		// Total explained variance should be close to 1.0 (allow some numerical error)
		totalVariance := 0.0
		for _, var_ := range result.ExplainedVarianceRatio {
			totalVariance += var_
		}
		if math.Abs(totalVariance-1.0) > 1e-6 {
			t.Errorf("Total explained variance ratio should be ~1.0, got %f", totalVariance)
		}

		// First component should explain most variance for diagonal data
		if result.ExplainedVarianceRatio[0] < 0.8 {
			t.Errorf("First component should explain >80%% variance, got %f", result.ExplainedVarianceRatio[0])
		}

		t.Logf("Explained variance ratio: %v", result.ExplainedVarianceRatio)
		t.Logf("Singular values: %v", result.SingularValues)
	})

	t.Run("3D PCA with dimensionality reduction", func(t *testing.T) {
		// Create 3D data that lies mostly in a 2D plane
		data := [][]float64{
			{1, 0, 1.01},
			{2, 1, 1.98},
			{3, 2, 3.02},
			{4, 3, 3.99},
			{0, -1, 0.02},
			{5, 4, 4.97},
		}

		dataArray := array.Empty(internal.Shape{len(data), len(data[0])}, internal.Float64)
		for i, row := range data {
			for j, val := range row {
				dataArray.Set(val, i, j)
			}
		}

		result, err := PCA(dataArray, 2) // Reduce to 2 dimensions
		if err != nil {
			t.Fatalf("PCA failed: %v", err)
		}

		// Should have 2 components
		if len(result.Components) != 2 {
			t.Errorf("Expected 2 components, got %d", len(result.Components))
		}

		// Should have transformed data
		if result.TransformedData == nil {
			t.Error("Expected transformed data")
		}

		// Transformed data should have correct shape
		expectedShape := internal.Shape{len(data), 2}
		if !shapeEqual(result.TransformedData.Shape(), expectedShape) {
			t.Errorf("Expected transformed data shape %v, got %v",
				expectedShape, result.TransformedData.Shape())
		}

		// First two components should explain most variance
		totalFirst2 := result.ExplainedVarianceRatio[0] + result.ExplainedVarianceRatio[1]
		if totalFirst2 < 0.95 {
			t.Errorf("First 2 components should explain >95%% variance, got %f", totalFirst2)
		}

		t.Logf("Explained variance ratio (2D): %v", result.ExplainedVarianceRatio)
	})

	t.Run("Parameter validation", func(t *testing.T) {
		dataArray := array.Empty(internal.Shape{2, 2}, internal.Float64)

		// Test nil array
		_, err := PCA(nil, 2)
		if err == nil {
			t.Error("Expected error for nil array")
		}

		// Test empty array
		empty := array.Empty(internal.Shape{0, 2}, internal.Float64)
		_, err = PCA(empty, 2)
		if err == nil {
			t.Error("Expected error for empty array")
		}

		// Test invalid number of components
		dataArray.Set(1.0, 0, 0)
		dataArray.Set(2.0, 0, 1)
		dataArray.Set(3.0, 1, 0)
		dataArray.Set(4.0, 1, 1)

		_, err = PCA(dataArray, 0)
		if err == nil {
			t.Error("Expected error for zero components")
		}

		_, err = PCA(dataArray, 3) // More than available features
		if err == nil {
			t.Error("Expected error for too many components")
		}

		// Test 1D array
		oneDArray := array.Empty(internal.Shape{5}, internal.Float64)
		_, err = PCA(oneDArray, 1)
		if err == nil {
			t.Error("Expected error for 1D array")
		}
	})

	t.Run("Standard dataset - Iris-like", func(t *testing.T) {
		// Create Iris-like 4D dataset
		data := [][]float64{
			{5.1, 3.5, 1.4, 0.2}, // Setosa-like
			{4.9, 3.0, 1.4, 0.2},
			{4.7, 3.2, 1.3, 0.2},
			{7.0, 3.2, 4.7, 1.4}, // Versicolor-like
			{6.4, 3.2, 4.5, 1.5},
			{6.9, 3.1, 4.9, 1.5},
			{6.3, 3.3, 6.0, 2.5}, // Virginica-like
			{5.8, 2.7, 5.1, 1.9},
			{7.1, 3.0, 5.9, 2.1},
		}

		dataArray := array.Empty(internal.Shape{len(data), len(data[0])}, internal.Float64)
		for i, row := range data {
			for j, val := range row {
				dataArray.Set(val, i, j)
			}
		}

		result, err := PCA(dataArray, 2) // Reduce to 2D for visualization
		if err != nil {
			t.Fatalf("PCA failed: %v", err)
		}

		// Verify result structure
		if len(result.Components) != 2 {
			t.Errorf("Expected 2 components, got %d", len(result.Components))
		}

		if len(result.ExplainedVarianceRatio) != 2 {
			t.Errorf("Expected 2 explained variance ratios, got %d", len(result.ExplainedVarianceRatio))
		}

		// Each component should be normalized (unit vector)
		for i, component := range result.Components {
			norm := 0.0
			for j := 0; j < component.Size(); j++ {
				val := component.At(j).(float64)
				norm += val * val
			}
			norm = math.Sqrt(norm)

			if math.Abs(norm-1.0) > 1e-10 {
				t.Errorf("Component %d should be normalized, got norm %f", i, norm)
			}
		}

		t.Logf("4D->2D PCA explained variance: %v", result.ExplainedVarianceRatio)
		t.Logf("Cumulative variance: %.3f",
			result.ExplainedVarianceRatio[0]+result.ExplainedVarianceRatio[1])
	})
}

func TestPCACenter(t *testing.T) {
	t.Run("Data centering verification", func(t *testing.T) {
		// Create data with non-zero mean
		data := [][]float64{
			{10, 20},
			{12, 22},
			{14, 24},
			{16, 26},
		}

		dataArray := array.Empty(internal.Shape{4, 2}, internal.Float64)
		for i, row := range data {
			for j, val := range row {
				dataArray.Set(val, i, j)
			}
		}

		result, err := PCA(dataArray, 2)
		if err != nil {
			t.Fatalf("PCA failed: %v", err)
		}

		// Mean should be stored
		if result.Mean == nil {
			t.Error("Expected mean to be computed and stored")
		}

		if result.Mean.Size() != 2 {
			t.Errorf("Expected mean size 2, got %d", result.Mean.Size())
		}

		// Verify mean values
		expectedMean := []float64{13.0, 23.0} // (10+12+14+16)/4, (20+22+24+26)/4
		for i, expected := range expectedMean {
			actual := result.Mean.At(i).(float64)
			if math.Abs(actual-expected) > 1e-10 {
				t.Errorf("Mean[%d]: expected %f, got %f", i, expected, actual)
			}
		}
	})
}

// Helper function to compare shapes
func shapeEqual(a, b internal.Shape) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}
