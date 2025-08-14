package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestPCATransform(t *testing.T) {
	t.Run("Transform new data using existing PCA", func(t *testing.T) {
		// Original training data
		trainData := [][]float64{
			{1, 1},
			{2, 2},
			{3, 3},
			{4, 4},
		}

		trainArray := array.Empty(internal.Shape{len(trainData), len(trainData[0])}, internal.Float64)
		for i, row := range trainData {
			for j, val := range row {
				trainArray.Set(val, i, j)
			}
		}

		// Fit PCA on training data
		pcaResult, err := PCA(trainArray, 2)
		if err != nil {
			t.Fatalf("PCA failed: %v", err)
		}

		// New test data (similar pattern)
		testData := [][]float64{
			{0.5, 0.5},
			{1.5, 1.5},
			{2.5, 2.5},
		}

		testArray := array.Empty(internal.Shape{len(testData), len(testData[0])}, internal.Float64)
		for i, row := range testData {
			for j, val := range row {
				testArray.Set(val, i, j)
			}
		}

		// Transform test data
		transformed, err := PCATransform(testArray, pcaResult)
		if err != nil {
			t.Fatalf("PCATransform failed: %v", err)
		}

		// Check transformed data shape
		expectedShape := internal.Shape{len(testData), 2}
		if !shapeEqual(transformed.Shape(), expectedShape) {
			t.Errorf("Expected shape %v, got %v", expectedShape, transformed.Shape())
		}

		// Transformed data should have similar pattern to training data
		// (for diagonal data, first component should be dominant)
		for i := 0; i < len(testData); i++ {
			pc1 := transformed.At(i, 0).(float64)
			pc2 := transformed.At(i, 1).(float64)

			// First component should be larger in magnitude than second
			if math.Abs(pc1) < math.Abs(pc2) {
				t.Errorf("For diagonal data, first component should dominate: PC1=%.3f, PC2=%.3f", pc1, pc2)
			}
		}

		t.Logf("Original test data: %v", testData)
		t.Logf("Transformed data: [%.3f, %.3f], [%.3f, %.3f], [%.3f, %.3f]",
			transformed.At(0, 0).(float64), transformed.At(0, 1).(float64),
			transformed.At(1, 0).(float64), transformed.At(1, 1).(float64),
			transformed.At(2, 0).(float64), transformed.At(2, 1).(float64))
	})

	t.Run("Parameter validation", func(t *testing.T) {
		// Create a valid PCA result
		dataArray := array.Empty(internal.Shape{2, 2}, internal.Float64)
		dataArray.Set(1.0, 0, 0)
		dataArray.Set(2.0, 0, 1)
		dataArray.Set(3.0, 1, 0)
		dataArray.Set(4.0, 1, 1)

		pcaResult, _ := PCA(dataArray, 2)

		// Test nil data
		_, err := PCATransform(nil, pcaResult)
		if err == nil {
			t.Error("Expected error for nil data")
		}

		// Test nil PCA result
		_, err = PCATransform(dataArray, nil)
		if err == nil {
			t.Error("Expected error for nil PCA result")
		}

		// Test mismatched dimensions
		wrongData := array.Empty(internal.Shape{2, 3}, internal.Float64) // Wrong number of features
		_, err = PCATransform(wrongData, pcaResult)
		if err == nil {
			t.Error("Expected error for mismatched dimensions")
		}

		// Test 1D array
		oneD := array.Empty(internal.Shape{5}, internal.Float64)
		_, err = PCATransform(oneD, pcaResult)
		if err == nil {
			t.Error("Expected error for 1D array")
		}
	})
}

func TestPCAInverseTransform(t *testing.T) {
	t.Run("Round-trip transformation", func(t *testing.T) {
		// Original data
		originalData := [][]float64{
			{1.5, 2.5},
			{2.5, 3.5},
			{3.5, 4.5},
			{4.5, 5.5},
		}

		originalArray := array.Empty(internal.Shape{len(originalData), len(originalData[0])}, internal.Float64)
		for i, row := range originalData {
			for j, val := range row {
				originalArray.Set(val, i, j)
			}
		}

		// Fit PCA
		pcaResult, err := PCA(originalArray, 2) // Keep all components
		if err != nil {
			t.Fatalf("PCA failed: %v", err)
		}

		// Inverse transform the transformed data (should get back original - mean)
		reconstructed, err := PCAInverseTransform(pcaResult.TransformedData, pcaResult)
		if err != nil {
			t.Fatalf("PCAInverseTransform failed: %v", err)
		}

		// Check shape
		if !shapeEqual(reconstructed.Shape(), originalArray.Shape()) {
			t.Errorf("Expected shape %v, got %v", originalArray.Shape(), reconstructed.Shape())
		}

		// Reconstructed data should be very close to original (within numerical precision)
		for i := 0; i < len(originalData); i++ {
			for j := 0; j < len(originalData[0]); j++ {
				original := originalData[i][j]
				reconstructedVal := reconstructed.At(i, j).(float64)

				if math.Abs(original-reconstructedVal) > 1e-10 {
					t.Errorf("Reconstruction error at [%d,%d]: original=%.6f, reconstructed=%.6f",
						i, j, original, reconstructedVal)
				}
			}
		}

		t.Logf("Round-trip transformation successful with max error < 1e-10")
	})

	t.Run("Dimensionality reduction round-trip", func(t *testing.T) {
		// 3D data that lies mostly in a 2D subspace
		originalData := [][]float64{
			{1, 2, 3.01}, // Small component in 3rd dimension
			{2, 4, 6.02},
			{3, 6, 9.01},
			{4, 8, 12.03},
		}

		originalArray := array.Empty(internal.Shape{len(originalData), len(originalData[0])}, internal.Float64)
		for i, row := range originalData {
			for j, val := range row {
				originalArray.Set(val, i, j)
			}
		}

		// Reduce to 2D
		pcaResult, err := PCA(originalArray, 2)
		if err != nil {
			t.Fatalf("PCA failed: %v", err)
		}

		// Inverse transform (this will lose some information since we reduced dimensions)
		reconstructed, err := PCAInverseTransform(pcaResult.TransformedData, pcaResult)
		if err != nil {
			t.Fatalf("PCAInverseTransform failed: %v", err)
		}

		// The reconstruction won't be perfect due to dimensionality reduction
		// But it should be close for the major components
		totalError := 0.0
		for i := 0; i < len(originalData); i++ {
			for j := 0; j < len(originalData[0]); j++ {
				original := originalData[i][j]
				reconstructedVal := reconstructed.At(i, j).(float64)
				error := math.Abs(original - reconstructedVal)
				totalError += error
			}
		}
		avgError := totalError / float64(len(originalData)*len(originalData[0]))

		// Should have low average error since most variance is captured by 2 components
		if avgError > 0.1 {
			t.Errorf("Average reconstruction error too high: %.3f", avgError)
		}

		t.Logf("Dimensionality reduction round-trip: average error = %.6f", avgError)
	})

	t.Run("Parameter validation", func(t *testing.T) {
		// Create valid inputs
		data := array.Empty(internal.Shape{2, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.0, 1, 0)
		data.Set(4.0, 1, 1)

		pcaResult, _ := PCA(data, 2)
		transformedData := pcaResult.TransformedData

		// Test nil transformed data
		_, err := PCAInverseTransform(nil, pcaResult)
		if err == nil {
			t.Error("Expected error for nil transformed data")
		}

		// Test nil PCA result
		_, err = PCAInverseTransform(transformedData, nil)
		if err == nil {
			t.Error("Expected error for nil PCA result")
		}

		// Test mismatched components
		wrongTransformed := array.Empty(internal.Shape{2, 1}, internal.Float64) // Wrong number of components
		_, err = PCAInverseTransform(wrongTransformed, pcaResult)
		if err == nil {
			t.Error("Expected error for mismatched component count")
		}

		// Test 1D array
		oneD := array.Empty(internal.Shape{5}, internal.Float64)
		_, err = PCAInverseTransform(oneD, pcaResult)
		if err == nil {
			t.Error("Expected error for 1D array")
		}
	})
}

func TestPCAWorkflow(t *testing.T) {
	t.Run("Complete PCA workflow", func(t *testing.T) {
		// Create training data - 4D with some structure
		trainData := [][]float64{
			{1, 2, 3, 4},
			{2, 4, 6, 8},
			{3, 6, 9, 12},
			{1.1, 2.2, 3.3, 4.4},
			{2.1, 4.2, 6.3, 8.4},
			{3.1, 6.2, 9.3, 12.4},
		}

		trainArray := array.Empty(internal.Shape{len(trainData), len(trainData[0])}, internal.Float64)
		for i, row := range trainData {
			for j, val := range row {
				trainArray.Set(val, i, j)
			}
		}

		// Step 1: Fit PCA and reduce to 2D
		pcaResult, err := PCA(trainArray, 2)
		if err != nil {
			t.Fatalf("PCA failed: %v", err)
		}

		// Verify first few components capture most variance
		cumVariance := pcaResult.ExplainedVarianceRatio[0] + pcaResult.ExplainedVarianceRatio[1]
		if cumVariance < 0.95 {
			t.Errorf("First 2 components should explain >95%% variance, got %.3f", cumVariance)
		}

		// Step 2: Transform new data
		testData := [][]float64{
			{1.5, 3.0, 4.5, 6.0},
			{2.5, 5.0, 7.5, 10.0},
		}

		testArray := array.Empty(internal.Shape{len(testData), len(testData[0])}, internal.Float64)
		for i, row := range testData {
			for j, val := range row {
				testArray.Set(val, i, j)
			}
		}

		transformedTest, err := PCATransform(testArray, pcaResult)
		if err != nil {
			t.Fatalf("PCATransform failed: %v", err)
		}

		// Step 3: Inverse transform to reconstruct
		reconstructed, err := PCAInverseTransform(transformedTest, pcaResult)
		if err != nil {
			t.Fatalf("PCAInverseTransform failed: %v", err)
		}

		// Step 4: Verify reconstruction quality
		totalError := 0.0
		for i := 0; i < len(testData); i++ {
			for j := 0; j < len(testData[0]); j++ {
				original := testData[i][j]
				reconstructedVal := reconstructed.At(i, j).(float64)
				error := math.Abs(original - reconstructedVal)
				totalError += error
			}
		}
		avgError := totalError / float64(len(testData)*len(testData[0]))

		// Should have reasonable reconstruction error
		if avgError > 1.0 {
			t.Errorf("Average reconstruction error too high: %.3f", avgError)
		}

		t.Logf("Complete workflow successful:")
		t.Logf("  Explained variance: %.1f%% + %.1f%% = %.1f%%",
			pcaResult.ExplainedVarianceRatio[0]*100,
			pcaResult.ExplainedVarianceRatio[1]*100,
			cumVariance*100)
		t.Logf("  Average reconstruction error: %.6f", avgError)
	})
}
