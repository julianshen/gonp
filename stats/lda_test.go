package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestLinearDiscriminantAnalysis tests LDA implementation
func TestLinearDiscriminantAnalysis(t *testing.T) {
	t.Run("Binary classification - simple case", func(t *testing.T) {
		// Simple 2D data with 2 classes
		// Class 0: points around (1, 1)
		// Class 1: points around (3, 3)
		X := array.Zeros(internal.Shape{6, 2}, internal.Float64)
		X.Set(1.0, 0, 0)
		X.Set(1.0, 0, 1) // Class 0
		X.Set(1.1, 1, 0)
		X.Set(0.9, 1, 1) // Class 0
		X.Set(0.9, 2, 0)
		X.Set(1.1, 2, 1) // Class 0
		X.Set(3.0, 3, 0)
		X.Set(3.0, 3, 1) // Class 1
		X.Set(2.9, 4, 0)
		X.Set(3.1, 4, 1) // Class 1
		X.Set(3.1, 5, 0)
		X.Set(2.9, 5, 1) // Class 1

		y, _ := array.FromSlice([]float64{0, 0, 0, 1, 1, 1})

		result, err := LinearDiscriminantAnalysis(X, y)
		if err != nil {
			t.Fatalf("LDA failed: %v", err)
		}

		// Check that we have the right structure
		if result == nil {
			t.Fatal("LDA result should not be nil")
		}

		// Should have 2 classes
		if len(result.Classes) != 2 {
			t.Errorf("Expected 2 classes, got %d", len(result.Classes))
		}

		// Should have proper class means
		if result.ClassMeans == nil {
			t.Error("Class means should not be nil")
		}

		// Should have discriminant coefficients
		if result.Coefficients == nil {
			t.Error("Coefficients should not be nil")
		}

		t.Logf("LDA Classes: %v", result.Classes)
		t.Logf("LDA completed successfully for binary classification")
	})

	t.Run("Multi-class classification", func(t *testing.T) {
		// 3 classes in 2D space
		X := array.Zeros(internal.Shape{9, 2}, internal.Float64)
		// Class 0: around (0, 0)
		X.Set(0.1, 0, 0)
		X.Set(0.1, 0, 1)
		X.Set(-0.1, 1, 0)
		X.Set(0.2, 1, 1)
		X.Set(0.2, 2, 0)
		X.Set(-0.1, 2, 1)

		// Class 1: around (2, 0)
		X.Set(2.1, 3, 0)
		X.Set(0.1, 3, 1)
		X.Set(1.9, 4, 0)
		X.Set(-0.1, 4, 1)
		X.Set(2.0, 5, 0)
		X.Set(0.2, 5, 1)

		// Class 2: around (1, 2)
		X.Set(1.1, 6, 0)
		X.Set(2.1, 6, 1)
		X.Set(0.9, 7, 0)
		X.Set(1.9, 7, 1)
		X.Set(1.0, 8, 0)
		X.Set(2.0, 8, 1)

		y, _ := array.FromSlice([]float64{0, 0, 0, 1, 1, 1, 2, 2, 2})

		result, err := LinearDiscriminantAnalysis(X, y)
		if err != nil {
			t.Fatalf("Multi-class LDA failed: %v", err)
		}

		// Should have 3 classes
		if len(result.Classes) != 3 {
			t.Errorf("Expected 3 classes, got %d", len(result.Classes))
		}

		// Check class means shape
		if result.ClassMeans.Shape()[0] != 3 || result.ClassMeans.Shape()[1] != 2 {
			t.Errorf("Class means should be 3x2, got %v", result.ClassMeans.Shape())
		}

		t.Logf("Multi-class LDA completed successfully")
	})

	t.Run("LDA prediction", func(t *testing.T) {
		// Simple training data
		X := array.Zeros(internal.Shape{4, 2}, internal.Float64)
		X.Set(1.0, 0, 0)
		X.Set(1.0, 0, 1) // Class 0
		X.Set(1.0, 1, 0)
		X.Set(1.0, 1, 1) // Class 0
		X.Set(3.0, 2, 0)
		X.Set(3.0, 2, 1) // Class 1
		X.Set(3.0, 3, 0)
		X.Set(3.0, 3, 1) // Class 1

		y, _ := array.FromSlice([]float64{0, 0, 1, 1})

		// Train LDA
		model, err := LinearDiscriminantAnalysis(X, y)
		if err != nil {
			t.Fatalf("LDA training failed: %v", err)
		}

		// Test prediction
		testX := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		testX.Set(1.1, 0, 0)
		testX.Set(1.1, 0, 1) // Should be class 0
		testX.Set(2.9, 1, 0)
		testX.Set(2.9, 1, 1) // Should be class 1

		predictions, err := LDAPredict(model, testX)
		if err != nil {
			t.Fatalf("LDA prediction failed: %v", err)
		}

		if predictions.Size() != 2 {
			t.Errorf("Expected 2 predictions, got %d", predictions.Size())
		}

		// Check that predictions are reasonable
		pred0 := int(predictions.At(0).(int64))
		pred1 := int(predictions.At(1).(int64))

		t.Logf("Predictions: [%d, %d]", pred0, pred1)

		// First point should be classified as class 0, second as class 1
		if pred0 != 0 {
			t.Errorf("Expected prediction 0 for first point, got %d", pred0)
		}
		if pred1 != 1 {
			t.Errorf("Expected prediction 1 for second point, got %d", pred1)
		}
	})

	t.Run("LDA probability prediction", func(t *testing.T) {
		// Simple training data
		X := array.Zeros(internal.Shape{4, 2}, internal.Float64)
		X.Set(0.0, 0, 0)
		X.Set(0.0, 0, 1) // Class 0
		X.Set(0.1, 1, 0)
		X.Set(0.1, 1, 1) // Class 0
		X.Set(2.0, 2, 0)
		X.Set(2.0, 2, 1) // Class 1
		X.Set(2.1, 3, 0)
		X.Set(2.1, 3, 1) // Class 1

		y, _ := array.FromSlice([]float64{0, 0, 1, 1})

		model, err := LinearDiscriminantAnalysis(X, y)
		if err != nil {
			t.Fatalf("LDA training failed: %v", err)
		}

		// Test probability prediction
		testX := array.Zeros(internal.Shape{1, 2}, internal.Float64)
		testX.Set(1.0, 0, 0)
		testX.Set(1.0, 0, 1) // Point in between classes

		probabilities, err := LDAPredictProba(model, testX)
		if err != nil {
			t.Fatalf("LDA probability prediction failed: %v", err)
		}

		// Should have probabilities for both classes
		if probabilities.Shape()[0] != 1 || probabilities.Shape()[1] != 2 {
			t.Errorf("Expected 1x2 probability matrix, got %v", probabilities.Shape())
		}

		// Probabilities should sum to 1
		prob0 := probabilities.At(0, 0).(float64)
		prob1 := probabilities.At(0, 1).(float64)

		if math.Abs(prob0+prob1-1.0) > 1e-10 {
			t.Errorf("Probabilities should sum to 1, got %f + %f = %f", prob0, prob1, prob0+prob1)
		}

		// Both probabilities should be positive
		if prob0 < 0 || prob1 < 0 {
			t.Errorf("Probabilities should be non-negative, got [%f, %f]", prob0, prob1)
		}

		t.Logf("Probabilities: [%.3f, %.3f]", prob0, prob1)
	})

	t.Run("Parameter validation", func(t *testing.T) {
		// Test nil inputs
		_, err := LinearDiscriminantAnalysis(nil, nil)
		if err == nil {
			t.Error("Expected error for nil inputs")
		}

		// Test mismatched sizes
		X := array.Ones(internal.Shape{3, 2}, internal.Float64)
		y, _ := array.FromSlice([]float64{0, 1}) // Wrong size

		_, err = LinearDiscriminantAnalysis(X, y)
		if err == nil {
			t.Error("Expected error for mismatched array sizes")
		}

		// Test insufficient data
		smallX := array.Ones(internal.Shape{1, 2}, internal.Float64)
		smallY, _ := array.FromSlice([]float64{0})

		_, err = LinearDiscriminantAnalysis(smallX, smallY)
		if err == nil {
			t.Error("Expected error for insufficient data")
		}
	})
}

// TestLDAEdgeCases tests edge cases for LDA
func TestLDAEdgeCases(t *testing.T) {
	t.Run("Single feature", func(t *testing.T) {
		// 1D classification problem
		X := array.Zeros(internal.Shape{4, 1}, internal.Float64)
		X.Set(1.0, 0, 0) // Class 0
		X.Set(1.1, 1, 0) // Class 0
		X.Set(3.0, 2, 0) // Class 1
		X.Set(3.1, 3, 0) // Class 1

		y, _ := array.FromSlice([]float64{0, 0, 1, 1})

		result, err := LinearDiscriminantAnalysis(X, y)
		if err != nil {
			t.Fatalf("1D LDA failed: %v", err)
		}

		if result.ClassMeans.Shape()[1] != 1 {
			t.Errorf("Expected 1 feature in class means, got %d", result.ClassMeans.Shape()[1])
		}

		t.Logf("1D LDA completed successfully")
	})

	t.Run("Equal class sizes", func(t *testing.T) {
		// Perfectly balanced classes
		X := array.Zeros(internal.Shape{6, 2}, internal.Float64)
		X.Set(1.0, 0, 0)
		X.Set(1.0, 0, 1) // Class 0
		X.Set(1.0, 1, 0)
		X.Set(1.0, 1, 1) // Class 0
		X.Set(1.0, 2, 0)
		X.Set(1.0, 2, 1) // Class 0
		X.Set(3.0, 3, 0)
		X.Set(3.0, 3, 1) // Class 1
		X.Set(3.0, 4, 0)
		X.Set(3.0, 4, 1) // Class 1
		X.Set(3.0, 5, 0)
		X.Set(3.0, 5, 1) // Class 1

		y, _ := array.FromSlice([]float64{0, 0, 0, 1, 1, 1})

		result, err := LinearDiscriminantAnalysis(X, y)
		if err != nil {
			t.Fatalf("Balanced LDA failed: %v", err)
		}

		// Both classes should have equal prior probabilities
		if len(result.Priors) != 2 {
			t.Errorf("Expected 2 priors, got %d", len(result.Priors))
		}

		for i, prior := range result.Priors {
			if math.Abs(prior-0.5) > 1e-10 {
				t.Errorf("Expected prior 0.5 for class %d, got %f", i, prior)
			}
		}

		t.Logf("Balanced LDA completed successfully")
	})

	t.Run("Imbalanced classes", func(t *testing.T) {
		// Imbalanced dataset: 1 sample in class 0, 3 samples in class 1
		X := array.Zeros(internal.Shape{4, 2}, internal.Float64)
		X.Set(1.0, 0, 0)
		X.Set(1.0, 0, 1) // Class 0 (minority)
		X.Set(3.0, 1, 0)
		X.Set(3.0, 1, 1) // Class 1
		X.Set(3.1, 2, 0)
		X.Set(3.1, 2, 1) // Class 1
		X.Set(2.9, 3, 0)
		X.Set(2.9, 3, 1) // Class 1

		y, _ := array.FromSlice([]float64{0, 1, 1, 1})

		result, err := LinearDiscriminantAnalysis(X, y)
		if err != nil {
			t.Fatalf("Imbalanced LDA failed: %v", err)
		}

		// Class 0 should have prior 0.25, class 1 should have prior 0.75
		expectedPriors := []float64{0.25, 0.75}
		for i, expectedPrior := range expectedPriors {
			if math.Abs(result.Priors[i]-expectedPrior) > 1e-10 {
				t.Errorf("Expected prior %f for class %d, got %f",
					expectedPrior, i, result.Priors[i])
			}
		}

		t.Logf("Imbalanced LDA completed successfully")
	})
}
