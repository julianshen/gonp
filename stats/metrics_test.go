package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
)

// TestClassificationMetrics tests classification evaluation metrics
func TestClassificationMetrics(t *testing.T) {
	t.Run("Perfect classifier", func(t *testing.T) {
		// Perfect predictions
		yTrue, _ := array.FromSlice([]float64{0, 1, 0, 1, 0, 1})
		yPred, _ := array.FromSlice([]float64{0, 1, 0, 1, 0, 1})

		accuracy, err := Accuracy(yTrue, yPred)
		if err != nil {
			t.Fatalf("Accuracy calculation failed: %v", err)
		}
		if accuracy != 1.0 {
			t.Errorf("Perfect classifier should have accuracy 1.0, got %.3f", accuracy)
		}

		precision, err := Precision(yTrue, yPred, 1.0)
		if err != nil {
			t.Fatalf("Precision calculation failed: %v", err)
		}
		if precision != 1.0 {
			t.Errorf("Perfect classifier should have precision 1.0, got %.3f", precision)
		}

		recall, err := Recall(yTrue, yPred, 1.0)
		if err != nil {
			t.Fatalf("Recall calculation failed: %v", err)
		}
		if recall != 1.0 {
			t.Errorf("Perfect classifier should have recall 1.0, got %.3f", recall)
		}

		f1, err := F1Score(yTrue, yPred, 1.0)
		if err != nil {
			t.Fatalf("F1 score calculation failed: %v", err)
		}
		if f1 != 1.0 {
			t.Errorf("Perfect classifier should have F1 score 1.0, got %.3f", f1)
		}

		t.Logf("Perfect classifier: accuracy=%.3f, precision=%.3f, recall=%.3f, f1=%.3f",
			accuracy, precision, recall, f1)
	})

	t.Run("Binary classification with errors", func(t *testing.T) {
		// Some misclassifications
		yTrue, _ := array.FromSlice([]float64{0, 0, 1, 1, 0, 1, 0, 1})
		yPred, _ := array.FromSlice([]float64{0, 1, 1, 1, 0, 0, 0, 1}) // 2 errors: pred[1]=1, pred[5]=0

		accuracy, err := Accuracy(yTrue, yPred)
		if err != nil {
			t.Fatalf("Accuracy calculation failed: %v", err)
		}
		expectedAccuracy := 6.0 / 8.0 // 6 correct out of 8
		if math.Abs(accuracy-expectedAccuracy) > 1e-10 {
			t.Errorf("Expected accuracy %.3f, got %.3f", expectedAccuracy, accuracy)
		}

		// For class 1 (positive class)
		precision, err := Precision(yTrue, yPred, 1.0)
		if err != nil {
			t.Fatalf("Precision calculation failed: %v", err)
		}

		recall, err := Recall(yTrue, yPred, 1.0)
		if err != nil {
			t.Fatalf("Recall calculation failed: %v", err)
		}

		f1, err := F1Score(yTrue, yPred, 1.0)
		if err != nil {
			t.Fatalf("F1 score calculation failed: %v", err)
		}

		// Manual calculation:
		// True class 1: indices 2, 3, 5, 7 (4 total)
		// Pred class 1: indices 1, 2, 3, 7 (4 total)
		// True Positives: 2, 3, 7 (3 total)
		// Precision = TP / (TP + FP) = 3 / 4 = 0.75
		// Recall = TP / (TP + FN) = 3 / 4 = 0.75
		// F1 = 2 * (P * R) / (P + R) = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75

		expectedPrecision := 0.75
		expectedRecall := 0.75
		expectedF1 := 0.75

		if math.Abs(precision-expectedPrecision) > 1e-10 {
			t.Errorf("Expected precision %.3f, got %.3f", expectedPrecision, precision)
		}
		if math.Abs(recall-expectedRecall) > 1e-10 {
			t.Errorf("Expected recall %.3f, got %.3f", expectedRecall, recall)
		}
		if math.Abs(f1-expectedF1) > 1e-10 {
			t.Errorf("Expected F1 %.3f, got %.3f", expectedF1, f1)
		}

		t.Logf("Binary classification: accuracy=%.3f, precision=%.3f, recall=%.3f, f1=%.3f",
			accuracy, precision, recall, f1)
	})

	t.Run("Multi-class classification", func(t *testing.T) {
		// 3-class problem
		yTrue, _ := array.FromSlice([]float64{0, 1, 2, 0, 1, 2, 0, 1, 2})
		yPred, _ := array.FromSlice([]float64{0, 1, 2, 0, 2, 2, 1, 1, 2}) // Some errors

		accuracy, err := Accuracy(yTrue, yPred)
		if err != nil {
			t.Fatalf("Accuracy calculation failed: %v", err)
		}

		// Calculate per-class metrics
		precision0, err := Precision(yTrue, yPred, 0.0)
		if err != nil {
			t.Fatalf("Precision for class 0 failed: %v", err)
		}

		precision1, err := Precision(yTrue, yPred, 1.0)
		if err != nil {
			t.Fatalf("Precision for class 1 failed: %v", err)
		}

		precision2, err := Precision(yTrue, yPred, 2.0)
		if err != nil {
			t.Fatalf("Precision for class 2 failed: %v", err)
		}

		t.Logf("Multi-class: accuracy=%.3f", accuracy)
		t.Logf("  Class 0: precision=%.3f", precision0)
		t.Logf("  Class 1: precision=%.3f", precision1)
		t.Logf("  Class 2: precision=%.3f", precision2)

		// Accuracy should be number of correct predictions / total
		// Correct: indices 0,1,2,3,5,7,8 = 7 out of 9
		expectedAccuracy := 7.0 / 9.0
		if math.Abs(accuracy-expectedAccuracy) > 1e-10 {
			t.Errorf("Expected accuracy %.3f, got %.3f", expectedAccuracy, accuracy)
		}
	})

	t.Run("Metrics parameter validation", func(t *testing.T) {
		yTrue, _ := array.FromSlice([]float64{0, 1, 0, 1})
		yPred, _ := array.FromSlice([]float64{0, 0, 1, 1})
		wrongSize, _ := array.FromSlice([]float64{0, 1, 0})

		// Test mismatched sizes
		_, err := Accuracy(yTrue, wrongSize)
		if err == nil {
			t.Error("Expected error for mismatched array sizes")
		}

		_, err = Precision(yTrue, wrongSize, 1.0)
		if err == nil {
			t.Error("Expected error for mismatched array sizes")
		}

		// Test nil inputs
		_, err = Accuracy(nil, yPred)
		if err == nil {
			t.Error("Expected error for nil yTrue")
		}

		_, err = Recall(yTrue, nil, 1.0)
		if err == nil {
			t.Error("Expected error for nil yPred")
		}
	})
}

// TestConfusionMatrix tests confusion matrix computation
func TestConfusionMatrix(t *testing.T) {
	t.Run("Binary confusion matrix", func(t *testing.T) {
		yTrue, _ := array.FromSlice([]float64{0, 0, 1, 1})
		yPred, _ := array.FromSlice([]float64{0, 1, 1, 1})

		cm, labels, err := ConfusionMatrix(yTrue, yPred)
		if err != nil {
			t.Fatalf("Confusion matrix failed: %v", err)
		}

		// Should have 2x2 matrix
		if cm.Shape()[0] != 2 || cm.Shape()[1] != 2 {
			t.Errorf("Expected 2x2 confusion matrix, got %v", cm.Shape())
		}

		// Should have labels [0, 1]
		if len(labels) != 2 || labels[0] != 0 || labels[1] != 1 {
			t.Errorf("Expected labels [0, 1], got %v", labels)
		}

		// Manual calculation:
		// True 0, Pred 0: 1 (index 0)
		// True 0, Pred 1: 1 (index 1)
		// True 1, Pred 0: 0
		// True 1, Pred 1: 2 (indices 2, 3)
		expectedCM := [][]int{
			{1, 1}, // True 0
			{0, 2}, // True 1
		}

		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				actual := int(cm.At(i, j).(int64))
				expected := expectedCM[i][j]
				if actual != expected {
					t.Errorf("CM[%d,%d] expected %d, got %d", i, j, expected, actual)
				}
			}
		}

		t.Logf("Binary confusion matrix:")
		for i := 0; i < 2; i++ {
			row := make([]int, 2)
			for j := 0; j < 2; j++ {
				row[j] = int(cm.At(i, j).(int64))
			}
			t.Logf("  True %d: %v", labels[i], row)
		}
	})

	t.Run("Multi-class confusion matrix", func(t *testing.T) {
		yTrue, _ := array.FromSlice([]float64{0, 1, 2, 0, 1, 2})
		yPred, _ := array.FromSlice([]float64{0, 1, 1, 2, 1, 2}) // Some misclassifications

		cm, labels, err := ConfusionMatrix(yTrue, yPred)
		if err != nil {
			t.Fatalf("Confusion matrix failed: %v", err)
		}

		// Should have 3x3 matrix
		if cm.Shape()[0] != 3 || cm.Shape()[1] != 3 {
			t.Errorf("Expected 3x3 confusion matrix, got %v", cm.Shape())
		}

		// Should have labels [0, 1, 2]
		if len(labels) != 3 {
			t.Errorf("Expected 3 labels, got %d", len(labels))
		}

		t.Logf("Multi-class confusion matrix:")
		for i := 0; i < 3; i++ {
			row := make([]int, 3)
			for j := 0; j < 3; j++ {
				row[j] = int(cm.At(i, j).(int64))
			}
			t.Logf("  True %d: %v", labels[i], row)
		}
	})

	t.Run("Confusion matrix parameter validation", func(t *testing.T) {
		yTrue, _ := array.FromSlice([]float64{0, 1, 0, 1})
		wrongSize, _ := array.FromSlice([]float64{0, 1, 0})

		// Test mismatched sizes
		_, _, err := ConfusionMatrix(yTrue, wrongSize)
		if err == nil {
			t.Error("Expected error for mismatched array sizes")
		}

		// Test nil inputs
		_, _, err = ConfusionMatrix(nil, yTrue)
		if err == nil {
			t.Error("Expected error for nil yTrue")
		}
	})
}

// TestClassificationReport tests comprehensive classification reporting
func TestClassificationReport(t *testing.T) {
	t.Run("Binary classification report", func(t *testing.T) {
		yTrue, _ := array.FromSlice([]float64{0, 0, 1, 1, 0, 1, 0, 1})
		yPred, _ := array.FromSlice([]float64{0, 1, 1, 1, 0, 0, 0, 1})

		report, err := GenerateClassificationReport(yTrue, yPred)
		if err != nil {
			t.Fatalf("Classification report failed: %v", err)
		}

		// Should have metrics for each class plus overall
		if report.Accuracy <= 0 || report.Accuracy > 1 {
			t.Errorf("Invalid accuracy: %f", report.Accuracy)
		}

		// Should have per-class metrics
		if len(report.PerClass) == 0 {
			t.Error("Per-class metrics should not be empty")
		}

		for class, metrics := range report.PerClass {
			if metrics.Precision < 0 || metrics.Precision > 1 {
				t.Errorf("Invalid precision for class %d: %f", class, metrics.Precision)
			}
			if metrics.Recall < 0 || metrics.Recall > 1 {
				t.Errorf("Invalid recall for class %d: %f", class, metrics.Recall)
			}
			if metrics.F1Score < 0 || metrics.F1Score > 1 {
				t.Errorf("Invalid F1 score for class %d: %f", class, metrics.F1Score)
			}
		}

		t.Logf("Classification Report:")
		t.Logf("  Overall Accuracy: %.3f", report.Accuracy)
		for class, metrics := range report.PerClass {
			t.Logf("  Class %d: P=%.3f, R=%.3f, F1=%.3f (support=%d)",
				class, metrics.Precision, metrics.Recall, metrics.F1Score, metrics.Support)
		}
		if report.MacroAvg != nil {
			t.Logf("  Macro Avg: P=%.3f, R=%.3f, F1=%.3f",
				report.MacroAvg.Precision, report.MacroAvg.Recall, report.MacroAvg.F1Score)
		}
		if report.WeightedAvg != nil {
			t.Logf("  Weighted Avg: P=%.3f, R=%.3f, F1=%.3f",
				report.WeightedAvg.Precision, report.WeightedAvg.Recall, report.WeightedAvg.F1Score)
		}
	})
}

// TestROCAUC tests ROC AUC computation
func TestROCAUC(t *testing.T) {
	t.Run("Perfect classifier ROC AUC", func(t *testing.T) {
		yTrue, _ := array.FromSlice([]float64{0, 0, 1, 1})
		yScores, _ := array.FromSlice([]float64{0.1, 0.2, 0.8, 0.9}) // Perfect separation

		auc, err := ROCAUC(yTrue, yScores)
		if err != nil {
			t.Fatalf("ROC AUC calculation failed: %v", err)
		}

		// Perfect classifier should have AUC = 1.0
		if math.Abs(auc-1.0) > 1e-10 {
			t.Errorf("Perfect classifier should have AUC 1.0, got %.6f", auc)
		}

		t.Logf("Perfect classifier ROC AUC: %.6f", auc)
	})

	t.Run("Random classifier ROC AUC", func(t *testing.T) {
		yTrue, _ := array.FromSlice([]float64{0, 1, 0, 1, 0, 1, 0, 1})
		yScores, _ := array.FromSlice([]float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}) // No discrimination

		auc, err := ROCAUC(yTrue, yScores)
		if err != nil {
			t.Fatalf("ROC AUC calculation failed: %v", err)
		}

		// Random classifier should have AUC around 0.5, but with identical scores,
		// the exact value depends on tie-breaking in sorting
		if auc < 0.3 || auc > 0.7 {
			t.Errorf("Random classifier should have AUC between 0.3-0.7, got %.6f", auc)
		}

		t.Logf("Random classifier ROC AUC: %.6f", auc)
	})

	t.Run("Realistic ROC AUC", func(t *testing.T) {
		yTrue, _ := array.FromSlice([]float64{0, 0, 0, 1, 1, 1})
		yScores, _ := array.FromSlice([]float64{0.2, 0.3, 0.6, 0.4, 0.7, 0.8}) // Some discrimination

		auc, err := ROCAUC(yTrue, yScores)
		if err != nil {
			t.Fatalf("ROC AUC calculation failed: %v", err)
		}

		// Should be between 0.5 and 1.0
		if auc < 0.5 || auc > 1.0 {
			t.Errorf("ROC AUC should be between 0.5 and 1.0, got %.6f", auc)
		}

		t.Logf("Realistic ROC AUC: %.6f", auc)
	})

	t.Run("ROC AUC parameter validation", func(t *testing.T) {
		yTrue, _ := array.FromSlice([]float64{0, 1, 0, 1})
		yScores, _ := array.FromSlice([]float64{0.2, 0.8, 0.3, 0.7})
		wrongSize, _ := array.FromSlice([]float64{0.1, 0.2, 0.3})

		// Test mismatched sizes
		_, err := ROCAUC(yTrue, wrongSize)
		if err == nil {
			t.Error("Expected error for mismatched array sizes")
		}

		// Test nil inputs
		_, err = ROCAUC(nil, yScores)
		if err == nil {
			t.Error("Expected error for nil yTrue")
		}

		// Test non-binary classification
		yMultiClass, _ := array.FromSlice([]float64{0, 1, 2, 1})
		_, err = ROCAUC(yMultiClass, yScores)
		if err == nil {
			t.Error("Expected error for non-binary classification")
		}
	})
}
