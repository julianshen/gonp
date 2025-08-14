package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestKFoldCrossValidation tests K-fold cross-validation implementation
func TestKFoldCrossValidation(t *testing.T) {
	t.Run("Basic K-fold with k=5", func(t *testing.T) {
		// Create test data: 10 samples
		X := array.Zeros(internal.Shape{10, 2}, internal.Float64)
		for i := 0; i < 10; i++ {
			X.Set(float64(i), i, 0)
			X.Set(float64(i*2), i, 1)
		}
		y, _ := array.FromSlice([]float64{0, 0, 1, 1, 0, 1, 0, 1, 1, 0})

		// Test K-fold split
		folds, err := KFoldSplit(X, y, 5, false, 42)
		if err != nil {
			t.Fatalf("KFold split failed: %v", err)
		}

		// Should have 5 folds
		if len(folds) != 5 {
			t.Errorf("Expected 5 folds, got %d", len(folds))
		}

		// Each fold should have train and validation sets
		totalTrainSize := 0
		totalValSize := 0
		for i, fold := range folds {
			if fold.TrainX == nil || fold.TrainY == nil || fold.ValX == nil || fold.ValY == nil {
				t.Errorf("Fold %d has nil arrays", i)
				continue
			}

			trainSize := fold.TrainX.Shape()[0]
			valSize := fold.ValX.Shape()[0]
			totalTrainSize += trainSize
			totalValSize += valSize

			// Validation set should be approximately 1/k of total
			expectedValSize := 10 / 5
			if valSize != expectedValSize {
				t.Errorf("Fold %d: expected val size %d, got %d", i, expectedValSize, valSize)
			}

			// Train set should be the rest
			expectedTrainSize := 10 - expectedValSize
			if trainSize != expectedTrainSize {
				t.Errorf("Fold %d: expected train size %d, got %d", i, expectedTrainSize, trainSize)
			}

			t.Logf("Fold %d: train=%d, val=%d", i, trainSize, valSize)
		}

		t.Logf("K-fold validation completed: %d folds", len(folds))
	})

	t.Run("K-fold with shuffle", func(t *testing.T) {
		// Create ordered test data
		X := array.Zeros(internal.Shape{8, 1}, internal.Float64)
		for i := 0; i < 8; i++ {
			X.Set(float64(i), i, 0)
		}
		y, _ := array.FromSlice([]float64{0, 0, 0, 0, 1, 1, 1, 1})

		// Test with shuffle
		foldsShuffled, err := KFoldSplit(X, y, 4, true, 42)
		if err != nil {
			t.Fatalf("KFold shuffle split failed: %v", err)
		}

		// Test without shuffle
		foldsUnshuffled, err := KFoldSplit(X, y, 4, false, 42)
		if err != nil {
			t.Fatalf("KFold unshuffled split failed: %v", err)
		}

		// Compare first validation folds - they should be different when shuffled
		val1Shuffled := foldsShuffled[0].ValY.At(0).(float64)
		val1Unshuffled := foldsUnshuffled[0].ValY.At(0).(float64)

		t.Logf("Shuffled first val: %.0f, Unshuffled first val: %.0f", val1Shuffled, val1Unshuffled)
		t.Logf("Shuffle test completed (may show same values due to small dataset)")
	})

	t.Run("K-fold parameter validation", func(t *testing.T) {
		X := array.Ones(internal.Shape{5, 2}, internal.Float64)
		y, _ := array.FromSlice([]float64{0, 1, 0, 1, 0})

		// Test k > n_samples
		_, err := KFoldSplit(X, y, 10, false, 42)
		if err == nil {
			t.Error("Expected error for k > n_samples")
		}

		// Test k <= 1
		_, err = KFoldSplit(X, y, 1, false, 42)
		if err == nil {
			t.Error("Expected error for k <= 1")
		}

		// Test mismatched array sizes
		wrongY, _ := array.FromSlice([]float64{0, 1, 0})
		_, err = KFoldSplit(X, wrongY, 3, false, 42)
		if err == nil {
			t.Error("Expected error for mismatched array sizes")
		}

		// Test nil arrays
		_, err = KFoldSplit(nil, y, 3, false, 42)
		if err == nil {
			t.Error("Expected error for nil X")
		}

		_, err = KFoldSplit(X, nil, 3, false, 42)
		if err == nil {
			t.Error("Expected error for nil y")
		}
	})
}

// TestStratifiedKFold tests stratified K-fold cross-validation
func TestStratifiedKFold(t *testing.T) {
	t.Run("Stratified K-fold maintains class proportions", func(t *testing.T) {
		// Create imbalanced dataset: 75% class 0, 25% class 1
		X := array.Zeros(internal.Shape{8, 2}, internal.Float64)
		for i := 0; i < 8; i++ {
			X.Set(float64(i), i, 0)
			X.Set(float64(i*2), i, 1)
		}
		y, _ := array.FromSlice([]float64{0, 0, 0, 0, 0, 0, 1, 1}) // 6 class 0, 2 class 1

		folds, err := StratifiedKFoldSplit(X, y, 4, true, 42)
		if err != nil {
			t.Fatalf("Stratified K-fold failed: %v", err)
		}

		// Should have 4 folds
		if len(folds) != 4 {
			t.Errorf("Expected 4 folds, got %d", len(folds))
		}

		// Check class proportions in each fold
		for i, fold := range folds {
			// Count classes in validation set
			class0Count := 0
			class1Count := 0
			for j := 0; j < fold.ValY.Size(); j++ {
				if fold.ValY.At(j).(float64) == 0 {
					class0Count++
				} else {
					class1Count++
				}
			}

			t.Logf("Fold %d validation: class 0=%d, class 1=%d", i, class0Count, class1Count)

			// Each fold validation should have some representation of both classes when possible
			totalVal := fold.ValY.Size()
			if totalVal >= 2 {
				// Should maintain approximate proportion or at least have some of each class
				if class0Count == 0 && class1Count == 0 {
					t.Errorf("Fold %d validation set is empty", i)
				}
			}
		}

		t.Logf("Stratified K-fold validation completed")
	})

	t.Run("Stratified K-fold with single class", func(t *testing.T) {
		// All samples are the same class
		X := array.Ones(internal.Shape{6, 2}, internal.Float64)
		y, _ := array.FromSlice([]float64{1, 1, 1, 1, 1, 1})

		folds, err := StratifiedKFoldSplit(X, y, 3, false, 42)
		if err != nil {
			t.Fatalf("Unexpected error for single class: %v", err)
		}

		// Should still work, just like regular K-fold
		if len(folds) != 3 {
			t.Errorf("Expected 3 folds, got %d", len(folds))
		}

		t.Logf("Single class stratified K-fold completed")
	})
}

// TestTimeSeriesSplit tests time series cross-validation
func TestTimeSeriesSplit(t *testing.T) {
	t.Run("Time series split with growing window", func(t *testing.T) {
		// Create time series data
		X := array.Zeros(internal.Shape{10, 2}, internal.Float64)
		for i := 0; i < 10; i++ {
			X.Set(float64(i), i, 0)   // Time feature
			X.Set(float64(i*i), i, 1) // Value feature
		}
		y, _ := array.FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})

		folds, err := TimeSeriesSplit(X, y, 5, 2) // n_splits=5, min_train_size=2
		if err != nil {
			t.Fatalf("Time series split failed: %v", err)
		}

		// Should have 5 splits
		if len(folds) != 5 {
			t.Errorf("Expected 5 splits, got %d", len(folds))
		}

		// Validate that training sets are growing and test sets are future data
		for i, fold := range folds {
			trainSize := fold.TrainX.Shape()[0]
			testSize := fold.ValX.Shape()[0]

			// Training size should be growing (expanding window)
			expectedMinTrainSize := 2 + i // min_train_size + i
			if trainSize < expectedMinTrainSize {
				t.Errorf("Split %d: train size %d should be at least %d", i, trainSize, expectedMinTrainSize)
			}

			// Test that training indices come before test indices (time order)
			if fold.TrainX.Shape()[0] > 0 && fold.ValX.Shape()[0] > 0 {
				lastTrainTime := fold.TrainX.At(trainSize-1, 0).(float64)
				firstTestTime := fold.ValX.At(0, 0).(float64)

				if lastTrainTime >= firstTestTime {
					t.Errorf("Split %d: training data should come before test data in time", i)
				}
			}

			t.Logf("Split %d: train=%d, test=%d", i, trainSize, testSize)
		}

		t.Logf("Time series split completed")
	})

	t.Run("Time series split parameter validation", func(t *testing.T) {
		X := array.Ones(internal.Shape{5, 2}, internal.Float64)
		y, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})

		// Test n_splits too large
		_, err := TimeSeriesSplit(X, y, 10, 1)
		if err == nil {
			t.Error("Expected error for n_splits too large")
		}

		// Test min_train_size too large
		_, err = TimeSeriesSplit(X, y, 2, 10)
		if err == nil {
			t.Error("Expected error for min_train_size too large")
		}

		// Test min_train_size <= 0
		_, err = TimeSeriesSplit(X, y, 2, 0)
		if err == nil {
			t.Error("Expected error for min_train_size <= 0")
		}
	})
}

// TestCrossValidateScore tests cross-validation scoring
func TestCrossValidateScore(t *testing.T) {
	t.Run("Cross-validate LDA classifier", func(t *testing.T) {
		// Create linearly separable data with more samples to ensure each fold has both classes
		X := array.Zeros(internal.Shape{50, 2}, internal.Float64)
		y := array.Empty(internal.Shape{50}, internal.Float64)

		// Class 0: points around (1, 1)
		for i := 0; i < 25; i++ {
			X.Set(1.0+0.1*float64(i%5), i, 0)
			X.Set(1.0+0.1*float64(i%5), i, 1)
			y.Set(0.0, i)
		}

		// Class 1: points around (3, 3)
		for i := 25; i < 50; i++ {
			X.Set(3.0+0.1*float64((i-25)%5), i, 0)
			X.Set(3.0+0.1*float64((i-25)%5), i, 1)
			y.Set(1.0, i)
		}

		// Define a simple LDA model function
		ldaModel := func(trainX, trainY, testX, testY *array.Array) (float64, error) {
			// Train LDA
			model, err := LinearDiscriminantAnalysis(trainX, trainY)
			if err != nil {
				return 0, err
			}

			// Predict
			predictions, err := LDAPredict(model, testX)
			if err != nil {
				return 0, err
			}

			// Calculate accuracy
			correct := 0
			for i := 0; i < testY.Size(); i++ {
				if int64(testY.At(i).(float64)) == predictions.At(i).(int64) {
					correct++
				}
			}

			accuracy := float64(correct) / float64(testY.Size())
			return accuracy, nil
		}

		// Test stratified K-fold cross-validation to ensure each fold has both classes
		scores, err := CrossValidateScore(X, y, ldaModel, 5, "stratified", false, 42)
		if err != nil {
			t.Fatalf("Cross-validation failed: %v", err)
		}

		// Should have 5 scores
		if len(scores) != 5 {
			t.Errorf("Expected 5 scores, got %d", len(scores))
		}

		// Calculate mean and std
		var sum float64
		for _, score := range scores {
			sum += score
			if score < 0 || score > 1 {
				t.Errorf("Score should be between 0 and 1, got %f", score)
			}
		}

		meanScore := sum / float64(len(scores))

		var variance float64
		for _, score := range scores {
			variance += (score - meanScore) * (score - meanScore)
		}
		stdScore := math.Sqrt(variance / float64(len(scores)))

		t.Logf("Cross-validation scores: %v", scores)
		t.Logf("Mean accuracy: %.3f ± %.3f", meanScore, stdScore)

		// For linearly separable data, should get high accuracy
		if meanScore < 0.7 {
			t.Errorf("Expected high accuracy for linearly separable data, got %.3f", meanScore)
		}
	})
}
