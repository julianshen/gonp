package stats

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// CVFold represents a single cross-validation fold
type CVFold struct {
	TrainX *array.Array // Training features
	TrainY *array.Array // Training targets
	ValX   *array.Array // Validation features
	ValY   *array.Array // Validation targets
}

// ModelFunc represents a machine learning model function for cross-validation
// It takes (trainX, trainY, testX, testY) and returns a score and error
type ModelFunc func(*array.Array, *array.Array, *array.Array, *array.Array) (float64, error)

// KFoldSplit performs K-fold cross-validation split
// X: feature matrix (n_samples x n_features)
// y: target vector (n_samples,)
// k: number of folds
// shuffle: whether to shuffle the data before splitting
// randomState: random seed for reproducibility
func KFoldSplit(X, y *array.Array, k int, shuffle bool, randomState int64) ([]*CVFold, error) {
	if X == nil || y == nil {
		return nil, errors.New("X and y cannot be nil")
	}

	if X.Ndim() != 2 || y.Ndim() != 1 {
		return nil, errors.New("X must be 2D and y must be 1D")
	}

	nSamples := X.Shape()[0]
	if y.Size() != nSamples {
		return nil, errors.New("X and y must have the same number of samples")
	}

	if k <= 1 {
		return nil, errors.New("k must be greater than 1")
	}

	if k > nSamples {
		return nil, fmt.Errorf("k (%d) cannot be greater than n_samples (%d)", k, nSamples)
	}

	// Create indices array
	indices := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		indices[i] = i
	}

	// Shuffle if requested
	if shuffle {
		rng := rand.New(rand.NewSource(randomState))
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	// Create folds
	folds := make([]*CVFold, k)
	foldSize := nSamples / k
	remainder := nSamples % k

	start := 0
	for i := 0; i < k; i++ {
		// Current fold size (add 1 to early folds if there's remainder)
		currentFoldSize := foldSize
		if i < remainder {
			currentFoldSize++
		}

		// Validation indices for this fold
		valIndices := indices[start : start+currentFoldSize]

		// Training indices (all others)
		trainIndices := make([]int, 0, nSamples-currentFoldSize)
		trainIndices = append(trainIndices, indices[:start]...)
		trainIndices = append(trainIndices, indices[start+currentFoldSize:]...)

		// Create training and validation arrays
		fold := &CVFold{}

		// Create training arrays
		fold.TrainX = extractRowsByIndices(X, trainIndices)
		fold.TrainY = extractRowsByIndices(y, trainIndices)

		// Create validation arrays
		fold.ValX = extractRowsByIndices(X, valIndices)
		fold.ValY = extractRowsByIndices(y, valIndices)

		folds[i] = fold
		start += currentFoldSize
	}

	return folds, nil
}

// StratifiedKFoldSplit performs stratified K-fold cross-validation split
// Maintains class distribution in each fold
func StratifiedKFoldSplit(X, y *array.Array, k int, shuffle bool, randomState int64) ([]*CVFold, error) {
	if X == nil || y == nil {
		return nil, errors.New("X and y cannot be nil")
	}

	if X.Ndim() != 2 || y.Ndim() != 1 {
		return nil, errors.New("X must be 2D and y must be 1D")
	}

	nSamples := X.Shape()[0]
	if y.Size() != nSamples {
		return nil, errors.New("X and y must have the same number of samples")
	}

	if k <= 1 {
		return nil, errors.New("k must be greater than 1")
	}

	if k > nSamples {
		return nil, fmt.Errorf("k (%d) cannot be greater than n_samples (%d)", k, nSamples)
	}

	// Get unique classes and their indices
	classIndices := make(map[int][]int)
	for i := 0; i < nSamples; i++ {
		class := int(convertToFloat64(y.At(i)))
		classIndices[class] = append(classIndices[class], i)
	}

	// If only one class, fall back to regular K-fold
	if len(classIndices) == 1 {
		return KFoldSplit(X, y, k, shuffle, randomState)
	}

	// Shuffle each class's indices if requested
	rng := rand.New(rand.NewSource(randomState))
	if shuffle {
		for class, indices := range classIndices {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
			classIndices[class] = indices
		}
	}

	// First, assign validation indices for each fold by distributing each class
	valIndicesByFold := make([][]int, k)

	// For each class, distribute samples across folds for validation
	for _, indices := range classIndices {
		classSize := len(indices)
		foldSize := classSize / k
		remainder := classSize % k

		start := 0
		for foldIdx := 0; foldIdx < k; foldIdx++ {
			// Current fold size for this class
			currentFoldSize := foldSize
			if foldIdx < remainder {
				currentFoldSize++
			}

			// Add indices to this fold's validation set
			end := start + currentFoldSize
			if end > classSize {
				end = classSize
			}

			valIndicesForClass := indices[start:end]
			valIndicesByFold[foldIdx] = append(valIndicesByFold[foldIdx], valIndicesForClass...)

			start = end
		}
	}

	// Now create training indices for each fold (everything not in that fold's validation)
	trainIndicesByFold := make([][]int, k)
	for foldIdx := 0; foldIdx < k; foldIdx++ {
		// Create a set of validation indices for quick lookup
		valIndexSet := make(map[int]bool)
		for _, idx := range valIndicesByFold[foldIdx] {
			valIndexSet[idx] = true
		}

		// Training indices are all sample indices not in validation for this fold
		for i := 0; i < nSamples; i++ {
			if !valIndexSet[i] {
				trainIndicesByFold[foldIdx] = append(trainIndicesByFold[foldIdx], i)
			}
		}
	}

	// Create folds using collected indices
	folds := make([]*CVFold, k)
	for i := 0; i < k; i++ {
		folds[i] = &CVFold{
			TrainX: extractRowsByIndices(X, trainIndicesByFold[i]),
			TrainY: extractRowsByIndices(y, trainIndicesByFold[i]),
			ValX:   extractRowsByIndices(X, valIndicesByFold[i]),
			ValY:   extractRowsByIndices(y, valIndicesByFold[i]),
		}
	}

	return folds, nil
}

// TimeSeriesSplit performs time series cross-validation split
// Creates expanding window splits where training data always comes before test data
func TimeSeriesSplit(X, y *array.Array, nSplits, minTrainSize int) ([]*CVFold, error) {
	if X == nil || y == nil {
		return nil, errors.New("X and y cannot be nil")
	}

	if X.Ndim() != 2 || y.Ndim() != 1 {
		return nil, errors.New("X must be 2D and y must be 1D")
	}

	nSamples := X.Shape()[0]
	if y.Size() != nSamples {
		return nil, errors.New("X and y must have the same number of samples")
	}

	if minTrainSize <= 0 {
		return nil, errors.New("minTrainSize must be greater than 0")
	}

	if minTrainSize >= nSamples {
		return nil, fmt.Errorf("minTrainSize (%d) must be less than n_samples (%d)", minTrainSize, nSamples)
	}

	if nSplits <= 0 {
		return nil, errors.New("nSplits must be greater than 0")
	}

	// Check if we have enough data for the requested splits
	maxTrainSize := nSamples - 1 // Need at least 1 sample for test
	if minTrainSize+nSplits-1 > maxTrainSize {
		return nil, fmt.Errorf("not enough samples for %d splits with minTrainSize=%d", nSplits, minTrainSize)
	}

	folds := make([]*CVFold, nSplits)

	for i := 0; i < nSplits; i++ {
		trainSize := minTrainSize + i
		testStart := trainSize
		testSize := 1 // Use 1 sample for test (can be adjusted)

		// Ensure we don't exceed sample bounds
		if testStart+testSize > nSamples {
			testSize = nSamples - testStart
		}

		if testSize <= 0 {
			return nil, fmt.Errorf("insufficient samples for split %d", i)
		}

		// Create training indices (0 to trainSize-1)
		trainIndices := make([]int, trainSize)
		for j := 0; j < trainSize; j++ {
			trainIndices[j] = j
		}

		// Create test indices (testStart to testStart+testSize-1)
		testIndices := make([]int, testSize)
		for j := 0; j < testSize; j++ {
			testIndices[j] = testStart + j
		}

		// Create fold
		fold := &CVFold{
			TrainX: extractRowsByIndices(X, trainIndices),
			TrainY: extractRowsByIndices(y, trainIndices),
			ValX:   extractRowsByIndices(X, testIndices),
			ValY:   extractRowsByIndices(y, testIndices),
		}

		folds[i] = fold
	}

	return folds, nil
}

// CrossValidateScore performs cross-validation with a given model function
func CrossValidateScore(X, y *array.Array, modelFunc ModelFunc, k int, cvType string, shuffle bool, randomState int64) ([]float64, error) {
	if modelFunc == nil {
		return nil, errors.New("modelFunc cannot be nil")
	}

	var folds []*CVFold
	var err error

	switch cvType {
	case "kfold":
		folds, err = KFoldSplit(X, y, k, shuffle, randomState)
	case "stratified":
		folds, err = StratifiedKFoldSplit(X, y, k, shuffle, randomState)
	case "timeseries":
		// For time series, use k as nSplits and assume minTrainSize = k
		folds, err = TimeSeriesSplit(X, y, k, k)
	default:
		return nil, fmt.Errorf("unsupported cv type: %s", cvType)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create folds: %v", err)
	}

	// Evaluate model on each fold
	scores := make([]float64, len(folds))
	for i, fold := range folds {
		score, err := modelFunc(fold.TrainX, fold.TrainY, fold.ValX, fold.ValY)
		if err != nil {
			return nil, fmt.Errorf("model evaluation failed on fold %d: %v", i, err)
		}
		scores[i] = score
	}

	return scores, nil
}

// Helper functions

// extractRowsByIndices extracts rows from an array by indices
func extractRowsByIndices(arr *array.Array, indices []int) *array.Array {
	if len(indices) == 0 {
		if arr.Ndim() == 1 {
			return array.Empty(internal.Shape{0}, arr.DType())
		} else {
			return array.Empty(internal.Shape{0, arr.Shape()[1]}, arr.DType())
		}
	}

	if arr.Ndim() == 1 {
		// 1D array
		result := array.Empty(internal.Shape{len(indices)}, arr.DType())
		for i, idx := range indices {
			result.Set(arr.At(idx), i)
		}
		return result
	} else {
		// 2D array
		nFeatures := arr.Shape()[1]
		result := array.Empty(internal.Shape{len(indices), nFeatures}, arr.DType())
		for i, idx := range indices {
			for j := 0; j < nFeatures; j++ {
				result.Set(arr.At(idx, j), i, j)
			}
		}
		return result
	}
}

// concatenateArraysVertical concatenates two arrays vertically (along rows)
func concatenateArraysVertical(arr1, arr2 *array.Array) *array.Array {
	if arr1 == nil {
		return arr2
	}
	if arr2 == nil {
		return arr1
	}

	if arr1.Ndim() != arr2.Ndim() {
		return arr1 // Error case, return first array
	}

	if arr1.Ndim() == 1 {
		// 1D arrays
		totalSize := arr1.Size() + arr2.Size()
		result := array.Empty(internal.Shape{totalSize}, arr1.DType())

		// Copy first array
		for i := 0; i < arr1.Size(); i++ {
			result.Set(arr1.At(i), i)
		}

		// Copy second array
		for i := 0; i < arr2.Size(); i++ {
			result.Set(arr2.At(i), arr1.Size()+i)
		}

		return result
	} else {
		// 2D arrays
		nRows1 := arr1.Shape()[0]
		nRows2 := arr2.Shape()[0]
		nCols := arr1.Shape()[1]

		if nCols != arr2.Shape()[1] {
			return arr1 // Error case, return first array
		}

		totalRows := nRows1 + nRows2
		result := array.Empty(internal.Shape{totalRows, nCols}, arr1.DType())

		// Copy first array
		for i := 0; i < nRows1; i++ {
			for j := 0; j < nCols; j++ {
				result.Set(arr1.At(i, j), i, j)
			}
		}

		// Copy second array
		for i := 0; i < nRows2; i++ {
			for j := 0; j < nCols; j++ {
				result.Set(arr2.At(i, j), nRows1+i, j)
			}
		}

		return result
	}
}
