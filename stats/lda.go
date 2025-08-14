package stats

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// LDAResult represents the result of Linear Discriminant Analysis
type LDAResult struct {
	Classes      []int        // Unique class labels
	ClassMeans   *array.Array // Mean vectors for each class (n_classes x n_features)
	Coefficients *array.Array // Linear discriminant coefficients
	Intercept    *array.Array // Intercept terms for each class
	Priors       []float64    // Prior probabilities for each class
	CovMatrix    *array.Array // Pooled within-class covariance matrix
	Explained    []float64    // Explained variance ratio for each component
}

// LinearDiscriminantAnalysis performs linear discriminant analysis for classification
// X: feature matrix (n_samples x n_features)
// y: target labels (n_samples,) - integer class labels
func LinearDiscriminantAnalysis(X, y *array.Array) (*LDAResult, error) {
	ctx := internal.StartProfiler("Stats.LinearDiscriminantAnalysis")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	// Input validation
	if X == nil || y == nil {
		return nil, errors.New("input arrays cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	if y.Ndim() != 1 {
		return nil, errors.New("y must be a 1D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if y.Size() != nSamples {
		return nil, errors.New("X and y must have the same number of samples")
	}

	if nSamples < 2 {
		return nil, errors.New("need at least 2 samples for LDA")
	}

	// Extract unique classes
	classes := extractUniqueClasses(y)
	nClasses := len(classes)

	if nClasses < 2 {
		return nil, errors.New("need at least 2 classes for LDA")
	}

	// Calculate class means and priors
	classMeans := array.Zeros(internal.Shape{nClasses, nFeatures}, internal.Float64)
	priors := make([]float64, nClasses)
	classCounts := make([]int, nClasses)

	for i, class := range classes {
		// Find samples belonging to this class
		classIndices := findClassIndices(y, class)
		classCounts[i] = len(classIndices)
		priors[i] = float64(classCounts[i]) / float64(nSamples)

		// Calculate class mean
		if len(classIndices) == 0 {
			continue
		}

		for j := 0; j < nFeatures; j++ {
			var sum float64
			for _, idx := range classIndices {
				sum += convertToFloat64(X.At(idx, j))
			}
			mean := sum / float64(len(classIndices))
			classMeans.Set(mean, i, j)
		}
	}

	// Calculate pooled within-class covariance matrix
	covMatrix, err := calculatePooledCovariance(X, y, classes, classMeans)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate covariance matrix: %v", err)
	}

	// Calculate discriminant coefficients using pseudo-inverse
	coefficients, intercept, explained, err := calculateDiscriminantCoefficients(
		classMeans, covMatrix, priors)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate discriminant coefficients: %v", err)
	}

	return &LDAResult{
		Classes:      classes,
		ClassMeans:   classMeans,
		Coefficients: coefficients,
		Intercept:    intercept,
		Priors:       priors,
		CovMatrix:    covMatrix,
		Explained:    explained,
	}, nil
}

// LDAPredict predicts class labels for new data using trained LDA model
func LDAPredict(model *LDAResult, X *array.Array) (*array.Array, error) {
	if model == nil {
		return nil, errors.New("model cannot be nil")
	}
	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nFeatures != model.ClassMeans.Shape()[1] {
		return nil, fmt.Errorf("X has %d features but model was trained on %d features",
			nFeatures, model.ClassMeans.Shape()[1])
	}

	predictions := array.Empty(internal.Shape{nSamples}, internal.Int64)

	// For each sample, calculate the discriminant score for each class
	for i := 0; i < nSamples; i++ {
		bestClass := model.Classes[0]
		bestScore := math.Inf(-1)

		for j, class := range model.Classes {
			score, err := calculateDiscriminantScore(X, i, model, j)
			if err != nil {
				return nil, fmt.Errorf("failed to calculate discriminant score: %v", err)
			}

			if score > bestScore {
				bestScore = score
				bestClass = class
			}
		}

		predictions.Set(int64(bestClass), i) // Ensure we set as int64
	}

	return predictions, nil
}

// LDAPredictProba predicts class probabilities for new data using trained LDA model
func LDAPredictProba(model *LDAResult, X *array.Array) (*array.Array, error) {
	if model == nil {
		return nil, errors.New("model cannot be nil")
	}
	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]
	nClasses := len(model.Classes)

	if nFeatures != model.ClassMeans.Shape()[1] {
		return nil, fmt.Errorf("X has %d features but model was trained on %d features",
			nFeatures, model.ClassMeans.Shape()[1])
	}

	probabilities := array.Zeros(internal.Shape{nSamples, nClasses}, internal.Float64)

	// For each sample, calculate probabilities using discriminant scores
	for i := 0; i < nSamples; i++ {
		scores := make([]float64, nClasses)
		maxScore := math.Inf(-1)

		// Calculate discriminant scores for all classes
		for j := range model.Classes {
			score, err := calculateDiscriminantScore(X, i, model, j)
			if err != nil {
				return nil, fmt.Errorf("failed to calculate discriminant score: %v", err)
			}
			scores[j] = score
			if score > maxScore {
				maxScore = score
			}
		}

		// Convert scores to probabilities using softmax
		var sumExp float64
		for j := range scores {
			scores[j] = math.Exp(scores[j] - maxScore) // Subtract max for numerical stability
			sumExp += scores[j]
		}

		// Normalize to get probabilities
		for j := range scores {
			probabilities.Set(scores[j]/sumExp, i, j)
		}
	}

	return probabilities, nil
}

// Helper functions

// extractUniqueClasses finds unique class labels from y array
func extractUniqueClasses(y *array.Array) []int {
	classMap := make(map[int]bool)
	for i := 0; i < y.Size(); i++ {
		class := int(convertToFloat64(y.At(i)))
		classMap[class] = true
	}

	var classes []int
	for class := range classMap {
		classes = append(classes, class)
	}
	sort.Ints(classes) // Sort for consistent ordering
	return classes
}

// findClassIndices returns indices where y equals the given class
func findClassIndices(y *array.Array, class int) []int {
	var indices []int
	for i := 0; i < y.Size(); i++ {
		if int(convertToFloat64(y.At(i))) == class {
			indices = append(indices, i)
		}
	}
	return indices
}

// calculatePooledCovariance calculates the pooled within-class covariance matrix
func calculatePooledCovariance(X, y *array.Array, classes []int, classMeans *array.Array) (*array.Array, error) {
	nFeatures := X.Shape()[1]
	nClasses := len(classes)

	// Initialize pooled covariance matrix
	covMatrix := array.Zeros(internal.Shape{nFeatures, nFeatures}, internal.Float64)

	totalWithinCount := 0

	// For each class, calculate within-class covariance and add to pool
	for i, class := range classes {
		classIndices := findClassIndices(y, class)
		if len(classIndices) <= 1 {
			continue // Skip classes with insufficient samples
		}

		// Calculate within-class covariance for this class
		classCov := array.Zeros(internal.Shape{nFeatures, nFeatures}, internal.Float64)

		for _, idx1 := range classIndices {
			for j := 0; j < nFeatures; j++ {
				for k := 0; k < nFeatures; k++ {
					xj := convertToFloat64(X.At(idx1, j))
					xk := convertToFloat64(X.At(idx1, k))
					meanJ := convertToFloat64(classMeans.At(i, j))
					meanK := convertToFloat64(classMeans.At(i, k))

					currentCov := convertToFloat64(classCov.At(j, k))
					newCov := currentCov + (xj-meanJ)*(xk-meanK)
					classCov.Set(newCov, j, k)
				}
			}
		}

		// Add this class's covariance to the pooled matrix
		for j := 0; j < nFeatures; j++ {
			for k := 0; k < nFeatures; k++ {
				pooledCov := convertToFloat64(covMatrix.At(j, k))
				classCovVal := convertToFloat64(classCov.At(j, k))
				covMatrix.Set(pooledCov+classCovVal, j, k)
			}
		}

		totalWithinCount += len(classIndices)
	}

	// Normalize by total within-class degrees of freedom
	dof := totalWithinCount - nClasses
	if dof <= 0 {
		return nil, errors.New("insufficient degrees of freedom for covariance calculation")
	}

	// Normalize covariance matrix
	for j := 0; j < nFeatures; j++ {
		for k := 0; k < nFeatures; k++ {
			currentCov := convertToFloat64(covMatrix.At(j, k))
			normalizedCov := currentCov / float64(dof)

			// Add small regularization to diagonal elements to prevent singularity
			if j == k && normalizedCov < 1e-10 {
				normalizedCov = 1e-6 // Reasonable regularization for diagonal
			}

			covMatrix.Set(normalizedCov, j, k)
		}
	}

	return covMatrix, nil
}

// calculateDiscriminantCoefficients calculates the linear discriminant coefficients
func calculateDiscriminantCoefficients(classMeans, covMatrix *array.Array, priors []float64) (*array.Array, *array.Array, []float64, error) {
	nClasses := classMeans.Shape()[0]
	nFeatures := classMeans.Shape()[1]

	// For binary classification, we can use a simplified approach
	if nClasses == 2 {
		return calculateBinaryDiscriminant(classMeans, covMatrix, priors)
	}

	// For multi-class, use eigenvalue decomposition of S_w^(-1) S_b
	// For now, implement a simplified version using pseudo-inverse
	coefficients := array.Zeros(internal.Shape{nFeatures, nClasses - 1}, internal.Float64)
	intercept := array.Zeros(internal.Shape{nClasses}, internal.Float64)

	// Calculate intercept terms (simplified)
	for i := 0; i < nClasses; i++ {
		intercept.Set(math.Log(priors[i]), i)
	}

	// Simplified coefficients (identity for now - would need proper implementation)
	for i := 0; i < nFeatures && i < nClasses-1; i++ {
		coefficients.Set(1.0, i, i)
	}

	// Placeholder explained variance
	explained := make([]float64, nClasses-1)
	for i := range explained {
		explained[i] = 1.0 / float64(nClasses-1)
	}

	return coefficients, intercept, explained, nil
}

// calculateBinaryDiscriminant calculates discriminant for binary classification
func calculateBinaryDiscriminant(classMeans, covMatrix *array.Array, priors []float64) (*array.Array, *array.Array, []float64, error) {
	nFeatures := classMeans.Shape()[1]

	// Calculate difference between class means
	meanDiff := array.Empty(internal.Shape{nFeatures}, internal.Float64)
	for i := 0; i < nFeatures; i++ {
		mean0 := convertToFloat64(classMeans.At(0, i))
		mean1 := convertToFloat64(classMeans.At(1, i))
		meanDiff.Set(mean1-mean0, i)
	}

	// For simplified implementation, use diagonal approximation of inverse covariance matrix
	// In practice, would use proper matrix inversion or regularization
	invCov := array.Zeros(internal.Shape{nFeatures, nFeatures}, internal.Float64)
	for i := 0; i < nFeatures; i++ {
		diagElement := convertToFloat64(covMatrix.At(i, i))
		if diagElement > 1e-10 {
			invCov.Set(1.0/diagElement, i, i)
		}
	}

	// Calculate discriminant coefficients: Σ^(-1) * (μ1 - μ0)
	coefficients := array.Empty(internal.Shape{nFeatures, 1}, internal.Float64)
	for i := 0; i < nFeatures; i++ {
		coef := 0.0
		for j := 0; j < nFeatures; j++ {
			invCovVal := convertToFloat64(invCov.At(i, j))
			meanDiffVal := convertToFloat64(meanDiff.At(j))
			coef += invCovVal * meanDiffVal
		}
		coefficients.Set(coef, i, 0)
	}

	// Calculate intercept
	intercept := array.Zeros(internal.Shape{2}, internal.Float64)
	intercept.Set(math.Log(priors[0]), 0)
	intercept.Set(math.Log(priors[1]), 1)

	explained := []float64{1.0} // Single discriminant component explains all variance

	return coefficients, intercept, explained, nil
}

// calculateDiscriminantScore calculates the discriminant score for a sample and class
func calculateDiscriminantScore(X *array.Array, sampleIdx int, model *LDAResult, classIdx int) (float64, error) {
	nFeatures := X.Shape()[1]

	// Start with log prior probability
	score := math.Log(model.Priors[classIdx])

	// Calculate the Gaussian discriminant function: log(P(x|y)) + log(P(y))
	// For multivariate Gaussian: log(P(x|y)) = -0.5 * (x-μ)^T Σ^(-1) (x-μ) - 0.5*log|Σ| - k/2*log(2π)
	// We skip the constant terms since they're the same for all classes

	var quadraticTerm float64
	var logDet float64

	for i := 0; i < nFeatures; i++ {
		x := convertToFloat64(X.At(sampleIdx, i))
		mean := convertToFloat64(model.ClassMeans.At(classIdx, i))
		variance := convertToFloat64(model.CovMatrix.At(i, i))

		if variance > 1e-10 {
			diff := x - mean
			quadraticTerm += (diff * diff) / variance
			logDet += math.Log(variance)
		} else {
			// Handle near-zero variance with regularization
			variance = 1e-3 // More reasonable regularization
			diff := x - mean
			quadraticTerm += (diff * diff) / variance
			logDet += math.Log(variance)
		}
	}

	// Combine terms: log(prior) - 0.5 * quadratic - 0.5 * log(det)
	score = score - 0.5*quadraticTerm - 0.5*logDet

	return score, nil
}
