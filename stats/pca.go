package stats

import (
	"errors"
	"fmt"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	gomath "github.com/julianshen/gonp/math"
)

// PrincipalComponentAnalysis performs dimensionality reduction using Principal Component Analysis
// following the scikit-learn pattern with Fit/Transform methods
type PrincipalComponentAnalysis struct {
	NComponents            int          // Number of components to keep
	Components             *array.Array // Principal component vectors (n_components x n_features)
	ExplainedVariance      []float64    // Variance explained by each component
	ExplainedVarianceRatio []float64    // Fraction of variance explained by each component
	SingularValues         []float64    // Singular values from SVD
	Mean                   *array.Array // Mean of the training data (for centering)
	fitted                 bool         // Whether the PCA has been fitted
}

// NewPCA creates a new PCA instance
func NewPCA(nComponents int) *PrincipalComponentAnalysis {
	return &PrincipalComponentAnalysis{
		NComponents: nComponents,
		fitted:      false,
	}
}

// Fit computes the principal components on training data
func (pca *PrincipalComponentAnalysis) Fit(X *array.Array) error {
	if X == nil {
		return errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nSamples < 2 {
		return errors.New("need at least 2 samples for PCA")
	}

	if pca.NComponents <= 0 {
		return errors.New("n_components must be greater than 0")
	}

	if pca.NComponents > nFeatures {
		return errors.New("n_components cannot exceed number of features")
	}

	// Step 1: Center the data (subtract mean from each feature)
	pca.Mean = array.Zeros(internal.Shape{nFeatures}, internal.Float64)

	// Compute mean for each feature
	for j := 0; j < nFeatures; j++ {
		sum := 0.0
		for i := 0; i < nSamples; i++ {
			sum += convertToFloat64(X.At(i, j))
		}
		mean := sum / float64(nSamples)
		pca.Mean.Set(mean, j)
	}

	// Create centered data matrix
	centeredData := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(X.At(i, j))
			mean := convertToFloat64(pca.Mean.At(j))
			centeredData.Set(value-mean, i, j)
		}
	}

	// Step 2: Perform SVD on centered data
	svdResult, err := gomath.SVD(centeredData)
	if err != nil {
		return fmt.Errorf("SVD decomposition failed: %v", err)
	}

	// Step 3: Extract principal components and compute explained variance
	pca.Components = array.Empty(internal.Shape{pca.NComponents, nFeatures}, internal.Float64)
	pca.ExplainedVariance = make([]float64, pca.NComponents)
	pca.SingularValues = make([]float64, pca.NComponents)

	// The principal components are the columns of V (right singular vectors)
	v := svdResult.V

	for i := 0; i < pca.NComponents; i++ {
		// Extract i-th principal component (i-th column of V)
		for j := 0; j < nFeatures; j++ {
			val := convertToFloat64(v.At(j, i))
			pca.Components.Set(val, i, j)
		}

		// Singular value and explained variance
		singularVal := convertToFloat64(svdResult.S.At(i))
		pca.SingularValues[i] = singularVal

		// Explained variance = (singular_value^2) / (n_samples - 1)
		pca.ExplainedVariance[i] = (singularVal * singularVal) / float64(nSamples-1)
	}

	// Step 4: Compute explained variance ratios
	totalVariance := 0.0
	for i := 0; i < svdResult.S.Size(); i++ {
		s := convertToFloat64(svdResult.S.At(i))
		totalVariance += (s * s) / float64(nSamples-1)
	}

	pca.ExplainedVarianceRatio = make([]float64, pca.NComponents)
	for i := 0; i < pca.NComponents; i++ {
		if totalVariance > 0 {
			pca.ExplainedVarianceRatio[i] = pca.ExplainedVariance[i] / totalVariance
		}
	}

	pca.fitted = true
	return nil
}

// Transform projects data onto the principal components
func (pca *PrincipalComponentAnalysis) Transform(X *array.Array) (*array.Array, error) {
	if !pca.fitted {
		return nil, errors.New("PCA must be fitted before transform")
	}

	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nFeatures != pca.Mean.Size() {
		return nil, errors.New("X has different number of features than fitted data")
	}

	// Center the data using fitted mean
	centeredData := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(X.At(i, j))
			mean := convertToFloat64(pca.Mean.At(j))
			centeredData.Set(value-mean, i, j)
		}
	}

	// Project onto principal components
	transformed := array.Empty(internal.Shape{nSamples, pca.NComponents}, internal.Float64)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < pca.NComponents; j++ {
			// Compute dot product of sample i with component j
			dotProduct := 0.0
			for k := 0; k < nFeatures; k++ {
				sampleVal := convertToFloat64(centeredData.At(i, k))
				componentVal := convertToFloat64(pca.Components.At(j, k))
				dotProduct += sampleVal * componentVal
			}
			transformed.Set(dotProduct, i, j)
		}
	}

	return transformed, nil
}

// FitTransform fits the PCA and transforms the data in one step
func (pca *PrincipalComponentAnalysis) FitTransform(X *array.Array) (*array.Array, error) {
	err := pca.Fit(X)
	if err != nil {
		return nil, err
	}
	return pca.Transform(X)
}

// InverseTransform transforms data from principal component space back to original space
func (pca *PrincipalComponentAnalysis) InverseTransform(X *array.Array) (*array.Array, error) {
	if !pca.fitted {
		return nil, errors.New("PCA must be fitted before inverse_transform")
	}

	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nComponents := X.Shape()[1]
	nFeatures := pca.Mean.Size()

	if nComponents != pca.NComponents {
		return nil, errors.New("X has different number of components than fitted PCA")
	}

	// Reconstruct data: X_reconstructed = X_transformed * Components
	reconstructed := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			// Compute sum over components
			sum := 0.0
			for k := 0; k < nComponents; k++ {
				transformedVal := convertToFloat64(X.At(i, k))
				componentVal := convertToFloat64(pca.Components.At(k, j))
				sum += transformedVal * componentVal
			}

			// Add back the mean
			meanVal := convertToFloat64(pca.Mean.At(j))
			reconstructed.Set(sum+meanVal, i, j)
		}
	}

	return reconstructed, nil
}

// PCAResult contains the results of Principal Component Analysis
type PCAResult struct {
	Components             []*array.Array // Principal component vectors
	ExplainedVariance      []float64      // Variance explained by each component
	ExplainedVarianceRatio []float64      // Fraction of variance explained by each component
	SingularValues         []float64      // Singular values from SVD
	Mean                   *array.Array   // Mean of the original data (for centering)
	TransformedData        *array.Array   // Data projected onto principal components
}

// PCA performs Principal Component Analysis on the input data
//
// Parameters:
//
//	data: Input data matrix (n_samples x n_features)
//	nComponents: Number of principal components to retain
//
// Returns: PCAResult with components, explained variance, and transformed data
func PCA(data *array.Array, nComponents int) (*PCAResult, error) {
	if data == nil {
		return nil, fmt.Errorf("data array cannot be nil")
	}

	if data.Ndim() != 2 {
		return nil, fmt.Errorf("data must be 2-dimensional, got %dD", data.Ndim())
	}

	shape := data.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	if nSamples == 0 || nFeatures == 0 {
		return nil, fmt.Errorf("data cannot be empty")
	}

	if nComponents <= 0 {
		return nil, fmt.Errorf("number of components must be positive, got %d", nComponents)
	}

	if nComponents > nFeatures {
		return nil, fmt.Errorf("number of components (%d) cannot exceed number of features (%d)",
			nComponents, nFeatures)
	}

	// Step 1: Center the data (subtract mean from each feature)
	mean := array.Empty(internal.Shape{nFeatures}, internal.Float64)

	// Compute mean for each feature
	for j := 0; j < nFeatures; j++ {
		sum := 0.0
		for i := 0; i < nSamples; i++ {
			val := convertToFloat64(data.At(i, j))
			sum += val
		}
		mean.Set(sum/float64(nSamples), j)
	}

	// Create centered data matrix
	centeredData := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			val := convertToFloat64(data.At(i, j))
			meanVal := convertToFloat64(mean.At(j))
			centeredData.Set(val-meanVal, i, j)
		}
	}

	// Step 2: Perform SVD on centered data
	// For PCA, we use SVD of X = U * S * V^T
	// Principal components are the columns of V (right singular vectors)
	svdResult, err := gomath.SVD(centeredData)
	if err != nil {
		return nil, fmt.Errorf("SVD decomposition failed: %v", err)
	}

	// Step 3: Extract principal components and compute explained variance
	components := make([]*array.Array, nComponents)
	explainedVariance := make([]float64, nComponents)
	singularValues := make([]float64, nComponents)

	// The principal components are the columns of V
	// In our SVDResult, V contains the right singular vectors as columns
	v := svdResult.V

	for i := 0; i < nComponents; i++ {
		// Extract i-th principal component (i-th column of V)
		component := array.Empty(internal.Shape{nFeatures}, internal.Float64)
		for j := 0; j < nFeatures; j++ {
			val := convertToFloat64(v.At(j, i))
			component.Set(val, j)
		}
		components[i] = component

		// Singular value and explained variance
		singularVal := convertToFloat64(svdResult.S.At(i))
		singularValues[i] = singularVal

		// Explained variance = (singular_value^2) / (n_samples - 1)
		explainedVariance[i] = (singularVal * singularVal) / float64(nSamples-1)
	}

	// Step 4: Compute explained variance ratios
	totalVariance := 0.0
	for i := 0; i < svdResult.S.Size(); i++ {
		s := convertToFloat64(svdResult.S.At(i))
		totalVariance += (s * s) / float64(nSamples-1)
	}

	explainedVarianceRatio := make([]float64, nComponents)
	for i := 0; i < nComponents; i++ {
		if totalVariance > 0 {
			explainedVarianceRatio[i] = explainedVariance[i] / totalVariance
		}
	}

	// Step 5: Transform data to principal component space
	transformedData, err := transformData(centeredData, components)
	if err != nil {
		return nil, fmt.Errorf("data transformation failed: %v", err)
	}

	result := &PCAResult{
		Components:             components,
		ExplainedVariance:      explainedVariance,
		ExplainedVarianceRatio: explainedVarianceRatio,
		SingularValues:         singularValues,
		Mean:                   mean,
		TransformedData:        transformedData,
	}

	return result, nil
}

// transformData projects the data onto the principal components
func transformData(data *array.Array, components []*array.Array) (*array.Array, error) {
	shape := data.Shape()
	nSamples := shape[0]
	nComponents := len(components)

	transformed := array.Empty(internal.Shape{nSamples, nComponents}, internal.Float64)

	// For each sample, compute dot product with each component
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nComponents; j++ {
			// Compute dot product of sample i with component j
			dotProduct := 0.0
			for k := 0; k < components[j].Size(); k++ {
				sampleVal := convertToFloat64(data.At(i, k))
				componentVal := convertToFloat64(components[j].At(k))
				dotProduct += sampleVal * componentVal
			}
			transformed.Set(dotProduct, i, j)
		}
	}

	return transformed, nil
}

// PCATransform applies PCA transformation to new data using existing PCA result
//
// Parameters:
//
//	data: New data to transform (same number of features as original)
//	pcaResult: Previously computed PCA result
//
// Returns: Transformed data in principal component space
func PCATransform(data *array.Array, pcaResult *PCAResult) (*array.Array, error) {
	if data == nil {
		return nil, fmt.Errorf("data array cannot be nil")
	}
	if pcaResult == nil {
		return nil, fmt.Errorf("PCA result cannot be nil")
	}

	if data.Ndim() != 2 {
		return nil, fmt.Errorf("data must be 2-dimensional")
	}

	shape := data.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	if nFeatures != pcaResult.Mean.Size() {
		return nil, fmt.Errorf("data features (%d) must match PCA features (%d)",
			nFeatures, pcaResult.Mean.Size())
	}

	// Center the new data using stored mean
	centeredData := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			val := convertToFloat64(data.At(i, j))
			meanVal := convertToFloat64(pcaResult.Mean.At(j))
			centeredData.Set(val-meanVal, i, j)
		}
	}

	// Transform using existing components
	return transformData(centeredData, pcaResult.Components)
}

// PCAInverseTransform transforms data from principal component space back to original space
//
// Parameters:
//
//	transformedData: Data in principal component space
//	pcaResult: PCA result containing components and mean
//
// Returns: Data reconstructed in original feature space
func PCAInverseTransform(transformedData *array.Array, pcaResult *PCAResult) (*array.Array, error) {
	if transformedData == nil {
		return nil, fmt.Errorf("transformed data cannot be nil")
	}
	if pcaResult == nil {
		return nil, fmt.Errorf("PCA result cannot be nil")
	}

	if transformedData.Ndim() != 2 {
		return nil, fmt.Errorf("transformed data must be 2-dimensional")
	}

	shape := transformedData.Shape()
	nSamples := shape[0]
	nComponents := shape[1]
	nFeatures := pcaResult.Mean.Size()

	if nComponents != len(pcaResult.Components) {
		return nil, fmt.Errorf("transformed data components (%d) must match PCA components (%d)",
			nComponents, len(pcaResult.Components))
	}

	// Reconstruct data: X_reconstructed = X_transformed * Components^T
	reconstructed := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			// Compute sum over components
			sum := 0.0
			for k := 0; k < nComponents; k++ {
				transformedVal := convertToFloat64(transformedData.At(i, k))
				componentVal := convertToFloat64(pcaResult.Components[k].At(j))
				sum += transformedVal * componentVal
			}

			// Add back the mean
			meanVal := convertToFloat64(pcaResult.Mean.At(j))
			reconstructed.Set(sum+meanVal, i, j)
		}
	}

	return reconstructed, nil
}
