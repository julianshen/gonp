package stats

import (
	"errors"
	"fmt"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// PolynomialFeatures generates polynomial features from input features
// following the scikit-learn pattern with Fit/Transform methods
type PolynomialFeatures struct {
	Degree          int      // Maximum degree of polynomial features
	InteractionOnly bool     // Include only interaction features (no powers)
	IncludeBias     bool     // Include bias column (constant term)
	InputFeatures   int      // Number of input features (learned during fit)
	OutputFeatures  int      // Number of output features (computed during fit)
	FeatureNames    []string // Names of generated features
	fitted          bool     // Whether the transformer has been fitted
}

// NewPolynomialFeatures creates a new polynomial feature generator
func NewPolynomialFeatures(degree int) *PolynomialFeatures {
	return &PolynomialFeatures{
		Degree:          degree,
		InteractionOnly: false,
		IncludeBias:     true,
		fitted:          false,
	}
}

// NewPolynomialFeaturesWithOptions creates a polynomial feature generator with custom options
func NewPolynomialFeaturesWithOptions(degree int, interactionOnly, includeBias bool) *PolynomialFeatures {
	return &PolynomialFeatures{
		Degree:          degree,
		InteractionOnly: interactionOnly,
		IncludeBias:     includeBias,
		fitted:          false,
	}
}

// Fit learns the structure of polynomial features from input data
func (pf *PolynomialFeatures) Fit(X *array.Array) error {
	if X == nil {
		return errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return errors.New("X must be a 2D array")
	}

	if pf.Degree < 1 {
		return errors.New("degree must be at least 1")
	}

	shape := X.Shape()
	pf.InputFeatures = shape[1]

	// Generate feature combinations and names
	pf.FeatureNames = []string{}

	// Add bias term if requested
	if pf.IncludeBias {
		pf.FeatureNames = append(pf.FeatureNames, "1")
	}

	// Generate all polynomial combinations up to the specified degree
	pf.generateFeatureCombinations()

	pf.OutputFeatures = len(pf.FeatureNames)
	pf.fitted = true

	return nil
}

// Transform applies polynomial feature transformation to data
func (pf *PolynomialFeatures) Transform(X *array.Array) (*array.Array, error) {
	if !pf.fitted {
		return nil, errors.New("PolynomialFeatures must be fitted before transform")
	}

	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	shape := X.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	if nFeatures != pf.InputFeatures {
		return nil, errors.New("X has different number of features than fitted data")
	}

	// Create output array
	output := array.Empty(internal.Shape{nSamples, pf.OutputFeatures}, internal.Float64)

	// Transform each sample
	for i := 0; i < nSamples; i++ {
		featureIdx := 0

		// Add bias term if requested
		if pf.IncludeBias {
			output.Set(1.0, i, featureIdx)
			featureIdx++
		}

		// Generate all polynomial combinations for this sample
		pf.transformSample(X, output, i, featureIdx)
	}

	return output, nil
}

// FitTransform fits the polynomial features and transforms the data in one step
func (pf *PolynomialFeatures) FitTransform(X *array.Array) (*array.Array, error) {
	err := pf.Fit(X)
	if err != nil {
		return nil, err
	}
	return pf.Transform(X)
}

// GetFeatureNames returns the names of the generated polynomial features
func (pf *PolynomialFeatures) GetFeatureNames() []string {
	if !pf.fitted {
		return nil
	}
	return pf.FeatureNames
}

// generateFeatureCombinations generates all polynomial feature combinations
func (pf *PolynomialFeatures) generateFeatureCombinations() {
	// Generate combinations for each degree from 1 to pf.Degree
	for degree := 1; degree <= pf.Degree; degree++ {
		pf.generateCombinationsOfDegree(degree)
	}
}

// generateCombinationsOfDegree generates all combinations of a specific degree
func (pf *PolynomialFeatures) generateCombinationsOfDegree(degree int) {
	combinations := make([][]int, 0)
	current := make([]int, 0, degree)

	pf.generateCombinationsRecursive(0, degree, current, &combinations)

	// Add feature names for each combination
	for _, combo := range combinations {
		if pf.InteractionOnly && degree > 1 {
			// For interaction-only, skip pure power terms (same feature repeated)
			if pf.hasSameFeatureRepeated(combo) {
				continue
			}
		}

		name := pf.generateFeatureName(combo)
		pf.FeatureNames = append(pf.FeatureNames, name)
	}
}

// generateCombinationsRecursive recursively generates all combinations
func (pf *PolynomialFeatures) generateCombinationsRecursive(start, remaining int, current []int, combinations *[][]int) {
	if remaining == 0 {
		combo := make([]int, len(current))
		copy(combo, current)
		*combinations = append(*combinations, combo)
		return
	}

	for i := start; i < pf.InputFeatures; i++ {
		current = append(current, i)
		pf.generateCombinationsRecursive(i, remaining-1, current, combinations)
		current = current[:len(current)-1]
	}
}

// hasSameFeatureRepeated checks if a combination has the same feature repeated
func (pf *PolynomialFeatures) hasSameFeatureRepeated(combo []int) bool {
	if len(combo) <= 1 {
		return false
	}

	for i := 1; i < len(combo); i++ {
		if combo[i] == combo[i-1] {
			return true
		}
	}
	return false
}

// generateFeatureName creates a descriptive name for a feature combination
func (pf *PolynomialFeatures) generateFeatureName(combo []int) string {
	if len(combo) == 0 {
		return "1"
	}

	if len(combo) == 1 {
		return fmt.Sprintf("x%d", combo[0])
	}

	// Count occurrences of each feature
	featureCounts := make(map[int]int)
	for _, feature := range combo {
		featureCounts[feature]++
	}

	// Build feature name
	name := ""
	for feature := 0; feature < pf.InputFeatures; feature++ {
		if count, exists := featureCounts[feature]; exists {
			if name != "" {
				name += " "
			}
			if count == 1 {
				name += fmt.Sprintf("x%d", feature)
			} else {
				name += fmt.Sprintf("x%d^%d", feature, count)
			}
		}
	}

	return name
}

// transformSample transforms a single sample by computing all polynomial combinations
func (pf *PolynomialFeatures) transformSample(X, output *array.Array, sampleIdx, startFeatureIdx int) {
	featureIdx := startFeatureIdx

	// Generate combinations for each degree from 1 to pf.Degree
	for degree := 1; degree <= pf.Degree; degree++ {
		combinations := make([][]int, 0)
		current := make([]int, 0, degree)

		pf.generateCombinationsRecursive(0, degree, current, &combinations)

		// Compute polynomial features for each combination
		for _, combo := range combinations {
			if pf.InteractionOnly && degree > 1 {
				// Skip pure power terms for interaction-only mode
				if pf.hasSameFeatureRepeated(combo) {
					continue
				}
			}

			// Compute the polynomial feature value
			value := 1.0
			for _, feature := range combo {
				featureValue := convertToFloat64(X.At(sampleIdx, feature))
				value *= featureValue
			}

			output.Set(value, sampleIdx, featureIdx)
			featureIdx++
		}
	}
}

// GetOutputFeatures returns the number of output features after transformation
func (pf *PolynomialFeatures) GetOutputFeatures() int {
	if !pf.fitted {
		return -1
	}
	return pf.OutputFeatures
}

// GetDegree returns the maximum degree of polynomial features
func (pf *PolynomialFeatures) GetDegree() int {
	return pf.Degree
}

// IsInteractionOnly returns whether only interaction features are included
func (pf *PolynomialFeatures) IsInteractionOnly() bool {
	return pf.InteractionOnly
}

// HasBias returns whether bias term is included
func (pf *PolynomialFeatures) HasBias() bool {
	return pf.IncludeBias
}
