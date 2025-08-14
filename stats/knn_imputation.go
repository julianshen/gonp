package stats

import (
	"errors"
	"math"
	"sort"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// KNNDistanceMetric defines the distance metric for KNN imputation
type KNNDistanceMetric string

const (
	// KNNEuclidean uses Euclidean distance for neighbor selection
	KNNEuclidean KNNDistanceMetric = "euclidean"
	// KNNManhattan uses Manhattan distance for neighbor selection
	KNNManhattan KNNDistanceMetric = "manhattan"
)

// KNNImputer performs imputation using K-Nearest Neighbors algorithm
// following the scikit-learn pattern with Fit/Transform methods
type KNNImputer struct {
	NNeighbors     int               // Number of neighbors to use for imputation
	DistanceMetric KNNDistanceMetric // Distance metric for finding neighbors
	MissingValues  float64           // Value that represents missing data
	fitted         bool              // Whether the imputer has been fitted
	data           *array.Array      // Training data for neighbor search
}

// neighborDistance holds a neighbor's index and distance for sorting
type neighborDistance struct {
	index    int     // Index of the neighbor in the dataset
	distance float64 // Distance to the target sample
}

// NewKNNImputer creates a new KNN imputer with default settings
func NewKNNImputer(nNeighbors int) *KNNImputer {
	return &KNNImputer{
		NNeighbors:     nNeighbors,
		DistanceMetric: KNNEuclidean,
		MissingValues:  math.NaN(),
		fitted:         false,
	}
}

// NewKNNImputerWithDistance creates a KNN imputer with specified distance metric
func NewKNNImputerWithDistance(nNeighbors int, distanceMetric string) *KNNImputer {
	var metric KNNDistanceMetric
	switch distanceMetric {
	case "euclidean":
		metric = KNNEuclidean
	case "manhattan":
		metric = KNNManhattan
	default:
		metric = KNNEuclidean
	}

	return &KNNImputer{
		NNeighbors:     nNeighbors,
		DistanceMetric: metric,
		MissingValues:  math.NaN(),
		fitted:         false,
	}
}

// NewKNNImputerWithOptions creates a KNN imputer with custom options
func NewKNNImputerWithOptions(nNeighbors int, distanceMetric KNNDistanceMetric, missingValues float64) *KNNImputer {
	return &KNNImputer{
		NNeighbors:     nNeighbors,
		DistanceMetric: distanceMetric,
		MissingValues:  missingValues,
		fitted:         false,
	}
}

// Fit prepares the KNN imputer by storing the reference data
func (knn *KNNImputer) Fit(X *array.Array) error {
	if X == nil {
		return errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return errors.New("X must be a 2D array")
	}

	shape := X.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	if nSamples == 0 || nFeatures == 0 {
		return errors.New("X cannot be empty")
	}

	if knn.NNeighbors <= 0 {
		return errors.New("number of neighbors must be positive")
	}

	// Count non-missing samples for validation
	nonMissingSamples := knn.countNonMissingSamples(X)
	if knn.NNeighbors >= nonMissingSamples {
		return errors.New("number of neighbors cannot exceed number of non-missing samples")
	}

	// Store reference data for neighbor search
	knn.data = array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(X.At(i, j))
			knn.data.Set(value, i, j)
		}
	}

	knn.fitted = true
	return nil
}

// Transform applies KNN imputation to the data
func (knn *KNNImputer) Transform(X *array.Array) (*array.Array, error) {
	if !knn.fitted {
		return nil, errors.New("KNNImputer must be fitted before transform")
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

	if nFeatures != knn.data.Shape()[1] {
		return nil, errors.New("X has different number of features than fitted data")
	}

	// Create output array as copy of input
	output := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(X.At(i, j))
			output.Set(value, i, j)
		}
	}

	// Impute missing values for each sample
	for i := 0; i < nSamples; i++ {
		err := knn.imputeSample(output, i)
		if err != nil {
			return nil, err
		}
	}

	return output, nil
}

// FitTransform fits the KNN imputer and transforms the data in one step
func (knn *KNNImputer) FitTransform(X *array.Array) (*array.Array, error) {
	err := knn.Fit(X)
	if err != nil {
		return nil, err
	}
	return knn.Transform(X)
}

// imputeSample imputes missing values for a single sample
func (knn *KNNImputer) imputeSample(data *array.Array, sampleIdx int) error {
	nFeatures := data.Shape()[1]

	// Find which features are missing for this sample
	missingFeatures := make([]int, 0)
	for j := 0; j < nFeatures; j++ {
		value := convertToFloat64(data.At(sampleIdx, j))
		if knn.isMissingValue(value) {
			missingFeatures = append(missingFeatures, j)
		}
	}

	// If no missing values, nothing to impute
	if len(missingFeatures) == 0 {
		return nil
	}

	// If all values are missing, fall back to simple imputation
	if len(missingFeatures) == nFeatures {
		return knn.imputeAllMissingFeatures(data, sampleIdx)
	}

	// Find k nearest neighbors based on non-missing features
	neighbors, err := knn.findNearestNeighbors(data, sampleIdx, missingFeatures)
	if err != nil {
		return err
	}

	// Impute each missing feature using neighbor averages
	for _, featureIdx := range missingFeatures {
		imputedValue := knn.computeImputedValue(neighbors, featureIdx)
		data.Set(imputedValue, sampleIdx, featureIdx)
	}

	return nil
}

// findNearestNeighbors finds k nearest neighbors for imputation
func (knn *KNNImputer) findNearestNeighbors(data *array.Array, sampleIdx int, missingFeatures []int) ([]int, error) {
	nSamples := knn.data.Shape()[0]

	// Compute distances to all other samples
	distances := make([]neighborDistance, 0, nSamples-1)

	for i := 0; i < nSamples; i++ {
		if i == sampleIdx {
			continue // Skip self
		}

		// Check if this potential neighbor has values for the features we need
		if !knn.isValidNeighbor(i, missingFeatures) {
			continue // Skip if neighbor has missing values in features we need
		}

		// Compute distance based on non-missing features
		distance, err := knn.computeDistance(data, sampleIdx, i, missingFeatures)
		if err != nil {
			return nil, err
		}

		distances = append(distances, neighborDistance{
			index:    i,
			distance: distance,
		})
	}

	if len(distances) < knn.NNeighbors {
		return nil, errors.New("insufficient neighbors with complete data for imputation")
	}

	// Sort by distance and select k nearest
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	neighbors := make([]int, knn.NNeighbors)
	for i := 0; i < knn.NNeighbors; i++ {
		neighbors[i] = distances[i].index
	}

	return neighbors, nil
}

// computeDistance computes distance between two samples using non-missing features
func (knn *KNNImputer) computeDistance(data *array.Array, idx1, idx2 int, missingFeatures []int) (float64, error) {
	nFeatures := data.Shape()[1]
	distance := 0.0
	usedFeatures := 0

	for j := 0; j < nFeatures; j++ {
		// Skip missing features for the sample being imputed
		skip := false
		for _, missingIdx := range missingFeatures {
			if j == missingIdx {
				skip = true
				break
			}
		}
		if skip {
			continue
		}

		val1 := convertToFloat64(data.At(idx1, j))
		val2 := convertToFloat64(knn.data.At(idx2, j))

		// Skip if either value is missing
		if knn.isMissingValue(val1) || knn.isMissingValue(val2) {
			continue
		}

		diff := val1 - val2

		switch knn.DistanceMetric {
		case KNNEuclidean:
			distance += diff * diff
		case KNNManhattan:
			distance += math.Abs(diff)
		default:
			return 0, errors.New("unsupported distance metric")
		}

		usedFeatures++
	}

	if usedFeatures == 0 {
		return math.Inf(1), nil // Infinite distance if no comparable features
	}

	if knn.DistanceMetric == KNNEuclidean {
		distance = math.Sqrt(distance)
	}

	return distance, nil
}

// isValidNeighbor checks if a potential neighbor has complete data for needed features
func (knn *KNNImputer) isValidNeighbor(neighborIdx int, missingFeatures []int) bool {
	// Check that neighbor has values for the features we want to impute
	for _, featureIdx := range missingFeatures {
		value := convertToFloat64(knn.data.At(neighborIdx, featureIdx))
		if knn.isMissingValue(value) {
			return false
		}
	}
	return true
}

// computeImputedValue computes the imputed value for a feature using neighbor average
func (knn *KNNImputer) computeImputedValue(neighbors []int, featureIdx int) float64 {
	sum := 0.0
	count := 0

	for _, neighborIdx := range neighbors {
		value := convertToFloat64(knn.data.At(neighborIdx, featureIdx))
		if !knn.isMissingValue(value) {
			sum += value
			count++
		}
	}

	if count == 0 {
		return 0.0 // Fallback if no neighbors have this feature
	}

	return sum / float64(count)
}

// imputeAllMissingFeatures handles the case where all features are missing
func (knn *KNNImputer) imputeAllMissingFeatures(data *array.Array, sampleIdx int) error {
	nFeatures := data.Shape()[1]

	// Use simple mean imputation from the reference data as fallback
	for j := 0; j < nFeatures; j++ {
		mean := knn.computeFeatureMean(j)
		data.Set(mean, sampleIdx, j)
	}

	return nil
}

// computeFeatureMean computes the mean of a feature from reference data
func (knn *KNNImputer) computeFeatureMean(featureIdx int) float64 {
	nSamples := knn.data.Shape()[0]
	sum := 0.0
	count := 0

	for i := 0; i < nSamples; i++ {
		value := convertToFloat64(knn.data.At(i, featureIdx))
		if !knn.isMissingValue(value) {
			sum += value
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	return sum / float64(count)
}

// countNonMissingSamples counts samples that have at least some non-missing values
func (knn *KNNImputer) countNonMissingSamples(X *array.Array) int {
	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]
	count := 0

	for i := 0; i < nSamples; i++ {
		hasNonMissing := false
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(X.At(i, j))
			if !knn.isMissingValue(value) {
				hasNonMissing = true
				break
			}
		}
		if hasNonMissing {
			count++
		}
	}

	return count
}

// isMissingValue checks if a value represents missing data
func (knn *KNNImputer) isMissingValue(value float64) bool {
	if math.IsNaN(knn.MissingValues) {
		return math.IsNaN(value)
	}
	return math.Abs(value-knn.MissingValues) < 1e-15
}

// GetNNeighbors returns the number of neighbors used for imputation
func (knn *KNNImputer) GetNNeighbors() int {
	return knn.NNeighbors
}

// GetDistanceMetric returns the distance metric used
func (knn *KNNImputer) GetDistanceMetric() KNNDistanceMetric {
	return knn.DistanceMetric
}

// GetMissingValues returns the missing value marker
func (knn *KNNImputer) GetMissingValues() float64 {
	return knn.MissingValues
}
