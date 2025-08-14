package stats

import (
	"errors"
	"math"
	"sort"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// ImputationStrategy defines the strategy for imputing missing values
type ImputationStrategy string

const (
	// ImputeMean fills missing values with the mean of the column
	ImputeMean ImputationStrategy = "mean"
	// ImputeMedian fills missing values with the median of the column
	ImputeMedian ImputationStrategy = "median"
	// ImputeMostFrequent fills missing values with the most frequent value in the column
	ImputeMostFrequent ImputationStrategy = "most_frequent"
	// ImputeConstant fills missing values with a user-specified constant
	ImputeConstant ImputationStrategy = "constant"
)

// SimpleImputer fills missing values using simple strategies
// following the scikit-learn pattern with Fit/Transform methods
type SimpleImputer struct {
	Strategy      ImputationStrategy // Strategy for imputation
	FillValue     float64            // Value to use for constant imputation
	Statistics    []float64          // Computed statistics per feature (mean, median, etc.)
	MissingValues float64            // Value that represents missing data (default: NaN)
	InputFeatures int                // Number of input features (learned during fit)
	fitted        bool               // Whether the imputer has been fitted
}

// NewSimpleImputer creates a new simple imputer with mean strategy
func NewSimpleImputer() *SimpleImputer {
	return &SimpleImputer{
		Strategy:      ImputeMean,
		FillValue:     0.0,
		MissingValues: math.NaN(),
		fitted:        false,
	}
}

// NewSimpleImputerWithStrategy creates a simple imputer with specified strategy
func NewSimpleImputerWithStrategy(strategy ImputationStrategy) *SimpleImputer {
	return &SimpleImputer{
		Strategy:      strategy,
		FillValue:     0.0,
		MissingValues: math.NaN(),
		fitted:        false,
	}
}

// NewSimpleImputerWithOptions creates a simple imputer with custom options
func NewSimpleImputerWithOptions(strategy ImputationStrategy, fillValue, missingValues float64) *SimpleImputer {
	return &SimpleImputer{
		Strategy:      strategy,
		FillValue:     fillValue,
		MissingValues: missingValues,
		fitted:        false,
	}
}

// Fit computes the imputation statistics from the input data
func (si *SimpleImputer) Fit(X *array.Array) error {
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

	si.InputFeatures = nFeatures
	si.Statistics = make([]float64, nFeatures)

	// Compute statistics for each feature
	for j := 0; j < nFeatures; j++ {
		switch si.Strategy {
		case ImputeMean:
			si.Statistics[j] = si.computeMean(X, j)
		case ImputeMedian:
			si.Statistics[j] = si.computeMedian(X, j)
		case ImputeMostFrequent:
			si.Statistics[j] = si.computeMostFrequent(X, j)
		case ImputeConstant:
			si.Statistics[j] = si.FillValue
		default:
			return errors.New("unsupported imputation strategy")
		}
	}

	si.fitted = true
	return nil
}

// Transform applies the fitted imputation to data
func (si *SimpleImputer) Transform(X *array.Array) (*array.Array, error) {
	if !si.fitted {
		return nil, errors.New("SimpleImputer must be fitted before transform")
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

	if nFeatures != si.InputFeatures {
		return nil, errors.New("X has different number of features than fitted data")
	}

	// Create output array as copy of input
	output := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(X.At(i, j))

			// Check if value is missing
			if si.isMissingValue(value) {
				// Replace with computed statistic
				output.Set(si.Statistics[j], i, j)
			} else {
				// Keep original value
				output.Set(value, i, j)
			}
		}
	}

	return output, nil
}

// FitTransform fits the imputer and transforms the data in one step
func (si *SimpleImputer) FitTransform(X *array.Array) (*array.Array, error) {
	err := si.Fit(X)
	if err != nil {
		return nil, err
	}
	return si.Transform(X)
}

// computeMean calculates the mean of non-missing values in a column
func (si *SimpleImputer) computeMean(X *array.Array, featureIdx int) float64 {
	sum := 0.0
	count := 0

	nSamples := X.Shape()[0]
	for i := 0; i < nSamples; i++ {
		value := convertToFloat64(X.At(i, featureIdx))
		if !si.isMissingValue(value) {
			sum += value
			count++
		}
	}

	if count == 0 {
		return 0.0 // All values are missing
	}

	return sum / float64(count)
}

// computeMedian calculates the median of non-missing values in a column
func (si *SimpleImputer) computeMedian(X *array.Array, featureIdx int) float64 {
	var values []float64

	nSamples := X.Shape()[0]
	for i := 0; i < nSamples; i++ {
		value := convertToFloat64(X.At(i, featureIdx))
		if !si.isMissingValue(value) {
			values = append(values, value)
		}
	}

	if len(values) == 0 {
		return 0.0 // All values are missing
	}

	sort.Float64s(values)
	n := len(values)

	if n%2 == 0 {
		// Even number of values - average of middle two
		return (values[n/2-1] + values[n/2]) / 2.0
	} else {
		// Odd number of values - middle value
		return values[n/2]
	}
}

// computeMostFrequent finds the most frequent non-missing value in a column
func (si *SimpleImputer) computeMostFrequent(X *array.Array, featureIdx int) float64 {
	valueCount := make(map[float64]int)

	nSamples := X.Shape()[0]
	for i := 0; i < nSamples; i++ {
		value := convertToFloat64(X.At(i, featureIdx))
		if !si.isMissingValue(value) {
			valueCount[value]++
		}
	}

	if len(valueCount) == 0 {
		return 0.0 // All values are missing
	}

	// Find most frequent value
	var mostFrequentValue float64
	maxCount := 0
	for value, count := range valueCount {
		if count > maxCount {
			maxCount = count
			mostFrequentValue = value
		}
	}

	return mostFrequentValue
}

// isMissingValue checks if a value represents missing data
func (si *SimpleImputer) isMissingValue(value float64) bool {
	if math.IsNaN(si.MissingValues) {
		return math.IsNaN(value)
	}
	return math.Abs(value-si.MissingValues) < 1e-15
}

// GetStatistics returns the computed imputation statistics
func (si *SimpleImputer) GetStatistics() []float64 {
	if !si.fitted {
		return nil
	}
	// Return a copy to prevent external modification
	stats := make([]float64, len(si.Statistics))
	copy(stats, si.Statistics)
	return stats
}

// GetStrategy returns the imputation strategy
func (si *SimpleImputer) GetStrategy() ImputationStrategy {
	return si.Strategy
}

// GetFillValue returns the fill value for constant imputation
func (si *SimpleImputer) GetFillValue() float64 {
	return si.FillValue
}

// GetMissingValues returns the value that represents missing data
func (si *SimpleImputer) GetMissingValues() float64 {
	return si.MissingValues
}

// SetFillValue sets the fill value for constant imputation
func (si *SimpleImputer) SetFillValue(fillValue float64) {
	si.FillValue = fillValue
	if si.fitted && si.Strategy == ImputeConstant {
		// Update statistics if already fitted with constant strategy
		for i := range si.Statistics {
			si.Statistics[i] = fillValue
		}
	}
}

// CountMissingValues counts the number of missing values per feature
func CountMissingValues(X *array.Array, missingValues float64) ([]int, error) {
	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	shape := X.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	counts := make([]int, nFeatures)

	for j := 0; j < nFeatures; j++ {
		for i := 0; i < nSamples; i++ {
			value := convertToFloat64(X.At(i, j))

			if math.IsNaN(missingValues) && math.IsNaN(value) {
				counts[j]++
			} else if !math.IsNaN(missingValues) && math.Abs(value-missingValues) < 1e-15 {
				counts[j]++
			}
		}
	}

	return counts, nil
}

// HasMissingValues checks if an array contains any missing values
func HasMissingValues(X *array.Array, missingValues float64) (bool, error) {
	if X == nil {
		return false, errors.New("X cannot be nil")
	}

	shape := X.Shape()
	if len(shape) == 1 {
		// 1D array
		nElements := shape[0]
		for i := 0; i < nElements; i++ {
			value := convertToFloat64(X.At(i))
			if math.IsNaN(missingValues) && math.IsNaN(value) {
				return true, nil
			} else if !math.IsNaN(missingValues) && math.Abs(value-missingValues) < 1e-15 {
				return true, nil
			}
		}
	} else if len(shape) == 2 {
		// 2D array
		nSamples := shape[0]
		nFeatures := shape[1]
		for i := 0; i < nSamples; i++ {
			for j := 0; j < nFeatures; j++ {
				value := convertToFloat64(X.At(i, j))
				if math.IsNaN(missingValues) && math.IsNaN(value) {
					return true, nil
				} else if !math.IsNaN(missingValues) && math.Abs(value-missingValues) < 1e-15 {
					return true, nil
				}
			}
		}
	} else {
		return false, errors.New("X must be 1D or 2D array")
	}

	return false, nil
}
