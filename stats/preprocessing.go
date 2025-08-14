package stats

import (
	"errors"
	"math"
	"sort"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// StandardScaler standardizes features by removing the mean and scaling to unit variance
type StandardScaler struct {
	Mean   *array.Array // Mean of each feature
	Scale  *array.Array // Standard deviation of each feature
	fitted bool
}

// NewStandardScaler creates a new StandardScaler
func NewStandardScaler() *StandardScaler {
	return &StandardScaler{
		fitted: false,
	}
}

// Fit computes the mean and standard deviation for later scaling
func (s *StandardScaler) Fit(X *array.Array) error {
	if X == nil {
		return errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nSamples < 2 {
		return errors.New("need at least 2 samples for StandardScaler")
	}

	// Compute mean for each feature
	s.Mean = array.Zeros(internal.Shape{nFeatures}, internal.Float64)
	for j := 0; j < nFeatures; j++ {
		sum := 0.0
		for i := 0; i < nSamples; i++ {
			sum += convertToFloat64(X.At(i, j))
		}
		mean := sum / float64(nSamples)
		s.Mean.Set(mean, j)
	}

	// Compute standard deviation for each feature
	s.Scale = array.Zeros(internal.Shape{nFeatures}, internal.Float64)
	for j := 0; j < nFeatures; j++ {
		mean := convertToFloat64(s.Mean.At(j))
		sumSq := 0.0
		for i := 0; i < nSamples; i++ {
			diff := convertToFloat64(X.At(i, j)) - mean
			sumSq += diff * diff
		}
		std := math.Sqrt(sumSq / float64(nSamples-1)) // Sample standard deviation
		if std < 1e-10 {
			std = 1.0 // Avoid division by zero for constant features
		}
		s.Scale.Set(std, j)
	}

	s.fitted = true
	return nil
}

// Transform standardizes the data using fitted parameters
func (s *StandardScaler) Transform(X *array.Array) (*array.Array, error) {
	if !s.fitted {
		return nil, errors.New("StandardScaler must be fitted before transform")
	}

	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nFeatures != s.Mean.Size() {
		return nil, errors.New("X has different number of features than fitted data")
	}

	// Create scaled array
	scaled := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(X.At(i, j))
			mean := convertToFloat64(s.Mean.At(j))
			scale := convertToFloat64(s.Scale.At(j))

			scaledValue := (value - mean) / scale
			scaled.Set(scaledValue, i, j)
		}
	}

	return scaled, nil
}

// FitTransform fits the scaler and transforms the data in one step
func (s *StandardScaler) FitTransform(X *array.Array) (*array.Array, error) {
	err := s.Fit(X)
	if err != nil {
		return nil, err
	}
	return s.Transform(X)
}

// InverseTransform transforms scaled data back to original scale
func (s *StandardScaler) InverseTransform(X *array.Array) (*array.Array, error) {
	if !s.fitted {
		return nil, errors.New("StandardScaler must be fitted before inverse_transform")
	}

	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nFeatures != s.Mean.Size() {
		return nil, errors.New("X has different number of features than fitted data")
	}

	// Create unscaled array
	unscaled := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			scaledValue := convertToFloat64(X.At(i, j))
			mean := convertToFloat64(s.Mean.At(j))
			scale := convertToFloat64(s.Scale.At(j))

			originalValue := scaledValue*scale + mean
			unscaled.Set(originalValue, i, j)
		}
	}

	return unscaled, nil
}

// MinMaxScaler scales features to a given range [min, max]
type MinMaxScaler struct {
	Min             *array.Array // Minimum value of each feature
	Scale           *array.Array // Scale factor for each feature
	FeatureRangeMin float64      // Target minimum value
	FeatureRangeMax float64      // Target maximum value
	fitted          bool
}

// NewMinMaxScaler creates a new MinMaxScaler with specified range
func NewMinMaxScaler(featureRangeMin, featureRangeMax float64) *MinMaxScaler {
	return &MinMaxScaler{
		FeatureRangeMin: featureRangeMin,
		FeatureRangeMax: featureRangeMax,
		fitted:          false,
	}
}

// Fit computes the min and range for later scaling
func (m *MinMaxScaler) Fit(X *array.Array) error {
	if X == nil {
		return errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return errors.New("X must be a 2D array")
	}

	if m.FeatureRangeMin >= m.FeatureRangeMax {
		return errors.New("feature_range_min must be less than feature_range_max")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nSamples == 0 {
		return errors.New("need at least 1 sample for MinMaxScaler")
	}

	// Compute min and max for each feature
	m.Min = array.Empty(internal.Shape{nFeatures}, internal.Float64)
	maxVals := array.Empty(internal.Shape{nFeatures}, internal.Float64)

	for j := 0; j < nFeatures; j++ {
		minVal := convertToFloat64(X.At(0, j))
		maxVal := minVal

		for i := 1; i < nSamples; i++ {
			val := convertToFloat64(X.At(i, j))
			if val < minVal {
				minVal = val
			}
			if val > maxVal {
				maxVal = val
			}
		}

		m.Min.Set(minVal, j)
		maxVals.Set(maxVal, j)
	}

	// Compute scale factors
	m.Scale = array.Empty(internal.Shape{nFeatures}, internal.Float64)
	targetRange := m.FeatureRangeMax - m.FeatureRangeMin

	for j := 0; j < nFeatures; j++ {
		minVal := convertToFloat64(m.Min.At(j))
		maxVal := convertToFloat64(maxVals.At(j))
		dataRange := maxVal - minVal

		var scale float64
		if dataRange < 1e-10 {
			scale = 1.0 // Avoid division by zero for constant features
		} else {
			scale = targetRange / dataRange
		}
		m.Scale.Set(scale, j)
	}

	m.fitted = true
	return nil
}

// Transform scales the data using fitted parameters
func (m *MinMaxScaler) Transform(X *array.Array) (*array.Array, error) {
	if !m.fitted {
		return nil, errors.New("MinMaxScaler must be fitted before transform")
	}

	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nFeatures != m.Min.Size() {
		return nil, errors.New("X has different number of features than fitted data")
	}

	// Create scaled array
	scaled := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(X.At(i, j))
			minVal := convertToFloat64(m.Min.At(j))
			scale := convertToFloat64(m.Scale.At(j))

			scaledValue := (value-minVal)*scale + m.FeatureRangeMin
			scaled.Set(scaledValue, i, j)
		}
	}

	return scaled, nil
}

// FitTransform fits the scaler and transforms the data in one step
func (m *MinMaxScaler) FitTransform(X *array.Array) (*array.Array, error) {
	err := m.Fit(X)
	if err != nil {
		return nil, err
	}
	return m.Transform(X)
}

// InverseTransform transforms scaled data back to original scale
func (m *MinMaxScaler) InverseTransform(X *array.Array) (*array.Array, error) {
	if !m.fitted {
		return nil, errors.New("MinMaxScaler must be fitted before inverse_transform")
	}

	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nFeatures != m.Min.Size() {
		return nil, errors.New("X has different number of features than fitted data")
	}

	// Create unscaled array
	unscaled := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			scaledValue := convertToFloat64(X.At(i, j))
			minVal := convertToFloat64(m.Min.At(j))
			scale := convertToFloat64(m.Scale.At(j))

			originalValue := (scaledValue-m.FeatureRangeMin)/scale + minVal
			unscaled.Set(originalValue, i, j)
		}
	}

	return unscaled, nil
}

// RobustScaler scales features using statistics that are robust to outliers
type RobustScaler struct {
	Median *array.Array // Median of each feature
	Scale  *array.Array // Interquartile range of each feature
	fitted bool
}

// NewRobustScaler creates a new RobustScaler
func NewRobustScaler() *RobustScaler {
	return &RobustScaler{
		fitted: false,
	}
}

// Fit computes the median and interquartile range for later scaling
func (r *RobustScaler) Fit(X *array.Array) error {
	if X == nil {
		return errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nSamples < 2 {
		return errors.New("need at least 2 samples for RobustScaler")
	}

	// Compute median and IQR for each feature
	r.Median = array.Empty(internal.Shape{nFeatures}, internal.Float64)
	r.Scale = array.Empty(internal.Shape{nFeatures}, internal.Float64)

	for j := 0; j < nFeatures; j++ {
		// Extract feature values
		values := make([]float64, nSamples)
		for i := 0; i < nSamples; i++ {
			values[i] = convertToFloat64(X.At(i, j))
		}

		// Sort values
		sort.Float64s(values)

		// Compute median
		var median float64
		if nSamples%2 == 0 {
			median = (values[nSamples/2-1] + values[nSamples/2]) / 2.0
		} else {
			median = values[nSamples/2]
		}
		r.Median.Set(median, j)

		// Compute interquartile range (IQR)
		q1Index := (nSamples - 1) / 4
		q3Index := 3 * (nSamples - 1) / 4

		// Handle edge cases for small samples
		if q1Index >= nSamples {
			q1Index = nSamples - 1
		}
		if q3Index >= nSamples {
			q3Index = nSamples - 1
		}

		q1 := values[q1Index]
		q3 := values[q3Index]
		iqr := q3 - q1

		if iqr < 1e-10 {
			iqr = 1.0 // Avoid division by zero for constant features
		}
		r.Scale.Set(iqr, j)
	}

	r.fitted = true
	return nil
}

// Transform scales the data using fitted parameters
func (r *RobustScaler) Transform(X *array.Array) (*array.Array, error) {
	if !r.fitted {
		return nil, errors.New("RobustScaler must be fitted before transform")
	}

	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nFeatures != r.Median.Size() {
		return nil, errors.New("X has different number of features than fitted data")
	}

	// Create scaled array
	scaled := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(X.At(i, j))
			median := convertToFloat64(r.Median.At(j))
			scale := convertToFloat64(r.Scale.At(j))

			scaledValue := (value - median) / scale
			scaled.Set(scaledValue, i, j)
		}
	}

	return scaled, nil
}

// FitTransform fits the scaler and transforms the data in one step
func (r *RobustScaler) FitTransform(X *array.Array) (*array.Array, error) {
	err := r.Fit(X)
	if err != nil {
		return nil, err
	}
	return r.Transform(X)
}

// InverseTransform transforms scaled data back to original scale
func (r *RobustScaler) InverseTransform(X *array.Array) (*array.Array, error) {
	if !r.fitted {
		return nil, errors.New("RobustScaler must be fitted before inverse_transform")
	}

	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	nSamples := X.Shape()[0]
	nFeatures := X.Shape()[1]

	if nFeatures != r.Median.Size() {
		return nil, errors.New("X has different number of features than fitted data")
	}

	// Create unscaled array
	unscaled := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			scaledValue := convertToFloat64(X.At(i, j))
			median := convertToFloat64(r.Median.At(j))
			scale := convertToFloat64(r.Scale.At(j))

			originalValue := scaledValue*scale + median
			unscaled.Set(originalValue, i, j)
		}
	}

	return unscaled, nil
}
