package stats

import (
	"errors"
	"math"
	"sort"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
)

// Descriptive statistics summary
type Description struct {
	Count  int
	Mean   float64
	Std    float64
	Min    float64
	Q25    float64 // 25th percentile
	Median float64 // 50th percentile
	Q75    float64 // 75th percentile
	Max    float64
}

// Mean calculates the arithmetic mean of an array
func Mean(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "Mean", "array"); err != nil {
		return 0, err
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "Mean", "array"); err != nil {
		return 0, err
	}

	sum := 0.0
	count := 0

	// Work with flattened array for easier iteration
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			sum += val
			count++
		}
	}

	if count == 0 {
		return math.NaN(), nil
	}

	return sum / float64(count), nil
}

// MeanSkipNaN calculates mean while skipping NaN values
func MeanSkipNaN(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "MeanSkipNaN", "array"); err != nil {
		return 0, err
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "MeanSkipNaN", "array"); err != nil {
		return 0, err
	}

	sum := 0.0
	count := 0

	// Work with flattened array for easier iteration
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			sum += val
			count++
		}
	}

	if count == 0 {
		return math.NaN(), nil
	}

	return sum / float64(count), nil
}

// Median calculates the median of an array
func Median(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "Median", "array"); err != nil {
		return 0, err
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "Median", "array"); err != nil {
		return 0, err
	}

	// Extract and sort non-NaN values
	values := make([]float64, 0, arr.Size())
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			values = append(values, val)
		}
	}

	if len(values) == 0 {
		return math.NaN(), nil
	}

	sort.Float64s(values)

	n := len(values)
	if n%2 == 0 {
		return (values[n/2-1] + values[n/2]) / 2.0, nil
	}
	return values[n/2], nil
}

// Std calculates the standard deviation
func Std(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "Std", "array"); err != nil {
		return 0, err
	}
	variance, err := Var(arr)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(variance), nil
}

// StdSkipNaN calculates standard deviation while skipping NaN values
func StdSkipNaN(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "StdSkipNaN", "array"); err != nil {
		return 0, err
	}
	variance, err := VarSkipNaN(arr)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(variance), nil
}

// Var calculates the variance
func Var(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "Var", "array"); err != nil {
		return 0, err
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "Var", "array"); err != nil {
		return 0, err
	}

	mean, err := Mean(arr)
	if err != nil {
		return 0, err
	}

	sumSquaredDiff := 0.0
	count := 0

	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			diff := val - mean
			sumSquaredDiff += diff * diff
			count++
		}
	}

	if count <= 1 {
		return 0, errors.New("need at least 2 values for variance")
	}

	return sumSquaredDiff / float64(count-1), nil
}

// VarSkipNaN calculates variance while skipping NaN values
func VarSkipNaN(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "VarSkipNaN", "array"); err != nil {
		return 0, err
	}
	return Var(arr) // Same implementation since Mean already skips NaN
}

// Min finds the minimum value
func Min(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "Min", "array"); err != nil {
		return 0, err
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "Min", "array"); err != nil {
		return 0, err
	}

	min := math.Inf(1)
	found := false

	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			if val < min {
				min = val
			}
			found = true
		}
	}

	if !found {
		return math.NaN(), nil
	}

	return min, nil
}

// Max finds the maximum value
func Max(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "Max", "array"); err != nil {
		return 0, err
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "Max", "array"); err != nil {
		return 0, err
	}

	max := math.Inf(-1)
	found := false

	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			if val > max {
				max = val
			}
			found = true
		}
	}

	if !found {
		return math.NaN(), nil
	}

	return max, nil
}

// Sum calculates the sum of all values
func Sum(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "Sum", "array"); err != nil {
		return 0, err
	}
	// Sum of empty array is 0, so no need to validate non-empty

	sum := 0.0
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			sum += val
		}
	}

	return sum, nil
}

// Quantile calculates the q-th quantile (0 <= q <= 1)
func Quantile(arr *array.Array, q float64) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "Quantile", "array"); err != nil {
		return 0, err
	}
	if q < 0 || q > 1 {
		return 0, internal.NewValidationErrorWithMsg("Quantile", "quantile must be between 0 and 1")
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "Quantile", "array"); err != nil {
		return 0, err
	}

	// Extract and sort non-NaN values
	values := make([]float64, 0, arr.Size())
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			values = append(values, val)
		}
	}

	if len(values) == 0 {
		return math.NaN(), nil
	}

	sort.Float64s(values)

	n := len(values)
	if q == 0 {
		return values[0], nil
	}
	if q == 1 {
		return values[n-1], nil
	}

	// Linear interpolation
	index := q * float64(n-1)
	lower := int(math.Floor(index))
	upper := int(math.Ceil(index))

	if lower == upper {
		return values[lower], nil
	}

	weight := index - float64(lower)
	return values[lower]*(1-weight) + values[upper]*weight, nil
}

// IQR calculates the interquartile range (Q75 - Q25)
func IQR(arr *array.Array) (float64, error) {
	if err := internal.QuickValidateNotNil(arr, "IQR", "array"); err != nil {
		return 0, err
	}
	q25, err := Quantile(arr, 0.25)
	if err != nil {
		return 0, err
	}

	q75, err := Quantile(arr, 0.75)
	if err != nil {
		return 0, err
	}

	return q75 - q25, nil
}

// Describe generates comprehensive descriptive statistics for a series
func Describe(s *series.Series) (*Description, error) {
	if err := internal.QuickValidateNotNil(s, "Describe", "series"); err != nil {
		return nil, err
	}
	desc := &Description{}

	// Convert series to array for calculations
	arr := s.Data()

	desc.Count = s.Len()

	var err error
	desc.Mean, err = Mean(arr)
	if err != nil {
		return nil, err
	}

	desc.Std, err = Std(arr)
	if err != nil {
		return nil, err
	}

	desc.Min, err = Min(arr)
	if err != nil {
		return nil, err
	}

	desc.Max, err = Max(arr)
	if err != nil {
		return nil, err
	}

	desc.Q25, err = Quantile(arr, 0.25)
	if err != nil {
		return nil, err
	}

	desc.Median, err = Quantile(arr, 0.5)
	if err != nil {
		return nil, err
	}

	desc.Q75, err = Quantile(arr, 0.75)
	if err != nil {
		return nil, err
	}

	return desc, nil
}

// Helper function to convert interface{} to float64
func convertToFloat64(val interface{}) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int8:
		return float64(v)
	case int16:
		return float64(v)
	case int32:
		return float64(v)
	case int64:
		return float64(v)
	case uint:
		return float64(v)
	case uint8:
		return float64(v)
	case uint16:
		return float64(v)
	case uint32:
		return float64(v)
	case uint64:
		return float64(v)
	default:
		return math.NaN()
	}
}
