package stats

import (
	"math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Skewness calculates the skewness (measure of asymmetry) of the distribution
func Skewness(arr *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.Skewness")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return 0, internal.NewValidationErrorWithMsg("Skewness", "array cannot be nil")
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "Skewness", "array"); err != nil {
		return 0, err
	}

	// Calculate mean and standard deviation
	mean, err := Mean(arr)
	if err != nil {
		return 0, err
	}

	std, err := Std(arr)
	if err != nil {
		return 0, err
	}

	if std == 0 {
		return math.NaN(), nil // Undefined for zero variance
	}

	// Calculate skewness: E[((X - μ) / σ)³]
	var skewnessSum float64
	var count int

	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			standardized := (val - mean) / std
			skewnessSum += standardized * standardized * standardized
			count++
		}
	}

	if count == 0 {
		return math.NaN(), nil
	}

	return skewnessSum / float64(count), nil
}

// Kurtosis calculates the kurtosis (measure of tail heaviness) of the distribution
func Kurtosis(arr *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.Kurtosis")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return 0, internal.NewValidationErrorWithMsg("Kurtosis", "array cannot be nil")
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "Kurtosis", "array"); err != nil {
		return 0, err
	}

	// Calculate mean and standard deviation
	mean, err := Mean(arr)
	if err != nil {
		return 0, err
	}

	std, err := Std(arr)
	if err != nil {
		return 0, err
	}

	if std == 0 {
		return math.NaN(), nil // Undefined for zero variance
	}

	// Calculate kurtosis: E[((X - μ) / σ)⁴]
	var kurtosisSum float64
	var count int

	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			standardized := (val - mean) / std
			fourthPower := standardized * standardized * standardized * standardized
			kurtosisSum += fourthPower
			count++
		}
	}

	if count == 0 {
		return math.NaN(), nil
	}

	return kurtosisSum / float64(count), nil
}

// ExcessKurtosis calculates the excess kurtosis (kurtosis - 3)
// Normal distribution has excess kurtosis of 0
func ExcessKurtosis(arr *array.Array) (float64, error) {
	kurtosis, err := Kurtosis(arr)
	if err != nil {
		return 0, err
	}
	return kurtosis - 3.0, nil
}

// Range calculates the range (max - min) of the array
func Range(arr *array.Array) (float64, error) {
	if arr == nil {
		return 0, internal.NewValidationErrorWithMsg("Range", "array cannot be nil")
	}

	min, err := Min(arr)
	if err != nil {
		return 0, err
	}

	max, err := Max(arr)
	if err != nil {
		return 0, err
	}

	if math.IsNaN(min) || math.IsNaN(max) {
		return math.NaN(), nil
	}

	return max - min, nil
}

// Mode finds the most frequently occurring value(s) in the array
// Returns the first mode found if multiple modes exist
func Mode(arr *array.Array) (float64, error) {
	if arr == nil {
		return 0, internal.NewValidationErrorWithMsg("Mode", "array cannot be nil")
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "Mode", "array"); err != nil {
		return 0, err
	}

	// Count frequencies
	frequencies := make(map[float64]int)
	flatArr := arr.Flatten()

	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			frequencies[val]++
		}
	}

	if len(frequencies) == 0 {
		return math.NaN(), nil
	}

	// Find mode
	var mode float64
	maxFreq := 0

	for value, freq := range frequencies {
		if freq > maxFreq {
			maxFreq = freq
			mode = value
		}
	}

	return mode, nil
}

// Percentile calculates the specified percentile (equivalent to Quantile but with 0-100 scale)
func Percentile(arr *array.Array, p float64) (float64, error) {
	if p < 0 || p > 100 {
		return 0, internal.NewValidationErrorWithMsg("Percentile", "percentile must be between 0 and 100")
	}
	return Quantile(arr, p/100.0)
}

// MeanAbsoluteDeviation calculates the mean absolute deviation from the mean
func MeanAbsoluteDeviation(arr *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.MeanAbsoluteDeviation")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return 0, internal.NewValidationErrorWithMsg("MeanAbsoluteDeviation", "array cannot be nil")
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "MeanAbsoluteDeviation", "array"); err != nil {
		return 0, err
	}

	mean, err := Mean(arr)
	if err != nil {
		return 0, err
	}

	var sum float64
	var count int

	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			sum += math.Abs(val - mean)
			count++
		}
	}

	if count == 0 {
		return math.NaN(), nil
	}

	return sum / float64(count), nil
}

// MedianAbsoluteDeviation calculates the median absolute deviation from the median
func MedianAbsoluteDeviation(arr *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.MedianAbsoluteDeviation")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return 0, internal.NewValidationErrorWithMsg("MedianAbsoluteDeviation", "array cannot be nil")
	}

	median, err := Median(arr)
	if err != nil {
		return 0, err
	}

	if math.IsNaN(median) {
		return math.NaN(), nil
	}

	// Calculate absolute deviations from median
	var deviations []float64
	flatArr := arr.Flatten()

	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			deviations = append(deviations, math.Abs(val-median))
		}
	}

	if len(deviations) == 0 {
		return math.NaN(), nil
	}

	// Create array from deviations and find median
	devArr, err := array.FromSlice(deviations)
	if err != nil {
		return 0, err
	}

	return Median(devArr)
}

// RootMeanSquare calculates the root mean square (quadratic mean)
func RootMeanSquare(arr *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.RootMeanSquare")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return 0, internal.NewValidationErrorWithMsg("RootMeanSquare", "array cannot be nil")
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "RootMeanSquare", "array"); err != nil {
		return 0, err
	}

	var sumSquares float64
	var count int

	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			sumSquares += val * val
			count++
		}
	}

	if count == 0 {
		return math.NaN(), nil
	}

	return math.Sqrt(sumSquares / float64(count)), nil
}

// GeometricMean calculates the geometric mean (only for positive values)
func GeometricMean(arr *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.GeometricMean")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return 0, internal.NewValidationErrorWithMsg("GeometricMean", "array cannot be nil")
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "GeometricMean", "array"); err != nil {
		return 0, err
	}

	var logSum float64
	var count int

	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) && val > 0 {
			logSum += math.Log(val)
			count++
		} else if val <= 0 {
			return math.NaN(), nil // Geometric mean undefined for non-positive values
		}
	}

	if count == 0 {
		return math.NaN(), nil
	}

	return math.Exp(logSum / float64(count)), nil
}

// HarmonicMean calculates the harmonic mean (only for positive values)
func HarmonicMean(arr *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.HarmonicMean")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return 0, internal.NewValidationErrorWithMsg("HarmonicMean", "array cannot be nil")
	}
	if err := internal.QuickValidateArrayNotEmpty(arr.Size(), "HarmonicMean", "array"); err != nil {
		return 0, err
	}

	var reciprocalSum float64
	var count int

	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) && val != 0 {
			reciprocalSum += 1.0 / val
			count++
		} else if val == 0 {
			return 0, nil // Harmonic mean is 0 if any value is 0
		}
	}

	if count == 0 {
		return math.NaN(), nil
	}

	return float64(count) / reciprocalSum, nil
}
