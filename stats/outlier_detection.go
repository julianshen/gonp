package stats

import (
	"fmt"
	"math"
	"sort"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// ZScoreOutlierDetection identifies outliers using the Z-score method
// Returns indices of outliers where |z-score| > threshold
//
// Parameters:
//
//	data: Input array
//	threshold: Z-score threshold (typically 2.0 or 3.0)
//
// Returns: Slice of indices where outliers are detected
func ZScoreOutlierDetection(data *array.Array, threshold float64) ([]int, error) {
	if data == nil {
		return nil, fmt.Errorf("data array cannot be nil")
	}
	if threshold <= 0 {
		return nil, fmt.Errorf("threshold must be positive, got %f", threshold)
	}

	n := data.Size()
	if n == 0 {
		return []int{}, nil
	}

	// Calculate mean and standard deviation
	var sum, sumSquares float64
	values := make([]float64, n)

	for i := 0; i < n; i++ {
		val := convertToFloat64(data.At(i))
		values[i] = val
		sum += val
		sumSquares += val * val
	}

	mean := sum / float64(n)
	variance := (sumSquares - sum*sum/float64(n)) / float64(n-1)
	stdDev := math.Sqrt(variance)

	// Avoid division by zero
	if stdDev < 1e-10 {
		return []int{}, nil // No outliers if no variation
	}

	// Find outliers
	var outliers []int
	for i, val := range values {
		zScore := math.Abs(val-mean) / stdDev
		if zScore > threshold {
			outliers = append(outliers, i)
		}
	}

	return outliers, nil
}

// IQROutlierDetection identifies outliers using the Interquartile Range method
// Returns indices of outliers beyond Q1 - k*IQR or Q3 + k*IQR
//
// Parameters:
//
//	data: Input array
//	k: IQR multiplier (typically 1.5)
//
// Returns: Slice of indices where outliers are detected
func IQROutlierDetection(data *array.Array, k float64) ([]int, error) {
	if data == nil {
		return nil, fmt.Errorf("data array cannot be nil")
	}
	if k < 0 {
		return nil, fmt.Errorf("k must be non-negative, got %f", k)
	}

	n := data.Size()
	if n < 4 {
		return []int{}, nil // Need at least 4 points for quartiles
	}

	// Create sorted copy for quartile calculation
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = convertToFloat64(data.At(i))
	}
	sort.Float64s(values)

	// Calculate quartiles
	q1Index := n / 4
	q3Index := 3 * n / 4

	var q1, q3 float64
	if n%4 == 0 {
		q1 = (values[q1Index-1] + values[q1Index]) / 2
		q3 = (values[q3Index-1] + values[q3Index]) / 2
	} else {
		q1 = values[q1Index]
		q3 = values[q3Index]
	}

	iqr := q3 - q1
	lowerBound := q1 - k*iqr
	upperBound := q3 + k*iqr

	// Find outliers in original array
	var outliers []int
	for i := 0; i < n; i++ {
		val := convertToFloat64(data.At(i))
		if val < lowerBound || val > upperBound {
			outliers = append(outliers, i)
		}
	}

	return outliers, nil
}

// ModifiedZScoreOutlierDetection identifies outliers using Modified Z-score based on MAD
// More robust than standard Z-score as it uses median instead of mean
//
// Parameters:
//
//	data: Input array
//	threshold: Modified Z-score threshold (typically 3.5)
//
// Returns: Slice of indices where outliers are detected
func ModifiedZScoreOutlierDetection(data *array.Array, threshold float64) ([]int, error) {
	if data == nil {
		return nil, fmt.Errorf("data array cannot be nil")
	}
	if threshold <= 0 {
		return nil, fmt.Errorf("threshold must be positive, got %f", threshold)
	}

	n := data.Size()
	if n == 0 {
		return []int{}, nil
	}

	// Convert to slice for easier manipulation
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = convertToFloat64(data.At(i))
	}

	// Calculate median
	sortedValues := make([]float64, n)
	copy(sortedValues, values)
	sort.Float64s(sortedValues)

	var median float64
	if n%2 == 0 {
		median = (sortedValues[n/2-1] + sortedValues[n/2]) / 2
	} else {
		median = sortedValues[n/2]
	}

	// Calculate MAD (Median Absolute Deviation)
	absDeviations := make([]float64, n)
	for i, val := range values {
		absDeviations[i] = math.Abs(val - median)
	}
	sort.Float64s(absDeviations)

	var mad float64
	if n%2 == 0 {
		mad = (absDeviations[n/2-1] + absDeviations[n/2]) / 2
	} else {
		mad = absDeviations[n/2]
	}

	// Avoid division by zero
	if mad < 1e-10 {
		return []int{}, nil
	}

	// Find outliers using modified Z-score
	var outliers []int
	for i, val := range values {
		modifiedZScore := 0.6745 * math.Abs(val-median) / mad
		if modifiedZScore > threshold {
			outliers = append(outliers, i)
		}
	}

	return outliers, nil
}

// RemoveOutliers creates a new array with outliers removed
//
// Parameters:
//
//	data: Original array
//	outlierIndices: Indices of elements to remove
//
// Returns: New array with outliers removed
func RemoveOutliers(data *array.Array, outlierIndices []int) (*array.Array, error) {
	if data == nil {
		return nil, fmt.Errorf("data array cannot be nil")
	}

	n := data.Size()
	if n == 0 {
		return array.Empty(internal.Shape{0}, data.DType()), nil
	}

	// Create set of indices to remove for O(1) lookup
	toRemove := make(map[int]bool)
	for _, idx := range outlierIndices {
		if idx >= 0 && idx < n {
			toRemove[idx] = true
		}
	}

	// Count remaining elements
	remaining := n - len(toRemove)
	if remaining <= 0 {
		return array.Empty(internal.Shape{0}, data.DType()), nil
	}

	// Create new array with remaining elements
	result := array.Empty(internal.Shape{remaining}, data.DType())
	j := 0
	for i := 0; i < n; i++ {
		if !toRemove[i] {
			result.Set(data.At(i), j)
			j++
		}
	}

	return result, nil
}
