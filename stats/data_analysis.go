package stats

import (
	"errors"
	"math"
	"sort"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// HistogramResult represents the result of histogram calculation
type HistogramResult struct {
	Counts   *array.Array // Histogram counts for each bin
	BinEdges *array.Array // Bin edges (n+1 edges for n bins)
	BinWidth float64      // Width of each bin (for uniform bins)
}

// Histogram computes a histogram of the data
func Histogram(arr *array.Array, bins interface{}) (*HistogramResult, error) {
	ctx := internal.StartProfiler("Stats.Histogram")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("Histogram", "array cannot be nil")
	}

	// Extract non-NaN values
	var values []float64
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			values = append(values, val)
		}
	}

	if len(values) == 0 {
		return nil, errors.New("no valid values for histogram")
	}

	// Determine bin edges
	var binEdges []float64
	switch b := bins.(type) {
	case int:
		binEdges = createUniformBins(values, b)
	case []float64:
		binEdges = b
	case *array.Array:
		binEdges = extractFloat64Values(b)
	default:
		return nil, errors.New("bins must be int, []float64, or *array.Array")
	}

	if len(binEdges) < 2 {
		return nil, errors.New("need at least 2 bin edges")
	}

	// Sort bin edges
	sort.Float64s(binEdges)

	// Count values in each bin
	nBins := len(binEdges) - 1
	counts := make([]float64, nBins)

	for _, value := range values {
		binIndex := findBinIndex(value, binEdges)
		if binIndex >= 0 && binIndex < nBins {
			counts[binIndex]++
		}
	}

	// Create result arrays
	countsArray, err := array.FromSlice(counts)
	if err != nil {
		return nil, err
	}

	edgesArray, err := array.FromSlice(binEdges)
	if err != nil {
		return nil, err
	}

	// Calculate bin width (for uniform bins)
	binWidth := 0.0
	if len(binEdges) > 1 {
		binWidth = binEdges[1] - binEdges[0]
	}

	return &HistogramResult{
		Counts:   countsArray,
		BinEdges: edgesArray,
		BinWidth: binWidth,
	}, nil
}

// Histogram2D computes a 2D histogram
type Histogram2DResult struct {
	Counts    *array.Array // 2D histogram counts
	XBinEdges *array.Array // X-axis bin edges
	YBinEdges *array.Array // Y-axis bin edges
}

// Histogram2D computes a 2D histogram of two datasets
func Histogram2D(x, y *array.Array, xBins, yBins interface{}) (*Histogram2DResult, error) {
	ctx := internal.StartProfiler("Stats.Histogram2D")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if x == nil || y == nil {
		return nil, internal.NewValidationErrorWithMsg("Histogram2D", "arrays cannot be nil")
	}

	if x.Size() != y.Size() {
		return nil, errors.New("x and y arrays must have the same size")
	}

	// Extract paired non-NaN values
	var xValues, yValues []float64
	xFlat := x.Flatten()
	yFlat := y.Flatten()

	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if !math.IsNaN(xVal) && !math.IsNaN(yVal) {
			xValues = append(xValues, xVal)
			yValues = append(yValues, yVal)
		}
	}

	if len(xValues) == 0 {
		return nil, errors.New("no valid paired values for 2D histogram")
	}

	// Determine bin edges for both dimensions
	var xBinEdges, yBinEdges []float64

	switch xb := xBins.(type) {
	case int:
		xBinEdges = createUniformBins(xValues, xb)
	case []float64:
		xBinEdges = xb
	case *array.Array:
		xBinEdges = extractFloat64Values(xb)
	default:
		return nil, errors.New("xBins must be int, []float64, or *array.Array")
	}

	switch yb := yBins.(type) {
	case int:
		yBinEdges = createUniformBins(yValues, yb)
	case []float64:
		yBinEdges = yb
	case *array.Array:
		yBinEdges = extractFloat64Values(yb)
	default:
		return nil, errors.New("yBins must be int, []float64, or *array.Array")
	}

	sort.Float64s(xBinEdges)
	sort.Float64s(yBinEdges)

	nXBins := len(xBinEdges) - 1
	nYBins := len(yBinEdges) - 1

	if nXBins < 1 || nYBins < 1 {
		return nil, errors.New("need at least 1 bin in each dimension")
	}

	// Count values in each 2D bin
	counts := array.Zeros(internal.Shape{nYBins, nXBins}, internal.Float64)

	for i := 0; i < len(xValues); i++ {
		xBinIndex := findBinIndex(xValues[i], xBinEdges)
		yBinIndex := findBinIndex(yValues[i], yBinEdges)

		if xBinIndex >= 0 && xBinIndex < nXBins && yBinIndex >= 0 && yBinIndex < nYBins {
			currentCount := convertToFloat64(counts.At(yBinIndex, xBinIndex))
			counts.Set(currentCount+1, yBinIndex, xBinIndex)
		}
	}

	// Create result arrays
	xEdgesArray, err := array.FromSlice(xBinEdges)
	if err != nil {
		return nil, err
	}

	yEdgesArray, err := array.FromSlice(yBinEdges)
	if err != nil {
		return nil, err
	}

	return &Histogram2DResult{
		Counts:    counts,
		XBinEdges: xEdgesArray,
		YBinEdges: yEdgesArray,
	}, nil
}

// Binning assigns values to discrete bins
func Binning(arr *array.Array, binEdges *array.Array) (*array.Array, error) {
	ctx := internal.StartProfiler("Stats.Binning")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil || binEdges == nil {
		return nil, internal.NewValidationErrorWithMsg("Binning", "arrays cannot be nil")
	}

	edges := extractFloat64Values(binEdges)
	if len(edges) < 2 {
		return nil, errors.New("need at least 2 bin edges")
	}

	sort.Float64s(edges)

	// Create result array with same shape as input
	result := array.Zeros(arr.Shape(), internal.Int64)

	flatArr := arr.Flatten()
	flatResult := result.Flatten()

	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))

		if math.IsNaN(val) {
			flatResult.Set(int64(-1), i) // Use -1 for NaN values
		} else {
			binIndex := findBinIndex(val, edges)
			flatResult.Set(int64(binIndex), i)
		}
	}

	return result, nil
}

// MovingAverage computes the moving average with specified window size
func MovingAverage(arr *array.Array, windowSize int) (*array.Array, error) {
	ctx := internal.StartProfiler("Stats.MovingAverage")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("MovingAverage", "array cannot be nil")
	}

	if windowSize <= 0 {
		return nil, internal.NewValidationErrorWithMsg("MovingAverage", "window size must be positive")
	}

	if arr.Shape().Ndim() != 1 {
		return nil, internal.NewValidationErrorWithMsg("MovingAverage", "array must be 1-dimensional")
	}

	size := arr.Size()
	if windowSize > size {
		return nil, errors.New("window size cannot be larger than array size")
	}

	// Calculate moving averages
	resultSize := size - windowSize + 1
	result := array.Zeros(internal.Shape{resultSize}, internal.Float64)

	for i := 0; i < resultSize; i++ {
		var sum float64
		var count int

		for j := 0; j < windowSize; j++ {
			val := convertToFloat64(arr.At(i + j))
			if !math.IsNaN(val) {
				sum += val
				count++
			}
		}

		if count > 0 {
			result.Set(sum/float64(count), i)
		} else {
			result.Set(math.NaN(), i)
		}
	}

	return result, nil
}

// MovingStd computes the moving standard deviation with specified window size
func MovingStd(arr *array.Array, windowSize int) (*array.Array, error) {
	ctx := internal.StartProfiler("Stats.MovingStd")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("MovingStd", "array cannot be nil")
	}

	if windowSize <= 1 {
		return nil, internal.NewValidationErrorWithMsg("MovingStd", "window size must be greater than 1")
	}

	if arr.Shape().Ndim() != 1 {
		return nil, internal.NewValidationErrorWithMsg("MovingStd", "array must be 1-dimensional")
	}

	size := arr.Size()
	if windowSize > size {
		return nil, errors.New("window size cannot be larger than array size")
	}

	resultSize := size - windowSize + 1
	result := array.Zeros(internal.Shape{resultSize}, internal.Float64)

	for i := 0; i < resultSize; i++ {
		// Extract window values
		var values []float64
		for j := 0; j < windowSize; j++ {
			val := convertToFloat64(arr.At(i + j))
			if !math.IsNaN(val) {
				values = append(values, val)
			}
		}

		if len(values) <= 1 {
			result.Set(math.NaN(), i)
			continue
		}

		// Calculate mean
		var mean float64
		for _, v := range values {
			mean += v
		}
		mean /= float64(len(values))

		// Calculate variance
		var variance float64
		for _, v := range values {
			diff := v - mean
			variance += diff * diff
		}
		variance /= float64(len(values) - 1)

		result.Set(math.Sqrt(variance), i)
	}

	return result, nil
}

// Helper functions

// createUniformBins creates uniformly spaced bin edges
func createUniformBins(values []float64, nBins int) []float64 {
	if len(values) == 0 || nBins <= 0 {
		return []float64{0, 1} // Default
	}

	min := values[0]
	max := values[0]

	for _, v := range values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	// Add small padding to include the maximum value
	if min == max {
		min -= 0.5
		max += 0.5
	} else {
		padding := (max - min) * 0.001
		min -= padding
		max += padding
	}

	binEdges := make([]float64, nBins+1)
	step := (max - min) / float64(nBins)

	for i := range binEdges {
		binEdges[i] = min + float64(i)*step
	}

	return binEdges
}

// findBinIndex finds which bin a value belongs to
func findBinIndex(value float64, binEdges []float64) int {
	// Binary search for efficiency
	left, right := 0, len(binEdges)-1

	// Handle edge cases
	if value < binEdges[0] {
		return -1 // Below range
	}
	if value >= binEdges[right] {
		return len(binEdges) - 2 // Last bin (or above range)
	}

	// Binary search
	for left < right-1 {
		mid := (left + right) / 2
		if value < binEdges[mid] {
			right = mid
		} else {
			left = mid
		}
	}

	return left
}

// extractFloat64Values extracts float64 values from an array
func extractFloat64Values(arr *array.Array) []float64 {
	values := make([]float64, arr.Size())
	flatArr := arr.Flatten()

	for i := 0; i < flatArr.Size(); i++ {
		values[i] = convertToFloat64(flatArr.At(i))
	}

	return values
}
