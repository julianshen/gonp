package stats

import (
	"errors"
	"math"
	"sort"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
)

// Correlation calculates the Pearson correlation coefficient
func Correlation(x, y *array.Array) (float64, error) {
	if x.Size() != y.Size() {
		return 0, errors.New("arrays must have the same size")
	}

	if x.Size() == 0 {
		return 0, errors.New("cannot calculate correlation of empty arrays")
	}

	// Extract paired non-NaN values
	var xVals, yVals []float64
	xFlat := x.Flatten()
	yFlat := y.Flatten()
	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if !math.IsNaN(xVal) && !math.IsNaN(yVal) {
			xVals = append(xVals, xVal)
			yVals = append(yVals, yVal)
		}
	}

	if len(xVals) < 2 {
		return 0, errors.New("need at least 2 valid pairs for correlation")
	}

	// Calculate means
	xMean := 0.0
	yMean := 0.0
	for i := range xVals {
		xMean += xVals[i]
		yMean += yVals[i]
	}
	xMean /= float64(len(xVals))
	yMean /= float64(len(yVals))

	// Calculate correlation
	numerator := 0.0
	xSumSq := 0.0
	ySumSq := 0.0

	for i := range xVals {
		xDiff := xVals[i] - xMean
		yDiff := yVals[i] - yMean

		numerator += xDiff * yDiff
		xSumSq += xDiff * xDiff
		ySumSq += yDiff * yDiff
	}

	denominator := math.Sqrt(xSumSq * ySumSq)
	if denominator == 0 {
		return 0, errors.New("cannot calculate correlation: zero variance")
	}

	return numerator / denominator, nil
}

// Covariance calculates the covariance between two arrays
func Covariance(x, y *array.Array) (float64, error) {
	if x.Size() != y.Size() {
		return 0, errors.New("arrays must have the same size")
	}

	if x.Size() == 0 {
		return 0, errors.New("cannot calculate covariance of empty arrays")
	}

	// Extract paired non-NaN values
	var xVals, yVals []float64
	xFlat := x.Flatten()
	yFlat := y.Flatten()
	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if !math.IsNaN(xVal) && !math.IsNaN(yVal) {
			xVals = append(xVals, xVal)
			yVals = append(yVals, yVal)
		}
	}

	if len(xVals) < 2 {
		return 0, errors.New("need at least 2 valid pairs for covariance")
	}

	// Calculate means
	xMean := 0.0
	yMean := 0.0
	for i := range xVals {
		xMean += xVals[i]
		yMean += yVals[i]
	}
	xMean /= float64(len(xVals))
	yMean /= float64(len(yVals))

	// Calculate covariance
	sum := 0.0
	for i := range xVals {
		sum += (xVals[i] - xMean) * (yVals[i] - yMean)
	}

	return sum / float64(len(xVals)-1), nil
}

// CovarianceMatrix calculates the covariance matrix for multiple series
func CovarianceMatrix(seriesList []*series.Series) (*array.Array, error) {
	if len(seriesList) == 0 {
		return nil, errors.New("need at least one series")
	}

	n := len(seriesList)

	// Initialize result matrix
	matrixData := make([]float64, n*n)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				// Diagonal: variance
				variance, err := Var(seriesList[i].Data())
				if err != nil {
					return nil, err
				}
				matrixData[i*n+j] = variance
			} else {
				// Off-diagonal: covariance
				cov, err := Covariance(seriesList[i].Data(), seriesList[j].Data())
				if err != nil {
					return nil, err
				}
				matrixData[i*n+j] = cov
			}
		}
	}

	// Create 2D array from flat data
	shape := internal.Shape{n, n}
	result := array.Empty(shape, internal.Float64)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			result.Set(matrixData[i*n+j], i, j)
		}
	}

	return result, nil
}

// CorrelationMatrix calculates the correlation matrix for multiple series
func CorrelationMatrix(seriesList []*series.Series) (*array.Array, error) {
	if len(seriesList) == 0 {
		return nil, errors.New("need at least one series")
	}

	n := len(seriesList)

	// Create matrix data based on actual size
	matrixData := make([]float64, n*n)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				// Diagonal: perfect correlation with self
				matrixData[i*n+j] = 1.0
			} else {
				// Off-diagonal: correlation
				corr, err := Correlation(seriesList[i].Data(), seriesList[j].Data())
				if err != nil {
					return nil, err
				}
				matrixData[i*n+j] = corr
			}
		}
	}

	// Reshape the flat data into a 2D array
	result := array.Empty(internal.Shape{n, n}, internal.Float64)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			result.Set(matrixData[i*n+j], i, j)
		}
	}

	return result, nil
}

// SpearmanCorrelation calculates the Spearman rank correlation coefficient
func SpearmanCorrelation(x, y *array.Array) (float64, error) {
	if x.Size() != y.Size() {
		return 0, errors.New("arrays must have the same size")
	}

	if x.Size() == 0 {
		return 0, errors.New("cannot calculate Spearman correlation of empty arrays")
	}

	// Extract paired non-NaN values with original indices
	type valuePair struct {
		x, y  float64
		index int
	}

	var pairs []valuePair
	xFlat := x.Flatten()
	yFlat := y.Flatten()
	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if !math.IsNaN(xVal) && !math.IsNaN(yVal) {
			pairs = append(pairs, valuePair{xVal, yVal, i})
		}
	}

	if len(pairs) < 2 {
		return 0, errors.New("need at least 2 valid pairs for Spearman correlation")
	}

	// Calculate ranks for x values
	xRanks := make([]float64, len(pairs))
	sortedPairs := make([]valuePair, len(pairs))
	copy(sortedPairs, pairs)

	sort.Slice(sortedPairs, func(i, j int) bool {
		return sortedPairs[i].x < sortedPairs[j].x
	})

	for i, pair := range sortedPairs {
		for j, originalPair := range pairs {
			if pair.index == originalPair.index {
				xRanks[j] = float64(i + 1)
				break
			}
		}
	}

	// Calculate ranks for y values
	yRanks := make([]float64, len(pairs))
	sort.Slice(sortedPairs, func(i, j int) bool {
		return sortedPairs[i].y < sortedPairs[j].y
	})

	for i, pair := range sortedPairs {
		for j, originalPair := range pairs {
			if pair.index == originalPair.index {
				yRanks[j] = float64(i + 1)
				break
			}
		}
	}

	// Calculate Pearson correlation of ranks
	xRankArr, _ := array.FromSlice(xRanks)
	yRankArr, _ := array.FromSlice(yRanks)

	return Correlation(xRankArr, yRankArr)
}

// KendallTau calculates Kendall's tau rank correlation coefficient
func KendallTau(x, y *array.Array) (float64, error) {
	if x.Size() != y.Size() {
		return 0, errors.New("arrays must have the same size")
	}

	if x.Size() == 0 {
		return 0, errors.New("cannot calculate Kendall tau of empty arrays")
	}

	// Extract paired non-NaN values
	var xVals, yVals []float64
	xFlat := x.Flatten()
	yFlat := y.Flatten()
	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if !math.IsNaN(xVal) && !math.IsNaN(yVal) {
			xVals = append(xVals, xVal)
			yVals = append(yVals, yVal)
		}
	}

	if len(xVals) < 2 {
		return 0, errors.New("need at least 2 valid pairs for Kendall tau")
	}

	n := len(xVals)
	concordant := 0
	discordant := 0

	// Count concordant and discordant pairs
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			xDiff := xVals[j] - xVals[i]
			yDiff := yVals[j] - yVals[i]

			if (xDiff > 0 && yDiff > 0) || (xDiff < 0 && yDiff < 0) {
				concordant++
			} else if (xDiff > 0 && yDiff < 0) || (xDiff < 0 && yDiff > 0) {
				discordant++
			}
			// Ties are ignored in this simple implementation
		}
	}

	totalPairs := n * (n - 1) / 2
	if totalPairs == 0 {
		return 0, errors.New("not enough pairs for Kendall tau")
	}

	return float64(concordant-discordant) / float64(totalPairs), nil
}
