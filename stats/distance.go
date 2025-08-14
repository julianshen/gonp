package stats

import (
	"errors"
	"math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// EuclideanDistance calculates the Euclidean distance between two arrays
func EuclideanDistance(x, y *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.EuclideanDistance")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if x == nil || y == nil {
		return 0, internal.NewValidationErrorWithMsg("EuclideanDistance", "arrays cannot be nil")
	}

	if x.Size() != y.Size() {
		return 0, errors.New("arrays must have the same size")
	}

	if x.Size() == 0 {
		return 0, errors.New("cannot calculate distance of empty arrays")
	}

	var sumSquaredDiff float64
	xFlat := x.Flatten()
	yFlat := y.Flatten()

	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if math.IsNaN(xVal) || math.IsNaN(yVal) {
			continue // Skip NaN values
		}

		diff := xVal - yVal
		sumSquaredDiff += diff * diff
	}

	return math.Sqrt(sumSquaredDiff), nil
}

// ManhattanDistance calculates the Manhattan (L1) distance between two arrays
func ManhattanDistance(x, y *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.ManhattanDistance")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if x == nil || y == nil {
		return 0, internal.NewValidationErrorWithMsg("ManhattanDistance", "arrays cannot be nil")
	}

	if x.Size() != y.Size() {
		return 0, errors.New("arrays must have the same size")
	}

	if x.Size() == 0 {
		return 0, errors.New("cannot calculate distance of empty arrays")
	}

	var sumAbsDiff float64
	xFlat := x.Flatten()
	yFlat := y.Flatten()

	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if math.IsNaN(xVal) || math.IsNaN(yVal) {
			continue // Skip NaN values
		}

		sumAbsDiff += math.Abs(xVal - yVal)
	}

	return sumAbsDiff, nil
}

// ChebyshevDistance calculates the Chebyshev (L∞) distance between two arrays
func ChebyshevDistance(x, y *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.ChebyshevDistance")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if x == nil || y == nil {
		return 0, internal.NewValidationErrorWithMsg("ChebyshevDistance", "arrays cannot be nil")
	}

	if x.Size() != y.Size() {
		return 0, errors.New("arrays must have the same size")
	}

	if x.Size() == 0 {
		return 0, errors.New("cannot calculate distance of empty arrays")
	}

	var maxDiff float64
	xFlat := x.Flatten()
	yFlat := y.Flatten()

	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if math.IsNaN(xVal) || math.IsNaN(yVal) {
			continue // Skip NaN values
		}

		diff := math.Abs(xVal - yVal)
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	return maxDiff, nil
}

// CosineDistance calculates the cosine distance (1 - cosine similarity) between two arrays
func CosineDistance(x, y *array.Array) (float64, error) {
	similarity, err := CosineSimilarity(x, y)
	if err != nil {
		return 0, err
	}
	return 1.0 - similarity, nil
}

// CosineSimilarity calculates the cosine similarity between two arrays
func CosineSimilarity(x, y *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.CosineSimilarity")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if x == nil || y == nil {
		return 0, internal.NewValidationErrorWithMsg("CosineSimilarity", "arrays cannot be nil")
	}

	if x.Size() != y.Size() {
		return 0, errors.New("arrays must have the same size")
	}

	if x.Size() == 0 {
		return 0, errors.New("cannot calculate similarity of empty arrays")
	}

	var dotProduct, xMagnitudeSquared, yMagnitudeSquared float64
	xFlat := x.Flatten()
	yFlat := y.Flatten()

	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if math.IsNaN(xVal) || math.IsNaN(yVal) {
			continue // Skip NaN values
		}

		dotProduct += xVal * yVal
		xMagnitudeSquared += xVal * xVal
		yMagnitudeSquared += yVal * yVal
	}

	magnitude := math.Sqrt(xMagnitudeSquared * yMagnitudeSquared)
	if magnitude == 0 {
		return math.NaN(), nil // Undefined for zero vectors
	}

	return dotProduct / magnitude, nil
}

// MinkowskiDistance calculates the Minkowski distance with parameter p
func MinkowskiDistance(x, y *array.Array, p float64) (float64, error) {
	ctx := internal.StartProfiler("Stats.MinkowskiDistance")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if x == nil || y == nil {
		return 0, internal.NewValidationErrorWithMsg("MinkowskiDistance", "arrays cannot be nil")
	}

	if p <= 0 {
		return 0, internal.NewValidationErrorWithMsg("MinkowskiDistance", "parameter p must be positive")
	}

	if x.Size() != y.Size() {
		return 0, errors.New("arrays must have the same size")
	}

	if x.Size() == 0 {
		return 0, errors.New("cannot calculate distance of empty arrays")
	}

	// Special cases for common values of p
	if p == 1 {
		return ManhattanDistance(x, y)
	}
	if p == 2 {
		return EuclideanDistance(x, y)
	}
	if math.IsInf(p, 1) {
		return ChebyshevDistance(x, y)
	}

	var sum float64
	xFlat := x.Flatten()
	yFlat := y.Flatten()

	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if math.IsNaN(xVal) || math.IsNaN(yVal) {
			continue // Skip NaN values
		}

		diff := math.Abs(xVal - yVal)
		sum += math.Pow(diff, p)
	}

	return math.Pow(sum, 1.0/p), nil
}

// HammingDistance calculates the Hamming distance (for discrete/binary data)
func HammingDistance(x, y *array.Array) (int, error) {
	ctx := internal.StartProfiler("Stats.HammingDistance")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if x == nil || y == nil {
		return 0, internal.NewValidationErrorWithMsg("HammingDistance", "arrays cannot be nil")
	}

	if x.Size() != y.Size() {
		return 0, errors.New("arrays must have the same size")
	}

	if x.Size() == 0 {
		return 0, errors.New("cannot calculate distance of empty arrays")
	}

	var differences int
	xFlat := x.Flatten()
	yFlat := y.Flatten()

	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if math.IsNaN(xVal) || math.IsNaN(yVal) {
			continue // Skip NaN values
		}

		if xVal != yVal {
			differences++
		}
	}

	return differences, nil
}

// JaccardSimilarity calculates the Jaccard similarity coefficient (for binary data)
func JaccardSimilarity(x, y *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Stats.JaccardSimilarity")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if x == nil || y == nil {
		return 0, internal.NewValidationErrorWithMsg("JaccardSimilarity", "arrays cannot be nil")
	}

	if x.Size() != y.Size() {
		return 0, errors.New("arrays must have the same size")
	}

	if x.Size() == 0 {
		return 0, errors.New("cannot calculate similarity of empty arrays")
	}

	var intersection, union int
	xFlat := x.Flatten()
	yFlat := y.Flatten()

	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if math.IsNaN(xVal) || math.IsNaN(yVal) {
			continue // Skip NaN values
		}

		// Treat as binary (non-zero = 1, zero = 0)
		xBinary := 0
		yBinary := 0
		if xVal != 0 {
			xBinary = 1
		}
		if yVal != 0 {
			yBinary = 1
		}

		if xBinary == 1 && yBinary == 1 {
			intersection++
		}
		if xBinary == 1 || yBinary == 1 {
			union++
		}
	}

	if union == 0 {
		return 1.0, nil // Both arrays are all zeros
	}

	return float64(intersection) / float64(union), nil
}

// JaccardDistance calculates the Jaccard distance (1 - Jaccard similarity)
func JaccardDistance(x, y *array.Array) (float64, error) {
	similarity, err := JaccardSimilarity(x, y)
	if err != nil {
		return 0, err
	}
	return 1.0 - similarity, nil
}
