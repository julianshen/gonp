package stats

import (
	"errors"
	"math/rand"
	"sort"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// StatisticFunc represents a function that computes a statistic from a sample
type StatisticFunc func(*array.Array) float64

// TestStatisticFunc represents a function that computes a test statistic from two samples
type TestStatisticFunc func(*array.Array, *array.Array) float64

// BootstrapSample generates a bootstrap sample by sampling with replacement
func BootstrapSample(data *array.Array, randomState int64) (*array.Array, []int, error) {
	if data == nil {
		return nil, nil, errors.New("data cannot be nil")
	}

	if data.Size() == 0 {
		return nil, nil, errors.New("data cannot be empty")
	}

	n := data.Size()
	rng := rand.New(rand.NewSource(randomState))

	// Generate bootstrap indices
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = rng.Intn(n)
	}

	// Create bootstrap sample
	sample := array.Empty(internal.Shape{n}, data.DType())
	for i, idx := range indices {
		sample.Set(data.At(idx), i)
	}

	return sample, indices, nil
}

// BootstrapResamples generates multiple bootstrap samples
func BootstrapResamples(data *array.Array, nResamples int, randomState int64) ([]*array.Array, error) {
	if data == nil {
		return nil, errors.New("data cannot be nil")
	}

	if nResamples <= 0 {
		return nil, errors.New("nResamples must be greater than 0")
	}

	resamples := make([]*array.Array, nResamples)
	rng := rand.New(rand.NewSource(randomState))

	for i := 0; i < nResamples; i++ {
		// Use a different seed for each resample
		seed := rng.Int63()
		sample, _, err := BootstrapSample(data, seed)
		if err != nil {
			return nil, err
		}
		resamples[i] = sample
	}

	return resamples, nil
}

// BootstrapConfidenceInterval computes confidence intervals using bootstrap resampling
// data: input dataset
// statFunc: function to compute the statistic of interest
// nBootstrap: number of bootstrap samples to generate
// confidenceLevel: confidence level (e.g., 0.95 for 95% CI)
// randomState: random seed for reproducibility
func BootstrapConfidenceInterval(data *array.Array, statFunc StatisticFunc, nBootstrap int, confidenceLevel float64, randomState int64) (float64, float64, error) {
	if data == nil {
		return 0, 0, errors.New("data cannot be nil")
	}

	if statFunc == nil {
		return 0, 0, errors.New("statistic function cannot be nil")
	}

	if nBootstrap <= 0 {
		return 0, 0, errors.New("nBootstrap must be greater than 0")
	}

	if confidenceLevel <= 0 || confidenceLevel >= 1 {
		return 0, 0, errors.New("confidenceLevel must be between 0 and 1")
	}

	// Generate bootstrap samples and compute statistics
	bootstrapStats := make([]float64, nBootstrap)
	rng := rand.New(rand.NewSource(randomState))

	for i := 0; i < nBootstrap; i++ {
		// Generate bootstrap sample
		seed := rng.Int63()
		sample, _, err := BootstrapSample(data, seed)
		if err != nil {
			return 0, 0, err
		}

		// Compute statistic for this sample
		bootstrapStats[i] = statFunc(sample)
	}

	// Sort the bootstrap statistics
	sort.Float64s(bootstrapStats)

	// Calculate confidence interval bounds
	alpha := 1.0 - confidenceLevel
	lowerPercentile := alpha / 2.0
	upperPercentile := 1.0 - alpha/2.0

	lowerIndex := int(lowerPercentile * float64(nBootstrap))
	upperIndex := int(upperPercentile * float64(nBootstrap))

	// Ensure indices are within bounds
	if lowerIndex < 0 {
		lowerIndex = 0
	}
	if upperIndex >= nBootstrap {
		upperIndex = nBootstrap - 1
	}

	lower := bootstrapStats[lowerIndex]
	upper := bootstrapStats[upperIndex]

	return lower, upper, nil
}

// BootstrapHypothesisTest performs a bootstrap hypothesis test for two samples
// group1, group2: the two samples to compare
// testStatFunc: function to compute the test statistic
// nBootstrap: number of bootstrap samples to generate
// randomState: random seed for reproducibility
// Returns: p-value, observed test statistic, error
func BootstrapHypothesisTest(group1, group2 *array.Array, testStatFunc TestStatisticFunc, nBootstrap int, randomState int64) (float64, float64, error) {
	if group1 == nil || group2 == nil {
		return 0, 0, errors.New("both groups must be non-nil")
	}

	if testStatFunc == nil {
		return 0, 0, errors.New("test statistic function cannot be nil")
	}

	if nBootstrap <= 0 {
		return 0, 0, errors.New("nBootstrap must be greater than 0")
	}

	// Compute observed test statistic
	observedStat := testStatFunc(group1, group2)

	// Combine both groups for permutation testing under null hypothesis
	combined := concatenateArrays(group1, group2)
	n1 := group1.Size()
	n2 := group2.Size()
	nTotal := combined.Size()

	if nTotal != n1+n2 {
		return 0, 0, errors.New("combined array size mismatch")
	}

	// Generate bootstrap distribution under null hypothesis
	bootstrapStats := make([]float64, nBootstrap)
	rng := rand.New(rand.NewSource(randomState))

	for i := 0; i < nBootstrap; i++ {
		// Randomly permute the combined sample
		indices := make([]int, nTotal)
		for j := 0; j < nTotal; j++ {
			indices[j] = j
		}
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		// Split permuted sample into two groups of original sizes
		bootGroup1 := array.Empty(internal.Shape{n1}, combined.DType())
		bootGroup2 := array.Empty(internal.Shape{n2}, combined.DType())

		for j := 0; j < n1; j++ {
			bootGroup1.Set(combined.At(indices[j]), j)
		}
		for j := 0; j < n2; j++ {
			bootGroup2.Set(combined.At(indices[n1+j]), j)
		}

		// Compute test statistic for permuted samples
		bootstrapStats[i] = testStatFunc(bootGroup1, bootGroup2)
	}

	// Calculate p-value (proportion of bootstrap stats >= observed stat)
	count := 0
	for _, stat := range bootstrapStats {
		if stat >= observedStat {
			count++
		}
	}

	pValue := float64(count) / float64(nBootstrap)

	return pValue, observedStat, nil
}

// Helper functions

// concatenateArrays concatenates two arrays vertically
func concatenateArrays(arr1, arr2 *array.Array) *array.Array {
	if arr1 == nil {
		return arr2
	}
	if arr2 == nil {
		return arr1
	}

	n1 := arr1.Size()
	n2 := arr2.Size()
	combined := array.Empty(internal.Shape{n1 + n2}, arr1.DType())

	// Copy first array
	for i := 0; i < n1; i++ {
		combined.Set(arr1.At(i), i)
	}

	// Copy second array
	for i := 0; i < n2; i++ {
		combined.Set(arr2.At(i), n1+i)
	}

	return combined
}
