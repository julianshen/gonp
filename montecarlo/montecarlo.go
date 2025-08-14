package montecarlo

import (
	"errors"
	"math"
	"math/rand"
	"time"
)

// Bounds represents integration bounds for one dimension
type Bounds struct {
	Low  float64
	High float64
}

// IntegrationResult represents the result of Monte Carlo integration
type IntegrationResult struct {
	Value              float64    // Estimated integral value
	StandardError      float64    // Standard error of the estimate
	ConfidenceInterval [2]float64 // Confidence interval
	Samples            int        // Number of samples used
	Variance           float64    // Sample variance
}

// IntegrationOptions provides options for Monte Carlo integration
type IntegrationOptions struct {
	Samples           int                // Number of samples to use
	ConfidenceLevel   float64            // Confidence level (default 0.95)
	RandomSeed        int64              // Random seed for reproducibility
	VarianceReduction string             // Variance reduction method
	ImportanceParams  map[string]float64 // Parameters for importance sampling
}

// ConvergenceDiagnostics contains convergence analysis results
type ConvergenceDiagnostics struct {
	RunningMeans  []float64 // Running means at different sample sizes
	RunningErrors []float64 // Running standard errors
	SampleSizes   []int     // Sample sizes for each measurement
	Converged     bool      // Whether convergence was achieved
}

// Integrate performs basic Monte Carlo integration
func Integrate(f func([]float64) float64, bounds []Bounds, samples int) (*IntegrationResult, error) {
	if f == nil {
		return nil, errors.New("function cannot be nil")
	}
	if len(bounds) == 0 {
		return nil, errors.New("bounds cannot be empty")
	}
	if samples <= 0 {
		return nil, errors.New("samples must be positive")
	}

	// Validate bounds
	for _, bound := range bounds {
		if bound.Low >= bound.High {
			return nil, errors.New("invalid bounds: low must be less than high")
		}
		if math.IsInf(bound.Low, 0) || math.IsInf(bound.High, 0) {
			return nil, errors.New("bounds cannot be infinite")
		}
	}

	// Initialize random number generator with current time
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Calculate volume of integration region
	volume := 1.0
	for _, bound := range bounds {
		volume *= (bound.High - bound.Low)
	}

	// Monte Carlo integration
	sum := 0.0
	sumSquares := 0.0
	validSamples := 0
	x := make([]float64, len(bounds))

	for i := 0; i < samples; i++ {
		// Generate random point in integration region
		for j, bound := range bounds {
			x[j] = bound.Low + rng.Float64()*(bound.High-bound.Low)
		}

		// Evaluate function
		fx := f(x)

		// Skip infinite or NaN values
		if math.IsInf(fx, 0) || math.IsNaN(fx) {
			continue
		}

		sum += fx
		sumSquares += fx * fx
		validSamples++
	}

	if validSamples == 0 {
		return nil, errors.New("no valid function evaluations")
	}

	// Calculate statistics
	mean := sum / float64(validSamples)
	meanSquares := sumSquares / float64(validSamples)
	variance := meanSquares - mean*mean

	if variance < 0 {
		variance = 0 // Numerical precision issues
	}

	// Estimate integral and standard error
	estimate := volume * mean
	standardError := volume * math.Sqrt(variance/float64(validSamples))

	// Calculate 95% confidence interval
	zValue := 1.96 // 95% confidence
	margin := zValue * standardError

	return &IntegrationResult{
		Value:              estimate,
		StandardError:      standardError,
		ConfidenceInterval: [2]float64{estimate - margin, estimate + margin},
		Samples:            validSamples,
		Variance:           variance,
	}, nil
}

// IntegrateWithOptions performs Monte Carlo integration with custom options
func IntegrateWithOptions(f func([]float64) float64, bounds []Bounds, options *IntegrationOptions) (*IntegrationResult, error) {
	if options == nil {
		return Integrate(f, bounds, 10000) // Default samples
	}

	// Set defaults
	samples := options.Samples
	if samples <= 0 {
		samples = 10000
	}

	confidenceLevel := options.ConfidenceLevel
	if confidenceLevel <= 0 || confidenceLevel >= 1 {
		confidenceLevel = 0.95
	}

	// Basic integration (importance sampling will be implemented later)
	result, err := integrateBasic(f, bounds, samples, options.RandomSeed, confidenceLevel)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// integrateBasic performs basic Monte Carlo integration with seed and confidence level
func integrateBasic(f func([]float64) float64, bounds []Bounds, samples int, seed int64, confidenceLevel float64) (*IntegrationResult, error) {
	if f == nil {
		return nil, errors.New("function cannot be nil")
	}
	if len(bounds) == 0 {
		return nil, errors.New("bounds cannot be empty")
	}
	if samples <= 0 {
		return nil, errors.New("samples must be positive")
	}

	// Validate bounds
	for _, bound := range bounds {
		if bound.Low >= bound.High {
			return nil, errors.New("invalid bounds: low must be less than high")
		}
		if math.IsInf(bound.Low, 0) || math.IsInf(bound.High, 0) {
			return nil, errors.New("bounds cannot be infinite")
		}
	}

	// Initialize random number generator
	var rng *rand.Rand
	if seed != 0 {
		rng = rand.New(rand.NewSource(seed))
	} else {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	// Calculate volume of integration region
	volume := 1.0
	for _, bound := range bounds {
		volume *= (bound.High - bound.Low)
	}

	// Monte Carlo integration
	sum := 0.0
	sumSquares := 0.0
	validSamples := 0
	x := make([]float64, len(bounds))

	for i := 0; i < samples; i++ {
		// Generate random point in integration region
		for j, bound := range bounds {
			x[j] = bound.Low + rng.Float64()*(bound.High-bound.Low)
		}

		// Evaluate function
		fx := f(x)

		// Skip infinite or NaN values
		if math.IsInf(fx, 0) || math.IsNaN(fx) {
			continue
		}

		sum += fx
		sumSquares += fx * fx
		validSamples++
	}

	if validSamples == 0 {
		return nil, errors.New("no valid function evaluations")
	}

	// Calculate statistics
	mean := sum / float64(validSamples)
	meanSquares := sumSquares / float64(validSamples)
	variance := meanSquares - mean*mean

	if variance < 0 {
		variance = 0 // Numerical precision issues
	}

	// Estimate integral and standard error
	estimate := volume * mean
	standardError := volume * math.Sqrt(variance/float64(validSamples))

	// Calculate confidence interval with specified level
	var zValue float64
	if confidenceLevel == 0.95 {
		zValue = 1.96
	} else if confidenceLevel == 0.99 {
		zValue = 2.576
	} else if confidenceLevel == 0.90 {
		zValue = 1.645
	} else {
		zValue = 1.96 // Default to 95%
	}

	margin := zValue * standardError

	return &IntegrationResult{
		Value:              estimate,
		StandardError:      standardError,
		ConfidenceInterval: [2]float64{estimate - margin, estimate + margin},
		Samples:            validSamples,
		Variance:           variance,
	}, nil
}

// AnalyzeConvergence analyzes the convergence of Monte Carlo integration
func AnalyzeConvergence(f func([]float64) float64, bounds []Bounds, maxSamples int) (*ConvergenceDiagnostics, error) {
	if f == nil {
		return nil, errors.New("function cannot be nil")
	}
	if len(bounds) == 0 {
		return nil, errors.New("bounds cannot be empty")
	}
	if maxSamples <= 10 {
		return nil, errors.New("maxSamples must be greater than 10")
	}

	// Validate bounds
	for _, bound := range bounds {
		if bound.Low >= bound.High {
			return nil, errors.New("invalid bounds: low must be less than high")
		}
	}

	// Initialize random number generator
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Calculate volume of integration region
	volume := 1.0
	for _, bound := range bounds {
		volume *= (bound.High - bound.Low)
	}

	// Collect samples for convergence analysis
	x := make([]float64, len(bounds))
	var values []float64

	for i := 0; i < maxSamples; i++ {
		// Generate random point in integration region
		for j, bound := range bounds {
			x[j] = bound.Low + rng.Float64()*(bound.High-bound.Low)
		}

		// Evaluate function
		fx := f(x)

		// Skip infinite or NaN values
		if math.IsInf(fx, 0) || math.IsNaN(fx) {
			continue
		}

		values = append(values, fx)
	}

	if len(values) < 10 {
		return nil, errors.New("insufficient valid function evaluations")
	}

	// Analyze convergence at different sample sizes
	checkPoints := []int{100, 500, 1000, 2500, 5000, 10000, 25000, 50000}
	var runningMeans []float64
	var runningErrors []float64
	var sampleSizes []int

	for _, n := range checkPoints {
		if n > len(values) {
			n = len(values)
		}

		// Calculate running statistics
		sum := 0.0
		sumSquares := 0.0

		for i := 0; i < n; i++ {
			sum += values[i]
			sumSquares += values[i] * values[i]
		}

		mean := sum / float64(n)
		meanSquares := sumSquares / float64(n)
		variance := meanSquares - mean*mean

		if variance < 0 {
			variance = 0
		}

		estimate := volume * mean
		standardError := volume * math.Sqrt(variance/float64(n))

		runningMeans = append(runningMeans, estimate)
		runningErrors = append(runningErrors, standardError)
		sampleSizes = append(sampleSizes, n)

		if n >= len(values) {
			break
		}
	}

	// Simple convergence check: error should decrease with sample size
	converged := true
	if len(runningErrors) >= 2 {
		lastError := runningErrors[len(runningErrors)-1]
		prevError := runningErrors[len(runningErrors)-2]
		if lastError > prevError {
			converged = false
		}
	}

	return &ConvergenceDiagnostics{
		RunningMeans:  runningMeans,
		RunningErrors: runningErrors,
		SampleSizes:   sampleSizes,
		Converged:     converged,
	}, nil
}
