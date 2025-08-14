package mcmc

import (
	"errors"
	"math"
	"math/rand"
	"time"
)

// MCMCChain represents a Markov Chain Monte Carlo sample chain
type MCMCChain struct {
	samples      [][]float64 // MCMC samples
	logPosterior []float64   // Log posterior values
	accepted     int         // Number of accepted proposals
	total        int         // Total number of proposals
	burnIn       int         // Burn-in period
	dimension    int         // Parameter dimension
}

// MCMCOptions provides options for MCMC sampling
type MCMCOptions struct {
	Samples          int     // Number of samples to generate
	BurnIn           int     // Burn-in period
	RandomSeed       int64   // Random seed for reproducibility
	AdaptiveProposal bool    // Use adaptive proposal tuning
	AdaptationPeriod int     // Period for proposal adaptation
	TargetAcceptRate float64 // Target acceptance rate (default 0.44 for 1D, 0.234 for multivariate)
}

// MCMCDiagnostics contains MCMC convergence diagnostics
type MCMCDiagnostics struct {
	EffectiveSampleSize  float64   // Effective sample size
	MCSE                 float64   // Monte Carlo standard error
	AutocorrelationTimes []float64 // Autocorrelation times for each parameter
	AcceptanceRate       float64   // Overall acceptance rate
}

// MetropolisHastings implements the Metropolis-Hastings algorithm
func MetropolisHastings(logPosterior func([]float64) float64, initialState []float64, proposalCov [][]float64, options *MCMCOptions) (*MCMCChain, error) {
	if logPosterior == nil {
		return nil, errors.New("log posterior function cannot be nil")
	}
	if len(initialState) == 0 {
		return nil, errors.New("initial state cannot be empty")
	}
	if options == nil {
		options = &MCMCOptions{Samples: 1000, BurnIn: 200}
	}
	if options.Samples <= 0 {
		return nil, errors.New("samples must be positive")
	}
	if options.BurnIn < 0 {
		return nil, errors.New("burn-in cannot be negative")
	}
	if options.BurnIn >= options.Samples {
		return nil, errors.New("burn-in cannot exceed total samples")
	}

	dimension := len(initialState)

	// Validate proposal covariance matrix
	if len(proposalCov) != dimension {
		return nil, errors.New("proposal covariance dimension mismatch")
	}
	for i, row := range proposalCov {
		if len(row) != dimension {
			return nil, errors.New("proposal covariance must be square matrix")
		}
		if proposalCov[i][i] <= 0 {
			return nil, errors.New("proposal covariance diagonal must be positive")
		}
	}

	// Initialize random number generator
	var rng *rand.Rand
	if options.RandomSeed != 0 {
		rng = rand.New(rand.NewSource(options.RandomSeed))
	} else {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	// Initialize chain
	chain := &MCMCChain{
		samples:      make([][]float64, 0, options.Samples),
		logPosterior: make([]float64, 0, options.Samples),
		accepted:     0,
		total:        0,
		burnIn:       options.BurnIn,
		dimension:    dimension,
	}

	// Current state
	current := make([]float64, dimension)
	copy(current, initialState)
	currentLogPost := logPosterior(current)

	// Check if initial state produces valid log posterior
	if math.IsInf(currentLogPost, 0) || math.IsNaN(currentLogPost) {
		return nil, errors.New("initial state produces invalid log posterior")
	}

	// MCMC sampling loop
	proposal := make([]float64, dimension)

	for i := 0; i < options.Samples; i++ {
		// Generate proposal using multivariate normal
		generateMultivariateNormal(current, proposalCov, proposal, rng)

		// Evaluate log posterior at proposal
		proposalLogPost := logPosterior(proposal)

		// Accept or reject proposal
		if !math.IsInf(proposalLogPost, 0) && !math.IsNaN(proposalLogPost) {
			// Metropolis-Hastings acceptance probability
			logAcceptProb := proposalLogPost - currentLogPost

			if logAcceptProb >= 0 || math.Log(rng.Float64()) < logAcceptProb {
				// Accept proposal
				copy(current, proposal)
				currentLogPost = proposalLogPost
				chain.accepted++
			}
		}

		chain.total++

		// Store sample (always store current state, whether proposal accepted or not)
		sample := make([]float64, dimension)
		copy(sample, current)
		chain.samples = append(chain.samples, sample)
		chain.logPosterior = append(chain.logPosterior, currentLogPost)

		// Adaptive proposal tuning (simplified version)
		if options.AdaptiveProposal && i > 0 && i%options.AdaptationPeriod == 0 && i < options.Samples/2 {
			acceptRate := float64(chain.accepted) / float64(chain.total)
			targetRate := options.TargetAcceptRate
			if targetRate == 0 {
				if dimension == 1 {
					targetRate = 0.44
				} else {
					targetRate = 0.234
				}
			}

			// Simple scaling adaptation
			scaleFactor := 1.0
			if acceptRate > targetRate {
				scaleFactor = 1.1
			} else if acceptRate < targetRate {
				scaleFactor = 0.9
			}

			// Scale proposal covariance
			for j := range proposalCov {
				for k := range proposalCov[j] {
					proposalCov[j][k] *= scaleFactor * scaleFactor
				}
			}
		}
	}

	return chain, nil
}

// GetSamples returns the MCMC samples (after burn-in)
func (c *MCMCChain) GetSamples() [][]float64 {
	if c == nil || len(c.samples) == 0 {
		return nil
	}

	if c.burnIn >= len(c.samples) {
		return nil
	}

	return c.samples[c.burnIn:]
}

// AcceptanceRate returns the acceptance rate
func (c *MCMCChain) AcceptanceRate() float64 {
	if c == nil || c.total == 0 {
		return 0.0
	}
	return float64(c.accepted) / float64(c.total)
}

// EffectiveSampleSize calculates the effective sample size
func (c *MCMCChain) EffectiveSampleSize() float64 {
	samples := c.GetSamples()
	if len(samples) == 0 {
		return 0.0
	}

	// Simple ESS calculation using autocorrelation
	// For multivariate, use the minimum ESS across dimensions
	minESS := math.Inf(1)

	for d := 0; d < c.dimension; d++ {
		// Extract single dimension
		x := make([]float64, len(samples))
		for i, sample := range samples {
			x[i] = sample[d]
		}

		// Calculate autocorrelation time
		autocorrTime := calculateAutocorrelationTime(x)
		ess := float64(len(x)) / (2.0*autocorrTime + 1.0)

		if ess < minESS {
			minESS = ess
		}
	}

	if math.IsInf(minESS, 1) {
		return float64(len(samples))
	}

	return math.Max(1.0, minESS)
}

// DiagnosticTests runs MCMC convergence diagnostics
func (c *MCMCChain) DiagnosticTests() *MCMCDiagnostics {
	if c == nil {
		return nil
	}

	samples := c.GetSamples()
	if len(samples) == 0 {
		return nil
	}

	ess := c.EffectiveSampleSize()
	autocorrTimes := make([]float64, c.dimension)

	// Calculate autocorrelation times for each dimension
	for d := 0; d < c.dimension; d++ {
		x := make([]float64, len(samples))
		for i, sample := range samples {
			x[i] = sample[d]
		}
		autocorrTimes[d] = calculateAutocorrelationTime(x)
	}

	// Calculate Monte Carlo standard error (for first dimension)
	var mcse float64
	if len(samples) > 0 {
		x := make([]float64, len(samples))
		for i, sample := range samples {
			x[i] = sample[0]
		}
		variance := calculateVariance(x)
		mcse = math.Sqrt(variance / ess)
	}

	return &MCMCDiagnostics{
		EffectiveSampleSize:  ess,
		MCSE:                 mcse,
		AutocorrelationTimes: autocorrTimes,
		AcceptanceRate:       c.AcceptanceRate(),
	}
}

// Thin returns a thinned version of the chain
func (c *MCMCChain) Thin(interval int) *MCMCChain {
	if c == nil || interval <= 0 {
		return nil
	}

	samples := c.GetSamples()
	if len(samples) == 0 {
		return nil
	}

	// Create thinned samples
	var thinnedSamples [][]float64
	var thinnedLogPost []float64

	for i := 0; i < len(samples); i += interval {
		thinnedSamples = append(thinnedSamples, samples[i])
		if i+c.burnIn < len(c.logPosterior) {
			thinnedLogPost = append(thinnedLogPost, c.logPosterior[i+c.burnIn])
		}
	}

	return &MCMCChain{
		samples:      thinnedSamples,
		logPosterior: thinnedLogPost,
		accepted:     c.accepted,
		total:        c.total,
		burnIn:       0, // Burn-in already applied
		dimension:    c.dimension,
	}
}

// GelmanRubinDiagnostic calculates the Gelman-Rubin diagnostic for multiple chains
func GelmanRubinDiagnostic(chains []*MCMCChain) (float64, error) {
	if len(chains) < 2 {
		return 0.0, errors.New("need at least 2 chains for Gelman-Rubin diagnostic")
	}

	// Check that all chains have same dimension
	dimension := chains[0].dimension
	for _, chain := range chains {
		if chain.dimension != dimension {
			return 0.0, errors.New("all chains must have same dimension")
		}
	}

	// For simplicity, calculate R-hat for first dimension only
	var chainMeans []float64
	var chainVars []float64
	var n int // samples per chain

	for _, chain := range chains {
		samples := chain.GetSamples()
		if len(samples) == 0 {
			return 0.0, errors.New("chain has no samples after burn-in")
		}

		if n == 0 {
			n = len(samples)
		} else if len(samples) != n {
			return 0.0, errors.New("all chains must have same length")
		}

		// Calculate mean and variance for this chain (first dimension)
		x := make([]float64, len(samples))
		for i, sample := range samples {
			x[i] = sample[0]
		}

		mean := calculateMean(x)
		variance := calculateVariance(x)

		chainMeans = append(chainMeans, mean)
		chainVars = append(chainVars, variance)
	}

	m := len(chains) // number of chains

	// Calculate between-chain and within-chain variances
	grandMean := 0.0
	for _, mean := range chainMeans {
		grandMean += mean
	}
	grandMean /= float64(m)

	// Between-chain variance
	B := 0.0
	for _, mean := range chainMeans {
		diff := mean - grandMean
		B += diff * diff
	}
	B = B * float64(n) / float64(m-1)

	// Within-chain variance
	W := 0.0
	for _, variance := range chainVars {
		W += variance
	}
	W /= float64(m)

	// Potential scale reduction factor
	varPlus := float64(n-1)/float64(n)*W + B/float64(n)
	rHat := math.Sqrt(varPlus / W)

	return rHat, nil
}

// CombineChains combines multiple MCMC chains into a single chain
func CombineChains(chains []*MCMCChain) *MCMCChain {
	if len(chains) == 0 {
		return nil
	}

	// Check dimension consistency
	dimension := chains[0].dimension
	for _, chain := range chains {
		if chain.dimension != dimension {
			return nil // Dimension mismatch
		}
	}

	var allSamples [][]float64
	var allLogPost []float64
	totalAccepted := 0
	totalTotal := 0

	for _, chain := range chains {
		samples := chain.GetSamples()
		allSamples = append(allSamples, samples...)

		// Add log posterior values (if available)
		for i := chain.burnIn; i < len(chain.logPosterior); i++ {
			allLogPost = append(allLogPost, chain.logPosterior[i])
		}

		totalAccepted += chain.accepted
		totalTotal += chain.total
	}

	return &MCMCChain{
		samples:      allSamples,
		logPosterior: allLogPost,
		accepted:     totalAccepted,
		total:        totalTotal,
		burnIn:       0, // Already burned in
		dimension:    dimension,
	}
}

// Helper functions

// generateMultivariateNormal generates a multivariate normal random vector
func generateMultivariateNormal(mean []float64, cov [][]float64, result []float64, rng *rand.Rand) {
	dimension := len(mean)

	// Generate independent standard normals
	z := make([]float64, dimension)
	for i := range z {
		z[i] = rng.NormFloat64()
	}

	// Simple implementation: assume diagonal covariance for now
	// For full covariance matrix, would need Cholesky decomposition
	for i := range result {
		result[i] = mean[i] + math.Sqrt(cov[i][i])*z[i]

		// Add off-diagonal correlation (simplified)
		for j := 0; j < i; j++ {
			if cov[i][j] != 0 {
				correlation := cov[i][j] / math.Sqrt(cov[i][i]*cov[j][j])
				result[i] += correlation * math.Sqrt(cov[i][i]) * z[j] * 0.1 // Simplified
			}
		}
	}
}

// calculateAutocorrelationTime estimates the autocorrelation time
func calculateAutocorrelationTime(x []float64) float64 {
	n := len(x)
	if n < 10 {
		return 1.0
	}

	// Calculate mean
	mean := calculateMean(x)

	// Calculate autocorrelation function
	maxLag := n / 4 // Don't go beyond n/4 lags
	autocorr := make([]float64, maxLag)

	// Lag 0 (variance)
	var0 := 0.0
	for _, val := range x {
		dev := val - mean
		var0 += dev * dev
	}
	var0 /= float64(n - 1)

	// Calculate autocorrelations
	for lag := 1; lag < maxLag && lag < n-1; lag++ {
		covar := 0.0
		count := 0

		for i := 0; i < n-lag; i++ {
			covar += (x[i] - mean) * (x[i+lag] - mean)
			count++
		}

		if count > 0 && var0 > 0 {
			autocorr[lag] = covar / (float64(count) * var0)
		}
	}

	// Find first lag where autocorrelation drops below 1/e ≈ 0.368
	threshold := 1.0 / math.E
	for lag := 1; lag < len(autocorr); lag++ {
		if math.Abs(autocorr[lag]) < threshold {
			return float64(lag)
		}
	}

	// If never drops below threshold, return conservative estimate
	return math.Min(float64(n)/4, 50.0)
}

// calculateMean calculates the mean of a slice
func calculateMean(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}

	sum := 0.0
	for _, val := range x {
		sum += val
	}
	return sum / float64(len(x))
}

// calculateVariance calculates the sample variance
func calculateVariance(x []float64) float64 {
	if len(x) < 2 {
		return 0
	}

	mean := calculateMean(x)
	sum := 0.0

	for _, val := range x {
		diff := val - mean
		sum += diff * diff
	}

	return sum / float64(len(x)-1)
}
