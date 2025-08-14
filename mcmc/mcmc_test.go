package mcmc

import (
	"math"
	"testing"
)

// TestMetropolisHastings tests the Metropolis-Hastings MCMC algorithm
func TestMetropolisHastings(t *testing.T) {
	t.Run("Sampling from standard normal distribution", func(t *testing.T) {
		// Log-posterior for standard normal: -0.5 * x^2 (up to constant)
		logPosterior := func(x []float64) float64 {
			return -0.5 * x[0] * x[0]
		}

		initialState := []float64{0.0}
		proposalCov := [][]float64{{1.0}} // Proposal standard deviation = 1.0

		// Run MCMC chain
		options := &MCMCOptions{
			Samples:    10000,
			BurnIn:     2000,
			RandomSeed: 42,
		}

		chain, err := MetropolisHastings(logPosterior, initialState, proposalCov, options)
		if err != nil {
			t.Fatalf("Metropolis-Hastings failed: %v", err)
		}

		// Check chain structure
		if chain == nil {
			t.Fatal("MCMC chain should not be nil")
		}

		samples := chain.GetSamples()
		if len(samples) != options.Samples-options.BurnIn {
			t.Errorf("Expected %d samples after burn-in, got %d",
				options.Samples-options.BurnIn, len(samples))
		}

		// Check acceptance rate (should be reasonable, e.g., 20-70%)
		acceptanceRate := chain.AcceptanceRate()
		if acceptanceRate < 0.1 || acceptanceRate > 0.9 {
			t.Logf("Acceptance rate %.3f may be suboptimal (expect 0.2-0.7)", acceptanceRate)
		}

		// Check sample statistics (should approximate standard normal)
		mean := calculateSampleMean(samples)
		variance := calculateSampleVariance(samples, mean)

		// Standard normal should have mean ≈ 0, variance ≈ 1
		if math.Abs(mean[0]) > 0.1 {
			t.Errorf("Sample mean %.4f too far from 0 (expected ≈ 0)", mean[0])
		}

		if math.Abs(variance-1.0) > 0.2 {
			t.Errorf("Sample variance %.4f too far from 1.0", variance)
		}

		t.Logf("Standard normal MCMC: mean=%.4f, var=%.4f, accept=%.3f",
			mean[0], variance, acceptanceRate)
	})

	t.Run("Sampling from bivariate normal distribution", func(t *testing.T) {
		// Log-posterior for bivariate normal with correlation ρ=0.5
		rho := 0.5
		logPosterior := func(x []float64) float64 {
			x1, x2 := x[0], x[1]
			det := 1.0 - rho*rho
			quadForm := (x1*x1 - 2*rho*x1*x2 + x2*x2) / det
			return -0.5 * quadForm
		}

		initialState := []float64{0.0, 0.0}
		proposalCov := [][]float64{
			{0.8, 0.0},
			{0.0, 0.8},
		} // Diagonal proposal covariance

		options := &MCMCOptions{
			Samples:    15000,
			BurnIn:     3000,
			RandomSeed: 123,
		}

		chain, err := MetropolisHastings(logPosterior, initialState, proposalCov, options)
		if err != nil {
			t.Fatalf("Bivariate MCMC failed: %v", err)
		}

		samples := chain.GetSamples()

		// Check dimensions
		if len(samples) == 0 || len(samples[0]) != 2 {
			t.Fatal("Expected 2D samples")
		}

		// Calculate sample statistics
		mean := calculateSampleMean(samples)
		sampleCorr := calculateCorrelation(samples)

		// Both means should be close to 0
		for i := 0; i < 2; i++ {
			if math.Abs(mean[i]) > 0.1 {
				t.Errorf("Mean[%d] = %.4f, expected ≈ 0", i, mean[i])
			}
		}

		// Sample correlation should be close to true correlation (0.5)
		if math.Abs(sampleCorr-rho) > 0.1 {
			t.Errorf("Sample correlation %.4f, expected ≈ %.1f", sampleCorr, rho)
		}

		t.Logf("Bivariate normal MCMC: mean=[%.4f, %.4f], corr=%.4f",
			mean[0], mean[1], sampleCorr)
	})

	t.Run("MCMC with adaptive proposal covariance", func(t *testing.T) {
		// Target: standard normal but with adaptive tuning
		logPosterior := func(x []float64) float64 {
			return -0.5 * x[0] * x[0]
		}

		initialState := []float64{0.0}
		proposalCov := [][]float64{{0.1}} // Start with small proposal

		options := &MCMCOptions{
			Samples:          8000,
			BurnIn:           1000,
			AdaptiveProposal: true,
			AdaptationPeriod: 500,
			TargetAcceptRate: 0.44, // Optimal for 1D
		}

		chain, err := MetropolisHastings(logPosterior, initialState, proposalCov, options)
		if err != nil {
			t.Fatalf("Adaptive MCMC failed: %v", err)
		}

		// Check that adaptation improved acceptance rate
		finalAcceptRate := chain.AcceptanceRate()
		if math.Abs(finalAcceptRate-options.TargetAcceptRate) > 0.15 {
			t.Logf("Final acceptance rate %.3f, target was %.2f",
				finalAcceptRate, options.TargetAcceptRate)
		}

		// Check effective sample size
		ess := chain.EffectiveSampleSize()
		if ess < 1000 { // Should have reasonable effective sample size
			t.Logf("Effective sample size %.0f may be low", ess)
		}

		t.Logf("Adaptive MCMC: accept=%.3f, ESS=%.0f", finalAcceptRate, ess)
	})
}

// TestMCMCDiagnostics tests MCMC convergence diagnostics
func TestMCMCDiagnostics(t *testing.T) {
	t.Run("Convergence diagnostics for single chain", func(t *testing.T) {
		// Target: standard normal
		logPosterior := func(x []float64) float64 {
			return -0.5 * x[0] * x[0]
		}

		initialState := []float64{0.0}
		proposalCov := [][]float64{{1.0}}

		options := &MCMCOptions{
			Samples: 5000,
			BurnIn:  1000,
		}

		chain, err := MetropolisHastings(logPosterior, initialState, proposalCov, options)
		if err != nil {
			t.Fatalf("MCMC failed: %v", err)
		}

		// Run convergence diagnostics
		diagnostics := chain.DiagnosticTests()

		// Check diagnostic structure
		if diagnostics.EffectiveSampleSize <= 0 {
			t.Error("Effective sample size should be positive")
		}

		if diagnostics.MCSE <= 0 {
			t.Error("Monte Carlo standard error should be positive")
		}

		if len(diagnostics.AutocorrelationTimes) == 0 {
			t.Error("Autocorrelation times should not be empty")
		}

		// Check that ESS is reasonable
		if diagnostics.EffectiveSampleSize > float64(len(chain.GetSamples())) {
			t.Error("ESS cannot exceed actual number of samples")
		}

		t.Logf("MCMC diagnostics: ESS=%.0f, MCSE=%.6f, autocorr=%.2f",
			diagnostics.EffectiveSampleSize, diagnostics.MCSE,
			diagnostics.AutocorrelationTimes[0])
	})

	t.Run("Gelman-Rubin diagnostic with multiple chains", func(t *testing.T) {
		// Target: standard normal
		logPosterior := func(x []float64) float64 {
			return -0.5 * x[0] * x[0]
		}

		proposalCov := [][]float64{{1.0}}

		// Run multiple chains from different starting points
		var chains []*MCMCChain
		startingPoints := [][]float64{{-2.0}, {0.0}, {2.0}}

		for _, start := range startingPoints {
			options := &MCMCOptions{
				Samples:    3000,
				BurnIn:     500,
				RandomSeed: int64(len(chains) + 1), // Different seeds
			}

			chain, err := MetropolisHastings(logPosterior, start, proposalCov, options)
			if err != nil {
				t.Fatalf("Chain %d failed: %v", len(chains), err)
			}
			chains = append(chains, chain)
		}

		// Calculate Gelman-Rubin diagnostic
		rHat, err := GelmanRubinDiagnostic(chains)
		if err != nil {
			t.Fatalf("Gelman-Rubin diagnostic failed: %v", err)
		}

		// R-hat should be close to 1.0 for converged chains
		if rHat > 1.2 {
			t.Errorf("R-hat = %.4f > 1.2, chains may not be converged", rHat)
		}

		if rHat < 0.95 {
			t.Errorf("R-hat = %.4f < 0.95, suspiciously low", rHat)
		}

		t.Logf("Gelman-Rubin diagnostic: R-hat = %.4f", rHat)
	})
}

// TestMCMCOptions tests MCMC parameter validation and options
func TestMCMCOptions(t *testing.T) {
	t.Run("Parameter validation", func(t *testing.T) {
		logPosterior := func(x []float64) float64 { return -x[0] * x[0] }
		initialState := []float64{0.0}
		proposalCov := [][]float64{{1.0}}

		// Test invalid number of samples
		_, err := MetropolisHastings(logPosterior, initialState, proposalCov, &MCMCOptions{
			Samples: 0,
		})
		if err == nil {
			t.Error("Should fail with zero samples")
		}

		// Test nil log posterior
		_, err = MetropolisHastings(nil, initialState, proposalCov, &MCMCOptions{
			Samples: 1000,
		})
		if err == nil {
			t.Error("Should fail with nil log posterior")
		}

		// Test mismatched dimensions
		_, err = MetropolisHastings(logPosterior, []float64{0.0, 0.0}, proposalCov, &MCMCOptions{
			Samples: 1000,
		})
		if err == nil {
			t.Error("Should fail with mismatched dimensions")
		}

		// Test invalid proposal covariance
		_, err = MetropolisHastings(logPosterior, initialState, [][]float64{{-1.0}}, &MCMCOptions{
			Samples: 1000,
		})
		if err == nil {
			t.Error("Should fail with negative variance in proposal")
		}

		t.Logf("Parameter validation tests passed")
	})

	t.Run("Different sampling options", func(t *testing.T) {
		logPosterior := func(x []float64) float64 { return -0.5 * x[0] * x[0] }
		initialState := []float64{0.0}
		proposalCov := [][]float64{{1.0}}

		// Test different burn-in periods
		for _, burnIn := range []int{0, 500, 1000} {
			options := &MCMCOptions{
				Samples: 2000,
				BurnIn:  burnIn,
			}

			chain, err := MetropolisHastings(logPosterior, initialState, proposalCov, options)
			if err != nil {
				t.Errorf("Failed with burn-in %d: %v", burnIn, err)
				continue
			}

			expectedSamples := options.Samples - burnIn
			if len(chain.GetSamples()) != expectedSamples {
				t.Errorf("Burn-in %d: expected %d samples, got %d",
					burnIn, expectedSamples, len(chain.GetSamples()))
			}
		}

		t.Logf("Burn-in option tests passed")
	})
}

// TestMCMCChainOperations tests chain manipulation and analysis
func TestMCMCChainOperations(t *testing.T) {
	t.Run("Chain thinning and subsampling", func(t *testing.T) {
		logPosterior := func(x []float64) float64 { return -0.5 * x[0] * x[0] }
		initialState := []float64{0.0}
		proposalCov := [][]float64{{1.0}}

		options := &MCMCOptions{
			Samples: 4000,
			BurnIn:  1000,
		}

		chain, err := MetropolisHastings(logPosterior, initialState, proposalCov, options)
		if err != nil {
			t.Fatalf("MCMC failed: %v", err)
		}

		// Test thinning
		thinnedChain := chain.Thin(5) // Keep every 5th sample
		expectedThinned := len(chain.GetSamples()) / 5

		if len(thinnedChain.GetSamples()) != expectedThinned {
			t.Errorf("Thinning: expected ~%d samples, got %d",
				expectedThinned, len(thinnedChain.GetSamples()))
		}

		// Test chain combination
		chain2, _ := MetropolisHastings(logPosterior, []float64{1.0}, proposalCov, options)
		combined := CombineChains([]*MCMCChain{chain, chain2})

		expectedCombined := len(chain.GetSamples()) + len(chain2.GetSamples())
		if len(combined.GetSamples()) != expectedCombined {
			t.Errorf("Chain combination: expected %d samples, got %d",
				expectedCombined, len(combined.GetSamples()))
		}

		t.Logf("Chain operations successful: thinned=%d, combined=%d",
			len(thinnedChain.GetSamples()), len(combined.GetSamples()))
	})
}

// Helper functions for testing

func calculateSampleMean(samples [][]float64) []float64 {
	if len(samples) == 0 {
		return nil
	}

	dim := len(samples[0])
	mean := make([]float64, dim)

	for _, sample := range samples {
		for i := range mean {
			mean[i] += sample[i]
		}
	}

	for i := range mean {
		mean[i] /= float64(len(samples))
	}

	return mean
}

func calculateSampleVariance(samples [][]float64, mean []float64) float64 {
	if len(samples) == 0 || len(samples[0]) == 0 {
		return 0
	}

	variance := 0.0
	for _, sample := range samples {
		diff := sample[0] - mean[0]
		variance += diff * diff
	}

	return variance / float64(len(samples)-1)
}

func calculateCorrelation(samples [][]float64) float64 {
	if len(samples) == 0 || len(samples[0]) < 2 {
		return 0
	}

	mean := calculateSampleMean(samples)

	var sumXY, sumXX, sumYY float64
	for _, sample := range samples {
		dx := sample[0] - mean[0]
		dy := sample[1] - mean[1]
		sumXY += dx * dy
		sumXX += dx * dx
		sumYY += dy * dy
	}

	return sumXY / math.Sqrt(sumXX*sumYY)
}
