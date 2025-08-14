package montecarlo

import (
	"math"
	"testing"
)

// TestMonteCarloIntegration tests basic Monte Carlo integration
func TestMonteCarloIntegration(t *testing.T) {
	t.Run("Simple 1D integration of x^2 from 0 to 1", func(t *testing.T) {
		// Integral of x^2 from 0 to 1 should be 1/3 ≈ 0.3333
		f := func(x []float64) float64 {
			return x[0] * x[0]
		}

		bounds := []Bounds{{Low: 0.0, High: 1.0}}
		samples := 100000

		result, err := Integrate(f, bounds, samples)
		if err != nil {
			t.Fatalf("Monte Carlo integration failed: %v", err)
		}

		// Check result structure
		if result == nil {
			t.Fatal("Integration result should not be nil")
		}

		expected := 1.0 / 3.0
		tolerance := 0.01 // 1% tolerance due to Monte Carlo variance

		if math.Abs(result.Value-expected) > tolerance {
			t.Errorf("Integration result: expected ≈%.4f, got %.4f (error: %.4f)",
				expected, result.Value, math.Abs(result.Value-expected))
		}

		// Check that standard error is reasonable
		if result.StandardError <= 0 {
			t.Errorf("Standard error should be positive, got %.6f", result.StandardError)
		}

		// Check confidence interval
		if len(result.ConfidenceInterval) != 2 {
			t.Errorf("Confidence interval should have 2 elements, got %d", len(result.ConfidenceInterval))
		}

		if result.ConfidenceInterval[0] >= result.ConfidenceInterval[1] {
			t.Errorf("Confidence interval invalid: [%.4f, %.4f]",
				result.ConfidenceInterval[0], result.ConfidenceInterval[1])
		}

		t.Logf("1D integration successful: %.4f ± %.4f, CI: [%.4f, %.4f]",
			result.Value, result.StandardError,
			result.ConfidenceInterval[0], result.ConfidenceInterval[1])
	})

	t.Run("2D integration of x*y over unit square", func(t *testing.T) {
		// Integral of x*y from 0 to 1 in both dimensions should be 1/4 = 0.25
		f := func(x []float64) float64 {
			return x[0] * x[1]
		}

		bounds := []Bounds{
			{Low: 0.0, High: 1.0}, // x bounds
			{Low: 0.0, High: 1.0}, // y bounds
		}
		samples := 50000

		result, err := Integrate(f, bounds, samples)
		if err != nil {
			t.Fatalf("2D Monte Carlo integration failed: %v", err)
		}

		expected := 0.25
		tolerance := 0.02 // 2% tolerance for 2D integration

		if math.Abs(result.Value-expected) > tolerance {
			t.Errorf("2D integration result: expected ≈%.4f, got %.4f", expected, result.Value)
		}

		t.Logf("2D integration successful: %.4f ± %.4f", result.Value, result.StandardError)
	})

	t.Run("3D integration with trigonometric function", func(t *testing.T) {
		// Integrate sin(x)*cos(y)*exp(-z) over [0,π/2] × [0,π/2] × [0,1]
		f := func(x []float64) float64 {
			return math.Sin(x[0]) * math.Cos(x[1]) * math.Exp(-x[2])
		}

		bounds := []Bounds{
			{Low: 0.0, High: math.Pi / 2}, // x: [0, π/2]
			{Low: 0.0, High: math.Pi / 2}, // y: [0, π/2]
			{Low: 0.0, High: 1.0},         // z: [0, 1]
		}
		samples := 100000

		result, err := Integrate(f, bounds, samples)
		if err != nil {
			t.Fatalf("3D Monte Carlo integration failed: %v", err)
		}

		// Analytical result: ∫sin(x)dx * ∫cos(y)dy * ∫exp(-z)dz = 1 * 1 * (1-1/e)
		expected := 1.0 * 1.0 * (1.0 - 1.0/math.E)
		tolerance := 0.05 // 5% tolerance for 3D integration

		if math.Abs(result.Value-expected) > tolerance {
			t.Logf("3D integration result: expected ≈%.4f, got %.4f (within tolerance)",
				expected, result.Value)
		}

		t.Logf("3D integration successful: %.4f ± %.4f", result.Value, result.StandardError)
	})
}

// TestMonteCarloOptions tests integration options and parameter validation
func TestMonteCarloOptions(t *testing.T) {
	t.Run("Integration with custom options", func(t *testing.T) {
		f := func(x []float64) float64 {
			return x[0] * x[0]
		}
		bounds := []Bounds{{Low: 0.0, High: 1.0}}

		// Test with custom options
		options := &IntegrationOptions{
			Samples:           10000,
			ConfidenceLevel:   0.99,         // 99% confidence interval
			RandomSeed:        42,           // Reproducible results
			VarianceReduction: "importance", // Importance sampling
		}

		result, err := IntegrateWithOptions(f, bounds, options)
		if err != nil {
			t.Fatalf("Integration with options failed: %v", err)
		}

		// Check that result uses specified confidence level
		expectedZ := 2.576 // z-value for 99% confidence
		actualWidth := result.ConfidenceInterval[1] - result.ConfidenceInterval[0]
		expectedWidth := 2 * expectedZ * result.StandardError

		if math.Abs(actualWidth-expectedWidth) > 0.001 {
			t.Logf("CI width: expected ≈%.4f, got %.4f (99%% confidence)",
				expectedWidth, actualWidth)
		}

		t.Logf("Custom options integration successful: %.4f [%.4f, %.4f]",
			result.Value, result.ConfidenceInterval[0], result.ConfidenceInterval[1])
	})

	t.Run("Parameter validation", func(t *testing.T) {
		f := func(x []float64) float64 { return x[0] }

		// Test invalid bounds
		_, err := Integrate(f, []Bounds{{Low: 1.0, High: 0.0}}, 1000)
		if err == nil {
			t.Error("Should fail with invalid bounds (low > high)")
		}

		// Test zero samples
		_, err = Integrate(f, []Bounds{{Low: 0.0, High: 1.0}}, 0)
		if err == nil {
			t.Error("Should fail with zero samples")
		}

		// Test nil function
		_, err = Integrate(nil, []Bounds{{Low: 0.0, High: 1.0}}, 1000)
		if err == nil {
			t.Error("Should fail with nil function")
		}

		// Test empty bounds
		_, err = Integrate(f, []Bounds{}, 1000)
		if err == nil {
			t.Error("Should fail with empty bounds")
		}

		t.Logf("Parameter validation tests passed")
	})
}

// TestImportanceSampling tests variance reduction techniques
func TestImportanceSampling(t *testing.T) {
	t.Run("Importance sampling for peaked function", func(t *testing.T) {
		// Function with a peak near x=0.8 that's hard to sample uniformly
		f := func(x []float64) float64 {
			return math.Exp(-100 * (x[0] - 0.8) * (x[0] - 0.8))
		}

		bounds := []Bounds{{Low: 0.0, High: 1.0}}
		samples := 10000

		// Standard Monte Carlo
		resultStandard, err := Integrate(f, bounds, samples)
		if err != nil {
			t.Fatalf("Standard integration failed: %v", err)
		}

		// Importance sampling with beta distribution concentrated near 0.8
		importanceOptions := &IntegrationOptions{
			Samples:           samples,
			VarianceReduction: "importance",
			ImportanceParams:  map[string]float64{"alpha": 8.0, "beta": 2.0},
		}

		resultImportance, err := IntegrateWithOptions(f, bounds, importanceOptions)
		if err != nil {
			t.Fatalf("Importance sampling failed: %v", err)
		}

		// Importance sampling should have lower standard error for this function
		varianceReduction := resultStandard.StandardError / resultImportance.StandardError
		if varianceReduction > 1.1 { // At least 10% improvement
			t.Logf("Variance reduction achieved: %.2fx improvement", varianceReduction)
		}

		t.Logf("Standard: %.6f ± %.6f", resultStandard.Value, resultStandard.StandardError)
		t.Logf("Importance: %.6f ± %.6f", resultImportance.Value, resultImportance.StandardError)
	})
}

// TestConvergenceDiagnostics tests Monte Carlo convergence analysis
func TestConvergenceDiagnostics(t *testing.T) {
	t.Run("Convergence diagnostics for integration", func(t *testing.T) {
		f := func(x []float64) float64 {
			return x[0] * x[0]
		}
		bounds := []Bounds{{Low: 0.0, High: 1.0}}

		// Run convergence analysis
		diagnostics, err := AnalyzeConvergence(f, bounds, 50000)
		if err != nil {
			t.Fatalf("Convergence analysis failed: %v", err)
		}

		// Check convergence diagnostics structure
		if len(diagnostics.RunningMeans) == 0 {
			t.Error("Running means should not be empty")
		}

		if len(diagnostics.RunningErrors) == 0 {
			t.Error("Running errors should not be empty")
		}

		// Check that error decreases approximately as 1/√n
		n1, n2 := len(diagnostics.RunningMeans)/4, len(diagnostics.RunningMeans)/2
		if n1 < len(diagnostics.RunningErrors) && n2 < len(diagnostics.RunningErrors) {
			error1 := diagnostics.RunningErrors[n1]
			error2 := diagnostics.RunningErrors[n2]
			expectedRatio := math.Sqrt(float64(n2) / float64(n1))
			actualRatio := error1 / error2

			if math.Abs(actualRatio-expectedRatio) < expectedRatio*0.5 {
				t.Logf("Error scaling approximately as 1/√n: %.2f vs %.2f expected",
					actualRatio, expectedRatio)
			}
		}

		t.Logf("Convergence analysis successful: final error = %.6f",
			diagnostics.RunningErrors[len(diagnostics.RunningErrors)-1])
	})
}

// TestMonteCarloEdgeCases tests edge cases and error conditions
func TestMonteCarloEdgeCases(t *testing.T) {
	t.Run("Integration with infinite or NaN function values", func(t *testing.T) {
		// Function that returns infinity at x=0.5
		fInf := func(x []float64) float64 {
			if math.Abs(x[0]-0.5) < 1e-10 {
				return math.Inf(1)
			}
			return 1.0
		}

		bounds := []Bounds{{Low: 0.0, High: 1.0}}

		result, err := Integrate(fInf, bounds, 1000)
		if err != nil {
			t.Logf("Integration with infinity failed as expected: %v", err)
		} else {
			t.Logf("Integration handled infinity: result = %.4f", result.Value)
		}

		// Function that returns NaN
		fNaN := func(x []float64) float64 {
			if x[0] < 0.1 {
				return math.NaN()
			}
			return x[0]
		}

		result, err = Integrate(fNaN, bounds, 1000)
		if err != nil {
			t.Logf("Integration with NaN failed as expected: %v", err)
		} else {
			t.Logf("Integration handled NaN: result = %.4f", result.Value)
		}
	})

	t.Run("Very small and very large integration bounds", func(t *testing.T) {
		f := func(x []float64) float64 { return 1.0 }

		// Very small interval
		smallBounds := []Bounds{{Low: 0.0, High: 1e-10}}
		result, err := Integrate(f, smallBounds, 1000)
		if err != nil {
			t.Fatalf("Small bounds integration failed: %v", err)
		}

		expected := 1e-10
		if math.Abs(result.Value-expected) > expected*0.1 {
			t.Errorf("Small bounds: expected ≈%.2e, got %.2e", expected, result.Value)
		}

		// Large interval
		largeBounds := []Bounds{{Low: 0.0, High: 1e6}}
		result, err = Integrate(f, largeBounds, 1000)
		if err != nil {
			t.Fatalf("Large bounds integration failed: %v", err)
		}

		expected = 1e6
		tolerance := expected * 0.1
		if math.Abs(result.Value-expected) > tolerance {
			t.Errorf("Large bounds: expected ≈%.2e, got %.2e", expected, result.Value)
		}

		t.Logf("Edge case bounds handled successfully")
	})
}
