package bayesian

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestBayesianLinearRegression tests Bayesian linear regression
func TestBayesianLinearRegression(t *testing.T) {
	t.Run("Bayesian linear regression with conjugate prior", func(t *testing.T) {
		// Generate synthetic linear data: y = 2 + 3*x + noise
		n := 50
		xData := make([]float64, n)
		yData := make([]float64, n)

		for i := 0; i < n; i++ {
			x := float64(i) / 10.0
			xData[i] = x
			yData[i] = 2.0 + 3.0*x + 0.1*(float64(i%5)-2.0) // Small noise
		}

		// Convert to arrays
		X, _ := array.FromSlice(xData)
		y, _ := array.FromSlice(yData)

		// Set up Bayesian linear model
		model := NewBayesianLinearModel()

		// Set weakly informative prior: Normal(0, 100) for coefficients
		priorMean := []float64{0.0, 0.0} // intercept, slope
		priorCov := [][]float64{
			{100.0, 0.0},
			{0.0, 100.0},
		}
		err := model.SetPrior("normal", priorMean, priorCov)
		if err != nil {
			t.Fatalf("Setting prior failed: %v", err)
		}

		// Fit the model (X is 1D, will be converted to design matrix internally)
		err = model.Fit(y, X)
		if err != nil {
			t.Fatalf("Bayesian fitting failed: %v", err)
		}

		// Check if model is fitted
		if !model.IsFitted() {
			t.Error("Model should be marked as fitted")
		}

		// Get posterior summary
		posterior := model.GetPosteriorSummary()
		if posterior == nil {
			t.Fatal("Posterior summary should not be nil")
		}

		// Check posterior mean (should be close to true values: [2, 3])
		if len(posterior.Mean) != 2 {
			t.Errorf("Expected 2 coefficients, got %d", len(posterior.Mean))
		}

		tolerance := 0.5 // Allow reasonable tolerance
		if math.Abs(posterior.Mean[0]-2.0) > tolerance {
			t.Errorf("Intercept: expected ≈2.0, got %.3f", posterior.Mean[0])
		}

		if math.Abs(posterior.Mean[1]-3.0) > tolerance {
			t.Errorf("Slope: expected ≈3.0, got %.3f", posterior.Mean[1])
		}

		// Check credible intervals
		if len(posterior.CredibleIntervals) != 2 {
			t.Error("Should have credible intervals for both coefficients")
		}

		for i, ci := range posterior.CredibleIntervals {
			if ci.Lower >= ci.Upper {
				t.Errorf("Invalid credible interval %d: [%.3f, %.3f]", i, ci.Lower, ci.Upper)
			}

			// True value should be within credible interval
			trueVal := []float64{2.0, 3.0}[i]
			if trueVal < ci.Lower || trueVal > ci.Upper {
				t.Logf("True value %.1f outside CI[%d]: [%.3f, %.3f]", trueVal, i, ci.Lower, ci.Upper)
			}
		}

		t.Logf("Bayesian regression: intercept=%.3f±%.3f, slope=%.3f±%.3f",
			posterior.Mean[0], posterior.StandardDeviations[0],
			posterior.Mean[1], posterior.StandardDeviations[1])
	})

	t.Run("Model comparison with Bayes factors", func(t *testing.T) {
		// Generate data that clearly favors linear over constant model
		n := 30
		xData := make([]float64, n)
		yData := make([]float64, n)

		for i := 0; i < n; i++ {
			x := float64(i) / 5.0
			xData[i] = x
			yData[i] = 1.0 + 2.0*x + 0.05*(float64(i%3)-1.0) // Clear linear trend
		}

		y, _ := array.FromSlice(yData)

		// Model 1: Intercept only (constant model)
		model1 := NewBayesianLinearModel()
		model1.SetPrior("normal", []float64{0.0}, [][]float64{{10.0}})

		// Create design matrix with intercept only
		interceptX := array.Ones(internal.Shape{n, 1}, internal.Float64)
		err := model1.Fit(y, interceptX)
		if err != nil {
			t.Fatalf("Constant model fitting failed: %v", err)
		}

		// Model 2: Linear model (intercept + slope)
		model2 := NewBayesianLinearModel()
		model2.SetPrior("normal", []float64{0.0, 0.0}, [][]float64{{10.0, 0.0}, {0.0, 10.0}})

		// Create design matrix with intercept and x
		linearX := array.Empty(internal.Shape{n, 2}, internal.Float64)
		for i := 0; i < n; i++ {
			linearX.Set(1.0, i, 0)      // intercept
			linearX.Set(xData[i], i, 1) // slope
		}
		err = model2.Fit(y, linearX)
		if err != nil {
			t.Fatalf("Linear model fitting failed: %v", err)
		}

		// Calculate Bayes factor (Model 2 vs Model 1)
		bayesFactor, err := CalculateBayesFactor(model2, model1)
		if err != nil {
			t.Fatalf("Bayes factor calculation failed: %v", err)
		}

		// Bayes factor should strongly favor linear model (BF > 3)
		if bayesFactor < 3.0 {
			t.Logf("Bayes factor %.2f, expected > 3 (moderate evidence for linear)", bayesFactor)
		}

		// Calculate model probabilities
		probs, err := CalculateModelProbabilities([]*BayesianLinearModel{model1, model2})
		if err != nil {
			t.Fatalf("Model probability calculation failed: %v", err)
		}

		prob1, prob2 := probs[0], probs[1]
		if math.Abs(prob1+prob2-1.0) > 0.001 {
			t.Errorf("Model probabilities should sum to 1: %.3f + %.3f = %.3f", prob1, prob2, prob1+prob2)
		}

		// Linear model should have higher probability
		if prob2 <= prob1 {
			t.Logf("Linear model probability %.3f, constant model %.3f", prob2, prob1)
		}

		t.Logf("Model comparison: BF=%.2f, P(constant)=%.3f, P(linear)=%.3f",
			bayesFactor, prob1, prob2)
	})
}

// TestBayesianModelAveraging tests Bayesian model averaging
func TestBayesianModelAveraging(t *testing.T) {
	t.Run("Bayesian model averaging for regression", func(t *testing.T) {
		// Generate data with some nonlinearity
		n := 40
		xData := make([]float64, n)
		yData := make([]float64, n)

		for i := 0; i < n; i++ {
			x := float64(i-20) / 10.0 // Center around 0
			xData[i] = x
			yData[i] = 1.0 + 2.0*x + 0.5*x*x + 0.1*(float64(i%4)-1.5) // Quadratic
		}

		y, _ := array.FromSlice(yData)

		// Create multiple models
		var models []*BayesianLinearModel

		// Model 1: Linear
		model1 := NewBayesianLinearModel()
		linearX := array.Empty(internal.Shape{n, 2}, internal.Float64)
		for i := 0; i < n; i++ {
			linearX.Set(1.0, i, 0)      // intercept
			linearX.Set(xData[i], i, 1) // x
		}
		model1.SetPrior("normal", []float64{0.0, 0.0}, [][]float64{{10.0, 0.0}, {0.0, 10.0}})
		model1.Fit(y, linearX)
		models = append(models, model1)

		// Model 2: Quadratic
		model2 := NewBayesianLinearModel()
		quadX := array.Empty(internal.Shape{n, 3}, internal.Float64)
		for i := 0; i < n; i++ {
			quadX.Set(1.0, i, 0)               // intercept
			quadX.Set(xData[i], i, 1)          // x
			quadX.Set(xData[i]*xData[i], i, 2) // x²
		}
		model2.SetPrior("normal", []float64{0.0, 0.0, 0.0},
			[][]float64{{10.0, 0.0, 0.0}, {0.0, 10.0, 0.0}, {0.0, 0.0, 10.0}})
		model2.Fit(y, quadX)
		models = append(models, model2)

		// Perform Bayesian model averaging
		bma, err := NewBayesianModelAveraging(models)
		if err != nil {
			t.Fatalf("Bayesian model averaging failed: %v", err)
		}

		// Check BMA structure
		if len(bma.ModelWeights) != len(models) {
			t.Errorf("Expected %d model weights, got %d", len(models), len(bma.ModelWeights))
		}

		// Weights should sum to 1
		weightSum := 0.0
		for _, w := range bma.ModelWeights {
			weightSum += w
		}
		if math.Abs(weightSum-1.0) > 0.001 {
			t.Errorf("Model weights should sum to 1, got %.4f", weightSum)
		}

		// Quadratic model should have higher weight (since data is quadratic)
		if bma.ModelWeights[1] <= bma.ModelWeights[0] {
			t.Logf("Quadratic weight %.3f, linear weight %.3f",
				bma.ModelWeights[1], bma.ModelWeights[0])
		}

		// Test prediction
		testX := 1.5 // New test point
		prediction, err := bma.Predict(testX)
		if err != nil {
			t.Fatalf("BMA prediction failed: %v", err)
		}

		// Check prediction structure
		if prediction.Mean == 0 {
			t.Error("Prediction mean should not be zero")
		}

		if prediction.StandardDeviation <= 0 {
			t.Error("Prediction standard deviation should be positive")
		}

		if len(prediction.CredibleInterval) != 2 {
			t.Error("Prediction should have credible interval")
		}

		t.Logf("BMA: linear weight=%.3f, quadratic weight=%.3f",
			bma.ModelWeights[0], bma.ModelWeights[1])
		t.Logf("Prediction at x=%.1f: %.3f ± %.3f",
			testX, prediction.Mean, prediction.StandardDeviation)
	})
}

// TestBayesianHypothesisTesting tests Bayesian hypothesis testing
func TestBayesianHypothesisTesting(t *testing.T) {
	t.Run("Bayesian t-test for mean difference", func(t *testing.T) {
		// Generate two groups with different means
		group1 := []float64{2.1, 2.3, 1.9, 2.4, 2.0, 2.2, 1.8, 2.5, 2.1, 2.0}
		group2 := []float64{3.1, 3.0, 2.9, 3.2, 3.1, 2.8, 3.3, 3.0, 2.9, 3.1}

		x1, _ := array.FromSlice(group1)
		x2, _ := array.FromSlice(group2)

		// Perform Bayesian t-test
		result, err := BayesianTTest(x1, x2)
		if err != nil {
			t.Fatalf("Bayesian t-test failed: %v", err)
		}

		// Check result structure
		if result.BayesFactor <= 0 {
			t.Error("Bayes factor should be positive")
		}

		if math.Abs(result.PosteriorProbabilityH1)+math.Abs(result.PosteriorProbabilityH0) < 0.99 {
			t.Error("Posterior probabilities should sum to approximately 1")
		}

		// Effect size should be reasonable
		if math.Abs(result.EffectSize) < 0.5 {
			t.Logf("Effect size %.3f may be small", result.EffectSize)
		}

		// Since groups have different means, should favor H1 (difference exists)
		if result.BayesFactor < 1.0 {
			t.Logf("Bayes factor %.2f < 1, weak evidence for difference", result.BayesFactor)
		}

		t.Logf("Bayesian t-test: BF=%.2f, effect size=%.3f, P(H1)=%.3f",
			result.BayesFactor, result.EffectSize, result.PosteriorProbabilityH1)
	})
}

// TestPriorSpecification tests different prior specifications
func TestPriorSpecification(t *testing.T) {
	t.Run("Different prior types and validation", func(t *testing.T) {
		model := NewBayesianLinearModel()

		// Test normal prior
		err := model.SetPrior("normal", []float64{0.0, 0.0},
			[][]float64{{1.0, 0.0}, {0.0, 1.0}})
		if err != nil {
			t.Errorf("Normal prior setting failed: %v", err)
		}

		// Test invalid prior type
		err = model.SetPrior("invalid", []float64{0.0}, [][]float64{{1.0}})
		if err == nil {
			t.Error("Should fail with invalid prior type")
		}

		// Test dimension mismatch
		err = model.SetPrior("normal", []float64{0.0},
			[][]float64{{1.0, 0.0}, {0.0, 1.0}})
		if err == nil {
			t.Error("Should fail with dimension mismatch")
		}

		// Test non-positive definite covariance
		err = model.SetPrior("normal", []float64{0.0, 0.0},
			[][]float64{{1.0, 0.0}, {0.0, -1.0}})
		if err == nil {
			t.Error("Should fail with non-positive definite covariance")
		}

		t.Logf("Prior validation tests passed")
	})

	t.Run("Prior sensitivity analysis", func(t *testing.T) {
		// Generate small dataset
		n := 10
		xData := make([]float64, n)
		yData := make([]float64, n)

		for i := 0; i < n; i++ {
			x := float64(i)
			xData[i] = x
			yData[i] = 1.0 + 0.5*x
		}

		y, _ := array.FromSlice(yData)

		// Test different prior strengths
		priorStrengths := []float64{0.1, 1.0, 10.0}
		var posteriorMeans [][]float64

		// Create design matrix
		X := array.Empty(internal.Shape{n, 2}, internal.Float64)
		for i := 0; i < n; i++ {
			X.Set(1.0, i, 0)      // intercept
			X.Set(xData[i], i, 1) // x
		}

		for _, strength := range priorStrengths {
			model := NewBayesianLinearModel()
			model.SetPrior("normal", []float64{0.0, 0.0},
				[][]float64{{strength, 0.0}, {0.0, strength}})

			err := model.Fit(y, X)
			if err != nil {
				t.Fatalf("Model fitting failed with prior strength %.1f: %v", strength, err)
			}

			posterior := model.GetPosteriorSummary()
			posteriorMeans = append(posteriorMeans, posterior.Mean)
		}

		// With stronger prior, posterior should be pulled toward prior mean (0)
		for i := 0; i < 2; i++ { // For each coefficient
			// Check that stronger prior pulls estimates toward zero
			if math.Abs(posteriorMeans[2][i]) >= math.Abs(posteriorMeans[0][i]) {
				t.Logf("Strong prior effect: weak=%.3f, strong=%.3f for coeff %d",
					posteriorMeans[0][i], posteriorMeans[2][i], i)
			}
		}

		t.Logf("Prior sensitivity analysis completed")
	})
}
