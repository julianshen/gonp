package bayesian

import (
	"errors"
	"math"

	"github.com/julianshen/gonp/array"
)

// BayesianLinearModel represents a Bayesian linear regression model
type BayesianLinearModel struct {
	priorMean     []float64   // Prior mean for coefficients
	priorCov      [][]float64 // Prior covariance matrix
	posteriorMean []float64   // Posterior mean
	posteriorCov  [][]float64 // Posterior covariance matrix
	fitted        bool        // Whether model has been fitted
	logEvidence   float64     // Log marginal likelihood
	nFeatures     int         // Number of features
	nObservations int         // Number of observations
	priorType     string      // Type of prior ("normal")
}

// PosteriorSummary contains posterior distribution summary statistics
type PosteriorSummary struct {
	Mean               []float64          // Posterior mean
	StandardDeviations []float64          // Posterior standard deviations
	CredibleIntervals  []CredibleInterval // Credible intervals for each coefficient
}

// CredibleInterval represents a credible interval
type CredibleInterval struct {
	Lower float64 // Lower bound
	Upper float64 // Upper bound
	Level float64 // Confidence level (e.g., 0.95)
}

// BayesianModelAveraging represents BMA results
type BayesianModelAveraging struct {
	Models       []*BayesianLinearModel // Models in the ensemble
	ModelWeights []float64              // Model weights (probabilities)
	LogEvidence  []float64              // Log evidence for each model
}

// BMAPredict represents a BMA prediction
type BMAPredict struct {
	Mean              float64    // Prediction mean
	StandardDeviation float64    // Prediction standard deviation
	CredibleInterval  [2]float64 // Prediction credible interval
}

// BayesianTTestResult represents results from Bayesian t-test
type BayesianTTestResult struct {
	BayesFactor            float64    // Bayes factor (H1 vs H0)
	PosteriorProbabilityH0 float64    // Posterior probability of H0 (no difference)
	PosteriorProbabilityH1 float64    // Posterior probability of H1 (difference exists)
	EffectSize             float64    // Standardized effect size
	CredibleInterval       [2]float64 // 95% credible interval for difference
}

// NewBayesianLinearModel creates a new Bayesian linear regression model
func NewBayesianLinearModel() *BayesianLinearModel {
	return &BayesianLinearModel{
		fitted:    false,
		priorType: "normal",
	}
}

// SetPrior sets the prior distribution for the model coefficients
func (m *BayesianLinearModel) SetPrior(priorType string, mean []float64, cov [][]float64) error {
	if priorType != "normal" {
		return errors.New("only 'normal' prior type is currently supported")
	}

	if len(mean) == 0 {
		return errors.New("prior mean cannot be empty")
	}

	if len(cov) != len(mean) {
		return errors.New("prior covariance matrix dimension must match mean dimension")
	}

	for i, row := range cov {
		if len(row) != len(mean) {
			return errors.New("prior covariance matrix must be square")
		}
		if cov[i][i] <= 0 {
			return errors.New("prior covariance diagonal elements must be positive")
		}
	}

	m.priorType = priorType
	m.priorMean = make([]float64, len(mean))
	copy(m.priorMean, mean)

	m.priorCov = make([][]float64, len(cov))
	for i, row := range cov {
		m.priorCov[i] = make([]float64, len(row))
		copy(m.priorCov[i], row)
	}

	m.nFeatures = len(mean)

	return nil
}

// Fit fits the Bayesian linear model to data
func (m *BayesianLinearModel) Fit(y, X *array.Array) error {
	if y == nil || X == nil {
		return errors.New("y and X cannot be nil")
	}

	// Get dimensions
	n := y.Size()
	if n == 0 {
		return errors.New("y cannot be empty")
	}

	// Handle both 1D and 2D X arrays
	var p int
	var XData [][]float64

	if X.Ndim() == 1 {
		// 1D X array - convert to 2D design matrix with intercept
		p = 2
		XData = make([][]float64, n)
		for i := 0; i < n; i++ {
			val, _ := X.At(i).(float64)
			XData[i] = []float64{1.0, val} // intercept, then X value
		}
	} else if X.Ndim() == 2 {
		// 2D X array - use directly
		shape := X.Shape()
		p = shape[1]
		XData = make([][]float64, n)
		for i := 0; i < n; i++ {
			XData[i] = make([]float64, p)
			for j := 0; j < p; j++ {
				val, _ := X.At(i, j).(float64)
				XData[i][j] = val
			}
		}
	} else {
		return errors.New("X must be 1D or 2D array")
	}

	// Check if prior is set and matches feature dimension
	if m.priorMean == nil || len(m.priorMean) != p {
		// Set default weak prior if not specified
		m.priorMean = make([]float64, p)
		m.priorCov = make([][]float64, p)
		for i := 0; i < p; i++ {
			m.priorCov[i] = make([]float64, p)
			for j := 0; j < p; j++ {
				if i == j {
					m.priorCov[i][j] = 100.0 // Weak prior
				}
			}
		}
		m.nFeatures = p
	}

	// Extract y data
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		val, _ := y.At(i).(float64)
		yData[i] = val
	}

	// Compute X'X and X'y
	XtX := make([][]float64, p)
	for i := 0; i < p; i++ {
		XtX[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += XData[k][i] * XData[k][j]
			}
			XtX[i][j] = sum
		}
	}

	Xty := make([]float64, p)
	for i := 0; i < p; i++ {
		sum := 0.0
		for k := 0; k < n; k++ {
			sum += XData[k][i] * yData[k]
		}
		Xty[i] = sum
	}

	// Compute prior precision matrix (inverse of covariance)
	priorPrecision, err := m.invertMatrix(m.priorCov)
	if err != nil {
		return errors.New("prior covariance matrix is singular")
	}

	// Compute posterior precision: prior_precision + X'X
	posteriorPrecision := make([][]float64, p)
	for i := 0; i < p; i++ {
		posteriorPrecision[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			posteriorPrecision[i][j] = priorPrecision[i][j] + XtX[i][j]
		}
	}

	// Compute posterior covariance (inverse of precision)
	posteriorCov, err := m.invertMatrix(posteriorPrecision)
	if err != nil {
		return errors.New("posterior precision matrix is singular")
	}

	// Compute posterior mean
	// posterior_mean = posterior_cov * (prior_precision * prior_mean + X'y)
	temp := make([]float64, p)
	for i := 0; i < p; i++ {
		sum := 0.0
		for j := 0; j < p; j++ {
			sum += priorPrecision[i][j] * m.priorMean[j]
		}
		temp[i] = sum + Xty[i]
	}

	posteriorMean := make([]float64, p)
	for i := 0; i < p; i++ {
		sum := 0.0
		for j := 0; j < p; j++ {
			sum += posteriorCov[i][j] * temp[j]
		}
		posteriorMean[i] = sum
	}

	// Calculate log evidence (marginal likelihood)
	logEvidence := m.calculateLogEvidence(yData, XData, posteriorMean, posteriorCov)

	// Store results
	m.posteriorMean = posteriorMean
	m.posteriorCov = posteriorCov
	m.logEvidence = logEvidence
	m.nObservations = n
	m.fitted = true

	return nil
}

// IsFitted returns whether the model has been fitted
func (m *BayesianLinearModel) IsFitted() bool {
	return m.fitted
}

// GetPosteriorSummary returns posterior distribution summary
func (m *BayesianLinearModel) GetPosteriorSummary() *PosteriorSummary {
	if !m.fitted {
		return nil
	}

	stdDeviations := make([]float64, len(m.posteriorMean))
	credibleIntervals := make([]CredibleInterval, len(m.posteriorMean))

	for i := range m.posteriorMean {
		stdDeviations[i] = math.Sqrt(m.posteriorCov[i][i])

		// 95% credible interval
		margin := 1.96 * stdDeviations[i]
		credibleIntervals[i] = CredibleInterval{
			Lower: m.posteriorMean[i] - margin,
			Upper: m.posteriorMean[i] + margin,
			Level: 0.95,
		}
	}

	return &PosteriorSummary{
		Mean:               m.posteriorMean,
		StandardDeviations: stdDeviations,
		CredibleIntervals:  credibleIntervals,
	}
}

// CalculateBayesFactor calculates Bayes factor between two models
func CalculateBayesFactor(model1, model2 *BayesianLinearModel) (float64, error) {
	if !model1.fitted || !model2.fitted {
		return 0.0, errors.New("both models must be fitted")
	}

	// Bayes factor = exp(log_evidence_1 - log_evidence_2)
	logBF := model1.logEvidence - model2.logEvidence

	return math.Exp(logBF), nil
}

// CalculateModelProbabilities calculates model probabilities from evidence
func CalculateModelProbabilities(models []*BayesianLinearModel) ([]float64, error) {
	if len(models) == 0 {
		return nil, errors.New("models cannot be empty")
	}

	// Check all models are fitted
	logEvidences := make([]float64, len(models))
	for i, model := range models {
		if !model.fitted {
			return nil, errors.New("all models must be fitted")
		}
		logEvidences[i] = model.logEvidence
	}

	// Find maximum log evidence for numerical stability
	maxLogEvidence := logEvidences[0]
	for _, logEv := range logEvidences {
		if logEv > maxLogEvidence {
			maxLogEvidence = logEv
		}
	}

	// Calculate unnormalized probabilities
	unnormalized := make([]float64, len(models))
	sum := 0.0
	for i, logEv := range logEvidences {
		unnormalized[i] = math.Exp(logEv - maxLogEvidence)
		sum += unnormalized[i]
	}

	// Normalize
	probabilities := make([]float64, len(models))
	for i := range probabilities {
		probabilities[i] = unnormalized[i] / sum
	}

	return probabilities, nil
}

// NewBayesianModelAveraging performs Bayesian model averaging
func NewBayesianModelAveraging(models []*BayesianLinearModel) (*BayesianModelAveraging, error) {
	if len(models) == 0 {
		return nil, errors.New("models cannot be empty")
	}

	// Calculate model probabilities
	modelWeights, err := CalculateModelProbabilities(models)
	if err != nil {
		return nil, err
	}

	// Extract log evidences
	logEvidence := make([]float64, len(models))
	for i, model := range models {
		logEvidence[i] = model.logEvidence
	}

	return &BayesianModelAveraging{
		Models:       models,
		ModelWeights: modelWeights,
		LogEvidence:  logEvidence,
	}, nil
}

// Predict makes a prediction using Bayesian model averaging
func (bma *BayesianModelAveraging) Predict(x float64) (*BMAPredict, error) {
	if len(bma.Models) == 0 {
		return nil, errors.New("no models available")
	}

	// Make predictions with each model
	predictions := make([]float64, len(bma.Models))
	variances := make([]float64, len(bma.Models))

	for i, model := range bma.Models {
		if !model.fitted {
			return nil, errors.New("all models must be fitted")
		}

		// Predict with this model
		// For linear model: y = β₀ + β₁x (for models with 2 coefficients)
		// For quadratic: y = β₀ + β₁x + β₂x² (for models with 3 coefficients)
		var pred float64
		if len(model.posteriorMean) == 2 {
			// Linear model
			pred = model.posteriorMean[0] + model.posteriorMean[1]*x
		} else if len(model.posteriorMean) == 3 {
			// Quadratic model
			pred = model.posteriorMean[0] + model.posteriorMean[1]*x + model.posteriorMean[2]*x*x
		} else {
			return nil, errors.New("unsupported model dimension for prediction")
		}

		predictions[i] = pred

		// Calculate prediction variance
		// For simplicity, use diagonal of posterior covariance scaled by design vector
		xVec := []float64{1.0, x}
		if len(model.posteriorMean) == 3 {
			xVec = append(xVec, x*x)
		}

		variance := 0.0
		for j := 0; j < len(xVec); j++ {
			for k := 0; k < len(xVec); k++ {
				variance += xVec[j] * model.posteriorCov[j][k] * xVec[k]
			}
		}
		variances[i] = variance
	}

	// Weighted average prediction
	weightedMean := 0.0
	for i := range predictions {
		weightedMean += bma.ModelWeights[i] * predictions[i]
	}

	// Calculate weighted variance
	weightedVariance := 0.0
	for i := range predictions {
		// Variance includes both model uncertainty and prediction uncertainty
		modelVariance := variances[i]
		predictionVariance := (predictions[i] - weightedMean) * (predictions[i] - weightedMean)
		weightedVariance += bma.ModelWeights[i] * (modelVariance + predictionVariance)
	}

	stdDev := math.Sqrt(weightedVariance)
	margin := 1.96 * stdDev

	return &BMAPredict{
		Mean:              weightedMean,
		StandardDeviation: stdDev,
		CredibleInterval:  [2]float64{weightedMean - margin, weightedMean + margin},
	}, nil
}

// BayesianTTest performs Bayesian t-test for comparing two groups
func BayesianTTest(group1, group2 *array.Array) (*BayesianTTestResult, error) {
	if group1 == nil || group2 == nil {
		return nil, errors.New("groups cannot be nil")
	}

	n1, n2 := group1.Size(), group2.Size()
	if n1 == 0 || n2 == 0 {
		return nil, errors.New("groups cannot be empty")
	}

	// Extract data
	data1 := make([]float64, n1)
	data2 := make([]float64, n2)

	for i := 0; i < n1; i++ {
		val, _ := group1.At(i).(float64)
		data1[i] = val
	}

	for i := 0; i < n2; i++ {
		val, _ := group2.At(i).(float64)
		data2[i] = val
	}

	// Calculate sample statistics
	mean1 := calculateMean(data1)
	mean2 := calculateMean(data2)
	var1 := calculateVariance(data1)
	var2 := calculateVariance(data2)

	// Pooled variance for effect size
	pooledVar := ((float64(n1-1))*var1 + (float64(n2-1))*var2) / float64(n1+n2-2)
	effectSize := (mean1 - mean2) / math.Sqrt(pooledVar)

	// Bayesian t-test using default prior
	// For simplicity, use approximation based on classical t-test
	// In practice, would use more sophisticated Bayesian model

	// Standard error for difference
	se := math.Sqrt(var1/float64(n1) + var2/float64(n2))

	// T-statistic
	t := (mean1 - mean2) / se

	// Approximate Bayes factor using BIC approximation
	// BF ≈ exp((t²-log(n))/2) where n is effective sample size
	nEff := float64(n1 + n2)
	logBF := (t*t - math.Log(nEff)) / 2.0
	bayesFactor := math.Exp(logBF)

	// Calculate posterior probabilities
	// P(H1|data) = BF / (1 + BF), P(H0|data) = 1 / (1 + BF)
	posteriorH1 := bayesFactor / (1.0 + bayesFactor)
	posteriorH0 := 1.0 / (1.0 + bayesFactor)

	// Credible interval for difference
	diff := mean1 - mean2
	margin := 1.96 * se

	return &BayesianTTestResult{
		BayesFactor:            bayesFactor,
		PosteriorProbabilityH0: posteriorH0,
		PosteriorProbabilityH1: posteriorH1,
		EffectSize:             effectSize,
		CredibleInterval:       [2]float64{diff - margin, diff + margin},
	}, nil
}

// Helper functions

// invertMatrix inverts a square matrix using Gaussian elimination
func (m *BayesianLinearModel) invertMatrix(matrix [][]float64) ([][]float64, error) {
	n := len(matrix)
	if n == 0 {
		return nil, errors.New("matrix cannot be empty")
	}

	// Create augmented matrix [A | I]
	augmented := make([][]float64, n)
	for i := 0; i < n; i++ {
		augmented[i] = make([]float64, 2*n)
		for j := 0; j < n; j++ {
			augmented[i][j] = matrix[i][j]
		}
		augmented[i][i+n] = 1.0 // Identity matrix
	}

	// Gaussian elimination
	for i := 0; i < n; i++ {
		// Find pivot
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(augmented[k][i]) > math.Abs(augmented[maxRow][i]) {
				maxRow = k
			}
		}

		// Swap rows
		if maxRow != i {
			augmented[i], augmented[maxRow] = augmented[maxRow], augmented[i]
		}

		// Check for singularity
		if math.Abs(augmented[i][i]) < 1e-12 {
			return nil, errors.New("matrix is singular")
		}

		// Scale pivot row
		pivot := augmented[i][i]
		for j := 0; j < 2*n; j++ {
			augmented[i][j] /= pivot
		}

		// Eliminate column
		for k := 0; k < n; k++ {
			if k != i {
				factor := augmented[k][i]
				for j := 0; j < 2*n; j++ {
					augmented[k][j] -= factor * augmented[i][j]
				}
			}
		}
	}

	// Extract inverse matrix
	inverse := make([][]float64, n)
	for i := 0; i < n; i++ {
		inverse[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			inverse[i][j] = augmented[i][j+n]
		}
	}

	return inverse, nil
}

// calculateLogEvidence calculates log marginal likelihood
func (m *BayesianLinearModel) calculateLogEvidence(y []float64, X [][]float64, posteriorMean []float64, posteriorCov [][]float64) float64 {
	// Simplified calculation - in practice would be more sophisticated
	n := len(y)
	p := len(posteriorMean)

	// Residual sum of squares
	rss := 0.0
	for i := 0; i < n; i++ {
		pred := 0.0
		for j := 0; j < p; j++ {
			pred += X[i][j] * posteriorMean[j]
		}
		residual := y[i] - pred
		rss += residual * residual
	}

	// Approximate log evidence using BIC-like formula
	logEvidence := -0.5 * (float64(n)*math.Log(2*math.Pi) + float64(n)*math.Log(rss/float64(n)) + float64(p)*math.Log(float64(n)))

	return logEvidence
}

// calculateMean calculates the mean of a slice
func calculateMean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}

	sum := 0.0
	for _, val := range data {
		sum += val
	}
	return sum / float64(len(data))
}

// calculateVariance calculates the sample variance
func calculateVariance(data []float64) float64 {
	if len(data) < 2 {
		return 0
	}

	mean := calculateMean(data)
	sumSquares := 0.0

	for _, val := range data {
		diff := val - mean
		sumSquares += diff * diff
	}

	return sumSquares / float64(len(data)-1)
}
