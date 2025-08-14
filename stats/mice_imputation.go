package stats

import (
	"errors"
	"math"
	"math/rand"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// MICEEstimator defines the regression estimator for MICE imputation
type MICEEstimator string

const (
	// MICELinearRegression uses linear regression for imputation
	MICELinearRegression MICEEstimator = "linear_regression"
	// MICERidgeRegression uses ridge regression for imputation
	MICERidgeRegression MICEEstimator = "ridge_regression"
	// MICEDecisionTree uses decision tree for imputation (future)
	MICEDecisionTree MICEEstimator = "decision_tree"
)

// MICEDiagnostics contains diagnostic information from MICE imputation
type MICEDiagnostics struct {
	Iterations         int       // Actual number of iterations performed
	ConvergenceHistory []float64 // Convergence metric over iterations
	Converged          bool      // Whether convergence was achieved
	EstimatorErrors    []string  // Any errors from estimator fitting
}

// MICEImputer performs Multiple Imputation by Chained Equations
// following the scikit-learn pattern with Fit/Transform methods
type MICEImputer struct {
	NImputations         int              // Number of multiple imputations to generate
	MaxIterations        int              // Maximum number of chained iterations
	ConvergenceTolerance float64          // Tolerance for convergence detection
	Estimator            MICEEstimator    // Regression estimator to use
	MissingValues        float64          // Value that represents missing data
	enableDiagnostics    bool             // Whether to collect diagnostic information
	fitted               bool             // Whether the imputer has been fitted
	data                 *array.Array     // Training data for imputation
	diagnostics          *MICEDiagnostics // Diagnostic information from last run
	actualIterations     int              // Actual iterations from last run
	converged            bool             // Whether last run converged
}

// NewMICEImputer creates a new MICE imputer with default settings
func NewMICEImputer(nImputations int) *MICEImputer {
	return &MICEImputer{
		NImputations:         nImputations,
		MaxIterations:        10,
		ConvergenceTolerance: 1e-3,
		Estimator:            MICELinearRegression,
		MissingValues:        math.NaN(),
		enableDiagnostics:    false,
		fitted:               false,
	}
}

// NewMICEImputerWithEstimator creates a MICE imputer with specified estimator
func NewMICEImputerWithEstimator(estimator string) *MICEImputer {
	var miceEstimator MICEEstimator
	switch estimator {
	case "linear_regression":
		miceEstimator = MICELinearRegression
	case "ridge_regression":
		miceEstimator = MICERidgeRegression
	case "decision_tree":
		miceEstimator = MICEDecisionTree
	default:
		// For invalid estimators, we still need to return something, but validation happens in Fit()
		miceEstimator = MICEEstimator(estimator) // Keep the invalid estimator for validation
	}

	return &MICEImputer{
		NImputations:         5, // Default number of imputations
		MaxIterations:        10,
		ConvergenceTolerance: 1e-3,
		Estimator:            miceEstimator,
		MissingValues:        math.NaN(),
		enableDiagnostics:    false,
		fitted:               false,
	}
}

// NewMICEImputerWithOptions creates a MICE imputer with custom options
func NewMICEImputerWithOptions(nImputations, maxIterations int, tolerance float64, estimator MICEEstimator, missingValues float64) *MICEImputer {
	return &MICEImputer{
		NImputations:         nImputations,
		MaxIterations:        maxIterations,
		ConvergenceTolerance: tolerance,
		Estimator:            estimator,
		MissingValues:        missingValues,
		enableDiagnostics:    false,
		fitted:               false,
	}
}

// Fit prepares the MICE imputer by storing the reference data
func (mice *MICEImputer) Fit(X *array.Array) error {
	if X == nil {
		return errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return errors.New("X must be a 2D array")
	}

	shape := X.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	if nSamples == 0 || nFeatures == 0 {
		return errors.New("X cannot be empty")
	}

	if mice.NImputations <= 0 {
		return errors.New("number of imputations must be positive")
	}

	if mice.MaxIterations <= 0 {
		return errors.New("max iterations must be positive")
	}

	if mice.ConvergenceTolerance < 0 {
		return errors.New("convergence tolerance must be non-negative")
	}

	// Validate estimator
	if mice.Estimator == MICEDecisionTree {
		return errors.New("decision tree estimator not yet implemented")
	}
	if mice.Estimator != MICELinearRegression && mice.Estimator != MICERidgeRegression && mice.Estimator != MICEDecisionTree {
		return errors.New("invalid estimator: " + string(mice.Estimator))
	}

	// Store reference data for imputation
	mice.data = array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(X.At(i, j))
			mice.data.Set(value, i, j)
		}
	}

	mice.fitted = true
	return nil
}

// Transform applies MICE imputation to generate multiple completed datasets
func (mice *MICEImputer) Transform(X *array.Array) ([]*array.Array, error) {
	if !mice.fitted {
		return nil, errors.New("MICEImputer must be fitted before transform")
	}

	if X == nil {
		return nil, errors.New("X cannot be nil")
	}

	if X.Ndim() != 2 {
		return nil, errors.New("X must be a 2D array")
	}

	shape := X.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	if nFeatures != mice.data.Shape()[1] {
		return nil, errors.New("X has different number of features than fitted data")
	}

	// Check if data has missing values
	hasMissing, err := HasMissingValues(X, mice.MissingValues)
	if err != nil {
		return nil, err
	}

	// Generate multiple imputed datasets
	datasets := make([]*array.Array, mice.NImputations)

	for i := 0; i < mice.NImputations; i++ {
		if !hasMissing {
			// If no missing values, return copy of original data
			dataset := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)
			for row := 0; row < nSamples; row++ {
				for col := 0; col < nFeatures; col++ {
					value := convertToFloat64(X.At(row, col))
					dataset.Set(value, row, col)
				}
			}
			datasets[i] = dataset
		} else {
			// Perform MICE imputation for this dataset
			dataset, err := mice.performMICEImputation(X)
			if err != nil {
				return nil, err
			}
			datasets[i] = dataset
		}
	}

	return datasets, nil
}

// FitTransform fits the MICE imputer and transforms the data in one step
func (mice *MICEImputer) FitTransform(X *array.Array) ([]*array.Array, error) {
	err := mice.Fit(X)
	if err != nil {
		return nil, err
	}
	return mice.Transform(X)
}

// performMICEImputation performs the actual MICE algorithm for one dataset
func (mice *MICEImputer) performMICEImputation(X *array.Array) (*array.Array, error) {
	shape := X.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	// Create working copy of data
	current := array.Empty(internal.Shape{nSamples, nFeatures}, internal.Float64)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(X.At(i, j))
			current.Set(value, i, j)
		}
	}

	// Initialize diagnostics if enabled
	var diagnostics *MICEDiagnostics
	if mice.enableDiagnostics {
		diagnostics = &MICEDiagnostics{
			Iterations:         0,
			ConvergenceHistory: make([]float64, 0),
			Converged:          false,
			EstimatorErrors:    make([]string, 0),
		}
	}

	// Find missing value patterns
	missingPattern := mice.findMissingPattern(current)

	// If no missing values, return original data
	if len(missingPattern.MissingFeatures) == 0 {
		mice.actualIterations = 0
		mice.converged = true
		if diagnostics != nil {
			diagnostics.Converged = true
			mice.diagnostics = diagnostics
		}
		return current, nil
	}

	// Special case: single feature - fall back to mean imputation
	if nFeatures == 1 {
		return mice.performSingleFeatureImputation(current)
	}

	// Initial imputation using simple methods
	err := mice.performInitialImputation(current, missingPattern)
	if err != nil {
		return nil, err
	}

	// Iterative chained equations
	previousImputations := make(map[int]*array.Array)
	for iter := 0; iter < mice.MaxIterations; iter++ {
		// Store previous state for convergence checking
		if iter > 0 {
			previousImputations[iter-1] = mice.copyArray(current)
		}

		// Perform one round of chained equations
		err := mice.performChainedIteration(current, missingPattern)
		if err != nil {
			if diagnostics != nil {
				diagnostics.EstimatorErrors = append(diagnostics.EstimatorErrors, err.Error())
			}
			// Continue with current imputation if estimator fails
		}

		// Check convergence
		if iter > 0 {
			converged, convergenceMetric := mice.checkConvergence(current, previousImputations[iter-1], missingPattern)

			if diagnostics != nil {
				diagnostics.ConvergenceHistory = append(diagnostics.ConvergenceHistory, convergenceMetric)
			}

			if converged {
				mice.actualIterations = iter + 1
				mice.converged = true
				if diagnostics != nil {
					diagnostics.Iterations = iter + 1
					diagnostics.Converged = true
					mice.diagnostics = diagnostics
				}
				break
			}
		}

		mice.actualIterations = iter + 1
		mice.converged = false
	}

	if diagnostics != nil && !mice.converged {
		diagnostics.Iterations = mice.MaxIterations
		diagnostics.Converged = false
		mice.diagnostics = diagnostics
	}

	return current, nil
}

// MissingPattern holds information about missing data patterns
type MissingPattern struct {
	MissingFeatures []int   // Features that have missing values
	MissingSamples  [][]int // For each feature, which samples are missing
	CompleteSamples []int   // Samples with no missing values
}

// findMissingPattern analyzes the missing data pattern
func (mice *MICEImputer) findMissingPattern(data *array.Array) *MissingPattern {
	shape := data.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	pattern := &MissingPattern{
		MissingFeatures: make([]int, 0),
		MissingSamples:  make([][]int, nFeatures),
		CompleteSamples: make([]int, 0),
	}

	// Find missing features and samples
	for j := 0; j < nFeatures; j++ {
		missingSamples := make([]int, 0)
		for i := 0; i < nSamples; i++ {
			value := convertToFloat64(data.At(i, j))
			if mice.isMissingValue(value) {
				missingSamples = append(missingSamples, i)
			}
		}

		if len(missingSamples) > 0 {
			pattern.MissingFeatures = append(pattern.MissingFeatures, j)
			pattern.MissingSamples[j] = missingSamples
		}
	}

	// Find complete samples
	for i := 0; i < nSamples; i++ {
		isComplete := true
		for j := 0; j < nFeatures; j++ {
			value := convertToFloat64(data.At(i, j))
			if mice.isMissingValue(value) {
				isComplete = false
				break
			}
		}
		if isComplete {
			pattern.CompleteSamples = append(pattern.CompleteSamples, i)
		}
	}

	return pattern
}

// performInitialImputation provides initial values using simple imputation
func (mice *MICEImputer) performInitialImputation(data *array.Array, pattern *MissingPattern) error {
	// Use mean imputation for initial values
	for _, featureIdx := range pattern.MissingFeatures {
		mean := mice.computeFeatureMean(data, featureIdx)
		for _, sampleIdx := range pattern.MissingSamples[featureIdx] {
			data.Set(mean, sampleIdx, featureIdx)
		}
	}
	return nil
}

// performChainedIteration performs one iteration of chained equations
func (mice *MICEImputer) performChainedIteration(data *array.Array, pattern *MissingPattern) error {
	// Iterate through each feature with missing values
	for _, targetFeature := range pattern.MissingFeatures {
		// Create predictor features (all other features)
		predictors := make([]int, 0, len(pattern.MissingFeatures)-1)
		for j := 0; j < data.Shape()[1]; j++ {
			if j != targetFeature {
				predictors = append(predictors, j)
			}
		}

		// Skip if no predictors available
		if len(predictors) == 0 {
			continue
		}

		// Fit regression model and impute missing values
		err := mice.fitAndImpute(data, targetFeature, predictors, pattern.MissingSamples[targetFeature])
		if err != nil {
			return err
		}
	}
	return nil
}

// fitAndImpute fits a regression model and imputes missing values for a feature
func (mice *MICEImputer) fitAndImpute(data *array.Array, targetFeature int, predictors []int, missingSamples []int) error {
	shape := data.Shape()
	nSamples := shape[0]

	// Collect complete cases for training
	trainingSamples := make([]int, 0)
	for i := 0; i < nSamples; i++ {
		// Skip samples that are in the missing list for target feature
		skip := false
		for _, missingIdx := range missingSamples {
			if i == missingIdx {
				skip = true
				break
			}
		}
		if skip {
			continue
		}

		// Check if all predictor features are available
		hasAllPredictors := true
		for _, predictorIdx := range predictors {
			value := convertToFloat64(data.At(i, predictorIdx))
			if mice.isMissingValue(value) {
				hasAllPredictors = false
				break
			}
		}

		if hasAllPredictors {
			trainingSamples = append(trainingSamples, i)
		}
	}

	// Need at least one training sample
	if len(trainingSamples) == 0 {
		return errors.New("no complete cases available for regression")
	}

	// Create training data
	X_train := array.Empty(internal.Shape{len(trainingSamples), len(predictors)}, internal.Float64)
	y_train := array.Empty(internal.Shape{len(trainingSamples)}, internal.Float64)

	for i, sampleIdx := range trainingSamples {
		for j, predictorIdx := range predictors {
			value := convertToFloat64(data.At(sampleIdx, predictorIdx))
			X_train.Set(value, i, j)
		}
		targetValue := convertToFloat64(data.At(sampleIdx, targetFeature))
		y_train.Set(targetValue, i)
	}

	// Fit regression model based on estimator type
	var predictions *array.Array
	var err error

	switch mice.Estimator {
	case MICELinearRegression:
		predictions, err = mice.fitLinearRegression(X_train, y_train, data, targetFeature, predictors, missingSamples)
	case MICERidgeRegression:
		predictions, err = mice.fitRidgeRegression(X_train, y_train, data, targetFeature, predictors, missingSamples)
	default:
		return errors.New("unsupported estimator")
	}

	if err != nil {
		return err
	}

	// Update missing values with predictions
	for i, sampleIdx := range missingSamples {
		if i < predictions.Size() {
			predictedValue := convertToFloat64(predictions.At(i))
			// Add small random noise to introduce variability
			noise := rand.NormFloat64() * 0.01 // Small Gaussian noise
			data.Set(predictedValue+noise, sampleIdx, targetFeature)
		}
	}

	return nil
}

// fitLinearRegression fits linear regression and makes predictions
func (mice *MICEImputer) fitLinearRegression(X_train, y_train, data *array.Array, targetFeature int, predictors []int, missingSamples []int) (*array.Array, error) {
	// Use existing linear regression implementation
	result, err := LinearRegression(y_train, X_train)
	if err != nil {
		return nil, err
	}

	// Create prediction data for missing samples
	X_pred := array.Empty(internal.Shape{len(missingSamples), len(predictors)}, internal.Float64)
	for i, sampleIdx := range missingSamples {
		for j, predictorIdx := range predictors {
			value := convertToFloat64(data.At(sampleIdx, predictorIdx))
			X_pred.Set(value, i, j)
		}
	}

	// Make predictions
	predictions := array.Empty(internal.Shape{len(missingSamples)}, internal.Float64)
	for i := 0; i < len(missingSamples); i++ {
		// Compute prediction: y = intercept + sum(coef * x)
		prediction := result.Intercept
		for j, coef := range result.Coefficients {
			x_val := convertToFloat64(X_pred.At(i, j))
			prediction += coef * x_val
		}
		predictions.Set(prediction, i)
	}

	return predictions, nil
}

// fitRidgeRegression fits ridge regression and makes predictions
func (mice *MICEImputer) fitRidgeRegression(X_train, y_train, data *array.Array, targetFeature int, predictors []int, missingSamples []int) (*array.Array, error) {
	// Use ridge regression with small regularization
	result, err := RidgeRegression(y_train, X_train, 0.1)
	if err != nil {
		// Fall back to linear regression if ridge fails
		return mice.fitLinearRegression(X_train, y_train, data, targetFeature, predictors, missingSamples)
	}

	// Create prediction data for missing samples
	X_pred := array.Empty(internal.Shape{len(missingSamples), len(predictors)}, internal.Float64)
	for i, sampleIdx := range missingSamples {
		for j, predictorIdx := range predictors {
			value := convertToFloat64(data.At(sampleIdx, predictorIdx))
			X_pred.Set(value, i, j)
		}
	}

	// Make predictions
	predictions := array.Empty(internal.Shape{len(missingSamples)}, internal.Float64)
	for i := 0; i < len(missingSamples); i++ {
		// Compute prediction: y = intercept + sum(coef * x)
		prediction := result.Intercept
		for j, coef := range result.Coefficients {
			x_val := convertToFloat64(X_pred.At(i, j))
			prediction += coef * x_val
		}
		predictions.Set(prediction, i)
	}

	return predictions, nil
}

// performSingleFeatureImputation handles the single feature case
func (mice *MICEImputer) performSingleFeatureImputation(data *array.Array) (*array.Array, error) {
	// Fall back to mean imputation for single feature
	mean := mice.computeFeatureMean(data, 0)

	nSamples := data.Shape()[0]
	for i := 0; i < nSamples; i++ {
		value := convertToFloat64(data.At(i, 0))
		if mice.isMissingValue(value) {
			data.Set(mean, i, 0)
		}
	}

	mice.actualIterations = 1
	mice.converged = true
	return data, nil
}

// checkConvergence checks if MICE has converged
func (mice *MICEImputer) checkConvergence(current, previous *array.Array, pattern *MissingPattern) (bool, float64) {
	// Compute mean absolute difference in imputed values
	totalDiff := 0.0
	count := 0

	for _, featureIdx := range pattern.MissingFeatures {
		for _, sampleIdx := range pattern.MissingSamples[featureIdx] {
			currentVal := convertToFloat64(current.At(sampleIdx, featureIdx))
			previousVal := convertToFloat64(previous.At(sampleIdx, featureIdx))

			diff := math.Abs(currentVal - previousVal)
			totalDiff += diff
			count++
		}
	}

	if count == 0 {
		return true, 0.0 // No missing values to check
	}

	meanDiff := totalDiff / float64(count)
	converged := meanDiff < mice.ConvergenceTolerance

	return converged, meanDiff
}

// computeFeatureMean computes the mean of non-missing values in a feature
func (mice *MICEImputer) computeFeatureMean(data *array.Array, featureIdx int) float64 {
	nSamples := data.Shape()[0]
	sum := 0.0
	count := 0

	for i := 0; i < nSamples; i++ {
		value := convertToFloat64(data.At(i, featureIdx))
		if !mice.isMissingValue(value) {
			sum += value
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	return sum / float64(count)
}

// copyArray creates a deep copy of an array
func (mice *MICEImputer) copyArray(source *array.Array) *array.Array {
	shape := source.Shape()
	copy := array.Empty(shape, internal.Float64)

	if len(shape) == 1 {
		for i := 0; i < shape[0]; i++ {
			value := convertToFloat64(source.At(i))
			copy.Set(value, i)
		}
	} else if len(shape) == 2 {
		for i := 0; i < shape[0]; i++ {
			for j := 0; j < shape[1]; j++ {
				value := convertToFloat64(source.At(i, j))
				copy.Set(value, i, j)
			}
		}
	}

	return copy
}

// isMissingValue checks if a value represents missing data
func (mice *MICEImputer) isMissingValue(value float64) bool {
	if math.IsNaN(mice.MissingValues) {
		return math.IsNaN(value)
	}
	return math.Abs(value-mice.MissingValues) < 1e-15
}

// Configuration methods
func (mice *MICEImputer) SetMaxIterations(maxIterations int) {
	mice.MaxIterations = maxIterations
}

func (mice *MICEImputer) SetConvergenceTolerance(tolerance float64) {
	mice.ConvergenceTolerance = tolerance
}

func (mice *MICEImputer) SetEnableDiagnostics(enable bool) {
	mice.enableDiagnostics = enable
}

// Getter methods
func (mice *MICEImputer) GetNImputations() int {
	return mice.NImputations
}

func (mice *MICEImputer) GetMaxIterations() int {
	return mice.MaxIterations
}

func (mice *MICEImputer) GetConvergenceTolerance() float64 {
	return mice.ConvergenceTolerance
}

func (mice *MICEImputer) GetEstimator() string {
	return string(mice.Estimator)
}

func (mice *MICEImputer) HasConverged() bool {
	return mice.converged
}

func (mice *MICEImputer) GetActualIterations() int {
	return mice.actualIterations
}

func (mice *MICEImputer) GetDiagnostics() *MICEDiagnostics {
	return mice.diagnostics
}
