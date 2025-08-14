package stats

import (
	"fmt"
	gmath "math"
	"strings"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/math"
)

// RegressionResult contains the results of linear regression analysis
type RegressionResult struct {
	Intercept        float64   // Intercept (constant term)
	Coefficients     []float64 // Regression coefficients
	RSquared         float64   // R-squared (coefficient of determination)
	AdjustedRSquared float64   // Adjusted R-squared
	StandardErrors   []float64 // Standard errors of coefficients
	TStats           []float64 // T-statistics for coefficients
	PValues          []float64 // P-values for coefficients
	MSE              float64   // Mean squared error
	RMSE             float64   // Root mean squared error
	DegreesOfFreedom int       // Degrees of freedom
	FStatistic       float64   // F-statistic for overall significance
	FPValue          float64   // P-value for F-statistic
}

// LinearRegression performs ordinary least squares linear regression
// y: dependent variable (1D array)
// X: independent variables (1D for simple, 2D for multiple regression)
// Returns RegressionResult with fitted model and statistics
func LinearRegression(y, X *array.Array) (*RegressionResult, error) {
	if y == nil || X == nil {
		return nil, fmt.Errorf("arrays cannot be nil")
	}

	// Handle dimensions
	var designMatrix *array.Array
	var err error

	if X.Ndim() == 1 {
		// Simple linear regression: add intercept column
		n := X.Size()
		if n != y.Size() {
			return nil, fmt.Errorf("X and y must have same number of observations: %d vs %d", n, y.Size())
		}

		// Create design matrix [1, X]
		designData := make([]float64, n*2)
		for i := 0; i < n; i++ {
			designData[i*2] = 1.0                 // Intercept column
			designData[i*2+1] = X.At(i).(float64) // X values
		}
		designMatrix, err = array.NewArrayWithShape(designData, internal.Shape{n, 2})
		if err != nil {
			return nil, fmt.Errorf("failed to create design matrix: %v", err)
		}
	} else if X.Ndim() == 2 {
		// Multiple regression: add intercept column if not present
		shape := X.Shape()
		n, p := shape[0], shape[1]

		if n != y.Size() {
			return nil, fmt.Errorf("X and y must have same number of observations: %d vs %d", n, y.Size())
		}

		// Create design matrix with intercept column [1, X]
		designData := make([]float64, n*(p+1))
		for i := 0; i < n; i++ {
			designData[i*(p+1)] = 1.0 // Intercept column
			for j := 0; j < p; j++ {
				designData[i*(p+1)+j+1] = X.At(i, j).(float64)
			}
		}
		designMatrix, err = array.NewArrayWithShape(designData, internal.Shape{n, p + 1})
		if err != nil {
			return nil, fmt.Errorf("failed to create design matrix: %v", err)
		}
	} else {
		return nil, fmt.Errorf("X must be 1D or 2D array, got %dD", X.Ndim())
	}

	if designMatrix.Size() == 0 {
		return nil, fmt.Errorf("empty design matrix")
	}

	// Solve normal equation: (X'X)^(-1) X'y
	beta, err := solveNormalEquation(designMatrix, y)
	if err != nil {
		return nil, fmt.Errorf("failed to solve normal equation: %v", err)
	}

	// Calculate fitted values and residuals
	yHat, err := predictValues(designMatrix, beta)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate fitted values: %v", err)
	}

	residuals, err := calculateResiduals(y, yHat)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate residuals: %v", err)
	}

	// Calculate regression statistics
	result := &RegressionResult{
		Intercept:    beta.At(0).(float64),
		Coefficients: make([]float64, beta.Size()-1),
	}

	// Extract coefficients (excluding intercept)
	for i := 1; i < beta.Size(); i++ {
		result.Coefficients[i-1] = beta.At(i).(float64)
	}

	// Calculate R-squared and other statistics
	err = calculateRegressionStatistics(result, y, yHat, residuals, designMatrix)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate statistics: %v", err)
	}

	return result, nil
}

// Predict generates predictions for new data using the fitted model
func (r *RegressionResult) Predict(X *array.Array) (*array.Array, error) {
	if X == nil {
		return nil, fmt.Errorf("X cannot be nil")
	}

	var designMatrix *array.Array
	var err error

	if X.Ndim() == 1 {
		// Simple regression: add intercept
		n := X.Size()
		designData := make([]float64, n*2)
		for i := 0; i < n; i++ {
			designData[i*2] = 1.0
			designData[i*2+1] = X.At(i).(float64)
		}
		designMatrix, err = array.NewArrayWithShape(designData, internal.Shape{n, 2})
	} else if X.Ndim() == 2 {
		// Multiple regression: add intercept
		shape := X.Shape()
		n, p := shape[0], shape[1]
		designData := make([]float64, n*(p+1))
		for i := 0; i < n; i++ {
			designData[i*(p+1)] = 1.0
			for j := 0; j < p; j++ {
				designData[i*(p+1)+j+1] = X.At(i, j).(float64)
			}
		}
		designMatrix, err = array.NewArrayWithShape(designData, internal.Shape{n, p + 1})
	} else {
		return nil, fmt.Errorf("X must be 1D or 2D array")
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create design matrix: %v", err)
	}

	// Create coefficient vector
	betaData := make([]float64, len(r.Coefficients)+1)
	betaData[0] = r.Intercept
	copy(betaData[1:], r.Coefficients)
	beta, err := array.FromSlice(betaData)
	if err != nil {
		return nil, fmt.Errorf("failed to create coefficient array: %v", err)
	}

	return predictValues(designMatrix, beta)
}

// Residuals calculates residuals for given X and y data
func (r *RegressionResult) Residuals(X, y *array.Array) (*array.Array, error) {
	predictions, err := r.Predict(X)
	if err != nil {
		return nil, fmt.Errorf("failed to generate predictions: %v", err)
	}

	return calculateResiduals(y, predictions)
}

// Helper functions

// solveNormalEquation solves (X'X)^(-1) X'y for coefficients
func solveNormalEquation(X, y *array.Array) (*array.Array, error) {
	// X' (transpose)
	XTranspose, err := math.Transpose(X)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose X: %v", err)
	}

	// X'X
	XTX, err := math.Dot(XTranspose, X)
	if err != nil {
		return nil, fmt.Errorf("failed to compute X'X: %v", err)
	}

	// X'y
	XTy, err := math.Dot(XTranspose, y.Reshape(internal.Shape{y.Size(), 1}))
	if err != nil {
		return nil, fmt.Errorf("failed to compute X'y: %v", err)
	}

	// Solve (X'X)^(-1) X'y
	beta, err := math.Solve(XTX, XTy)
	if err != nil {
		return nil, fmt.Errorf("failed to solve linear system: %v", err)
	}

	// Flatten to 1D
	return beta.Reshape(internal.Shape{beta.Size()}), nil
}

// predictValues calculates X * beta
func predictValues(X, beta *array.Array) (*array.Array, error) {
	betaMatrix := beta.Reshape(internal.Shape{beta.Size(), 1})
	predictions, err := math.Dot(X, betaMatrix)
	if err != nil {
		return nil, err
	}

	return predictions.Reshape(internal.Shape{predictions.Size()}), nil
}

// calculateResiduals computes y - yHat
func calculateResiduals(y, yHat *array.Array) (*array.Array, error) {
	if y.Size() != yHat.Size() {
		return nil, fmt.Errorf("size mismatch: y=%d, yHat=%d", y.Size(), yHat.Size())
	}

	residuals := array.Empty(internal.Shape{y.Size()}, internal.Float64)
	for i := 0; i < y.Size(); i++ {
		res := y.At(i).(float64) - yHat.At(i).(float64)
		residuals.Set(res, i)
	}

	return residuals, nil
}

// calculateRegressionStatistics computes all regression statistics
func calculateRegressionStatistics(result *RegressionResult, y, yHat, residuals, X *array.Array) error {
	n := y.Size()
	p := X.Shape()[1] - 1 // Number of predictors (excluding intercept)

	// Degrees of freedom
	result.DegreesOfFreedom = n - p - 1

	// Mean Squared Error
	rss := 0.0 // Residual sum of squares
	for i := 0; i < residuals.Size(); i++ {
		res := residuals.At(i).(float64)
		rss += res * res
	}
	result.MSE = rss / float64(result.DegreesOfFreedom)
	result.RMSE = gmath.Sqrt(result.MSE)

	// R-squared
	var err error
	result.RSquared, err = RSquared(y, yHat)
	if err != nil {
		return fmt.Errorf("failed to calculate R-squared: %v", err)
	}

	// Adjusted R-squared
	result.AdjustedRSquared = 1 - (1-result.RSquared)*float64(n-1)/float64(result.DegreesOfFreedom)

	// Calculate standard errors and t-statistics
	err = calculateInferenceStatistics(result, X, residuals)
	if err != nil {
		return fmt.Errorf("failed to calculate inference statistics: %v", err)
	}

	// F-statistic for overall significance
	mss := result.RSquared * getTotalSumSquares(y) / float64(p) // Mean square regression
	result.FStatistic = mss / result.MSE
	result.FPValue = fDistributionPValue(result.FStatistic, p, result.DegreesOfFreedom)

	return nil
}

// calculateRidgeRegressionStatistics computes statistics for ridge regression
func calculateRidgeRegressionStatistics(result *RegressionResult, y, yHat, residuals, X *array.Array, alpha float64) error {
	n := y.Size()
	p := X.Shape()[1] - 1 // Number of predictors (excluding intercept)

	// Degrees of freedom
	result.DegreesOfFreedom = n - p - 1

	// Mean Squared Error
	rss := 0.0 // Residual sum of squares
	for i := 0; i < residuals.Size(); i++ {
		res := residuals.At(i).(float64)
		rss += res * res
	}
	result.MSE = rss / float64(result.DegreesOfFreedom)
	result.RMSE = gmath.Sqrt(result.MSE)

	// R-squared
	var err error
	result.RSquared, err = RSquared(y, yHat)
	if err != nil {
		return fmt.Errorf("failed to calculate R-squared: %v", err)
	}

	// Adjusted R-squared
	result.AdjustedRSquared = 1 - (1-result.RSquared)*float64(n-1)/float64(result.DegreesOfFreedom)

	// Calculate standard errors using the regularized covariance matrix
	err = calculateRidgeInferenceStatistics(result, X, alpha)
	if err != nil {
		return fmt.Errorf("failed to calculate ridge inference statistics: %v", err)
	}

	// F-statistic for overall significance
	mss := result.RSquared * getTotalSumSquares(y) / float64(p) // Mean square regression
	result.FStatistic = mss / result.MSE
	result.FPValue = fDistributionPValue(result.FStatistic, p, result.DegreesOfFreedom)

	return nil
}

// calculateRidgeInferenceStatistics computes standard errors for ridge regression using regularized covariance matrix
func calculateRidgeInferenceStatistics(result *RegressionResult, X *array.Array, alpha float64) error {
	// Calculate (X'X + αI)^(-1) for standard errors
	XTranspose, err := math.Transpose(X)
	if err != nil {
		return err
	}

	XTX, err := math.Dot(XTranspose, X)
	if err != nil {
		return err
	}

	// Add regularization: X'X + αI
	shape := XTX.Shape()
	n := shape[0]
	for i := 0; i < n; i++ {
		currentVal := XTX.At(i, i).(float64)
		if i == 0 && alpha > 0 {
			// Don't regularize intercept term
			continue
		} else {
			XTX.Set(currentVal+alpha, i, i)
		}
	}

	// Create identity matrix for solving (X'X + αI)^(-1)
	identityData := make([]float64, n*n)
	for i := 0; i < n; i++ {
		identityData[i*n+i] = 1.0
	}
	identity, err := array.NewArrayWithShape(identityData, internal.Shape{n, n})
	if err != nil {
		return err
	}

	// Solve (X'X + αI)^(-1)
	XTXInv, err := math.Solve(XTX, identity)
	if err != nil {
		// If still singular, set standard errors to NaN
		numCoeffs := len(result.Coefficients)
		result.StandardErrors = make([]float64, numCoeffs)
		result.TStats = make([]float64, numCoeffs)
		result.PValues = make([]float64, numCoeffs)

		for i := 0; i < numCoeffs; i++ {
			result.StandardErrors[i] = gmath.NaN()
			result.TStats[i] = gmath.NaN()
			result.PValues[i] = gmath.NaN()
		}
		return nil
	}

	// Standard errors are sqrt(diag((X'X + αI)^(-1) * MSE))
	numCoeffs := len(result.Coefficients) + 1            // Including intercept
	result.StandardErrors = make([]float64, numCoeffs-1) // Excluding intercept
	result.TStats = make([]float64, numCoeffs-1)
	result.PValues = make([]float64, numCoeffs-1)

	for i := 1; i < numCoeffs; i++ { // Skip intercept
		diagonal := XTXInv.At(i, i).(float64)

		// Edge case: Check for negative diagonal elements
		if diagonal <= 0 {
			result.StandardErrors[i-1] = gmath.NaN()
			result.TStats[i-1] = gmath.NaN()
			result.PValues[i-1] = gmath.NaN()
			continue
		}

		variance := diagonal * result.MSE

		// Edge case: Check for negative variance
		if variance < 0 {
			result.StandardErrors[i-1] = gmath.NaN()
			result.TStats[i-1] = gmath.NaN()
			result.PValues[i-1] = gmath.NaN()
			continue
		}

		se := gmath.Sqrt(variance)
		result.StandardErrors[i-1] = se

		// T-statistic
		if se > 0 {
			result.TStats[i-1] = result.Coefficients[i-1] / se

			// P-value (two-tailed t-test)
			tStatAbs := gmath.Abs(result.TStats[i-1])
			if gmath.IsInf(tStatAbs, 0) || gmath.IsNaN(tStatAbs) {
				result.PValues[i-1] = gmath.NaN()
			} else if tStatAbs > 100 {
				result.PValues[i-1] = 0.0 // Extremely significant
			} else {
				result.PValues[i-1] = tDistributionPValue(tStatAbs, result.DegreesOfFreedom)
			}
		} else {
			result.TStats[i-1] = gmath.NaN()
			result.PValues[i-1] = gmath.NaN()
		}
	}

	return nil
}

// calculateInferenceStatistics computes standard errors, t-stats, and p-values with edge case handling
func calculateInferenceStatistics(result *RegressionResult, X, residuals *array.Array) error {
	// Calculate (X'X)^(-1) for standard errors using solve with identity matrix
	XTranspose, err := math.Transpose(X)
	if err != nil {
		return err
	}

	XTX, err := math.Dot(XTranspose, X)
	if err != nil {
		return err
	}

	// Check for numerical stability - calculate condition number
	conditionNumber, err := calculateConditionNumber(XTX)
	if err == nil && conditionNumber > 1e12 {
		return fmt.Errorf("design matrix is ill-conditioned (condition number: %.2e), results may be unreliable", conditionNumber)
	}

	// Create identity matrix for solving (X'X)^(-1)
	shape := XTX.Shape()
	n := shape[0]
	identityData := make([]float64, n*n)
	for i := 0; i < n; i++ {
		identityData[i*n+i] = 1.0
	}
	identity, err := array.NewArrayWithShape(identityData, internal.Shape{n, n})
	if err != nil {
		return err
	}

	// Attempt to solve X'X * Inv = I to get (X'X)^(-1)
	XTXInv, err := math.Solve(XTX, identity)
	if err != nil {
		// Handle singular matrix case with regularization
		return handleSingularMatrix(result, XTX, identity)
	}

	// Standard errors are sqrt(diag((X'X)^(-1) * MSE))
	numCoeffs := len(result.Coefficients) + 1            // Including intercept
	result.StandardErrors = make([]float64, numCoeffs-1) // Excluding intercept
	result.TStats = make([]float64, numCoeffs-1)
	result.PValues = make([]float64, numCoeffs-1)

	for i := 1; i < numCoeffs; i++ { // Skip intercept
		diagonal := XTXInv.At(i, i).(float64)

		// Edge case: Check for negative or zero diagonal elements
		if diagonal <= 0 {
			result.StandardErrors[i-1] = gmath.NaN()
			result.TStats[i-1] = gmath.NaN()
			result.PValues[i-1] = gmath.NaN()
			continue
		}

		variance := diagonal * result.MSE

		// Edge case: Check for zero or very small MSE (perfect fit)
		if result.MSE <= 0 || result.MSE < 1e-29 {
			// For perfect fit, standard errors are effectively zero
			result.StandardErrors[i-1] = 0.0
			// T-statistic is undefined for perfect fit (coefficient / 0)
			if gmath.Abs(result.Coefficients[i-1]) < 1e-15 {
				result.TStats[i-1] = 0.0
			} else {
				result.TStats[i-1] = gmath.Inf(1) // Infinite t-stat for non-zero coefficient
			}
			result.PValues[i-1] = 0.0 // Perfect significance
			continue
		}

		// Edge case: Check for negative variance (should not happen mathematically)
		if variance < 0 {
			result.StandardErrors[i-1] = gmath.NaN()
			result.TStats[i-1] = gmath.NaN()
			result.PValues[i-1] = gmath.NaN()
			continue
		}

		se := gmath.Sqrt(variance)

		// Edge case: Check for extremely small standard errors or NaN from tiny variance
		if gmath.IsNaN(se) || se < 1e-14 || variance < 1e-25 {
			// For very small MSE, treat as perfect fit
			result.StandardErrors[i-1] = 0.0
			if gmath.Abs(result.Coefficients[i-1]) < 1e-15 {
				result.TStats[i-1] = 0.0
			} else {
				result.TStats[i-1] = gmath.Inf(1)
			}
			result.PValues[i-1] = 0.0
		} else {
			result.StandardErrors[i-1] = se
			// T-statistic with safe division
			result.TStats[i-1] = result.Coefficients[i-1] / se

			// P-value (two-tailed t-test) with bounds checking
			tStatAbs := gmath.Abs(result.TStats[i-1])
			if gmath.IsInf(tStatAbs, 0) || gmath.IsNaN(tStatAbs) {
				result.PValues[i-1] = gmath.NaN()
			} else if tStatAbs > 100 {
				result.PValues[i-1] = 0.0 // Extremely significant
			} else {
				result.PValues[i-1] = tDistributionPValue(tStatAbs, result.DegreesOfFreedom)
			}
		}
	}

	return nil
}

// Regression metrics functions

// RSquared calculates the coefficient of determination
func RSquared(yActual, yPredicted *array.Array) (float64, error) {
	if yActual.Size() != yPredicted.Size() {
		return 0, fmt.Errorf("arrays must have same size")
	}

	// Calculate mean of actual values
	yMean, err := Mean(yActual)
	if err != nil {
		return 0, err
	}

	// Calculate total sum of squares and residual sum of squares
	tss := 0.0 // Total sum of squares
	rss := 0.0 // Residual sum of squares

	for i := 0; i < yActual.Size(); i++ {
		actual := yActual.At(i).(float64)
		predicted := yPredicted.At(i).(float64)

		tss += (actual - yMean) * (actual - yMean)
		rss += (actual - predicted) * (actual - predicted)
	}

	if tss == 0 {
		return 1.0, nil // Perfect fit when no variance in y
	}

	return 1 - (rss / tss), nil
}

// MeanSquaredError calculates MSE between actual and predicted values
func MeanSquaredError(yActual, yPredicted *array.Array) (float64, error) {
	if yActual.Size() != yPredicted.Size() {
		return 0, fmt.Errorf("arrays must have same size")
	}

	sumSquaredErrors := 0.0
	n := yActual.Size()

	for i := 0; i < n; i++ {
		actual := yActual.At(i).(float64)
		predicted := yPredicted.At(i).(float64)
		error := actual - predicted
		sumSquaredErrors += error * error
	}

	return sumSquaredErrors / float64(n), nil
}

// MeanAbsoluteError calculates MAE between actual and predicted values
func MeanAbsoluteError(yActual, yPredicted *array.Array) (float64, error) {
	if yActual.Size() != yPredicted.Size() {
		return 0, fmt.Errorf("arrays must have same size")
	}

	sumAbsoluteErrors := 0.0
	n := yActual.Size()

	for i := 0; i < n; i++ {
		actual := yActual.At(i).(float64)
		predicted := yPredicted.At(i).(float64)
		error := gmath.Abs(actual - predicted)
		sumAbsoluteErrors += error
	}

	return sumAbsoluteErrors / float64(n), nil
}

// Helper functions for statistical distributions

// getTotalSumSquares calculates the total sum of squares for y
func getTotalSumSquares(y *array.Array) float64 {
	yMean, _ := Mean(y)
	tss := 0.0
	for i := 0; i < y.Size(); i++ {
		diff := y.At(i).(float64) - yMean
		tss += diff * diff
	}
	return tss
}

// tDistributionPValue calculates two-tailed p-value for t-distribution
// This is a simplified approximation
func tDistributionPValue(t float64, df int) float64 {
	if df <= 0 {
		return 1.0
	}

	// Simplified approximation using normal distribution for large df
	if df > 30 {
		return 2 * (1 - normalCDF(t))
	}

	// For smaller df, use a rough approximation
	// In practice, you'd use a proper t-distribution function
	return 2 * (1 - normalCDF(t*gmath.Sqrt(float64(df)/(float64(df)+t*t))))
}

// fDistributionPValue calculates p-value for F-distribution (simplified)
func fDistributionPValue(f float64, df1, df2 int) float64 {
	if f <= 0 || df1 <= 0 || df2 <= 0 {
		return 1.0
	}

	// Very simplified approximation - in practice use proper F-distribution
	// For now, just indicate significance for reasonable F values
	if f > 4.0 {
		return 0.01 // Likely significant
	} else if f > 2.0 {
		return 0.05 // Moderately significant
	}
	return 0.2 // Not significant
}

// normalCDF function already exists in tests.go

// calculateConditionNumber estimates the condition number of a matrix
func calculateConditionNumber(A *array.Array) (float64, error) {
	if A.Ndim() != 2 {
		return 0, fmt.Errorf("matrix must be 2D")
	}

	shape := A.Shape()
	if shape[0] != shape[1] {
		return 0, fmt.Errorf("matrix must be square")
	}

	// Simple condition number estimation using matrix norms
	// Check for exactly zero rows/columns first (singular matrix)
	n := shape[0]
	for i := 0; i < n; i++ {
		rowSum := 0.0
		colSum := 0.0
		for j := 0; j < n; j++ {
			rowSum += gmath.Abs(A.At(i, j).(float64))
			colSum += gmath.Abs(A.At(j, i).(float64))
		}
		if rowSum < 1e-15 || colSum < 1e-15 {
			return gmath.Inf(1), nil // Singular matrix (zero row or column)
		}
	}

	// Calculate rough condition number using matrix norms
	maxElement := 0.0
	minNonZero := gmath.Inf(1)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			val := gmath.Abs(A.At(i, j).(float64))
			if val > maxElement {
				maxElement = val
			}
			if val > 1e-15 && val < minNonZero {
				minNonZero = val
			}
		}
	}

	if minNonZero == gmath.Inf(1) || maxElement == 0 {
		return gmath.Inf(1), nil // All zero matrix
	}

	// Rough estimate - proper calculation would use singular values
	return maxElement / minNonZero, nil
}

// handleSingularMatrix handles the case when X'X matrix is singular
func handleSingularMatrix(result *RegressionResult, XTX, identity *array.Array) error {
	// For singular matrices, we cannot compute standard errors reliably
	// Set all standard errors, t-statistics, and p-values to NaN

	numCoeffs := len(result.Coefficients)
	result.StandardErrors = make([]float64, numCoeffs)
	result.TStats = make([]float64, numCoeffs)
	result.PValues = make([]float64, numCoeffs)

	for i := 0; i < numCoeffs; i++ {
		result.StandardErrors[i] = gmath.NaN()
		result.TStats[i] = gmath.NaN()
		result.PValues[i] = gmath.NaN()
	}

	return fmt.Errorf("design matrix is singular - cannot compute reliable standard errors, coefficients may be unreliable")
}

// RidgeRegression performs ridge regression with L2 regularization
// This is useful when dealing with singular or ill-conditioned matrices
func RidgeRegression(y, X *array.Array, alpha float64) (*RegressionResult, error) {
	if y == nil || X == nil {
		return nil, fmt.Errorf("arrays cannot be nil")
	}

	if alpha < 0 {
		return nil, fmt.Errorf("regularization parameter alpha must be non-negative, got %f", alpha)
	}

	// Handle dimensions same as LinearRegression
	var designMatrix *array.Array
	var err error

	if X.Ndim() == 1 {
		// Simple linear regression: add intercept column
		n := X.Size()
		if n != y.Size() {
			return nil, fmt.Errorf("X and y must have same number of observations: %d vs %d", n, y.Size())
		}

		// Create design matrix [1, X]
		designData := make([]float64, n*2)
		for i := 0; i < n; i++ {
			designData[i*2] = 1.0                 // Intercept column
			designData[i*2+1] = X.At(i).(float64) // X values
		}
		designMatrix, err = array.NewArrayWithShape(designData, internal.Shape{n, 2})
		if err != nil {
			return nil, fmt.Errorf("failed to create design matrix: %v", err)
		}
	} else if X.Ndim() == 2 {
		// Multiple regression: add intercept column if not present
		shape := X.Shape()
		n, p := shape[0], shape[1]

		if n != y.Size() {
			return nil, fmt.Errorf("X and y must have same number of observations: %d vs %d", n, y.Size())
		}

		// Create design matrix with intercept column [1, X]
		designData := make([]float64, n*(p+1))
		for i := 0; i < n; i++ {
			designData[i*(p+1)] = 1.0 // Intercept column
			for j := 0; j < p; j++ {
				designData[i*(p+1)+j+1] = X.At(i, j).(float64)
			}
		}
		designMatrix, err = array.NewArrayWithShape(designData, internal.Shape{n, p + 1})
		if err != nil {
			return nil, fmt.Errorf("failed to create design matrix: %v", err)
		}
	} else {
		return nil, fmt.Errorf("X must be 1D or 2D array, got %dD", X.Ndim())
	}

	if designMatrix.Size() == 0 {
		return nil, fmt.Errorf("empty design matrix")
	}

	// Solve ridge regression: (X'X + αI)^(-1) X'y
	beta, err := solveRidgeEquation(designMatrix, y, alpha)
	if err != nil {
		return nil, fmt.Errorf("failed to solve ridge regression equation: %v", err)
	}

	// Calculate fitted values and residuals
	yHat, err := predictValues(designMatrix, beta)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate fitted values: %v", err)
	}

	residuals, err := calculateResiduals(y, yHat)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate residuals: %v", err)
	}

	// Calculate regression statistics
	result := &RegressionResult{
		Intercept:    beta.At(0).(float64),
		Coefficients: make([]float64, beta.Size()-1),
	}

	// Extract coefficients (excluding intercept)
	for i := 1; i < beta.Size(); i++ {
		result.Coefficients[i-1] = beta.At(i).(float64)
	}

	// Calculate R-squared and other statistics for ridge regression
	err = calculateRidgeRegressionStatistics(result, y, yHat, residuals, designMatrix, alpha)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate statistics: %v", err)
	}

	return result, nil
}

// solveRidgeEquation solves (X'X + αI)^(-1) X'y for ridge regression coefficients
func solveRidgeEquation(X, y *array.Array, alpha float64) (*array.Array, error) {
	// X' (transpose)
	XTranspose, err := math.Transpose(X)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose X: %v", err)
	}

	// X'X
	XTX, err := math.Dot(XTranspose, X)
	if err != nil {
		return nil, fmt.Errorf("failed to compute X'X: %v", err)
	}

	// Add regularization: X'X + αI
	shape := XTX.Shape()
	n := shape[0]
	for i := 0; i < n; i++ {
		// Add alpha to diagonal elements (except intercept if alpha > 0)
		currentVal := XTX.At(i, i).(float64)
		if i == 0 && alpha > 0 {
			// Don't regularize intercept term
			continue
		} else {
			XTX.Set(currentVal+alpha, i, i)
		}
	}

	// X'y
	XTy, err := math.Dot(XTranspose, y.Reshape(internal.Shape{y.Size(), 1}))
	if err != nil {
		return nil, fmt.Errorf("failed to compute X'y: %v", err)
	}

	// Solve (X'X + αI)^(-1) X'y
	beta, err := math.Solve(XTX, XTy)
	if err != nil {
		return nil, fmt.Errorf("failed to solve regularized linear system: %v", err)
	}

	// Flatten to 1D
	return beta.Reshape(internal.Shape{beta.Size()}), nil
}

// LinearRegressionWithFallback attempts standard OLS first, then falls back to ridge regression if singular
func LinearRegressionWithFallback(y, X *array.Array, ridgeAlpha float64) (*RegressionResult, error) {
	// First attempt standard OLS
	result, err := LinearRegression(y, X)
	if err != nil {
		// Check if it's a singularity issue
		if strings.Contains(err.Error(), "singular") || strings.Contains(err.Error(), "rank deficient") {
			// Fall back to ridge regression
			fmt.Printf("Warning: Standard OLS failed due to singularity, falling back to ridge regression with α=%.6f\n", ridgeAlpha)
			return RidgeRegression(y, X, ridgeAlpha)
		}
		// If it's a different error, return it
		return nil, err
	}

	// Standard OLS succeeded
	return result, nil
}
