package stats

import (
	"fmt"
	"math"

	"github.com/julianshen/gonp/array"
)

// HuberRegressionResult contains the results of Huber robust regression
type HuberRegressionResult struct {
	Coefficients []float64 // Regression coefficients (excluding intercept)
	Intercept    float64   // Intercept term
	RSquared     float64   // Coefficient of determination
	Converged    bool      // Whether the algorithm converged
	Iterations   int       // Number of iterations used
	Scale        float64   // Robust scale estimate
}

// TukeyBisquareRegressionResult contains the results of Tukey bisquare robust regression
type TukeyBisquareRegressionResult struct {
	Coefficients []float64 // Regression coefficients (excluding intercept)
	Intercept    float64   // Intercept term
	RSquared     float64   // Coefficient of determination
	Converged    bool      // Whether the algorithm converged
	Iterations   int       // Number of iterations used
	Scale        float64   // Robust scale estimate
}

// HuberRegression performs Huber robust regression using iteratively reweighted least squares (IRLS)
// This method is robust against outliers by using Huber's loss function
//
// Parameters:
//
//	y: Response variable array
//	X: Predictor variable array (can be 1D or 2D)
//	k: Tuning parameter for Huber loss (typically 1.345 for 95% efficiency)
//
// Returns: HuberRegressionResult with coefficients, diagnostics, and convergence info
func HuberRegression(y, X *array.Array, k float64) (*HuberRegressionResult, error) {
	// Input validation
	if y == nil {
		return nil, fmt.Errorf("y array cannot be nil")
	}
	if X == nil {
		return nil, fmt.Errorf("X array cannot be nil")
	}
	if k <= 0 {
		return nil, fmt.Errorf("tuning parameter k must be positive, got %f", k)
	}

	// Get dimensions
	n := y.Size()
	if n == 0 {
		return nil, fmt.Errorf("empty y array")
	}

	// Handle X dimensionality
	var numFeatures int
	if X.Ndim() == 1 {
		if X.Size() != n {
			return nil, fmt.Errorf("X size (%d) must match y size (%d)", X.Size(), n)
		}
		numFeatures = 1
	} else if X.Ndim() == 2 {
		shape := X.Shape()
		if shape[0] != n {
			return nil, fmt.Errorf("X rows (%d) must match y size (%d)", shape[0], n)
		}
		numFeatures = shape[1]
	} else {
		return nil, fmt.Errorf("X must be 1D or 2D array, got %dD", X.Ndim())
	}

	// Start with OLS as initial estimate
	olsResult, err := LinearRegression(y, X)
	if err != nil {
		return nil, fmt.Errorf("failed to compute initial OLS estimate: %v", err)
	}

	// IRLS parameters
	maxIterations := 50
	tolerance := 1e-6

	// Current estimates
	coefficients := make([]float64, len(olsResult.Coefficients))
	copy(coefficients, olsResult.Coefficients)
	intercept := olsResult.Intercept

	var converged bool
	var iterations int
	var scale float64

	for iterations = 0; iterations < maxIterations; iterations++ {
		// Compute residuals
		residuals := make([]float64, n)
		for i := 0; i < n; i++ {
			yVal := convertToFloat64(y.At(i))

			predicted := intercept
			for j := 0; j < numFeatures; j++ {
				var xVal float64
				if X.Ndim() == 1 {
					xVal = convertToFloat64(X.At(i))
				} else {
					xVal = convertToFloat64(X.At(i, j))
				}
				predicted += coefficients[j] * xVal
			}
			residuals[i] = yVal - predicted
		}

		// Compute robust scale estimate (MAD - Median Absolute Deviation)
		absResiduals := make([]float64, n)
		for i := 0; i < n; i++ {
			absResiduals[i] = math.Abs(residuals[i])
		}
		scale = medianAbsoluteDeviation(absResiduals)

		// Avoid division by zero
		if scale < 1e-10 {
			scale = 1e-10
		}

		// Compute Huber weights
		weights := make([]float64, n)
		for i := 0; i < n; i++ {
			normalizedRes := math.Abs(residuals[i]) / scale
			if normalizedRes <= k {
				weights[i] = 1.0
			} else {
				weights[i] = k / normalizedRes
			}
		}

		// Solve weighted least squares
		newIntercept, newCoefficients, err := weightedLeastSquares(y, X, weights)
		if err != nil {
			return nil, fmt.Errorf("weighted least squares failed at iteration %d: %v", iterations, err)
		}

		// Check convergence
		interceptChange := math.Abs(newIntercept - intercept)
		maxCoeffChange := 0.0
		for j := 0; j < len(coefficients); j++ {
			change := math.Abs(newCoefficients[j] - coefficients[j])
			if change > maxCoeffChange {
				maxCoeffChange = change
			}
		}

		if interceptChange < tolerance && maxCoeffChange < tolerance {
			converged = true
			break
		}

		// Update estimates
		intercept = newIntercept
		coefficients = newCoefficients
	}

	// Compute R-squared
	rSquared := computeRSquared(y, X, intercept, coefficients)

	result := &HuberRegressionResult{
		Coefficients: coefficients,
		Intercept:    intercept,
		RSquared:     rSquared,
		Converged:    converged,
		Iterations:   iterations + 1,
		Scale:        scale,
	}

	return result, nil
}

// TukeyBisquareRegression performs Tukey bisquare robust regression using IRLS
// This method completely discards outliers beyond a threshold, making it very robust
//
// Parameters:
//
//	y: Response variable array
//	X: Predictor variable array (can be 1D or 2D)
//	c: Tuning parameter for Tukey bisquare (typically 4.685 for 95% efficiency)
//
// Returns: TukeyBisquareRegressionResult with coefficients, diagnostics, and convergence info
func TukeyBisquareRegression(y, X *array.Array, c float64) (*TukeyBisquareRegressionResult, error) {
	// Input validation
	if y == nil {
		return nil, fmt.Errorf("y array cannot be nil")
	}
	if X == nil {
		return nil, fmt.Errorf("X array cannot be nil")
	}
	if c <= 0 {
		return nil, fmt.Errorf("tuning parameter c must be positive, got %f", c)
	}

	// Get dimensions
	n := y.Size()
	if n == 0 {
		return nil, fmt.Errorf("empty y array")
	}

	// Handle X dimensionality
	var numFeatures int
	if X.Ndim() == 1 {
		if X.Size() != n {
			return nil, fmt.Errorf("X size (%d) must match y size (%d)", X.Size(), n)
		}
		numFeatures = 1
	} else if X.Ndim() == 2 {
		shape := X.Shape()
		if shape[0] != n {
			return nil, fmt.Errorf("X rows (%d) must match y size (%d)", shape[0], n)
		}
		numFeatures = shape[1]
	} else {
		return nil, fmt.Errorf("X must be 1D or 2D array, got %dD", X.Ndim())
	}

	// Start with OLS as initial estimate
	olsResult, err := LinearRegression(y, X)
	if err != nil {
		return nil, fmt.Errorf("failed to compute initial OLS estimate: %v", err)
	}

	// IRLS parameters
	maxIterations := 50
	tolerance := 1e-6

	// Current estimates
	coefficients := make([]float64, len(olsResult.Coefficients))
	copy(coefficients, olsResult.Coefficients)
	intercept := olsResult.Intercept

	var converged bool
	var iterations int
	var scale float64

	for iterations = 0; iterations < maxIterations; iterations++ {
		// Compute residuals
		residuals := make([]float64, n)
		for i := 0; i < n; i++ {
			yVal := convertToFloat64(y.At(i))

			predicted := intercept
			for j := 0; j < numFeatures; j++ {
				var xVal float64
				if X.Ndim() == 1 {
					xVal = convertToFloat64(X.At(i))
				} else {
					xVal = convertToFloat64(X.At(i, j))
				}
				predicted += coefficients[j] * xVal
			}
			residuals[i] = yVal - predicted
		}

		// Compute robust scale estimate (MAD)
		absResiduals := make([]float64, n)
		for i := 0; i < n; i++ {
			absResiduals[i] = math.Abs(residuals[i])
		}
		scale = medianAbsoluteDeviation(absResiduals)

		if scale < 1e-10 {
			scale = 1e-10
		}

		// Compute Tukey bisquare weights
		weights := make([]float64, n)
		for i := 0; i < n; i++ {
			u := math.Abs(residuals[i]) / scale
			if u <= c {
				// Tukey bisquare weight function
				uOverC := u / c
				weights[i] = (1 - uOverC*uOverC) * (1 - uOverC*uOverC)
			} else {
				weights[i] = 0.0 // Complete rejection of outliers
			}
		}

		// Solve weighted least squares
		newIntercept, newCoefficients, err := weightedLeastSquares(y, X, weights)
		if err != nil {
			return nil, fmt.Errorf("weighted least squares failed at iteration %d: %v", iterations, err)
		}

		// Check convergence
		interceptChange := math.Abs(newIntercept - intercept)
		maxCoeffChange := 0.0
		for j := 0; j < len(coefficients); j++ {
			change := math.Abs(newCoefficients[j] - coefficients[j])
			if change > maxCoeffChange {
				maxCoeffChange = change
			}
		}

		if interceptChange < tolerance && maxCoeffChange < tolerance {
			converged = true
			break
		}

		// Update estimates
		intercept = newIntercept
		coefficients = newCoefficients
	}

	// Compute R-squared
	rSquared := computeRSquared(y, X, intercept, coefficients)

	result := &TukeyBisquareRegressionResult{
		Coefficients: coefficients,
		Intercept:    intercept,
		RSquared:     rSquared,
		Converged:    converged,
		Iterations:   iterations + 1,
		Scale:        scale,
	}

	return result, nil
}

// Helper functions for robust regression

// medianAbsoluteDeviation computes the median absolute deviation scaled by 1.4826 for consistency
func medianAbsoluteDeviation(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}

	// Make a copy and sort it
	sorted := make([]float64, len(data))
	copy(sorted, data)
	quickSort(sorted, 0, len(sorted)-1)

	// Find median
	median := findMedian(sorted)

	// Compute absolute deviations from median
	absDeviations := make([]float64, len(data))
	for i, val := range data {
		absDeviations[i] = math.Abs(val - median)
	}

	// Sort absolute deviations
	quickSort(absDeviations, 0, len(absDeviations)-1)

	// Return MAD scaled by 1.4826 (for consistency with standard deviation)
	return findMedian(absDeviations) * 1.4826
}

// findMedian finds the median of a sorted array
func findMedian(sorted []float64) float64 {
	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

// quickSort sorts a float64 slice in place
func quickSort(arr []float64, low, high int) {
	if low < high {
		pi := partition(arr, low, high)
		quickSort(arr, low, pi-1)
		quickSort(arr, pi+1, high)
	}
}

func partition(arr []float64, low, high int) int {
	pivot := arr[high]
	i := low - 1

	for j := low; j <= high-1; j++ {
		if arr[j] < pivot {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i + 1
}

// weightedLeastSquares solves weighted least squares problem
func weightedLeastSquares(y, X *array.Array, weights []float64) (float64, []float64, error) {
	n := y.Size()

	// Handle X dimensionality
	var numFeatures int
	if X.Ndim() == 1 {
		numFeatures = 1
	} else {
		numFeatures = X.Shape()[1]
	}

	// Create weighted design matrix with intercept column
	// W^(1/2) * [1, X]
	designCols := numFeatures + 1 // +1 for intercept

	// Build normal equations: (X^T W X) beta = X^T W y
	XtWX := make([][]float64, designCols)
	for i := range XtWX {
		XtWX[i] = make([]float64, designCols)
	}

	XtWy := make([]float64, designCols)

	for i := 0; i < n; i++ {
		w := weights[i]
		yVal := convertToFloat64(y.At(i))

		// Design matrix row: [1, x1, x2, ...]
		designRow := make([]float64, designCols)
		designRow[0] = 1.0 // intercept

		for j := 0; j < numFeatures; j++ {
			if X.Ndim() == 1 {
				designRow[j+1] = convertToFloat64(X.At(i))
			} else {
				designRow[j+1] = convertToFloat64(X.At(i, j))
			}
		}

		// Add to normal equations
		for j := 0; j < designCols; j++ {
			// XtWy
			XtWy[j] += w * designRow[j] * yVal

			// XtWX
			for k := 0; k < designCols; k++ {
				XtWX[j][k] += w * designRow[j] * designRow[k]
			}
		}
	}

	// Solve the normal equations using simple Gaussian elimination
	solution, err := gaussianElimination(XtWX, XtWy)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to solve normal equations: %v", err)
	}

	// Extract intercept and coefficients
	intercept := solution[0]
	coefficients := solution[1:]

	return intercept, coefficients, nil
}

// gaussianElimination solves Ax = b using Gaussian elimination with partial pivoting
func gaussianElimination(A [][]float64, b []float64) ([]float64, error) {
	n := len(A)
	if len(b) != n {
		return nil, fmt.Errorf("dimension mismatch")
	}

	// Create augmented matrix
	augmented := make([][]float64, n)
	for i := 0; i < n; i++ {
		augmented[i] = make([]float64, n+1)
		copy(augmented[i][:n], A[i])
		augmented[i][n] = b[i]
	}

	// Forward elimination with partial pivoting
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
			return nil, fmt.Errorf("singular matrix")
		}

		// Eliminate column
		for k := i + 1; k < n; k++ {
			factor := augmented[k][i] / augmented[i][i]
			for j := i; j <= n; j++ {
				augmented[k][j] -= factor * augmented[i][j]
			}
		}
	}

	// Back substitution
	solution := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		solution[i] = augmented[i][n]
		for j := i + 1; j < n; j++ {
			solution[i] -= augmented[i][j] * solution[j]
		}
		solution[i] /= augmented[i][i]
	}

	return solution, nil
}

// computeRSquared computes R-squared for the regression
func computeRSquared(y, X *array.Array, intercept float64, coefficients []float64) float64 {
	n := y.Size()
	var sumSquaredResiduals, totalSumSquares float64

	// Compute mean of y
	var yMean float64
	for i := 0; i < n; i++ {
		yMean += convertToFloat64(y.At(i))
	}
	yMean /= float64(n)

	// Compute sums
	for i := 0; i < n; i++ {
		yVal := convertToFloat64(y.At(i))

		// Predicted value
		predicted := intercept
		numFeatures := len(coefficients)
		for j := 0; j < numFeatures; j++ {
			if X.Ndim() == 1 {
				predicted += coefficients[j] * convertToFloat64(X.At(i))
			} else {
				predicted += coefficients[j] * convertToFloat64(X.At(i, j))
			}
		}

		// Residual
		residual := yVal - predicted
		sumSquaredResiduals += residual * residual

		// Total sum of squares
		totalSumSquares += (yVal - yMean) * (yVal - yMean)
	}

	if totalSumSquares == 0 {
		return 1.0 // Perfect fit when all y values are the same
	}

	return 1.0 - (sumSquaredResiduals / totalSumSquares)
}
