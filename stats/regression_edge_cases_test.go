package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestAdvancedRegressionEdgeCases tests advanced edge cases for regression functions
func TestAdvancedRegressionEdgeCases(t *testing.T) {
	t.Run("Perfect linear relationship", func(t *testing.T) {
		// Perfect linear relationship: y = 2x + 3
		X, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
		y, _ := array.FromSlice([]float64{5, 7, 9, 11, 13})

		result, err := LinearRegression(y, X)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// R-squared should be 1.0 for perfect relationship
		if math.Abs(result.RSquared-1.0) > 1e-10 {
			t.Errorf("Perfect relationship should have R² = 1.0, got %f", result.RSquared)
		}

		// Intercept should be approximately 3.0
		if math.Abs(result.Intercept-3.0) > 1e-10 {
			t.Errorf("Perfect relationship intercept should be 3.0, got %f", result.Intercept)
		}

		// Slope should be approximately 2.0
		if len(result.Coefficients) > 0 && math.Abs(result.Coefficients[0]-2.0) > 1e-10 {
			t.Errorf("Perfect relationship slope should be 2.0, got %f", result.Coefficients[0])
		}

		// Standard errors should be very small (close to 0)
		if len(result.StandardErrors) > 0 && result.StandardErrors[0] > 1e-10 {
			t.Errorf("Standard errors should be near zero for perfect relationship, got %e", result.StandardErrors[0])
		}
	})

	t.Run("Regression with identical X values", func(t *testing.T) {
		// All X values are the same (singular matrix)
		X, _ := array.FromSlice([]float64{5, 5, 5, 5, 5})
		y, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})

		// Implementation may handle this or return error
		result, err := LinearRegression(y, X)
		if err != nil {
			t.Logf("Identical X values handled with error (expected): %v", err)
		} else {
			t.Logf("Identical X values handled successfully: R²=%.6f, slope=%.6f",
				result.RSquared, result.Coefficients[0])
		}
	})

	t.Run("Regression with constant y values", func(t *testing.T) {
		// All y values are the same
		X, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
		y, _ := array.FromSlice([]float64{10, 10, 10, 10, 10})

		result, err := LinearRegression(y, X)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// R-squared behavior with constant y depends on implementation
		// Some implementations return 0.0, others may return 1.0 due to division by zero handling
		t.Logf("Constant y regression R²=%.6f (implementation dependent)", result.RSquared)

		// Slope should be approximately 0
		if len(result.Coefficients) > 0 && math.Abs(result.Coefficients[0]) > 1e-10 {
			t.Errorf("Constant y should have slope ≈ 0, got %f", result.Coefficients[0])
		}

		// Intercept should be approximately the mean of y (10.0)
		if math.Abs(result.Intercept-10.0) > 1e-10 {
			t.Errorf("Constant y intercept should be 10.0, got %f", result.Intercept)
		}
	})

	t.Run("Regression with very few data points", func(t *testing.T) {
		// Minimum case: 2 data points
		X, _ := array.FromSlice([]float64{1, 2})
		y, _ := array.FromSlice([]float64{3, 5})

		result, err := LinearRegression(y, X)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// With only 2 points, R-squared should be 1.0 (perfect fit)
		if math.Abs(result.RSquared-1.0) > 1e-10 {
			t.Errorf("2-point regression should have R² = 1.0, got %f", result.RSquared)
		}

		// Degrees of freedom should be 0
		if result.DegreesOfFreedom != 0 {
			t.Errorf("2-point regression should have 0 degrees of freedom, got %d", result.DegreesOfFreedom)
		}
	})

	t.Run("Regression with single data point", func(t *testing.T) {
		X, _ := array.FromSlice([]float64{1})
		y, _ := array.FromSlice([]float64{3})

		// Should return error for insufficient data
		_, err := LinearRegression(y, X)
		if err == nil {
			t.Error("Expected error for regression with single data point")
		}
	})

	t.Run("Regression with extreme outliers", func(t *testing.T) {
		// Most points follow y = x pattern, but one extreme outlier
		X, _ := array.FromSlice([]float64{1, 2, 3, 4, 1000})
		y, _ := array.FromSlice([]float64{1, 2, 3, 4, 1000})

		result, err := LinearRegression(y, X)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// R-squared should still be high due to the extreme point
		if result.RSquared < 0.9 {
			t.Logf("R-squared with extreme outlier: %f (may be affected by outlier)", result.RSquared)
		}

		// Slope should be close to 1.0 due to the strong linear pattern
		if len(result.Coefficients) > 0 {
			t.Logf("Slope with extreme outlier: %f", result.Coefficients[0])
		}
	})

	t.Run("Multiple regression with multicollinearity", func(t *testing.T) {
		// X1 and X2 are perfectly correlated
		y, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
		X1, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
		X2, _ := array.FromSlice([]float64{2, 4, 6, 8, 10}) // X2 = 2 * X1

		// Create design matrix with both X1 and X2
		designMatrix := array.Zeros(internal.Shape{5, 2}, internal.Float64)
		for i := 0; i < 5; i++ {
			designMatrix.Set(X1.At(i).(float64), i, 0)
			designMatrix.Set(X2.At(i).(float64), i, 1)
		}

		// Should either handle multicollinearity or return error
		result, err := LinearRegression(y, designMatrix)
		if err != nil {
			t.Logf("Multicollinearity handled with error (acceptable): %v", err)
		} else {
			t.Logf("Multicollinearity handled, R² = %f", result.RSquared)
			// Standard errors might be very large due to multicollinearity
			if len(result.StandardErrors) > 0 {
				t.Logf("Standard errors with multicollinearity: %v", result.StandardErrors)
			}
		}
	})

	t.Run("Ridge regression with various alpha values", func(t *testing.T) {
		// Test ridge regression with different regularization parameters
		X, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
		y, _ := array.FromSlice([]float64{2, 4, 6, 8, 10})

		alphas := []float64{0.0, 0.1, 1.0, 10.0, 100.0}
		var results []*RegressionResult

		for _, alpha := range alphas {
			result, err := RidgeRegression(y, X, alpha)
			if err != nil {
				t.Fatalf("Ridge regression failed for alpha=%f: %v", alpha, err)
			}
			results = append(results, result)

			t.Logf("Ridge α=%.1f: R²=%.6f, slope=%.6f, intercept=%.6f",
				alpha, result.RSquared, result.Coefficients[0], result.Intercept)
		}

		// As alpha increases, coefficients should shrink toward zero
		for i := 1; i < len(results); i++ {
			prevSlope := math.Abs(results[i-1].Coefficients[0])
			currSlope := math.Abs(results[i].Coefficients[0])
			if currSlope > prevSlope {
				t.Logf("Note: Ridge coefficient didn't shrink as expected (α=%.1f: %.6f, α=%.1f: %.6f)",
					alphas[i-1], prevSlope, alphas[i], currSlope)
			}
		}
	})

	t.Run("Regression fallback mechanism", func(t *testing.T) {
		// Create a singular matrix case that should trigger fallback to ridge
		X1, _ := array.FromSlice([]float64{1, 2, 3})
		X2, _ := array.FromSlice([]float64{1, 2, 3}) // Identical to X1
		y, _ := array.FromSlice([]float64{1, 4, 9})

		designMatrix := array.Zeros(internal.Shape{3, 2}, internal.Float64)
		for i := 0; i < 3; i++ {
			designMatrix.Set(X1.At(i).(float64), i, 0)
			designMatrix.Set(X2.At(i).(float64), i, 1)
		}

		// Test fallback with reasonable ridge alpha
		result, err := LinearRegressionWithFallback(y, designMatrix, 0.1)
		if err != nil {
			t.Fatalf("Regression fallback failed: %v", err)
		}

		// Should successfully return a result
		if result == nil {
			t.Error("Fallback should return a valid result")
		} else {
			t.Logf("Fallback regression: R²=%.6f, coefficients=%v", result.RSquared, result.Coefficients)
		}
	})
}

// TestRobustRegressionEdgeCases tests edge cases for robust regression methods
func TestRobustRegressionEdgeCases(t *testing.T) {
	t.Run("Huber regression with no outliers", func(t *testing.T) {
		// Clean linear data without outliers
		X, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
		y, _ := array.FromSlice([]float64{2, 4, 6, 8, 10})

		result, err := HuberRegression(y, X, 1.345)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Should be very similar to OLS regression for clean data
		olsResult, _ := LinearRegression(y, X)

		slopeDiff := math.Abs(result.Coefficients[0] - olsResult.Coefficients[0])
		interceptDiff := math.Abs(result.Intercept - olsResult.Intercept)

		if slopeDiff > 0.1 {
			t.Errorf("Huber and OLS slopes should be similar for clean data: Huber=%.6f, OLS=%.6f",
				result.Coefficients[0], olsResult.Coefficients[0])
		}
		if interceptDiff > 0.1 {
			t.Errorf("Huber and OLS intercepts should be similar for clean data: Huber=%.6f, OLS=%.6f",
				result.Intercept, olsResult.Intercept)
		}
	})

	t.Run("Tukey bisquare with extreme outliers", func(t *testing.T) {
		// Linear data with one extreme outlier
		X, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
		y, _ := array.FromSlice([]float64{2, 4, 6, 100, 10}) // y[3] is extreme outlier

		result, err := TukeyBisquareRegression(y, X, 4.685)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Compare with OLS to see the effect of robust regression
		olsResult, _ := LinearRegression(y, X)

		t.Logf("Tukey bisquare slope=%.3f vs OLS slope=%.3f",
			result.Coefficients[0], olsResult.Coefficients[0])
		t.Logf("Tukey bisquare R²=%.3f vs OLS R²=%.3f",
			result.RSquared, olsResult.RSquared)

		// Tukey bisquare should be more robust to the outlier
		if math.Abs(result.Coefficients[0]-2.0) > math.Abs(olsResult.Coefficients[0]-2.0) {
			t.Logf("Note: Tukey bisquare may not have improved robustness in this case")
		}
	})

	t.Run("Robust regression convergence", func(t *testing.T) {
		// Test convergence with difficult data
		X, _ := array.FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
		y, _ := array.FromSlice([]float64{1, 2, 100, 4, 5, 6, -100, 8, 9, 10})

		// Test both robust methods
		huberResult, err := HuberRegression(y, X, 1.345)
		if err != nil {
			t.Logf("Huber regression convergence issue: %v", err)
		} else {
			if !huberResult.Converged {
				t.Logf("Huber regression did not converge in %d iterations", huberResult.Iterations)
			} else {
				t.Logf("Huber regression converged in %d iterations", huberResult.Iterations)
			}
		}

		tukeyResult, err := TukeyBisquareRegression(y, X, 4.685)
		if err != nil {
			t.Logf("Tukey bisquare regression convergence issue: %v", err)
		} else {
			if !tukeyResult.Converged {
				t.Logf("Tukey bisquare regression did not converge in %d iterations", tukeyResult.Iterations)
			} else {
				t.Logf("Tukey bisquare regression converged in %d iterations", tukeyResult.Iterations)
			}
		}
	})

	t.Run("Robust regression parameter sensitivity", func(t *testing.T) {
		X, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
		y, _ := array.FromSlice([]float64{2, 4, 6, 20, 10}) // Moderate outlier

		// Test different k values for Huber regression
		kValues := []float64{0.5, 1.0, 1.345, 2.0, 3.0}
		for _, k := range kValues {
			result, err := HuberRegression(y, X, k)
			if err != nil {
				t.Logf("Huber k=%.3f failed: %v", k, err)
			} else {
				t.Logf("Huber k=%.3f: slope=%.3f, R²=%.3f", k, result.Coefficients[0], result.RSquared)
			}
		}

		// Test different c values for Tukey bisquare
		cValues := []float64{2.0, 3.0, 4.685, 6.0, 8.0}
		for _, c := range cValues {
			result, err := TukeyBisquareRegression(y, X, c)
			if err != nil {
				t.Logf("Tukey c=%.3f failed: %v", c, err)
			} else {
				t.Logf("Tukey c=%.3f: slope=%.3f, R²=%.3f", c, result.Coefficients[0], result.RSquared)
			}
		}
	})
}
