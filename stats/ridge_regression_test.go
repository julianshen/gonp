package stats

import (
	"math"
	"strings"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestRidgeRegression(t *testing.T) {
	t.Run("Simple ridge regression", func(t *testing.T) {
		// Create simple test data
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{2, 4, 6, 8, 10} // y = 2*x exactly

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		// Test with small regularization
		result, err := RidgeRegression(yArr, xArr, 0.1)
		if err != nil {
			t.Fatalf("RidgeRegression failed: %v", err)
		}

		// Should be close to OLS result (y = 2*x + 0)
		if math.Abs(result.Intercept) > 0.1 {
			t.Errorf("Expected intercept near 0, got %f", result.Intercept)
		}

		if len(result.Coefficients) != 1 {
			t.Fatalf("Expected 1 coefficient, got %d", len(result.Coefficients))
		}

		if math.Abs(result.Coefficients[0]-2.0) > 0.1 {
			t.Errorf("Expected coefficient near 2.0, got %f", result.Coefficients[0])
		}

		// R-squared should be high
		if result.RSquared < 0.99 {
			t.Errorf("Expected high R-squared, got %f", result.RSquared)
		}
	})

	t.Run("Ridge regression with perfect correlation", func(t *testing.T) {
		// Create perfectly correlated X variables (problematic for OLS)
		x1 := []float64{1, 2, 3, 4, 5}
		x2 := []float64{2, 4, 6, 8, 10} // x2 = 2*x1, perfect correlation
		y := []float64{1, 2, 3, 4, 5}

		X := array.Empty(internal.Shape{5, 2}, internal.Float64)
		yArr, _ := array.FromSlice(y)

		for i := 0; i < 5; i++ {
			X.Set(x1[i], i, 0)
			X.Set(x2[i], i, 1)
		}

		// Ridge regression should handle this better than OLS
		result, err := RidgeRegression(yArr, X, 1.0)
		if err != nil {
			t.Fatalf("RidgeRegression failed: %v", err)
		}

		// Should produce finite coefficients
		for i, coeff := range result.Coefficients {
			if math.IsInf(coeff, 0) || math.IsNaN(coeff) {
				t.Errorf("Coefficient %d should be finite, got %f", i, coeff)
			}
		}

		// Intercept should be finite
		if math.IsInf(result.Intercept, 0) || math.IsNaN(result.Intercept) {
			t.Errorf("Intercept should be finite, got %f", result.Intercept)
		}
	})

	t.Run("Ridge regression shrinkage effect", func(t *testing.T) {
		// Create data where ridge should shrink coefficients
		x1 := []float64{1, 2, 3, 4, 5}
		x2 := []float64{1.1, 2.1, 3.1, 4.1, 5.1} // Slightly correlated with x1
		y := []float64{10, 20, 30, 40, 50}       // Large coefficients without regularization

		X := array.Empty(internal.Shape{5, 2}, internal.Float64)
		yArr, _ := array.FromSlice(y)

		for i := 0; i < 5; i++ {
			X.Set(x1[i], i, 0)
			X.Set(x2[i], i, 1)
		}

		// Compare ridge with different alpha values
		resultLow, err := RidgeRegression(yArr, X, 0.01)
		if err != nil {
			t.Fatalf("RidgeRegression with low alpha failed: %v", err)
		}

		resultHigh, err := RidgeRegression(yArr, X, 10.0)
		if err != nil {
			t.Fatalf("RidgeRegression with high alpha failed: %v", err)
		}

		// Higher alpha should produce smaller (more shrunken) coefficients
		for i := 0; i < len(resultLow.Coefficients); i++ {
			lowCoeff := math.Abs(resultLow.Coefficients[i])
			highCoeff := math.Abs(resultHigh.Coefficients[i])

			if highCoeff > lowCoeff {
				t.Errorf("Higher regularization should shrink coefficients: low α coeff[%d]=%.3f, high α coeff[%d]=%.3f",
					i, lowCoeff, i, highCoeff)
			}
		}
	})

	t.Run("Ridge regression with zero alpha equals OLS", func(t *testing.T) {
		// Ridge with α=0 should equal OLS (when OLS is solvable)
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{3, 5, 7, 9, 11} // y = 2*x + 1

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		ridgeResult, err := RidgeRegression(yArr, xArr, 0.0)
		if err != nil {
			t.Fatalf("RidgeRegression with alpha=0 failed: %v", err)
		}

		olsResult, err := LinearRegression(yArr, xArr)
		if err != nil {
			t.Fatalf("LinearRegression failed: %v", err)
		}

		// Coefficients should be very close
		if math.Abs(ridgeResult.Intercept-olsResult.Intercept) > 1e-10 {
			t.Errorf("Intercepts should match: ridge=%.10f, ols=%.10f", ridgeResult.Intercept, olsResult.Intercept)
		}

		for i := 0; i < len(ridgeResult.Coefficients); i++ {
			if math.Abs(ridgeResult.Coefficients[i]-olsResult.Coefficients[i]) > 1e-10 {
				t.Errorf("Coefficient %d should match: ridge=%.10f, ols=%.10f",
					i, ridgeResult.Coefficients[i], olsResult.Coefficients[i])
			}
		}
	})

	t.Run("Ridge regression parameter validation", func(t *testing.T) {
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{2, 4, 6, 8, 10}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		// Test negative alpha
		_, err := RidgeRegression(yArr, xArr, -0.1)
		if err == nil {
			t.Error("Expected error for negative alpha")
		}

		// Test nil arrays
		_, err = RidgeRegression(nil, xArr, 0.1)
		if err == nil {
			t.Error("Expected error for nil y array")
		}

		_, err = RidgeRegression(yArr, nil, 0.1)
		if err == nil {
			t.Error("Expected error for nil X array")
		}
	})
}

func TestLinearRegressionWithFallback(t *testing.T) {
	t.Run("Fallback to ridge for singular matrix", func(t *testing.T) {
		// Create perfectly correlated X variables that will cause OLS to fail
		x1 := []float64{1, 2, 3, 4, 5}
		x2 := []float64{2, 4, 6, 8, 10} // x2 = 2*x1, perfect correlation
		y := []float64{1, 2, 3, 4, 5}

		X := array.Empty(internal.Shape{5, 2}, internal.Float64)
		yArr, _ := array.FromSlice(y)

		for i := 0; i < 5; i++ {
			X.Set(x1[i], i, 0)
			X.Set(x2[i], i, 1)
		}

		// This should fall back to ridge regression
		result, err := LinearRegressionWithFallback(yArr, X, 1.0)
		if err != nil {
			t.Fatalf("LinearRegressionWithFallback failed: %v", err)
		}

		// Should produce finite coefficients
		if math.IsInf(result.Intercept, 0) || math.IsNaN(result.Intercept) {
			t.Errorf("Intercept should be finite, got %f", result.Intercept)
		}

		for i, coeff := range result.Coefficients {
			if math.IsInf(coeff, 0) || math.IsNaN(coeff) {
				t.Errorf("Coefficient %d should be finite, got %f", i, coeff)
			}
		}
	})

	t.Run("Use OLS when matrix is well-conditioned", func(t *testing.T) {
		// Create well-conditioned data that OLS should handle fine
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{3, 5, 7, 9, 11} // y = 2*x + 1

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		result, err := LinearRegressionWithFallback(yArr, xArr, 1.0)
		if err != nil {
			t.Fatalf("LinearRegressionWithFallback failed: %v", err)
		}

		// Should be very close to exact solution y = 2*x + 1
		if math.Abs(result.Intercept-1.0) > 1e-10 {
			t.Errorf("Expected intercept near 1.0, got %.10f", result.Intercept)
		}

		if math.Abs(result.Coefficients[0]-2.0) > 1e-10 {
			t.Errorf("Expected coefficient near 2.0, got %.10f", result.Coefficients[0])
		}
	})

	t.Run("Propagate non-singularity errors", func(t *testing.T) {
		// Test with mismatched array sizes (should not trigger fallback)
		x := []float64{1, 2, 3}
		y := []float64{1, 2, 3, 4, 5} // Different size

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		_, err := LinearRegressionWithFallback(yArr, xArr, 1.0)
		if err == nil {
			t.Error("Expected error for mismatched array sizes")
		}

		// Error should not mention ridge regression fallback
		if strings.Contains(err.Error(), "ridge regression") {
			t.Error("Should not fall back to ridge for non-singularity errors")
		}
	})
}

func TestRidgeRegressionMultipleVariables(t *testing.T) {
	t.Run("Multiple regression with ridge", func(t *testing.T) {
		// Create multiple regression data
		// y = 2*x1 + 3*x2 + 1 + noise
		x1 := []float64{1, 2, 3, 4, 5, 6}
		x2 := []float64{2, 1, 4, 3, 6, 5}
		y := []float64{7, 8, 16, 15, 23, 20} // Approximately 2*x1 + 3*x2 + 1

		X := array.Empty(internal.Shape{6, 2}, internal.Float64)
		yArr, _ := array.FromSlice(y)

		for i := 0; i < 6; i++ {
			X.Set(x1[i], i, 0)
			X.Set(x2[i], i, 1)
		}

		result, err := RidgeRegression(yArr, X, 0.1)
		if err != nil {
			t.Fatalf("Multiple ridge regression failed: %v", err)
		}

		// Should have 2 coefficients
		if len(result.Coefficients) != 2 {
			t.Fatalf("Expected 2 coefficients, got %d", len(result.Coefficients))
		}

		// Coefficients should be reasonably close to true values (2, 3), but shrunken by regularization
		// With ridge regression, we expect some shrinkage toward zero
		if math.Abs(result.Coefficients[0]-2.0) > 1.0 {
			t.Errorf("First coefficient should be reasonably close to 2.0 (with shrinkage), got %f", result.Coefficients[0])
		}

		if math.Abs(result.Coefficients[1]-3.0) > 1.0 {
			t.Errorf("Second coefficient should be reasonably close to 3.0 (with shrinkage), got %f", result.Coefficients[1])
		}

		// Intercept should be reasonable (ridge typically doesn't shrink intercept as much)
		if math.Abs(result.Intercept-1.0) > 2.0 {
			t.Errorf("Intercept should be reasonably close to 1.0, got %f", result.Intercept)
		}

		// R-squared should be decent
		if result.RSquared < 0.8 {
			t.Errorf("Expected reasonable R-squared, got %f", result.RSquared)
		}
	})
}

func BenchmarkRidgeRegression(b *testing.B) {
	// Create test data
	n := 1000
	x := make([]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = float64(i)
		y[i] = 2.0*float64(i) + 1.0 + 0.1*float64(i%10) // Add some noise
	}

	xArr, _ := array.FromSlice(x)
	yArr, _ := array.FromSlice(y)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := RidgeRegression(yArr, xArr, 0.1)
		if err != nil {
			b.Fatalf("Benchmark failed: %v", err)
		}
	}
}
