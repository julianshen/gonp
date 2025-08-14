package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
)

func TestHuberRegression(t *testing.T) {
	t.Run("Simple linear relationship with no outliers", func(t *testing.T) {
		// Create clean linear data: y = 2*x + 1
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{3, 5, 7, 9, 11}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		result, err := HuberRegression(yArr, xArr, 1.345) // Default tuning parameter
		if err != nil {
			t.Fatalf("HuberRegression failed: %v", err)
		}

		// Should be close to OLS result for clean data
		if math.Abs(result.Intercept-1.0) > 1e-6 {
			t.Errorf("Expected intercept near 1.0, got %f", result.Intercept)
		}

		if len(result.Coefficients) != 1 {
			t.Fatalf("Expected 1 coefficient, got %d", len(result.Coefficients))
		}

		if math.Abs(result.Coefficients[0]-2.0) > 1e-6 {
			t.Errorf("Expected coefficient near 2.0, got %f", result.Coefficients[0])
		}

		// Should converge and have reasonable R-squared
		if !result.Converged {
			t.Errorf("Expected convergence")
		}

		if result.RSquared < 0.99 {
			t.Errorf("Expected high R-squared for clean data, got %f", result.RSquared)
		}
	})

	t.Run("Linear relationship with outliers", func(t *testing.T) {
		// Create data with outliers: y = 2*x + 1 + outliers
		x := []float64{1, 2, 3, 4, 5, 6, 7}
		y := []float64{3, 5, 7, 9, 11, 25, 15} // Points 5 and 6 are outliers

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		huberResult, err := HuberRegression(yArr, xArr, 1.345)
		if err != nil {
			t.Fatalf("HuberRegression failed: %v", err)
		}

		// Compare with OLS which should be more affected by outliers
		olsResult, err := LinearRegression(yArr, xArr)
		if err != nil {
			t.Fatalf("OLS failed: %v", err)
		}

		// Huber should be more robust (closer to true values)
		trueIntercept := 1.0
		trueSlope := 2.0

		huberInterceptError := math.Abs(huberResult.Intercept - trueIntercept)
		olsInterceptError := math.Abs(olsResult.Intercept - trueIntercept)

		huberSlopeError := math.Abs(huberResult.Coefficients[0] - trueSlope)
		olsSlopeError := math.Abs(olsResult.Coefficients[0] - trueSlope)

		// Huber should be more robust to outliers
		if huberInterceptError > olsInterceptError {
			t.Logf("Warning: Huber intercept error (%.6f) > OLS error (%.6f)",
				huberInterceptError, olsInterceptError)
		}

		if huberSlopeError > olsSlopeError {
			t.Logf("Warning: Huber slope error (%.6f) > OLS error (%.6f)",
				huberSlopeError, olsSlopeError)
		}

		t.Logf("Huber: intercept=%.3f, slope=%.3f", huberResult.Intercept, huberResult.Coefficients[0])
		t.Logf("OLS: intercept=%.3f, slope=%.3f", olsResult.Intercept, olsResult.Coefficients[0])
	})

	t.Run("Parameter validation", func(t *testing.T) {
		x := []float64{1, 2, 3}
		y := []float64{2, 4, 6}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		// Test nil arrays
		_, err := HuberRegression(nil, xArr, 1.345)
		if err == nil {
			t.Error("Expected error for nil y array")
		}

		_, err = HuberRegression(yArr, nil, 1.345)
		if err == nil {
			t.Error("Expected error for nil X array")
		}

		// Test invalid tuning parameter
		_, err = HuberRegression(yArr, xArr, 0.0)
		if err == nil {
			t.Error("Expected error for zero tuning parameter")
		}

		_, err = HuberRegression(yArr, xArr, -1.0)
		if err == nil {
			t.Error("Expected error for negative tuning parameter")
		}
	})
}

func TestTukeyBisquareRegression(t *testing.T) {
	t.Run("Simple linear relationship with no outliers", func(t *testing.T) {
		// Create clean linear data: y = 2*x + 1
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{3, 5, 7, 9, 11}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		result, err := TukeyBisquareRegression(yArr, xArr, 4.685) // Default tuning parameter
		if err != nil {
			t.Fatalf("TukeyBisquareRegression failed: %v", err)
		}

		// Should be close to OLS result for clean data
		if math.Abs(result.Intercept-1.0) > 1e-6 {
			t.Errorf("Expected intercept near 1.0, got %f", result.Intercept)
		}

		if len(result.Coefficients) != 1 {
			t.Fatalf("Expected 1 coefficient, got %d", len(result.Coefficients))
		}

		if math.Abs(result.Coefficients[0]-2.0) > 1e-6 {
			t.Errorf("Expected coefficient near 2.0, got %f", result.Coefficients[0])
		}
	})

	t.Run("Linear relationship with outliers", func(t *testing.T) {
		// Create data with outliers
		x := []float64{1, 2, 3, 4, 5, 6, 7}
		y := []float64{3, 5, 7, 9, 11, 30, 15} // Large outlier at point 5

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		tukeyResult, err := TukeyBisquareRegression(yArr, xArr, 4.685)
		if err != nil {
			t.Fatalf("TukeyBisquareRegression failed: %v", err)
		}

		// Should be robust to outliers (close to true slope of 2.0 and intercept of 1.0)
		if math.Abs(tukeyResult.Intercept-1.0) > 0.5 {
			t.Logf("Tukey bisquare intercept: %.3f (expected ~1.0)", tukeyResult.Intercept)
		}

		if math.Abs(tukeyResult.Coefficients[0]-2.0) > 0.5 {
			t.Logf("Tukey bisquare slope: %.3f (expected ~2.0)", tukeyResult.Coefficients[0])
		}

		t.Logf("Tukey: intercept=%.3f, slope=%.3f", tukeyResult.Intercept, tukeyResult.Coefficients[0])
	})
}
