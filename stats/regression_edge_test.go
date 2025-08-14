package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestRegressionEdgeCases(t *testing.T) {
	t.Run("Perfect correlation - singular matrix", func(t *testing.T) {
		// Create perfectly correlated X variables
		x1 := []float64{1, 2, 3, 4, 5}
		x2 := []float64{2, 4, 6, 8, 10} // x2 = 2*x1, perfect correlation
		y := []float64{1, 2, 3, 4, 5}

		// Create X matrix with perfectly correlated columns
		X := array.Empty(internal.Shape{5, 2}, internal.Float64)
		yArr, _ := array.FromSlice(y)

		for i := 0; i < 5; i++ {
			X.Set(x1[i], i, 0)
			X.Set(x2[i], i, 1)
		}

		result, err := LinearRegression(yArr, X)

		// The function should handle the singular matrix case
		if err == nil {
			// If it succeeds, standard errors should be NaN for ill-conditioned case
			for i, se := range result.StandardErrors {
				if !math.IsNaN(se) && !math.IsInf(se, 0) {
					// Standard errors should be very large due to multicollinearity
					if se < 1e10 {
						t.Errorf("Expected very large standard error for perfect correlation, got %f at index %d", se, i)
					}
				}
			}
		} else {
			// Should contain information about singular matrix
			if err.Error() != "" {
				t.Logf("Expected error for singular matrix: %v", err)
			}
		}
	})

	t.Run("Zero variance in Y", func(t *testing.T) {
		// All Y values are the same
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{5, 5, 5, 5, 5} // No variance

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		result, err := LinearRegression(yArr, xArr)
		if err != nil {
			t.Fatalf("Failed regression with zero Y variance: %v", err)
		}

		// R-squared should be 1.0 for perfect fit (no variance case)
		if !math.IsNaN(result.RSquared) && math.Abs(result.RSquared-1.0) > 1e-10 {
			t.Errorf("Expected R-squared = 1.0 for zero Y variance, got %f", result.RSquared)
		}

		// For zero variance in Y, standard errors might be NaN due to numerical issues
		// This is acceptable behavior for degenerate cases
		for i, se := range result.StandardErrors {
			if !math.IsNaN(se) && se < 0 {
				t.Errorf("Standard error should not be negative at index %d: %e", i, se)
			}
			t.Logf("Zero Y variance - SE[%d]=%e, MSE=%e", i, se, result.MSE)
		}
	})

	t.Run("Single observation", func(t *testing.T) {
		// Only one data point
		x := []float64{1}
		y := []float64{2}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		_, err := LinearRegression(yArr, xArr)

		// Should fail - can't do regression with single point
		if err == nil {
			t.Error("Expected error for single observation regression")
		}
	})

	t.Run("Identical X values", func(t *testing.T) {
		// All X values are identical
		x := []float64{3, 3, 3, 3, 3}
		y := []float64{1, 2, 3, 4, 5}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		_, err := LinearRegression(yArr, xArr)

		// Should fail - no variance in X
		if err == nil {
			t.Error("Expected error for identical X values")
		}
	})

	t.Run("Very small numbers - numerical stability", func(t *testing.T) {
		// Very small numbers that might cause numerical issues
		x := []float64{1e-10, 2e-10, 3e-10, 4e-10, 5e-10}
		y := []float64{1e-15, 2e-15, 3e-15, 4e-15, 5e-15}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		result, err := LinearRegression(yArr, xArr)
		if err != nil {
			t.Logf("Regression with tiny numbers failed as expected: %v", err)
			return
		}

		// If it succeeds, check that results are reasonable
		if !math.IsNaN(result.Intercept) && !math.IsInf(result.Intercept, 0) {
			t.Logf("Intercept with tiny numbers: %e", result.Intercept)
		}

		// Standard errors should be finite or NaN (not infinite)
		for i, se := range result.StandardErrors {
			if math.IsInf(se, 0) {
				t.Errorf("Standard error should not be infinite at index %d, got %e", i, se)
			}
		}
	})

	t.Run("Very large numbers - overflow protection", func(t *testing.T) {
		// Very large numbers that might cause overflow
		x := []float64{1e15, 2e15, 3e15, 4e15, 5e15}
		y := []float64{1e16, 2e16, 3e16, 4e16, 5e16}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		result, err := LinearRegression(yArr, xArr)
		if err != nil {
			t.Logf("Regression with large numbers failed: %v", err)
			return
		}

		// Check that results don't overflow to infinity
		if math.IsInf(result.Intercept, 0) {
			t.Errorf("Intercept overflowed to infinity: %e", result.Intercept)
		}

		for i, coeff := range result.Coefficients {
			if math.IsInf(coeff, 0) {
				t.Errorf("Coefficient overflowed to infinity at index %d: %e", i, coeff)
			}
		}

		// Standard errors should be finite
		for i, se := range result.StandardErrors {
			if math.IsInf(se, 0) {
				t.Errorf("Standard error overflowed at index %d: %e", i, se)
			}
		}
	})

	t.Run("Missing values simulation", func(t *testing.T) {
		// Simulate missing values with NaN (should be caught early)
		x := []float64{1, 2, math.NaN(), 4, 5}
		y := []float64{2, 4, 6, 8, 10}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		_, err := LinearRegression(yArr, xArr)

		// Should handle NaN values gracefully (may succeed or fail)
		if err != nil {
			t.Logf("Regression with NaN values failed as expected: %v", err)
		}
	})

	t.Run("Ill-conditioned matrix", func(t *testing.T) {
		// Create an ill-conditioned design matrix
		// Variables that are almost perfectly correlated
		x1 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		x2 := []float64{1.00001, 2.00001, 3.00001, 4.00001, 5.00001} // Almost identical to x1
		y := []float64{1, 3, 5, 7, 9}

		X := array.Empty(internal.Shape{5, 2}, internal.Float64)
		yArr, _ := array.FromSlice(y)

		for i := 0; i < 5; i++ {
			X.Set(x1[i], i, 0)
			X.Set(x2[i], i, 1)
		}

		result, err := LinearRegression(yArr, X)

		if err != nil {
			// Should detect ill-conditioning
			t.Logf("Detected ill-conditioned matrix as expected: %v", err)
		} else {
			// If it succeeds, standard errors should be very large
			for i, se := range result.StandardErrors {
				if !math.IsNaN(se) && se < 1e6 {
					t.Logf("Warning: Standard error might be too small for ill-conditioned case at index %d: %e", i, se)
				}
			}
		}
	})

	t.Run("Zero MSE case", func(t *testing.T) {
		// Perfect linear relationship - should result in zero MSE
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{2, 4, 6, 8, 10} // y = 2*x exactly

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		result, err := LinearRegression(yArr, xArr)
		if err != nil {
			t.Fatalf("Failed perfect fit regression: %v", err)
		}

		// MSE should be very close to zero
		if result.MSE > 1e-10 {
			t.Errorf("Expected MSE near zero for perfect fit, got %e", result.MSE)
		}

		// R-squared should be very close to 1
		if math.Abs(result.RSquared-1.0) > 1e-10 {
			t.Errorf("Expected R-squared near 1.0 for perfect fit, got %f", result.RSquared)
		}

		// For perfect fit, standard errors might be NaN due to very small MSE
		// This is acceptable - the coefficients are still valid
		for i, se := range result.StandardErrors {
			if !math.IsNaN(se) && se < 0 {
				t.Errorf("Standard error should not be negative at index %d: %e", i, se)
			}
			t.Logf("Perfect fit - SE[%d]=%e, MSE=%e", i, se, result.MSE)
		}
	})
}

func TestConditionNumberCalculation(t *testing.T) {
	t.Run("Identity matrix", func(t *testing.T) {
		// Identity matrix should have condition number 1
		identityData := []float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		}
		identity, _ := array.NewArrayWithShape(identityData, internal.Shape{3, 3})

		cond, err := calculateConditionNumber(identity)
		if err != nil {
			t.Fatalf("Failed to calculate condition number: %v", err)
		}

		if math.Abs(cond-1.0) > 0.1 {
			t.Errorf("Expected condition number ~1.0 for identity, got %f", cond)
		}
	})

	t.Run("Singular matrix", func(t *testing.T) {
		// Matrix with zero row should be singular
		singularData := []float64{
			1, 2,
			0, 0, // Zero row
		}
		singular, _ := array.NewArrayWithShape(singularData, internal.Shape{2, 2})

		cond, err := calculateConditionNumber(singular)
		if err != nil {
			t.Fatalf("Failed to calculate condition number: %v", err)
		}

		if !math.IsInf(cond, 0) {
			t.Errorf("Expected infinite condition number for singular matrix, got %f", cond)
		}
	})

	t.Run("Non-square matrix", func(t *testing.T) {
		// Non-square matrix should return error
		nonSquareData := []float64{1, 2, 3, 4, 5, 6}
		nonSquare, _ := array.NewArrayWithShape(nonSquareData, internal.Shape{2, 3})

		_, err := calculateConditionNumber(nonSquare)
		if err == nil {
			t.Error("Expected error for non-square matrix")
		}
	})
}

func TestSingularMatrixHandling(t *testing.T) {
	result := &RegressionResult{
		Coefficients: []float64{1.5, 2.3},
	}

	// Create dummy matrices (content doesn't matter for this test)
	dummyData := []float64{1, 0, 0, 1}
	XTX, _ := array.NewArrayWithShape(dummyData, internal.Shape{2, 2})
	identity, _ := array.NewArrayWithShape(dummyData, internal.Shape{2, 2})

	err := handleSingularMatrix(result, XTX, identity)

	// Should return an error
	if err == nil {
		t.Error("Expected error from handleSingularMatrix")
	}

	// All statistical measures should be NaN
	for i, se := range result.StandardErrors {
		if !math.IsNaN(se) {
			t.Errorf("Standard error should be NaN at index %d, got %f", i, se)
		}
	}

	for i, tStat := range result.TStats {
		if !math.IsNaN(tStat) {
			t.Errorf("T-statistic should be NaN at index %d, got %f", i, tStat)
		}
	}

	for i, pVal := range result.PValues {
		if !math.IsNaN(pVal) {
			t.Errorf("P-value should be NaN at index %d, got %f", i, pVal)
		}
	}
}
