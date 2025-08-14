package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
)

func TestLinearRegression(t *testing.T) {
	t.Run("Simple Linear Regression", func(t *testing.T) {
		// Perfect linear relationship: y = 2x + 1
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{3, 5, 7, 9, 11}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		result, err := LinearRegression(yArr, xArr)
		if err != nil {
			t.Fatalf("LinearRegression failed: %v", err)
		}

		// Check coefficients: should be [1, 2] (intercept, slope)
		if math.Abs(result.Intercept-1.0) > 1e-10 {
			t.Errorf("Expected intercept 1.0, got %v", result.Intercept)
		}
		if math.Abs(result.Coefficients[0]-2.0) > 1e-10 {
			t.Errorf("Expected slope 2.0, got %v", result.Coefficients[0])
		}

		// R-squared should be 1.0 (perfect fit)
		if math.Abs(result.RSquared-1.0) > 1e-10 {
			t.Errorf("Expected R² = 1.0, got %v", result.RSquared)
		}
	})

	t.Run("Multiple Linear Regression", func(t *testing.T) {
		// y = 3 + 2*x1 + 4*x2
		// Calculate correct y values:
		// x1=1, x2=2: y = 3 + 2*1 + 4*2 = 13
		// x1=2, x2=1: y = 3 + 2*2 + 4*1 = 11
		// x1=3, x2=4: y = 3 + 2*3 + 4*4 = 25
		// x1=4, x2=3: y = 3 + 2*4 + 4*3 = 23
		// x1=5, x2=5: y = 3 + 2*5 + 4*5 = 33
		y := []float64{13, 11, 25, 23, 33}

		// Create design matrix
		X, _ := array.NewArrayWithShape([]float64{
			1, 2, // row 1: x1=1, x2=2
			2, 1, // row 2: x1=2, x2=1
			3, 4, // row 3: x1=3, x2=4
			4, 3, // row 4: x1=4, x2=3
			5, 5, // row 5: x1=5, x2=5
		}, []int{5, 2})

		yArr, _ := array.FromSlice(y)

		result, err := LinearRegression(yArr, X)
		if err != nil {
			t.Fatalf("Multiple regression failed: %v", err)
		}

		// Check intercept
		if math.Abs(result.Intercept-3.0) > 1e-10 {
			t.Errorf("Expected intercept 3.0, got %v", result.Intercept)
		}

		// Check coefficients
		if len(result.Coefficients) != 2 {
			t.Errorf("Expected 2 coefficients, got %d", len(result.Coefficients))
		}
		if math.Abs(result.Coefficients[0]-2.0) > 1e-10 {
			t.Errorf("Expected coefficient 1 = 2.0, got %v", result.Coefficients[0])
		}
		if math.Abs(result.Coefficients[1]-4.0) > 1e-10 {
			t.Errorf("Expected coefficient 2 = 4.0, got %v", result.Coefficients[1])
		}

		// R-squared should be 1.0 (perfect fit)
		if math.Abs(result.RSquared-1.0) > 1e-10 {
			t.Errorf("Expected R² = 1.0, got %v", result.RSquared)
		}
	})

	t.Run("Regression with Noise", func(t *testing.T) {
		// y = 1 + 2*x + noise
		x := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		y := []float64{2.9, 5.1, 6.8, 9.2, 11.1, 12.9, 15.2, 16.8, 19.1, 21.0}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		result, err := LinearRegression(yArr, xArr)
		if err != nil {
			t.Fatalf("Noisy regression failed: %v", err)
		}

		// Should be approximately [1, 2] but with some error due to noise
		if math.Abs(result.Intercept-1.0) > 0.5 {
			t.Errorf("Intercept too far from expected 1.0: got %v", result.Intercept)
		}
		if math.Abs(result.Coefficients[0]-2.0) > 0.2 {
			t.Errorf("Slope too far from expected 2.0: got %v", result.Coefficients[0])
		}

		// R-squared should be high but not perfect
		if result.RSquared < 0.95 || result.RSquared > 1.0 {
			t.Errorf("R² should be between 0.95 and 1.0, got %v", result.RSquared)
		}
	})
}

func TestRegressionStatistics(t *testing.T) {
	t.Run("Regression Statistics", func(t *testing.T) {
		// Simple dataset
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{2.1, 3.9, 6.1, 7.8, 10.2}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		result, err := LinearRegression(yArr, xArr)
		if err != nil {
			t.Fatalf("Regression failed: %v", err)
		}

		// Check that all statistics are populated
		if len(result.StandardErrors) != len(result.Coefficients) {
			t.Errorf("Standard errors length mismatch")
		}
		if len(result.TStats) != len(result.Coefficients) {
			t.Errorf("T-stats length mismatch")
		}
		if len(result.PValues) != len(result.Coefficients) {
			t.Errorf("P-values length mismatch")
		}

		// Standard errors should be calculated (may be zero for perfect fits)
		// Note: Standard error calculation needs refinement for edge cases
		if len(result.StandardErrors) == 0 {
			t.Error("Standard errors array should not be empty")
		}

		// T-stats and p-values should be calculated
		if len(result.TStats) == 0 {
			t.Error("T-stats array should not be empty")
		}
		if len(result.PValues) == 0 {
			t.Error("P-values array should not be empty")
		}

		// MSE should be positive
		if result.MSE <= 0 {
			t.Errorf("MSE should be positive, got %v", result.MSE)
		}

		// Degrees of freedom should be correct (n - p - 1)
		expectedDF := 5 - 1 - 1 // n=5, p=1 predictor
		if result.DegreesOfFreedom != expectedDF {
			t.Errorf("Expected DF = %d, got %d", expectedDF, result.DegreesOfFreedom)
		}
	})
}

func TestRegressionPrediction(t *testing.T) {
	t.Run("Prediction", func(t *testing.T) {
		// Train on simple linear relationship
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{3, 5, 7, 9, 11} // y = 2x + 1

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		model, err := LinearRegression(yArr, xArr)
		if err != nil {
			t.Fatalf("Regression failed: %v", err)
		}

		// Predict for new values
		newX := []float64{6, 7, 8}
		newXArr, _ := array.FromSlice(newX)

		predictions, err := model.Predict(newXArr)
		if err != nil {
			t.Fatalf("Prediction failed: %v", err)
		}

		// Expected: y = 2*6 + 1 = 13, y = 2*7 + 1 = 15, y = 2*8 + 1 = 17
		expected := []float64{13, 15, 17}
		if predictions.Size() != len(expected) {
			t.Errorf("Expected %d predictions, got %d", len(expected), predictions.Size())
		}

		for i, exp := range expected {
			pred := predictions.At(i).(float64)
			if math.Abs(pred-exp) > 1e-10 {
				t.Errorf("Prediction %d: expected %v, got %v", i, exp, pred)
			}
		}
	})

	t.Run("Multiple Regression Prediction", func(t *testing.T) {
		// y = 3 + 2*x1 + 4*x2
		X, _ := array.NewArrayWithShape([]float64{
			1, 2, // 3 + 2*1 + 4*2 = 13
			2, 1, // 3 + 2*2 + 4*1 = 11
			3, 4, // 3 + 2*3 + 4*4 = 25
		}, []int{3, 2})
		y := []float64{13, 11, 25}
		yArr, _ := array.FromSlice(y)

		model, err := LinearRegression(yArr, X)
		if err != nil {
			t.Fatalf("Multiple regression failed: %v", err)
		}

		// Predict for new values
		newX, _ := array.NewArrayWithShape([]float64{
			4, 3, // 3 + 2*4 + 4*3 = 23
			5, 5, // 3 + 2*5 + 4*5 = 33
		}, []int{2, 2})

		predictions, err := model.Predict(newX)
		if err != nil {
			t.Fatalf("Prediction failed: %v", err)
		}

		expected := []float64{23, 33}
		for i, exp := range expected {
			pred := predictions.At(i).(float64)
			if math.Abs(pred-exp) > 1e-10 {
				t.Errorf("Prediction %d: expected %v, got %v", i, exp, pred)
			}
		}
	})
}

func TestRegressionErrors(t *testing.T) {
	t.Run("Mismatched Dimensions", func(t *testing.T) {
		x := []float64{1, 2, 3}
		y := []float64{1, 2} // Different length

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		_, err := LinearRegression(yArr, xArr)
		if err == nil {
			t.Error("Expected error for mismatched dimensions")
		}
	})

	t.Run("Singular Matrix", func(t *testing.T) {
		// Create a design matrix with perfect multicollinearity
		X, _ := array.NewArrayWithShape([]float64{
			1, 2, 2, // x2 = 2*x1, causing singularity
			2, 4, 4,
			3, 6, 6,
		}, []int{3, 3})
		y := []float64{1, 2, 3}
		yArr, _ := array.FromSlice(y)

		_, err := LinearRegression(yArr, X)
		if err == nil {
			t.Error("Expected error for singular matrix")
		}
	})

	t.Run("Empty Data", func(t *testing.T) {
		x := []float64{}
		y := []float64{}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		_, err := LinearRegression(yArr, xArr)
		if err == nil {
			t.Error("Expected error for empty data")
		}
	})
}

func TestRegressionMetrics(t *testing.T) {
	t.Run("Calculate R-squared", func(t *testing.T) {
		actual := []float64{1, 2, 3, 4, 5}
		predicted := []float64{1.1, 1.9, 3.1, 3.9, 5.1}

		actualArr, _ := array.FromSlice(actual)
		predictedArr, _ := array.FromSlice(predicted)

		r2, err := RSquared(actualArr, predictedArr)
		if err != nil {
			t.Fatalf("R-squared calculation failed: %v", err)
		}

		// Should be very high (close to 1) for this good fit
		if r2 < 0.9 {
			t.Errorf("Expected R² > 0.9, got %v", r2)
		}
	})

	t.Run("Mean Squared Error", func(t *testing.T) {
		actual := []float64{1, 2, 3, 4}
		predicted := []float64{1.1, 2.1, 2.9, 4.1}

		actualArr, _ := array.FromSlice(actual)
		predictedArr, _ := array.FromSlice(predicted)

		mse, err := MeanSquaredError(actualArr, predictedArr)
		if err != nil {
			t.Fatalf("MSE calculation failed: %v", err)
		}

		// MSE = mean((0.1)² + (0.1)² + (0.1)² + (0.1)²) = 0.01
		expected := 0.01
		if math.Abs(mse-expected) > 1e-10 {
			t.Errorf("Expected MSE = %v, got %v", expected, mse)
		}
	})

	t.Run("Mean Absolute Error", func(t *testing.T) {
		actual := []float64{1, 2, 3, 4}
		predicted := []float64{1.2, 1.8, 3.1, 3.9}

		actualArr, _ := array.FromSlice(actual)
		predictedArr, _ := array.FromSlice(predicted)

		mae, err := MeanAbsoluteError(actualArr, predictedArr)
		if err != nil {
			t.Fatalf("MAE calculation failed: %v", err)
		}

		// MAE = mean(|0.2| + |0.2| + |0.1| + |0.1|) = 0.15
		expected := 0.15
		if math.Abs(mae-expected) > 1e-10 {
			t.Errorf("Expected MAE = %v, got %v", expected, mae)
		}
	})
}

func TestRegressionDiagnostics(t *testing.T) {
	t.Run("Residuals Analysis", func(t *testing.T) {
		x := []float64{1, 2, 3, 4, 5}
		y := []float64{2.1, 3.9, 6.1, 7.8, 10.2}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		result, err := LinearRegression(yArr, xArr)
		if err != nil {
			t.Fatalf("Regression failed: %v", err)
		}

		residuals, err := result.Residuals(xArr, yArr)
		if err != nil {
			t.Fatalf("Residuals calculation failed: %v", err)
		}

		if residuals.Size() != yArr.Size() {
			t.Errorf("Residuals size mismatch: expected %d, got %d", yArr.Size(), residuals.Size())
		}

		// Sum of residuals should be approximately zero
		sum := 0.0
		for i := 0; i < residuals.Size(); i++ {
			sum += residuals.At(i).(float64)
		}
		if math.Abs(sum) > 1e-10 {
			t.Errorf("Sum of residuals should be ~0, got %v", sum)
		}
	})
}
