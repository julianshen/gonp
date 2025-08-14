package math

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
)

func TestPolynomialCreation(t *testing.T) {
	t.Run("Create Polynomial from Coefficients", func(t *testing.T) {
		// Test creating polynomial p(x) = 2x² + 3x + 1
		coeffs := []float64{1, 3, 2} // constant, linear, quadratic
		poly := NewPolynomial(coeffs)

		if poly.Degree() != 2 {
			t.Errorf("Expected degree 2, got %d", poly.Degree())
		}

		expected := []float64{1, 3, 2}
		actual := poly.Coefficients()
		if len(actual) != len(expected) {
			t.Errorf("Expected %d coefficients, got %d", len(expected), len(actual))
		}

		for i, exp := range expected {
			if math.Abs(actual[i]-exp) > 1e-10 {
				t.Errorf("Coefficient %d: expected %f, got %f", i, exp, actual[i])
			}
		}
	})

	t.Run("Create Zero Polynomial", func(t *testing.T) {
		coeffs := []float64{0}
		poly := NewPolynomial(coeffs)

		if poly.Degree() != 0 {
			t.Errorf("Expected degree 0 for zero polynomial, got %d", poly.Degree())
		}
	})

	t.Run("Create Polynomial with Leading Zeros", func(t *testing.T) {
		coeffs := []float64{1, 2, 0, 0} // Should be trimmed to [1, 2]
		poly := NewPolynomial(coeffs)

		if poly.Degree() != 1 {
			t.Errorf("Expected degree 1 after trimming, got %d", poly.Degree())
		}
	})
}

func TestPolynomialEvaluation(t *testing.T) {
	t.Run("Evaluate Simple Polynomial", func(t *testing.T) {
		// p(x) = 2x² + 3x + 1
		coeffs := []float64{1, 3, 2}
		poly := NewPolynomial(coeffs)

		// Test p(0) = 1
		result := poly.Evaluate(0)
		if math.Abs(result-1) > 1e-10 {
			t.Errorf("p(0): expected 1, got %f", result)
		}

		// Test p(1) = 2 + 3 + 1 = 6
		result = poly.Evaluate(1)
		if math.Abs(result-6) > 1e-10 {
			t.Errorf("p(1): expected 6, got %f", result)
		}

		// Test p(2) = 8 + 6 + 1 = 15
		result = poly.Evaluate(2)
		if math.Abs(result-15) > 1e-10 {
			t.Errorf("p(2): expected 15, got %f", result)
		}
	})

	t.Run("Evaluate Array of Values", func(t *testing.T) {
		coeffs := []float64{1, 2} // p(x) = 2x + 1
		poly := NewPolynomial(coeffs)

		xValues, _ := array.FromSlice([]float64{0, 1, 2, 3})
		results, err := poly.EvaluateArray(xValues)
		if err != nil {
			t.Fatalf("EvaluateArray failed: %v", err)
		}

		expected := []float64{1, 3, 5, 7}
		for i, exp := range expected {
			actual := convertToFloat64(results.At(i))
			if math.Abs(actual-exp) > 1e-10 {
				t.Errorf("Index %d: expected %f, got %f", i, exp, actual)
			}
		}
	})
}

func TestPolynomialArithmetic(t *testing.T) {
	t.Run("Add Polynomials", func(t *testing.T) {
		// p1(x) = 2x + 1
		// p2(x) = x² + 3x
		// p1 + p2 = x² + 5x + 1
		p1 := NewPolynomial([]float64{1, 2})
		p2 := NewPolynomial([]float64{0, 3, 1})

		sum, err := p1.Add(p2)
		if err != nil {
			t.Fatalf("Add failed: %v", err)
		}

		expected := []float64{1, 5, 1}
		actual := sum.Coefficients()
		if len(actual) != len(expected) {
			t.Errorf("Expected %d coefficients, got %d", len(expected), len(actual))
		}

		for i, exp := range expected {
			if math.Abs(actual[i]-exp) > 1e-10 {
				t.Errorf("Coefficient %d: expected %f, got %f", i, exp, actual[i])
			}
		}
	})

	t.Run("Subtract Polynomials", func(t *testing.T) {
		// p1(x) = x² + 2x + 1
		// p2(x) = x + 1
		// p1 - p2 = x² + x
		p1 := NewPolynomial([]float64{1, 2, 1})
		p2 := NewPolynomial([]float64{1, 1})

		diff, err := p1.Subtract(p2)
		if err != nil {
			t.Fatalf("Subtract failed: %v", err)
		}

		expected := []float64{0, 1, 1}
		actual := diff.Coefficients()

		for i, exp := range expected {
			if math.Abs(actual[i]-exp) > 1e-10 {
				t.Errorf("Coefficient %d: expected %f, got %f", i, exp, actual[i])
			}
		}
	})

	t.Run("Multiply Polynomials", func(t *testing.T) {
		// p1(x) = x + 1
		// p2(x) = x + 2
		// p1 * p2 = x² + 3x + 2
		p1 := NewPolynomial([]float64{1, 1})
		p2 := NewPolynomial([]float64{2, 1})

		product, err := p1.Multiply(p2)
		if err != nil {
			t.Fatalf("Multiply failed: %v", err)
		}

		expected := []float64{2, 3, 1}
		actual := product.Coefficients()

		for i, exp := range expected {
			if math.Abs(actual[i]-exp) > 1e-10 {
				t.Errorf("Coefficient %d: expected %f, got %f", i, exp, actual[i])
			}
		}
	})

	t.Run("Scale Polynomial", func(t *testing.T) {
		// p(x) = 2x + 1, scale by 3: 6x + 3
		poly := NewPolynomial([]float64{1, 2})
		scaled := poly.Scale(3)

		expected := []float64{3, 6}
		actual := scaled.Coefficients()

		for i, exp := range expected {
			if math.Abs(actual[i]-exp) > 1e-10 {
				t.Errorf("Coefficient %d: expected %f, got %f", i, exp, actual[i])
			}
		}
	})
}

func TestPolynomialDerivative(t *testing.T) {
	t.Run("Derivative of Quadratic", func(t *testing.T) {
		// p(x) = 2x² + 3x + 1
		// p'(x) = 4x + 3
		poly := NewPolynomial([]float64{1, 3, 2})
		derivative := poly.Derivative()

		expected := []float64{3, 4}
		actual := derivative.Coefficients()

		for i, exp := range expected {
			if math.Abs(actual[i]-exp) > 1e-10 {
				t.Errorf("Coefficient %d: expected %f, got %f", i, exp, actual[i])
			}
		}
	})

	t.Run("Derivative of Constant", func(t *testing.T) {
		// p(x) = 5, p'(x) = 0
		poly := NewPolynomial([]float64{5})
		derivative := poly.Derivative()

		if derivative.Degree() != 0 {
			t.Errorf("Expected degree 0, got %d", derivative.Degree())
		}

		if math.Abs(derivative.Coefficients()[0]) > 1e-10 {
			t.Errorf("Expected zero derivative, got %f", derivative.Coefficients()[0])
		}
	})

	t.Run("Second Derivative", func(t *testing.T) {
		// p(x) = x³ + 2x² + 3x + 1
		// p'(x) = 3x² + 4x + 3
		// p''(x) = 6x + 4
		poly := NewPolynomial([]float64{1, 3, 2, 1})
		secondDerivative := poly.Derivative().Derivative()

		expected := []float64{4, 6}
		actual := secondDerivative.Coefficients()

		for i, exp := range expected {
			if math.Abs(actual[i]-exp) > 1e-10 {
				t.Errorf("Coefficient %d: expected %f, got %f", i, exp, actual[i])
			}
		}
	})
}

func TestPolynomialIntegral(t *testing.T) {
	t.Run("Integrate Quadratic", func(t *testing.T) {
		// p(x) = 2x² + 3x + 1
		// ∫p(x)dx = (2/3)x³ + (3/2)x² + x + C
		poly := NewPolynomial([]float64{1, 3, 2})
		integral := poly.Integral(0) // C = 0

		expected := []float64{0, 1, 1.5, 2.0 / 3.0}
		actual := integral.Coefficients()

		for i, exp := range expected {
			if math.Abs(actual[i]-exp) > 1e-10 {
				t.Errorf("Coefficient %d: expected %f, got %f", i, exp, actual[i])
			}
		}
	})

	t.Run("Definite Integral", func(t *testing.T) {
		// p(x) = x², integrate from 0 to 2
		// ∫₀² x² dx = [x³/3]₀² = 8/3
		poly := NewPolynomial([]float64{0, 0, 1})
		result, err := poly.DefiniteIntegral(0, 2)
		if err != nil {
			t.Fatalf("DefiniteIntegral failed: %v", err)
		}

		expected := 8.0 / 3.0
		if math.Abs(result-expected) > 1e-10 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})
}

func TestPolynomialFitting(t *testing.T) {
	t.Run("Linear Fit", func(t *testing.T) {
		// Fit to y = 2x + 1
		xData := []float64{0, 1, 2, 3}
		yData := []float64{1, 3, 5, 7}

		xArr, _ := array.FromSlice(xData)
		yArr, _ := array.FromSlice(yData)

		poly, err := PolynomialFit(xArr, yArr, 1)
		if err != nil {
			t.Fatalf("PolynomialFit failed: %v", err)
		}

		// Check coefficients
		coeffs := poly.Coefficients()
		if math.Abs(coeffs[0]-1) > 1e-10 { // constant term
			t.Errorf("Expected constant term 1, got %f", coeffs[0])
		}
		if math.Abs(coeffs[1]-2) > 1e-10 { // linear term
			t.Errorf("Expected linear term 2, got %f", coeffs[1])
		}
	})

	t.Run("Quadratic Fit", func(t *testing.T) {
		// Fit to y = x² + 2x + 1
		xData := []float64{0, 1, 2, 3, 4}
		yData := []float64{1, 4, 9, 16, 25}

		xArr, _ := array.FromSlice(xData)
		yArr, _ := array.FromSlice(yData)

		poly, err := PolynomialFit(xArr, yArr, 2)
		if err != nil {
			t.Fatalf("PolynomialFit failed: %v", err)
		}

		// Test the fit by evaluating at test points
		for i, x := range xData {
			predicted := poly.Evaluate(x)
			if math.Abs(predicted-yData[i]) > 1e-8 {
				t.Errorf("At x=%f: expected %f, got %f", x, yData[i], predicted)
			}
		}
	})

	t.Run("Overfit Warning", func(t *testing.T) {
		// Try to fit degree 5 to 3 points (should fail)
		xData := []float64{0, 1, 2}
		yData := []float64{1, 2, 3}

		xArr, _ := array.FromSlice(xData)
		yArr, _ := array.FromSlice(yData)

		_, err := PolynomialFit(xArr, yArr, 5)
		if err == nil {
			t.Error("Expected error for overfitting, got none")
		}
	})
}

func TestPolynomialRoots(t *testing.T) {
	t.Run("Linear Roots", func(t *testing.T) {
		// p(x) = 2x - 4, root at x = 2
		poly := NewPolynomial([]float64{-4, 2})
		roots, err := poly.FindRoots()
		if err != nil {
			t.Fatalf("FindRoots failed: %v", err)
		}

		if len(roots) != 1 {
			t.Errorf("Expected 1 root, got %d", len(roots))
		}

		if math.Abs(roots[0]-2) > 1e-10 {
			t.Errorf("Expected root 2, got %f", roots[0])
		}
	})

	t.Run("Quadratic Roots", func(t *testing.T) {
		// p(x) = x² - 5x + 6 = (x-2)(x-3), roots at x = 2, 3
		poly := NewPolynomial([]float64{6, -5, 1})
		roots, err := poly.FindRoots()
		if err != nil {
			t.Fatalf("FindRoots failed: %v", err)
		}

		if len(roots) != 2 {
			t.Errorf("Expected 2 roots, got %d", len(roots))
		}

		// Sort roots for comparison
		if roots[0] > roots[1] {
			roots[0], roots[1] = roots[1], roots[0]
		}

		if math.Abs(roots[0]-2) > 1e-10 {
			t.Errorf("Expected first root 2, got %f", roots[0])
		}
		if math.Abs(roots[1]-3) > 1e-10 {
			t.Errorf("Expected second root 3, got %f", roots[1])
		}
	})

	t.Run("No Real Roots", func(t *testing.T) {
		// p(x) = x² + 1, no real roots
		poly := NewPolynomial([]float64{1, 0, 1})
		roots, err := poly.FindRoots()
		if err != nil {
			t.Fatalf("FindRoots failed: %v", err)
		}

		if len(roots) != 0 {
			t.Errorf("Expected 0 real roots, got %d", len(roots))
		}
	})
}

func TestPolynomialSpecialOperations(t *testing.T) {
	t.Run("Composition", func(t *testing.T) {
		// p(x) = x + 1, q(x) = 2x
		// p(q(x)) = 2x + 1
		p := NewPolynomial([]float64{1, 1})
		q := NewPolynomial([]float64{0, 2})

		composition, err := p.Compose(q)
		if err != nil {
			t.Fatalf("Compose failed: %v", err)
		}

		expected := []float64{1, 2}
		actual := composition.Coefficients()

		for i, exp := range expected {
			if math.Abs(actual[i]-exp) > 1e-10 {
				t.Errorf("Coefficient %d: expected %f, got %f", i, exp, actual[i])
			}
		}
	})

	t.Run("Division", func(t *testing.T) {
		// Divide x³ - 1 by x - 1
		// Result: quotient = x² + x + 1, remainder = 0
		dividend := NewPolynomial([]float64{-1, 0, 0, 1})
		divisor := NewPolynomial([]float64{-1, 1})

		quotient, remainder, err := dividend.Divide(divisor)
		if err != nil {
			t.Fatalf("Divide failed: %v", err)
		}

		// Check quotient: x² + x + 1
		expectedQuotient := []float64{1, 1, 1}
		actualQuotient := quotient.Coefficients()

		for i, exp := range expectedQuotient {
			if math.Abs(actualQuotient[i]-exp) > 1e-10 {
				t.Errorf("Quotient coefficient %d: expected %f, got %f", i, exp, actualQuotient[i])
			}
		}

		// Check remainder: 0
		if remainder.Degree() != 0 || math.Abs(remainder.Coefficients()[0]) > 1e-10 {
			t.Errorf("Expected zero remainder, got degree %d with value %f",
				remainder.Degree(), remainder.Coefficients()[0])
		}
	})

	t.Run("GCD", func(t *testing.T) {
		// GCD of (x² - 1) and (x - 1) should be (x - 1)
		p1 := NewPolynomial([]float64{-1, 0, 1}) // x² - 1
		p2 := NewPolynomial([]float64{-1, 1})    // x - 1

		gcd, err := p1.GCD(p2)
		if err != nil {
			t.Fatalf("GCD failed: %v", err)
		}

		// Result should be x - 1 (up to scaling)
		if gcd.Degree() != 1 {
			t.Errorf("Expected GCD degree 1, got %d", gcd.Degree())
		}

		// Normalize by leading coefficient
		coeffs := gcd.Coefficients()
		if coeffs[1] != 0 {
			scale := 1.0 / coeffs[1]
			for i := range coeffs {
				coeffs[i] *= scale
			}
		}

		if math.Abs(coeffs[0]+1) > 1e-10 || math.Abs(coeffs[1]-1) > 1e-10 {
			t.Errorf("Expected GCD coefficients [-1, 1], got %v", coeffs)
		}
	})
}

func TestPolynomialUtilities(t *testing.T) {
	t.Run("String Representation", func(t *testing.T) {
		poly := NewPolynomial([]float64{1, -2, 3})
		str := poly.String()

		// Should contain terms like 3x², -2x, +1
		if str == "" {
			t.Error("String representation should not be empty")
		}
	})

	t.Run("Equal Polynomials", func(t *testing.T) {
		p1 := NewPolynomial([]float64{1, 2, 3})
		p2 := NewPolynomial([]float64{1, 2, 3})
		p3 := NewPolynomial([]float64{1, 2, 4})

		if !p1.Equal(p2) {
			t.Error("Identical polynomials should be equal")
		}

		if p1.Equal(p3) {
			t.Error("Different polynomials should not be equal")
		}
	})

	t.Run("Copy Polynomial", func(t *testing.T) {
		original := NewPolynomial([]float64{1, 2, 3})
		copy := original.Copy()

		if !original.Equal(copy) {
			t.Error("Copy should be equal to original")
		}

		// Modify copy
		copy = copy.Scale(2)
		if original.Equal(copy) {
			t.Error("Modified copy should not equal original")
		}
	})
}

func BenchmarkPolynomialOperations(b *testing.B) {
	poly1 := NewPolynomial([]float64{1, 2, 3, 4, 5})
	poly2 := NewPolynomial([]float64{5, 4, 3, 2, 1})

	b.Run("Evaluation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = poly1.Evaluate(2.5)
		}
	})

	b.Run("Addition", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = poly1.Add(poly2)
		}
	})

	b.Run("Multiplication", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = poly1.Multiply(poly2)
		}
	})

	b.Run("Derivative", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = poly1.Derivative()
		}
	})
}
