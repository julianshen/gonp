package math

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Polynomial represents a polynomial with floating-point coefficients
// Coefficients are stored in ascending order of powers: [c₀, c₁, c₂, ...] for c₀ + c₁x + c₂x² + ...
type Polynomial struct {
	coeffs []float64
}

// NewPolynomial creates a new polynomial from coefficients
// Coefficients are in ascending order of powers: [constant, linear, quadratic, ...]
func NewPolynomial(coefficients []float64) *Polynomial {
	if len(coefficients) == 0 {
		return &Polynomial{coeffs: []float64{0}}
	}

	// Remove leading zeros
	coeffs := make([]float64, len(coefficients))
	copy(coeffs, coefficients)

	// Find the highest non-zero coefficient
	lastNonZero := len(coeffs) - 1
	for lastNonZero > 0 && math.Abs(coeffs[lastNonZero]) < 1e-15 {
		lastNonZero--
	}

	// Keep at least one coefficient (even if it's zero)
	if lastNonZero == 0 && math.Abs(coeffs[0]) < 1e-15 {
		return &Polynomial{coeffs: []float64{0}}
	}

	return &Polynomial{coeffs: coeffs[:lastNonZero+1]}
}

// Degree returns the degree of the polynomial
func (p *Polynomial) Degree() int {
	return len(p.coeffs) - 1
}

// Coefficients returns a copy of the polynomial coefficients
func (p *Polynomial) Coefficients() []float64 {
	result := make([]float64, len(p.coeffs))
	copy(result, p.coeffs)
	return result
}

// Evaluate evaluates the polynomial at a given point using Horner's method
func (p *Polynomial) Evaluate(x float64) float64 {
	if len(p.coeffs) == 0 {
		return 0
	}

	// Horner's method: start from highest degree
	result := p.coeffs[len(p.coeffs)-1]
	for i := len(p.coeffs) - 2; i >= 0; i-- {
		result = result*x + p.coeffs[i]
	}

	return result
}

// EvaluateArray evaluates the polynomial at multiple points
func (p *Polynomial) EvaluateArray(xValues *array.Array) (*array.Array, error) {
	if xValues == nil {
		return nil, internal.NewValidationErrorWithMsg("Polynomial.EvaluateArray", "x values cannot be nil")
	}

	flatX := xValues.Flatten()
	results := make([]float64, flatX.Size())

	for i := 0; i < flatX.Size(); i++ {
		x := convertToFloat64(flatX.At(i))
		results[i] = p.Evaluate(x)
	}

	result, err := array.FromSlice(results)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// Add adds two polynomials
func (p *Polynomial) Add(other *Polynomial) (*Polynomial, error) {
	if other == nil {
		return nil, errors.New("cannot add nil polynomial")
	}

	maxLen := len(p.coeffs)
	if len(other.coeffs) > maxLen {
		maxLen = len(other.coeffs)
	}

	result := make([]float64, maxLen)

	for i := 0; i < maxLen; i++ {
		var coeff1, coeff2 float64
		if i < len(p.coeffs) {
			coeff1 = p.coeffs[i]
		}
		if i < len(other.coeffs) {
			coeff2 = other.coeffs[i]
		}
		result[i] = coeff1 + coeff2
	}

	return NewPolynomial(result), nil
}

// Subtract subtracts another polynomial from this one
func (p *Polynomial) Subtract(other *Polynomial) (*Polynomial, error) {
	if other == nil {
		return nil, errors.New("cannot subtract nil polynomial")
	}

	maxLen := len(p.coeffs)
	if len(other.coeffs) > maxLen {
		maxLen = len(other.coeffs)
	}

	result := make([]float64, maxLen)

	for i := 0; i < maxLen; i++ {
		var coeff1, coeff2 float64
		if i < len(p.coeffs) {
			coeff1 = p.coeffs[i]
		}
		if i < len(other.coeffs) {
			coeff2 = other.coeffs[i]
		}
		result[i] = coeff1 - coeff2
	}

	return NewPolynomial(result), nil
}

// Multiply multiplies two polynomials
func (p *Polynomial) Multiply(other *Polynomial) (*Polynomial, error) {
	if other == nil {
		return nil, errors.New("cannot multiply by nil polynomial")
	}

	if len(p.coeffs) == 1 && p.coeffs[0] == 0 {
		return NewPolynomial([]float64{0}), nil
	}
	if len(other.coeffs) == 1 && other.coeffs[0] == 0 {
		return NewPolynomial([]float64{0}), nil
	}

	resultLen := len(p.coeffs) + len(other.coeffs) - 1
	result := make([]float64, resultLen)

	for i := 0; i < len(p.coeffs); i++ {
		for j := 0; j < len(other.coeffs); j++ {
			result[i+j] += p.coeffs[i] * other.coeffs[j]
		}
	}

	return NewPolynomial(result), nil
}

// Scale multiplies the polynomial by a scalar
func (p *Polynomial) Scale(factor float64) *Polynomial {
	result := make([]float64, len(p.coeffs))
	for i, coeff := range p.coeffs {
		result[i] = coeff * factor
	}
	return NewPolynomial(result)
}

// Derivative returns the derivative of the polynomial
func (p *Polynomial) Derivative() *Polynomial {
	if len(p.coeffs) <= 1 {
		return NewPolynomial([]float64{0})
	}

	result := make([]float64, len(p.coeffs)-1)
	for i := 1; i < len(p.coeffs); i++ {
		result[i-1] = float64(i) * p.coeffs[i]
	}

	return NewPolynomial(result)
}

// Integral returns the integral of the polynomial with given constant of integration
func (p *Polynomial) Integral(constant float64) *Polynomial {
	result := make([]float64, len(p.coeffs)+1)
	result[0] = constant

	for i, coeff := range p.coeffs {
		result[i+1] = coeff / float64(i+1)
	}

	return NewPolynomial(result)
}

// DefiniteIntegral computes the definite integral from a to b
func (p *Polynomial) DefiniteIntegral(a, b float64) (float64, error) {
	antiderivative := p.Integral(0)
	return antiderivative.Evaluate(b) - antiderivative.Evaluate(a), nil
}

// FindRoots finds the real roots of the polynomial
func (p *Polynomial) FindRoots() ([]float64, error) {
	degree := p.Degree()

	if degree == 0 {
		if math.Abs(p.coeffs[0]) < 1e-15 {
			return nil, errors.New("constant zero polynomial has infinitely many roots")
		}
		return []float64{}, nil // No roots for non-zero constant
	}

	if degree == 1 {
		// Linear: ax + b = 0 => x = -b/a
		if math.Abs(p.coeffs[1]) < 1e-15 {
			return nil, errors.New("degenerate linear polynomial")
		}
		root := -p.coeffs[0] / p.coeffs[1]
		return []float64{root}, nil
	}

	if degree == 2 {
		// Quadratic formula
		return p.solveQuadratic()
	}

	// For higher degrees, use numerical methods (simplified implementation)
	return p.findRootsNumerical()
}

// solveQuadratic solves quadratic equations using the quadratic formula
func (p *Polynomial) solveQuadratic() ([]float64, error) {
	if len(p.coeffs) < 3 {
		return nil, errors.New("not a quadratic polynomial")
	}

	a := p.coeffs[2]
	b := p.coeffs[1]
	c := p.coeffs[0]

	if math.Abs(a) < 1e-15 {
		return nil, errors.New("degenerate quadratic polynomial")
	}

	discriminant := b*b - 4*a*c

	if discriminant < 0 {
		return []float64{}, nil // No real roots
	}

	if math.Abs(discriminant) < 1e-15 {
		// One root (repeated)
		root := -b / (2 * a)
		return []float64{root}, nil
	}

	// Two distinct roots
	sqrtD := math.Sqrt(discriminant)
	root1 := (-b + sqrtD) / (2 * a)
	root2 := (-b - sqrtD) / (2 * a)

	if root1 > root2 {
		return []float64{root2, root1}, nil
	}
	return []float64{root1, root2}, nil
}

// findRootsNumerical finds roots using numerical methods (simplified)
func (p *Polynomial) findRootsNumerical() ([]float64, error) {
	// This is a simplified implementation using the Newton-Raphson method
	// In practice, you'd use more sophisticated methods like Durand-Kerner or Jenkins-Traub

	var roots []float64
	derivative := p.Derivative()

	// Try different starting points
	candidates := []float64{-10, -1, 0, 1, 10}
	tolerance := 1e-10
	maxIterations := 100

	for _, start := range candidates {
		root, found := p.newtonRaphson(start, derivative, tolerance, maxIterations)
		if found {
			// Check if this root is already in our list
			isNew := true
			for _, existingRoot := range roots {
				if math.Abs(root-existingRoot) < tolerance {
					isNew = false
					break
				}
			}
			if isNew && math.Abs(p.Evaluate(root)) < tolerance {
				roots = append(roots, root)
			}
		}
	}

	return roots, nil
}

// newtonRaphson performs Newton-Raphson root finding
func (p *Polynomial) newtonRaphson(x0 float64, derivative *Polynomial, tolerance float64, maxIter int) (float64, bool) {
	x := x0

	for i := 0; i < maxIter; i++ {
		fx := p.Evaluate(x)
		fpx := derivative.Evaluate(x)

		if math.Abs(fpx) < 1e-15 {
			break // Derivative too small
		}

		newX := x - fx/fpx

		if math.Abs(newX-x) < tolerance {
			return newX, true
		}

		x = newX
	}

	return x, false
}

// Compose computes the composition p(q(x))
func (p *Polynomial) Compose(q *Polynomial) (*Polynomial, error) {
	if q == nil {
		return nil, errors.New("cannot compose with nil polynomial")
	}

	if len(p.coeffs) == 0 {
		return NewPolynomial([]float64{0}), nil
	}

	// Start with the constant term
	result := NewPolynomial([]float64{p.coeffs[0]})

	// Build up powers of q(x)
	qPower := NewPolynomial([]float64{1}) // q(x)⁰ = 1

	for i := 1; i < len(p.coeffs); i++ {
		// qPower = q(x)ⁱ
		var err error
		qPower, err = qPower.Multiply(q)
		if err != nil {
			return nil, err
		}

		// Add p.coeffs[i] * q(x)ⁱ to result
		term := qPower.Scale(p.coeffs[i])
		result, err = result.Add(term)
		if err != nil {
			return nil, err
		}
	}

	return result, nil
}

// Divide performs polynomial long division
func (p *Polynomial) Divide(divisor *Polynomial) (*Polynomial, *Polynomial, error) {
	if divisor == nil {
		return nil, nil, errors.New("cannot divide by nil polynomial")
	}

	if divisor.Degree() == 0 && math.Abs(divisor.coeffs[0]) < 1e-15 {
		return nil, nil, errors.New("cannot divide by zero polynomial")
	}

	if p.Degree() < divisor.Degree() {
		// Quotient is 0, remainder is p
		return NewPolynomial([]float64{0}), p.Copy(), nil
	}

	dividend := p.Copy()
	quotientCoeffs := make([]float64, p.Degree()-divisor.Degree()+1)

	for dividend.Degree() >= divisor.Degree() && !dividend.isZero() {
		// Leading coefficient division
		leadCoeffQuotient := dividend.coeffs[dividend.Degree()] / divisor.coeffs[divisor.Degree()]
		degreeQuotient := dividend.Degree() - divisor.Degree()

		quotientCoeffs[degreeQuotient] = leadCoeffQuotient

		// Create monomial term
		termCoeffs := make([]float64, degreeQuotient+1)
		termCoeffs[degreeQuotient] = leadCoeffQuotient
		term := NewPolynomial(termCoeffs)

		// Multiply divisor by term
		product, err := divisor.Multiply(term)
		if err != nil {
			return nil, nil, err
		}

		// Subtract from dividend
		dividend, err = dividend.Subtract(product)
		if err != nil {
			return nil, nil, err
		}
	}

	quotient := NewPolynomial(quotientCoeffs)
	remainder := dividend

	return quotient, remainder, nil
}

// GCD computes the greatest common divisor of two polynomials using Euclidean algorithm
func (p *Polynomial) GCD(other *Polynomial) (*Polynomial, error) {
	if other == nil {
		return nil, errors.New("cannot compute GCD with nil polynomial")
	}

	a := p.Copy()
	b := other.Copy()

	for !b.isZero() {
		_, remainder, err := a.Divide(b)
		if err != nil {
			return nil, err
		}
		a = b
		b = remainder
	}

	// Normalize by leading coefficient
	if len(a.coeffs) > 0 && a.coeffs[a.Degree()] != 0 {
		leadCoeff := a.coeffs[a.Degree()]
		a = a.Scale(1.0 / leadCoeff)
	}

	return a, nil
}

// Equal checks if two polynomials are equal
func (p *Polynomial) Equal(other *Polynomial) bool {
	if other == nil {
		return false
	}

	if len(p.coeffs) != len(other.coeffs) {
		return false
	}

	for i, coeff := range p.coeffs {
		if math.Abs(coeff-other.coeffs[i]) > 1e-15 {
			return false
		}
	}

	return true
}

// Copy creates a deep copy of the polynomial
func (p *Polynomial) Copy() *Polynomial {
	coeffs := make([]float64, len(p.coeffs))
	copy(coeffs, p.coeffs)
	return &Polynomial{coeffs: coeffs}
}

// isZero checks if the polynomial is zero
func (p *Polynomial) isZero() bool {
	for _, coeff := range p.coeffs {
		if math.Abs(coeff) > 1e-15 {
			return false
		}
	}
	return true
}

// String returns a string representation of the polynomial
func (p *Polynomial) String() string {
	if len(p.coeffs) == 0 {
		return "0"
	}

	if p.isZero() {
		return "0"
	}

	var terms []string

	for i := len(p.coeffs) - 1; i >= 0; i-- {
		coeff := p.coeffs[i]
		if math.Abs(coeff) < 1e-15 {
			continue
		}

		var term string
		absCoeff := math.Abs(coeff)

		if i == 0 {
			// Constant term
			if coeff > 0 && len(terms) > 0 {
				term = fmt.Sprintf("+%.3g", coeff)
			} else {
				term = fmt.Sprintf("%.3g", coeff)
			}
		} else if i == 1 {
			// Linear term
			if absCoeff == 1 {
				if coeff > 0 && len(terms) > 0 {
					term = "+x"
				} else if coeff > 0 {
					term = "x"
				} else {
					term = "-x"
				}
			} else {
				if coeff > 0 && len(terms) > 0 {
					term = fmt.Sprintf("+%.3gx", coeff)
				} else {
					term = fmt.Sprintf("%.3gx", coeff)
				}
			}
		} else {
			// Higher degree terms
			if absCoeff == 1 {
				if coeff > 0 && len(terms) > 0 {
					term = fmt.Sprintf("+x^%d", i)
				} else if coeff > 0 {
					term = fmt.Sprintf("x^%d", i)
				} else {
					term = fmt.Sprintf("-x^%d", i)
				}
			} else {
				if coeff > 0 && len(terms) > 0 {
					term = fmt.Sprintf("+%.3gx^%d", coeff, i)
				} else {
					term = fmt.Sprintf("%.3gx^%d", coeff, i)
				}
			}
		}

		terms = append(terms, term)
	}

	if len(terms) == 0 {
		return "0"
	}

	return strings.Join(terms, "")
}

// PolynomialFit performs polynomial fitting using least squares
func PolynomialFit(xData, yData *array.Array, degree int) (*Polynomial, error) {
	ctx := internal.StartProfiler("Math.PolynomialFit")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if xData == nil || yData == nil {
		return nil, internal.NewValidationErrorWithMsg("PolynomialFit", "data arrays cannot be nil")
	}

	if xData.Size() != yData.Size() {
		return nil, errors.New("x and y data must have the same size")
	}

	n := xData.Size()
	if n <= degree {
		return nil, errors.New("need more data points than polynomial degree")
	}

	if degree < 0 {
		return nil, errors.New("polynomial degree must be non-negative")
	}

	// Flatten arrays
	xFlat := xData.Flatten()
	yFlat := yData.Flatten()

	// Build Vandermonde matrix
	A := make([][]float64, n)
	for i := 0; i < n; i++ {
		A[i] = make([]float64, degree+1)
		x := convertToFloat64(xFlat.At(i))
		A[i][0] = 1.0 // x^0
		for j := 1; j <= degree; j++ {
			A[i][j] = A[i][j-1] * x // x^j
		}
	}

	// Build target vector
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		b[i] = convertToFloat64(yFlat.At(i))
	}

	// Solve normal equations: A^T A x = A^T b
	ATA := make([][]float64, degree+1)
	ATb := make([]float64, degree+1)

	for i := 0; i <= degree; i++ {
		ATA[i] = make([]float64, degree+1)
		for j := 0; j <= degree; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += A[k][i] * A[k][j]
			}
			ATA[i][j] = sum
		}

		sum := 0.0
		for k := 0; k < n; k++ {
			sum += A[k][i] * b[k]
		}
		ATb[i] = sum
	}

	// Convert to arrays and solve
	ATAArray := matrixToArray(ATA)
	ATbArray, err := array.FromSlice(ATb)
	if err != nil {
		return nil, err
	}

	coeffsArray, err := Solve(ATAArray, ATbArray)
	if err != nil {
		return nil, fmt.Errorf("failed to solve normal equations: %v", err)
	}

	// Extract coefficients
	coeffs := make([]float64, degree+1)
	coeffsFlat := coeffsArray.Flatten()
	for i := 0; i <= degree; i++ {
		coeffs[i] = convertToFloat64(coeffsFlat.At(i))
	}

	return NewPolynomial(coeffs), nil
}

// matrixToArray converts 2D slice to Array
func matrixToArray(matrix [][]float64) *array.Array {
	if len(matrix) == 0 {
		return nil
	}

	rows := len(matrix)
	cols := len(matrix[0])
	data := make([]float64, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = matrix[i][j]
		}
	}

	arr, _ := array.NewArrayWithShape(data, []int{rows, cols})
	return arr
}
