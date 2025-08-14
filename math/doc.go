// Package math provides mathematical functions and operations for GoNP arrays.
//
// # Overview
//
// The math package extends Go's standard math library to work with GoNP arrays,
// providing vectorized mathematical operations, linear algebra functions, and
// universal functions (ufuncs) that operate element-wise on arrays.
//
// All functions in this package are optimized for performance using:
//   - SIMD instructions (AVX, AVX2, AVX-512) where available
//   - Multi-threading for large arrays
//   - Efficient memory access patterns
//   - Automatic broadcasting for compatible shapes
//
// # Quick Start
//
// Basic mathematical operations:
//
//	import "github.com/julianshen/gonp/array"
//	import "github.com/julianshen/gonp/math"
//
//	// Create test data
//	arr, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
//
//	// Universal functions (element-wise operations)
//	sines := math.Sin(arr)           // Sine of each element
//	exponentials := math.Exp(arr)    // e^x for each element
//	logarithms := math.Log(arr)      // Natural log of each element
//
//	// Power and root functions
//	squares := math.Square(arr)      // x^2 for each element
//	roots := math.Sqrt(arr)          // √x for each element
//	powers := math.Pow(arr, 2.5)     // x^2.5 for each element
//
//	// Rounding functions
//	rounded := math.Round(arr)       // Round to nearest integer
//	ceiling := math.Ceil(arr)        // Round up
//	floor := math.Floor(arr)         // Round down
//
// # Universal Functions (UFuncs)
//
// Universal functions operate element-wise on arrays and support broadcasting:
//
// ## Trigonometric Functions
//
//	// Basic trigonometric functions
//	sines := math.Sin(angles)        // Sine
//	cosines := math.Cos(angles)      // Cosine
//	tangents := math.Tan(angles)     // Tangent
//
//	// Inverse trigonometric functions
//	arcsines := math.Asin(values)    // Arcsine
//	arccosines := math.Acos(values)  // Arccosine
//	arctangents := math.Atan(values) // Arctangent
//	atan2_vals := math.Atan2(y, x)   // Two-argument arctangent
//
//	// Hyperbolic functions
//	sinh_vals := math.Sinh(arr)      // Hyperbolic sine
//	cosh_vals := math.Cosh(arr)      // Hyperbolic cosine
//	tanh_vals := math.Tanh(arr)      // Hyperbolic tangent
//
// ## Exponential and Logarithmic Functions
//
//	// Exponential functions
//	exp_vals := math.Exp(arr)        // e^x
//	exp2_vals := math.Exp2(arr)      // 2^x
//	expm1_vals := math.Expm1(arr)    // e^x - 1 (accurate for small x)
//
//	// Logarithmic functions
//	log_vals := math.Log(arr)        // Natural logarithm
//	log10_vals := math.Log10(arr)    // Base-10 logarithm
//	log2_vals := math.Log2(arr)      // Base-2 logarithm
//	log1p_vals := math.Log1p(arr)    // ln(1+x) (accurate for small x)
//
// ## Power and Root Functions
//
//	// Power functions
//	squares := math.Square(arr)      // x^2
//	cubes := math.Power(arr, 3)      // x^3
//	powers := math.Pow(base, exp)    // base^exp (element-wise)
//
//	// Root functions
//	sqrt_vals := math.Sqrt(arr)      // Square root
//	cbrt_vals := math.Cbrt(arr)      // Cube root
//
// ## Rounding and Comparison Functions
//
//	// Rounding
//	rounded := math.Round(arr)       // Round to nearest integer
//	truncated := math.Trunc(arr)     // Truncate toward zero
//	ceiling := math.Ceil(arr)        // Round up
//	floor := math.Floor(arr)         // Round down
//
//	// Comparison and selection
//	maximums := math.Maximum(a, b)   // Element-wise maximum
//	minimums := math.Minimum(a, b)   // Element-wise minimum
//	absolute := math.Abs(arr)        // Absolute value
//	signs := math.Sign(arr)          // Sign of each element
//
// # Linear Algebra Functions
//
// The package provides comprehensive linear algebra operations:
//
// ## Matrix Operations
//
//	// Matrix multiplication
//	result := math.Dot(a, b)         // Matrix multiplication
//	inner := math.Inner(a, b)        // Inner product
//	outer := math.Outer(a, b)        // Outer product
//
//	// Matrix properties
//	det := math.Det(matrix)          // Determinant
//	trace := math.Trace(matrix)      // Trace (sum of diagonal)
//	rank := math.Rank(matrix)        // Matrix rank
//	cond := math.Cond(matrix)        // Condition number
//
//	// Matrix norms
//	frobenius := math.Norm(matrix)           // Frobenius norm
//	spectral := math.Norm(matrix, "spec")    // Spectral norm
//	nuclear := math.Norm(matrix, "nuc")      // Nuclear norm
//
// ## Matrix Decompositions
//
//	// Singular Value Decomposition
//	u, s, vt := math.SVD(matrix)     // SVD: A = U * S * V^T
//
//	// QR Decomposition
//	q, r := math.QR(matrix)          // QR: A = Q * R
//
//	// Cholesky Decomposition
//	l := math.Cholesky(matrix)       // A = L * L^T (for positive definite)
//
//	// LU Decomposition with pivoting
//	p, l, u := math.LU(matrix)       // PA = LU
//
//	// Eigenvalue Decomposition
//	eigenvals, eigenvecs := math.Eig(matrix)  // A * v = λ * v
//
// ## Linear System Solving
//
//	// Solve linear systems Ax = b
//	solution := math.Solve(A, b)     // Solve Ax = b
//
//	// Least squares solutions
//	x, residuals := math.LstSq(A, b) // Minimize ||Ax - b||^2
//
//	// Matrix inversion
//	inverse := math.Inv(matrix)      // A^(-1)
//	pinverse := math.Pinv(matrix)    // Moore-Penrose pseudoinverse
//
// # Polynomial Operations
//
// Complete polynomial algebra and root finding:
//
//	// Create polynomials
//	p := math.NewPolynomial([]float64{1, -2, 1})  // x^2 - 2x + 1
//
//	// Polynomial operations
//	sum := p.Add(q)                  // Polynomial addition
//	product := p.Multiply(q)         // Polynomial multiplication
//	quotient, remainder := p.Divide(q) // Polynomial division
//	composed := p.Compose(q)         // Composition: p(q(x))
//
//	// Evaluation and root finding
//	values := p.Evaluate(x_vals)     // Evaluate at array of points
//	derivative := p.Derivative()     // First derivative
//	integral := p.Integral()         // Indefinite integral
//	roots := p.Roots()               // Find polynomial roots
//
//	// Polynomial fitting
//	coeffs := math.PolyFit(x, y, degree)  // Fit polynomial to data
//
// # Performance Features
//
// ## SIMD Optimization
//
// Mathematical functions automatically use SIMD instructions when available:
//
//   - Operates on 4 float64 values simultaneously with AVX
//   - Up to 8 float64 values with AVX-512
//   - 2-4x performance improvement over scalar operations
//   - Automatic fallback for unsupported operations
//
// ## Memory Efficiency
//
//	// In-place operations (when possible)
//	math.SinInPlace(arr)             // Modifies arr directly
//	math.ExpInPlace(arr)             // Saves memory allocation
//
//	// Pre-allocated output arrays
//	result := array.Empty(arr.Shape(), arr.DType())
//	math.Sin(arr, result)            // Store result in existing array
//
// ## Broadcasting
//
// Operations between arrays of different but compatible shapes:
//
//	matrix := array.FromSlice([][]float64{{1, 2, 3}, {4, 5, 6}})  // [2, 3]
//	vector := array.FromSlice([]float64{10, 20, 30})              // [3]
//	result := math.Add(matrix, vector)  // Broadcasting: vector applied to each row
//
// # Error Handling
//
// Mathematical functions handle edge cases appropriately:
//
//	// Domain errors
//	result := math.Sqrt(negative_array)  // Returns NaN for negative inputs
//	logs := math.Log(zero_array)         // Returns -Inf for zero inputs
//
//	// Overflow/underflow
//	large := math.Exp(very_large_array)  // Returns +Inf for overflow
//	small := math.Exp(very_small_array)  // Returns 0 for underflow
//
//	// Invalid operations
//	result, err := math.Solve(singular_matrix, b)  // Returns error
//	if err != nil {
//		log.Printf("Matrix is singular: %v", err)
//	}
//
// # Advanced Usage
//
// ## Custom Universal Functions
//
//	// Define custom element-wise functions
//	sigmoid := func(x interface{}) interface{} {
//		val := x.(float64)
//		return 1.0 / (1.0 + math.Exp(-val))
//	}
//
//	// Apply to arrays
//	result := math.ApplyUnary(arr, sigmoid)
//
// ## Numerical Stability
//
// The package includes numerically stable implementations:
//
//   - Expm1/Log1p for accurate computation near zero
//   - Kahan summation for reduced floating-point errors
//   - Pivoting in matrix decompositions
//   - Condition number checking for ill-conditioned matrices
//
// ## Integration with Statistics
//
//	// Mathematical functions work seamlessly with stats package
//	data := generateRandomData()
//	transformed := math.Log(data)        // Log transform
//	normalized := stats.Normalize(transformed)  // Z-score normalization
//	correlation := stats.Correlation(data, transformed)
//
// # Migration from NumPy
//
// Common NumPy mathematical functions and their GoNP equivalents:
//
//	# NumPy                          // GoNP
//	import numpy as np               import "github.com/julianshen/gonp/math"
//
//	np.sin(arr)                      math.Sin(arr)
//	np.exp(arr)                      math.Exp(arr)
//	np.sqrt(arr)                     math.Sqrt(arr)
//	np.power(arr, 2)                 math.Pow(arr, 2)
//
//	np.dot(a, b)                     math.Dot(a, b)
//	np.linalg.svd(matrix)            math.SVD(matrix)
//	np.linalg.solve(A, b)            math.Solve(A, b)
//	np.linalg.inv(matrix)            math.Inv(matrix)
//
//	np.polyfit(x, y, deg)            math.PolyFit(x, y, deg)
//	np.roots(coeffs)                 math.NewPolynomial(coeffs).Roots()
package math
