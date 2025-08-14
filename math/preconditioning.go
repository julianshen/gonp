package math

import (
	"fmt"
	"math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Preconditioner interface defines methods for matrix preconditioning
type Preconditioner interface {
	// Apply applies the preconditioner to a vector: result = M^(-1) * x
	Apply(x *array.Array) (*array.Array, error)

	// Solve solves M * y = x for y, where M is the preconditioning matrix
	Solve(x *array.Array) (*array.Array, error)

	// GetDiagonal returns the diagonal of the preconditioning matrix (if applicable)
	GetDiagonal() *array.Array
}

// JacobiPreconditioner implements Jacobi (diagonal) preconditioning
type JacobiPreconditioner struct {
	diagonal *array.Array // Inverse of diagonal elements
	n        int          // Matrix size
}

// NewJacobiPreconditioner creates a new Jacobi preconditioner from matrix A
// The Jacobi preconditioner is M = diag(A), where M^(-1) = diag(1/A[i,i])
func NewJacobiPreconditioner(A *array.Array) (*JacobiPreconditioner, error) {
	if A == nil {
		return nil, fmt.Errorf("matrix cannot be nil")
	}

	shape := A.Shape()
	if shape.Ndim() != 2 {
		return nil, fmt.Errorf("matrix must be 2-dimensional")
	}

	if shape[0] != shape[1] {
		return nil, fmt.Errorf("matrix must be square")
	}

	n := shape[0]
	diagonal := array.Empty(internal.Shape{n}, internal.Float64)

	// Extract diagonal elements and compute their reciprocals
	for i := 0; i < n; i++ {
		diagElement := convertToFloat64(A.At(i, i))
		if diagElement == 0 {
			return nil, fmt.Errorf("matrix has zero diagonal element at position %d", i)
		}

		// Store the reciprocal for efficient application
		diagonal.Set(1.0/diagElement, i)
	}

	return &JacobiPreconditioner{
		diagonal: diagonal,
		n:        n,
	}, nil
}

// Apply applies the Jacobi preconditioner to vector x
// Result[i] = x[i] / A[i,i] (where A[i,i] are the original diagonal elements)
func (jp *JacobiPreconditioner) Apply(x *array.Array) (*array.Array, error) {
	if x == nil {
		return nil, fmt.Errorf("input vector cannot be nil")
	}

	if x.Ndim() != 1 {
		return nil, fmt.Errorf("input must be a vector (1D array)")
	}

	if x.Size() != jp.n {
		return nil, fmt.Errorf("vector size (%d) must match matrix size (%d)", x.Size(), jp.n)
	}

	result := array.Empty(internal.Shape{jp.n}, internal.Float64)

	for i := 0; i < jp.n; i++ {
		xi := convertToFloat64(x.At(i))
		invDiag := convertToFloat64(jp.diagonal.At(i))
		result.Set(xi*invDiag, i)
	}

	return result, nil
}

// Solve solves M * y = x for y, which is equivalent to Apply for diagonal preconditioners
func (jp *JacobiPreconditioner) Solve(x *array.Array) (*array.Array, error) {
	return jp.Apply(x)
}

// GetDiagonal returns the diagonal of the inverse preconditioning matrix
func (jp *JacobiPreconditioner) GetDiagonal() *array.Array {
	return jp.diagonal
}

// PreconditionedConjugateGradient solves A*x = b using preconditioned conjugate gradient
func PreconditionedConjugateGradient(A *array.Array, b *array.Array, preconditioner Preconditioner, options *IterativeSolverOptions) *IterativeSolverResult {
	if options == nil {
		options = DefaultIterativeSolverOptions()
	}

	result := &IterativeSolverResult{
		Converged: false,
	}

	if A == nil {
		result.Error = fmt.Errorf("matrix A cannot be nil")
		return result
	}
	if b == nil {
		result.Error = fmt.Errorf("vector b cannot be nil")
		return result
	}
	if preconditioner == nil {
		// Fall back to standard CG if no preconditioner provided
		return ConjugateGradient(A, b, options)
	}

	shape := A.Shape()
	if shape.Ndim() != 2 || shape[0] != shape[1] {
		result.Error = fmt.Errorf("matrix A must be square")
		return result
	}

	n := shape[0]
	if b.Size() != n {
		result.Error = fmt.Errorf("vector b size must match matrix A size")
		return result
	}

	// Initialize solution vector
	var x *array.Array
	x = array.Zeros(internal.Shape{n}, internal.Float64)

	// Compute initial residual: r = b - A*x
	Ax, err := Dot(A, x)
	if err != nil {
		result.Error = fmt.Errorf("failed to compute A*x: %v", err)
		return result
	}

	r := array.Empty(internal.Shape{n}, internal.Float64)
	for i := 0; i < n; i++ {
		bi := convertToFloat64(b.At(i))
		axi := convertToFloat64(Ax.At(i))
		r.Set(bi-axi, i)
	}

	// Apply preconditioner: z = M^(-1) * r
	z, err := preconditioner.Apply(r)
	if err != nil {
		result.Error = fmt.Errorf("failed to apply preconditioner: %v", err)
		return result
	}

	// Initial search direction: p = z
	p := z.Copy()

	// Initial inner product: rzOld = r^T * z
	rzOld := 0.0
	for i := 0; i < n; i++ {
		ri := convertToFloat64(r.At(i))
		zi := convertToFloat64(z.At(i))
		rzOld += ri * zi
	}

	var iterations int
	var residualNorm float64

	for iterations = 0; iterations < options.MaxIterations; iterations++ {
		// Compute residual norm for convergence check
		residualNorm = 0.0
		for i := 0; i < n; i++ {
			ri := convertToFloat64(r.At(i))
			residualNorm += ri * ri
		}
		residualNorm = math.Sqrt(residualNorm)

		// Check for convergence
		if residualNorm < options.Tolerance {
			break
		}

		// Compute A*p
		Ap, err := Dot(A, p)
		if err != nil {
			result.Error = fmt.Errorf("failed to compute A*p at iteration %d: %v", iterations, err)
			return result
		}

		// Compute alpha = (r^T * z) / (p^T * A * p)
		pAp := 0.0
		for i := 0; i < n; i++ {
			pi := convertToFloat64(p.At(i))
			api := convertToFloat64(Ap.At(i))
			pAp += pi * api
		}

		if pAp == 0 {
			result.Error = fmt.Errorf("matrix is not positive definite (p^T*A*p = 0)")
			return result
		}

		alpha := rzOld / pAp

		// Update solution: x = x + alpha * p
		for i := 0; i < n; i++ {
			xi := convertToFloat64(x.At(i))
			pi := convertToFloat64(p.At(i))
			x.Set(xi+alpha*pi, i)
		}

		// Update residual: r = r - alpha * A * p
		for i := 0; i < n; i++ {
			ri := convertToFloat64(r.At(i))
			api := convertToFloat64(Ap.At(i))
			r.Set(ri-alpha*api, i)
		}

		// Apply preconditioner: z = M^(-1) * r
		z, err = preconditioner.Apply(r)
		if err != nil {
			result.Error = fmt.Errorf("failed to apply preconditioner at iteration %d: %v", iterations, err)
			return result
		}

		// Compute new inner product: rzNew = r^T * z
		rzNew := 0.0
		for i := 0; i < n; i++ {
			ri := convertToFloat64(r.At(i))
			zi := convertToFloat64(z.At(i))
			rzNew += ri * zi
		}

		// Compute beta = rzNew / rzOld
		beta := rzNew / rzOld

		// Update search direction: p = z + beta * p
		for i := 0; i < n; i++ {
			zi := convertToFloat64(z.At(i))
			pi := convertToFloat64(p.At(i))
			p.Set(zi+beta*pi, i)
		}

		rzOld = rzNew
	}

	// Final residual norm calculation
	residualNorm = 0.0
	for i := 0; i < n; i++ {
		ri := convertToFloat64(r.At(i))
		residualNorm += ri * ri
	}
	residualNorm = math.Sqrt(residualNorm)

	result.Solution = x
	result.Iterations = iterations
	result.Residual = residualNorm
	result.Converged = residualNorm < options.Tolerance

	return result
}
