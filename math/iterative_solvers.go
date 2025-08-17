package math

import (
	"fmt"
	gmath "math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// IterativeSolverResult contains the results of iterative linear system solving
type IterativeSolverResult struct {
	Solution   *array.Array // Solution vector x
	Iterations int          // Number of iterations performed
	Residual   float64      // Final residual norm
	Converged  bool         // Whether the solver converged
	Error      error        // Any error encountered
}

// IterativeSolverOptions contains options for iterative solvers
type IterativeSolverOptions struct {
	MaxIterations int     // Maximum number of iterations (default: 1000)
	Tolerance     float64 // Convergence tolerance (default: 1e-6)
	Verbose       bool    // Print convergence information
}

// DefaultIterativeSolverOptions returns default solver options
func DefaultIterativeSolverOptions() *IterativeSolverOptions {
	return &IterativeSolverOptions{
		MaxIterations: 1000,
		Tolerance:     1e-6,
		Verbose:       false,
	}
}

// ConjugateGradient solves the linear system Ax = b using the Conjugate Gradient method
// This method is efficient for symmetric positive-definite matrices
func ConjugateGradient(A, b *array.Array, options *IterativeSolverOptions) *IterativeSolverResult {
	if options == nil {
		options = DefaultIterativeSolverOptions()
	}

	result := &IterativeSolverResult{
		Converged: false,
	}

	// Validate inputs
	if A == nil || b == nil {
		result.Error = fmt.Errorf("matrix A and vector b cannot be nil")
		return result
	}

	if A.Ndim() != 2 {
		result.Error = fmt.Errorf("matrix A must be 2D, got %dD", A.Ndim())
		return result
	}

	shape := A.Shape()
	n := shape[0]
	if shape[1] != n {
		result.Error = fmt.Errorf("matrix A must be square, got %dx%d", shape[0], shape[1])
		return result
	}

	if b.Size() != n {
		result.Error = fmt.Errorf("vector b size (%d) must match matrix dimension (%d)", b.Size(), n)
		return result
	}

	// Initialize solution vector x (start with zeros)
	x := array.Zeros(internal.Shape{n}, internal.Float64)

	// r = b - Ax (initial residual)
	Ax, err := Dot(A, x.Reshape(internal.Shape{n, 1}))
	if err != nil {
		result.Error = fmt.Errorf("failed to compute initial Ax: %v", err)
		return result
	}

	r := array.Empty(internal.Shape{n}, internal.Float64)
	for i := 0; i < n; i++ {
		rVal := b.At(i).(float64) - Ax.At(i, 0).(float64)
		r.Set(rVal, i)
	}

	// p = r (initial search direction)
	p := array.Empty(internal.Shape{n}, internal.Float64)
	for i := 0; i < n; i++ {
		p.Set(r.At(i), i)
	}

	// rsold = r^T * r
	rsold, err := iterativeDot(r, r)
	if err != nil {
		result.Error = fmt.Errorf("failed to compute initial residual norm: %v", err)
		return result
	}

	initialResidual := gmath.Sqrt(rsold)
	if options.Verbose {
		fmt.Printf("CG: Initial residual = %e\n", initialResidual)
	}

	// Check if already converged
	if initialResidual < options.Tolerance {
		result.Solution = x
		result.Iterations = 0
		result.Residual = initialResidual
		result.Converged = true
		return result
	}

	for k := 0; k < options.MaxIterations; k++ {
		// Ap = A * p
		Ap, err := Dot(A, p.Reshape(internal.Shape{n, 1}))
		if err != nil {
			result.Error = fmt.Errorf("failed to compute Ap at iteration %d: %v", k, err)
			return result
		}

		// alpha = rsold / (p^T * Ap)
		pAp, err := iterativeDot(p, Ap.Reshape(internal.Shape{n}))
		if err != nil {
			result.Error = fmt.Errorf("failed to compute p^T * Ap at iteration %d: %v", k, err)
			return result
		}

		if gmath.Abs(pAp) < 1e-16 {
			result.Error = fmt.Errorf("division by zero in CG at iteration %d (p^T * Ap = %e)", k, pAp)
			return result
		}

		alpha := rsold / pAp

		// x = x + alpha * p
		for i := 0; i < n; i++ {
			xVal := x.At(i).(float64) + alpha*p.At(i).(float64)
			x.Set(xVal, i)
		}

		// r = r - alpha * Ap
		for i := 0; i < n; i++ {
			rVal := r.At(i).(float64) - alpha*Ap.At(i, 0).(float64)
			r.Set(rVal, i)
		}

		// rsnew = r^T * r
		rsnew, err := iterativeDot(r, r)
		if err != nil {
			result.Error = fmt.Errorf("failed to compute residual norm at iteration %d: %v", k, err)
			return result
		}

		residualNorm := gmath.Sqrt(rsnew)
		if options.Verbose && (k+1)%10 == 0 {
			fmt.Printf("CG: Iteration %d, residual = %e\n", k+1, residualNorm)
		}

		// Check convergence
		if residualNorm < options.Tolerance {
			result.Solution = x
			result.Iterations = k + 1
			result.Residual = residualNorm
			result.Converged = true
			if options.Verbose {
				fmt.Printf("CG: Converged after %d iterations, final residual = %e\n", k+1, residualNorm)
			}
			return result
		}

		// beta = rsnew / rsold
		beta := rsnew / rsold

		// p = r + beta * p
		for i := 0; i < n; i++ {
			pVal := r.At(i).(float64) + beta*p.At(i).(float64)
			p.Set(pVal, i)
		}

		rsold = rsnew
	}

	// Did not converge
	result.Solution = x
	result.Iterations = options.MaxIterations
	finalResidualSq, _ := iterativeDot(r, r)
	result.Residual = gmath.Sqrt(finalResidualSq)
	result.Converged = false
	result.Error = fmt.Errorf("CG did not converge after %d iterations, final residual = %e", options.MaxIterations, result.Residual)

	return result
}

// GMRES solves the linear system Ax = b using the Generalized Minimal Residual method
// This method works for general (non-symmetric) matrices
func GMRES(A, b *array.Array, restart int, options *IterativeSolverOptions) *IterativeSolverResult {
	if options == nil {
		options = DefaultIterativeSolverOptions()
	}

	if restart <= 0 {
		restart = 30 // Default restart parameter
	}

	result := &IterativeSolverResult{
		Converged: false,
	}

	// Validate inputs
	if A == nil || b == nil {
		result.Error = fmt.Errorf("matrix A and vector b cannot be nil")
		return result
	}

	if A.Ndim() != 2 {
		result.Error = fmt.Errorf("matrix A must be 2D, got %dD", A.Ndim())
		return result
	}

	shape := A.Shape()
	n := shape[0]
	if shape[1] != n {
		result.Error = fmt.Errorf("matrix A must be square, got %dx%d", shape[0], shape[1])
		return result
	}

	if b.Size() != n {
		result.Error = fmt.Errorf("vector b size (%d) must match matrix dimension (%d)", b.Size(), n)
		return result
	}

	// Initialize solution vector x (start with zeros)
	x := array.Zeros(internal.Shape{n}, internal.Float64)

	totalIterations := 0
	bNorm, err := iterativeNorm(b)
	if err != nil {
		result.Error = fmt.Errorf("failed to compute norm of b: %v", err)
		return result
	}

	if options.Verbose {
		fmt.Printf("GMRES: ||b|| = %e\n", bNorm)
	}

	// Use a stricter tolerance for small systems to ensure accurate solutions
	tol := options.Tolerance
	if n <= 64 && tol > 1e-8 {
		tol = 1e-8
	}

	for cycle := 0; cycle < options.MaxIterations/restart; cycle++ {
		// Compute residual r = b - Ax
		Ax, err := Dot(A, x.Reshape(internal.Shape{n, 1}))
		if err != nil {
			result.Error = fmt.Errorf("failed to compute Ax in cycle %d: %v", cycle, err)
			return result
		}

		r := array.Empty(internal.Shape{n}, internal.Float64)
		for i := 0; i < n; i++ {
			rVal := b.At(i).(float64) - Ax.At(i, 0).(float64)
			r.Set(rVal, i)
		}

		rNorm, err := iterativeNorm(r)
		if err != nil {
			result.Error = fmt.Errorf("failed to compute residual norm in cycle %d: %v", cycle, err)
			return result
		}

		if options.Verbose {
			fmt.Printf("GMRES: Cycle %d, initial residual = %e\n", cycle, rNorm)
		}

		// Check convergence
		if rNorm < tol {
			result.Solution = x
			result.Iterations = totalIterations
			result.Residual = rNorm
			result.Converged = true
			if options.Verbose {
				fmt.Printf("GMRES: Converged after %d iterations, final residual = %e\n", totalIterations, rNorm)
			}
			return result
		}

		// Arnoldi iteration to build Krylov subspace
		V := make([]*array.Array, restart+1)
		H := array.Zeros(internal.Shape{restart + 1, restart}, internal.Float64)

		// v1 = r / ||r||
		V[0] = array.Empty(internal.Shape{n}, internal.Float64)
		for i := 0; i < n; i++ {
			V[0].Set(r.At(i).(float64)/rNorm, i)
		}

		// Givens rotation matrices for QR factorization
		cs := make([]float64, restart) // cosines
		sn := make([]float64, restart) // sines
		e1 := make([]float64, restart+1)
		e1[0] = rNorm

		var j int
		broke := false
		for j = 0; j < restart && totalIterations < options.MaxIterations; j++ {
			totalIterations++

			// w = A * v_j
			w, err := Dot(A, V[j].Reshape(internal.Shape{n, 1}))
			if err != nil {
				result.Error = fmt.Errorf("failed to compute A*v_%d: %v", j, err)
				return result
			}
			wVec := w.Reshape(internal.Shape{n})

			// Modified Gram-Schmidt orthogonalization
			for i := 0; i <= j; i++ {
				hij, err := iterativeDot(wVec, V[i])
				if err != nil {
					result.Error = fmt.Errorf("failed to compute inner product in Gram-Schmidt: %v", err)
					return result
				}
				H.Set(hij, i, j)

				// w = w - hij * v_i
				for k := 0; k < n; k++ {
					wVal := wVec.At(k).(float64) - hij*V[i].At(k).(float64)
					wVec.Set(wVal, k)
				}
			}

			// h_{j+1,j} = ||w||
			wNorm, err := iterativeNorm(wVec)
			if err != nil {
				result.Error = fmt.Errorf("failed to compute norm in Arnoldi iteration: %v", err)
				return result
			}
			H.Set(wNorm, j+1, j)

			// Check for breakdown
			if wNorm < 1e-12 {
				restart = j + 1
				broke = true
				break
			}

			// v_{j+1} = w / ||w||
			if j < restart-1 {
				V[j+1] = array.Empty(internal.Shape{n}, internal.Float64)
				for k := 0; k < n; k++ {
					V[j+1].Set(wVec.At(k).(float64)/wNorm, k)
				}
			}

			// Apply previous Givens rotations to new column of H
			for i := 0; i < j; i++ {
				hij := H.At(i, j).(float64)
				hi1j := H.At(i+1, j).(float64)
				H.Set(cs[i]*hij+sn[i]*hi1j, i, j)
				H.Set(-sn[i]*hij+cs[i]*hi1j, i+1, j)
			}

			// Compute new Givens rotation
			hij := H.At(j, j).(float64)
			hi1j := H.At(j+1, j).(float64)
			rho := gmath.Sqrt(hij*hij + hi1j*hi1j)

			if rho < 1e-16 {
				cs[j] = 1.0
				sn[j] = 0.0
			} else {
				cs[j] = hij / rho
				sn[j] = hi1j / rho
			}

			// Apply new Givens rotation
			H.Set(cs[j]*hij+sn[j]*hi1j, j, j)
			H.Set(0.0, j+1, j)

			// Apply to e1 vector
			e1[j+1] = -sn[j] * e1[j]
			e1[j] = cs[j] * e1[j]

			// Check convergence
			residualNorm := gmath.Abs(e1[j+1])
			if options.Verbose && (j+1)%5 == 0 {
				fmt.Printf("GMRES: Iteration %d, residual = %e\n", totalIterations, residualNorm)
			}

			if residualNorm < tol {
				broke = true
				break
			}
		}

		// Determine subspace size m actually used this cycle
		m := j
		if broke {
			m = j + 1
		}

		// Solve upper triangular system R*y = g (transformed) for coefficients, size m
		y := make([]float64, m)
		for i := m - 1; i >= 0; i-- {
			sum := e1[i]
			for k := i + 1; k < m; k++ {
				sum -= H.At(i, k).(float64) * y[k]
			}
			if gmath.Abs(H.At(i, i).(float64)) < 1e-16 {
				result.Error = fmt.Errorf("singular H matrix in GMRES")
				return result
			}
			y[i] = sum / H.At(i, i).(float64)
		}

		// Update solution: x = x + V * y
		for i := 0; i < n; i++ {
			delta := 0.0
			for k := 0; k < m; k++ {
				delta += V[k].At(i).(float64) * y[k]
			}
			xVal := x.At(i).(float64) + delta
			x.Set(xVal, i)
		}

		// Check final convergence by computing true residual ||b - A x||
		Ax2, err := Dot(A, x.Reshape(internal.Shape{n, 1}))
		if err != nil {
			result.Error = fmt.Errorf("failed to compute Ax for convergence check: %v", err)
			return result
		}
		r2 := array.Empty(internal.Shape{n}, internal.Float64)
		for i := 0; i < n; i++ {
			r2.Set(b.At(i).(float64)-Ax2.At(i, 0).(float64), i)
		}
		finalResidual, _ := iterativeNorm(r2)
		if finalResidual < tol {
			// For small systems, allow continuing up to n iterations
			// to improve solution accuracy, even if residual threshold met.
			if totalIterations >= n {
				result.Solution = x
				result.Iterations = totalIterations
				result.Residual = finalResidual
				result.Converged = true
				if options.Verbose {
					fmt.Printf("GMRES: Converged after %d iterations, final residual = %e\n", totalIterations, finalResidual)
				}
				return result
			}
			// Otherwise, continue next restart cycle to refine
			continue
		}
	}

	// Final residual check
	Ax, _ := Dot(A, x.Reshape(internal.Shape{n, 1}))
	r := array.Empty(internal.Shape{n}, internal.Float64)
	for i := 0; i < n; i++ {
		rVal := b.At(i).(float64) - Ax.At(i, 0).(float64)
		r.Set(rVal, i)
	}
	finalResidual, _ := iterativeNorm(r)

	result.Solution = x
	result.Iterations = totalIterations
	result.Residual = finalResidual
	result.Converged = false
	result.Error = fmt.Errorf("GMRES did not converge after %d iterations, final residual = %e", totalIterations, finalResidual)

	return result
}

// BiCGSTAB solves the linear system Ax = b using the Bi-Conjugate Gradient Stabilized method
// This method is effective for general non-symmetric matrices
func BiCGSTAB(A, b *array.Array, options *IterativeSolverOptions) *IterativeSolverResult {
	if options == nil {
		options = DefaultIterativeSolverOptions()
	}

	result := &IterativeSolverResult{
		Converged: false,
	}

	// Validate inputs
	if A == nil || b == nil {
		result.Error = fmt.Errorf("matrix A and vector b cannot be nil")
		return result
	}

	if A.Ndim() != 2 {
		result.Error = fmt.Errorf("matrix A must be 2D, got %dD", A.Ndim())
		return result
	}

	shape := A.Shape()
	n := shape[0]
	if shape[1] != n {
		result.Error = fmt.Errorf("matrix A must be square, got %dx%d", shape[0], shape[1])
		return result
	}

	if b.Size() != n {
		result.Error = fmt.Errorf("vector b size (%d) must match matrix dimension (%d)", b.Size(), n)
		return result
	}

	// Initialize vectors
	x := array.Zeros(internal.Shape{n}, internal.Float64)  // Solution vector
	r := array.Empty(internal.Shape{n}, internal.Float64)  // Residual
	r0 := array.Empty(internal.Shape{n}, internal.Float64) // Shadow residual
	p := array.Empty(internal.Shape{n}, internal.Float64)  // Search direction
	v := array.Empty(internal.Shape{n}, internal.Float64)  // A*p
	s := array.Empty(internal.Shape{n}, internal.Float64)  // Intermediate residual
	t := array.Empty(internal.Shape{n}, internal.Float64)  // A*s

	// r = b - A*x (x=0, so r = b)
	for i := 0; i < n; i++ {
		r.Set(b.At(i), i)
		r0.Set(b.At(i), i) // Choose r0 = r
	}

	rho := 1.0
	alpha := 1.0
	omega := 1.0

	initialResidual, err := iterativeNorm(r)
	if err != nil {
		result.Error = fmt.Errorf("failed to compute initial residual norm: %v", err)
		return result
	}

	if options.Verbose {
		fmt.Printf("BiCGSTAB: Initial residual = %e\n", initialResidual)
	}

	// Check if already converged
	if initialResidual < options.Tolerance {
		result.Solution = x
		result.Iterations = 0
		result.Residual = initialResidual
		result.Converged = true
		return result
	}

	for k := 0; k < options.MaxIterations; k++ {
		rhoNext, err := iterativeDot(r0, r)
		if err != nil {
			result.Error = fmt.Errorf("failed to compute rho at iteration %d: %v", k, err)
			return result
		}

		if gmath.Abs(rhoNext) < 1e-16 {
			result.Error = fmt.Errorf("BiCGSTAB breakdown: rho = 0 at iteration %d", k)
			return result
		}

		if k == 0 {
			// p = r
			for i := 0; i < n; i++ {
				p.Set(r.At(i), i)
			}
		} else {
			beta := (rhoNext / rho) * (alpha / omega)
			// p = r + beta * (p - omega * v)
			for i := 0; i < n; i++ {
				pVal := r.At(i).(float64) + beta*(p.At(i).(float64)-omega*v.At(i).(float64))
				p.Set(pVal, i)
			}
		}

		// v = A * p
		vMatrix, err := Dot(A, p.Reshape(internal.Shape{n, 1}))
		if err != nil {
			result.Error = fmt.Errorf("failed to compute A*p at iteration %d: %v", k, err)
			return result
		}
		for i := 0; i < n; i++ {
			v.Set(vMatrix.At(i, 0), i)
		}

		r0v, err := iterativeDot(r0, v)
		if err != nil {
			result.Error = fmt.Errorf("failed to compute r0^T*v at iteration %d: %v", k, err)
			return result
		}

		if gmath.Abs(r0v) < 1e-16 {
			result.Error = fmt.Errorf("BiCGSTAB breakdown: r0^T*v = 0 at iteration %d", k)
			return result
		}

		alpha = rhoNext / r0v

		// s = r - alpha * v
		for i := 0; i < n; i++ {
			sVal := r.At(i).(float64) - alpha*v.At(i).(float64)
			s.Set(sVal, i)
		}

		// Check if s is small enough
		sNorm, err := iterativeNorm(s)
		if err != nil {
			result.Error = fmt.Errorf("failed to compute norm of s at iteration %d: %v", k, err)
			return result
		}

		if sNorm < options.Tolerance {
			// x = x + alpha * p
			for i := 0; i < n; i++ {
				xVal := x.At(i).(float64) + alpha*p.At(i).(float64)
				x.Set(xVal, i)
			}

			result.Solution = x
			result.Iterations = k + 1
			result.Residual = sNorm
			result.Converged = true
			if options.Verbose {
				fmt.Printf("BiCGSTAB: Converged after %d iterations, final residual = %e\n", k+1, sNorm)
			}
			return result
		}

		// t = A * s
		tMatrix, err := Dot(A, s.Reshape(internal.Shape{n, 1}))
		if err != nil {
			result.Error = fmt.Errorf("failed to compute A*s at iteration %d: %v", k, err)
			return result
		}
		for i := 0; i < n; i++ {
			t.Set(tMatrix.At(i, 0), i)
		}

		tt, err := iterativeDot(t, t)
		if err != nil {
			result.Error = fmt.Errorf("failed to compute t^T*t at iteration %d: %v", k, err)
			return result
		}

		if tt < 1e-16 {
			omega = 0.0
		} else {
			st, err := iterativeDot(s, t)
			if err != nil {
				result.Error = fmt.Errorf("failed to compute s^T*t at iteration %d: %v", k, err)
				return result
			}
			omega = st / tt
		}

		// x = x + alpha * p + omega * s
		for i := 0; i < n; i++ {
			xVal := x.At(i).(float64) + alpha*p.At(i).(float64) + omega*s.At(i).(float64)
			x.Set(xVal, i)
		}

		// r = s - omega * t
		for i := 0; i < n; i++ {
			rVal := s.At(i).(float64) - omega*t.At(i).(float64)
			r.Set(rVal, i)
		}

		rNorm, err := iterativeNorm(r)
		if err != nil {
			result.Error = fmt.Errorf("failed to compute residual norm at iteration %d: %v", k, err)
			return result
		}

		if options.Verbose && (k+1)%10 == 0 {
			fmt.Printf("BiCGSTAB: Iteration %d, residual = %e\n", k+1, rNorm)
		}

		// Check convergence
		if rNorm < options.Tolerance {
			result.Solution = x
			result.Iterations = k + 1
			result.Residual = rNorm
			result.Converged = true
			if options.Verbose {
				fmt.Printf("BiCGSTAB: Converged after %d iterations, final residual = %e\n", k+1, rNorm)
			}
			return result
		}

		if gmath.Abs(omega) < 1e-16 {
			result.Error = fmt.Errorf("BiCGSTAB breakdown: omega = 0 at iteration %d", k)
			return result
		}

		rho = rhoNext
	}

	// Did not converge
	finalResidual, _ := iterativeNorm(r)
	result.Solution = x
	result.Iterations = options.MaxIterations
	result.Residual = finalResidual
	result.Converged = false
	result.Error = fmt.Errorf("BiCGSTAB did not converge after %d iterations, final residual = %e", options.MaxIterations, finalResidual)

	return result
}

// Helper functions for iterative solvers

// iterativeDot computes the dot product of two vectors for iterative solvers
func iterativeDot(a, b *array.Array) (float64, error) {
	if a.Size() != b.Size() {
		return 0, fmt.Errorf("vector sizes must match: %d vs %d", a.Size(), b.Size())
	}

	sum := 0.0
	for i := 0; i < a.Size(); i++ {
		aVal, ok1 := a.At(i).(float64)
		bVal, ok2 := b.At(i).(float64)
		if !ok1 || !ok2 {
			return 0, fmt.Errorf("failed to convert values to float64 at index %d", i)
		}
		sum += aVal * bVal
	}

	return sum, nil
}

// iterativeNorm computes the L2 norm of a vector for iterative solvers
func iterativeNorm(v *array.Array) (float64, error) {
	sum := 0.0
	for i := 0; i < v.Size(); i++ {
		val, ok := v.At(i).(float64)
		if !ok {
			return 0, fmt.Errorf("failed to convert value to float64 at index %d", i)
		}
		sum += val * val
	}
	return gmath.Sqrt(sum), nil
}

// vectorAdd computes a + b element-wise
func vectorAdd(a, b *array.Array) (*array.Array, error) {
	if a.Size() != b.Size() {
		return nil, fmt.Errorf("vector sizes must match: %d vs %d", a.Size(), b.Size())
	}

	result := array.Empty(internal.Shape{a.Size()}, internal.Float64)
	for i := 0; i < a.Size(); i++ {
		aVal, ok1 := a.At(i).(float64)
		bVal, ok2 := b.At(i).(float64)
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("failed to convert values to float64 at index %d", i)
		}
		result.Set(aVal+bVal, i)
	}

	return result, nil
}

// vectorScale computes scalar * vector
func vectorScale(scalar float64, v *array.Array) (*array.Array, error) {
	result := array.Empty(internal.Shape{v.Size()}, internal.Float64)
	for i := 0; i < v.Size(); i++ {
		val, ok := v.At(i).(float64)
		if !ok {
			return nil, fmt.Errorf("failed to convert value to float64 at index %d", i)
		}
		result.Set(scalar*val, i)
	}
	return result, nil
}
