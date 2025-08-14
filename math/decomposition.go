package math

import (
	"fmt"
	gomath "math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// LUResult represents the result of LU decomposition
type LUResult struct {
	L *array.Array // Lower triangular matrix
	U *array.Array // Upper triangular matrix
	P *array.Array // Permutation matrix
}

// QRResult represents the result of QR decomposition
type QRResult struct {
	Q *array.Array // Orthogonal matrix
	R *array.Array // Upper triangular matrix
}

// SVDResult represents the result of Singular Value Decomposition
type SVDResult struct {
	U *array.Array // Left singular vectors
	S *array.Array // Singular values
	V *array.Array // Right singular vectors (V^T)
}

// EigenResult represents the result of eigenvalue decomposition
type EigenResult struct {
	Values  *array.Array // Eigenvalues
	Vectors *array.Array // Eigenvectors (columns)
}

// LU computes the LU decomposition of a square matrix A = P*L*U
func LU(arr *array.Array) (*LUResult, error) {
	ctx := internal.StartProfiler("Math.LU")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("LU", "array cannot be nil")
	}

	shape := arr.Shape()
	if shape.Ndim() != 2 {
		return nil, internal.NewValidationErrorWithMsg("LU", "array must be 2-dimensional")
	}

	if shape[0] != shape[1] {
		return nil, internal.NewValidationErrorWithMsg("LU", "array must be square")
	}

	n := shape[0]

	// Create working copy for decomposition - this will become U
	A := arr.Copy()

	// Keep track of row permutations
	perm := make([]int, n)
	for i := 0; i < n; i++ {
		perm[i] = i
	}

	// Initialize L as zeros (diagonal will be set to 1 later)
	L := array.Zeros(internal.Shape{n, n}, arr.DType())

	// Perform LU decomposition with partial pivoting
	for k := 0; k < n-1; k++ {
		// Find pivot in column k from row k onwards
		pivotRow := k
		maxVal := gomath.Abs(convertToFloat64(A.At(k, k)))

		for i := k + 1; i < n; i++ {
			val := gomath.Abs(convertToFloat64(A.At(i, k)))
			if val > maxVal {
				maxVal = val
				pivotRow = i
			}
		}

		// Check for singularity
		if maxVal < 1e-14 {
			return nil, internal.NewValidationErrorWithMsg("LU",
				fmt.Sprintf("matrix is singular at column %d", k))
		}

		// Swap rows if needed
		if pivotRow != k {
			swapRows(A, k, pivotRow)
			swapRowsPartial(L, k, pivotRow, k) // Only swap up to column k-1
			// Track permutation
			perm[k], perm[pivotRow] = perm[pivotRow], perm[k]
		}

		// Elimination
		pivot := convertToFloat64(A.At(k, k))
		for i := k + 1; i < n; i++ {
			factor := convertToFloat64(A.At(i, k)) / pivot
			L.Set(factor, i, k) // Store multiplier in L

			// Update row i
			for j := k + 1; j < n; j++ {
				currentVal := convertToFloat64(A.At(i, j))
				pivotVal := convertToFloat64(A.At(k, j))
				newVal := currentVal - factor*pivotVal
				A.Set(newVal, i, j)
			}
			// Zero out below diagonal
			A.Set(0.0, i, k)
		}
	}

	// Check final diagonal element for singularity
	finalPivot := convertToFloat64(A.At(n-1, n-1))
	if gomath.Abs(finalPivot) < 1e-14 {
		return nil, internal.NewValidationErrorWithMsg("LU",
			fmt.Sprintf("matrix is singular at final pivot"))
	}

	// Set diagonal of L to 1
	for i := 0; i < n; i++ {
		L.Set(1.0, i, i)
	}

	// A now contains U
	U := A

	// Create permutation matrix P
	P := array.Zeros(internal.Shape{n, n}, arr.DType())
	for i := 0; i < n; i++ {
		P.Set(1.0, i, perm[i])
	}

	internal.DebugVerbose("LU decomposition completed for %dx%d matrix", n, n)
	return &LUResult{L: L, U: U, P: P}, nil
}

// QR computes the QR decomposition of a matrix A = Q*R using Gram-Schmidt
func QR(arr *array.Array) (*QRResult, error) {
	ctx := internal.StartProfiler("Math.QR")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("QR", "array cannot be nil")
	}

	shape := arr.Shape()
	if shape.Ndim() != 2 {
		return nil, internal.NewValidationErrorWithMsg("QR", "array must be 2-dimensional")
	}

	m, n := shape[0], shape[1]
	minDim := m
	if n < minDim {
		minDim = n
	}

	Q := array.Zeros(internal.Shape{m, minDim}, arr.DType())
	R := array.Zeros(internal.Shape{minDim, n}, arr.DType())

	// Modified Gram-Schmidt process
	for j := 0; j < minDim; j++ {
		// Extract column j
		col := extractColumn(arr, j)

		// Orthogonalize against previous columns
		for i := 0; i < j; i++ {
			qCol := extractColumn(Q, i)
			rij := vectorDotProduct(qCol, col)
			R.Set(rij, i, j)

			// col = col - R[i,j] * Q[:,i]
			for k := 0; k < m; k++ {
				oldVal := convertToFloat64(col.At(k))
				qVal := convertToFloat64(qCol.At(k))
				newVal := oldVal - rij*qVal
				col.Set(newVal, k)
			}
		}

		// Normalize
		norm := vectorNorm(col)
		if norm < 1e-14 {
			return nil, internal.NewValidationErrorWithMsg("QR",
				fmt.Sprintf("matrix is rank deficient at column %d", j))
		}

		R.Set(norm, j, j)

		// Store normalized column in Q
		for k := 0; k < m; k++ {
			qVal := convertToFloat64(col.At(k)) / norm
			Q.Set(qVal, k, j)
		}

		// Compute R[j, j+1:n]
		for k := j + 1; k < n; k++ {
			colK := extractColumn(arr, k)
			qCol := extractColumn(Q, j)
			rjk := vectorDotProduct(qCol, colK)
			R.Set(rjk, j, k)
		}
	}

	internal.DebugVerbose("QR decomposition completed for %dx%d matrix", m, n)
	return &QRResult{Q: Q, R: R}, nil
}

// Chol computes the Cholesky decomposition of a positive definite matrix A = L*L^T
func Chol(arr *array.Array) (*array.Array, error) {
	ctx := internal.StartProfiler("Math.Chol")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("Chol", "array cannot be nil")
	}

	shape := arr.Shape()
	if shape.Ndim() != 2 {
		return nil, internal.NewValidationErrorWithMsg("Chol", "array must be 2-dimensional")
	}

	if shape[0] != shape[1] {
		return nil, internal.NewValidationErrorWithMsg("Chol", "array must be square")
	}

	n := shape[0]
	L := array.Zeros(internal.Shape{n, n}, arr.DType())

	// Cholesky-Banachiewicz algorithm
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			if i == j {
				// Diagonal elements
				sum := 0.0
				for k := 0; k < j; k++ {
					lval := convertToFloat64(L.At(j, k))
					sum += lval * lval
				}
				aij := convertToFloat64(arr.At(i, j))
				val := aij - sum
				if val <= 0 {
					return nil, internal.NewValidationErrorWithMsg("Chol",
						"matrix is not positive definite")
				}
				L.Set(gomath.Sqrt(val), i, j)
			} else {
				// Off-diagonal elements
				sum := 0.0
				for k := 0; k < j; k++ {
					lik := convertToFloat64(L.At(i, k))
					ljk := convertToFloat64(L.At(j, k))
					sum += lik * ljk
				}
				aij := convertToFloat64(arr.At(i, j))
				ljj := convertToFloat64(L.At(j, j))
				val := (aij - sum) / ljj
				L.Set(val, i, j)
			}
		}
	}

	internal.DebugVerbose("Cholesky decomposition completed for %dx%d matrix", n, n)
	return L, nil
}

// PowerMethod computes the dominant eigenvalue and eigenvector using power iteration
func PowerMethod(arr *array.Array, maxIter int, tolerance float64) (*EigenResult, error) {
	ctx := internal.StartProfiler("Math.PowerMethod")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("PowerMethod", "array cannot be nil")
	}

	shape := arr.Shape()
	if shape.Ndim() != 2 {
		return nil, internal.NewValidationErrorWithMsg("PowerMethod", "array must be 2-dimensional")
	}

	if shape[0] != shape[1] {
		return nil, internal.NewValidationErrorWithMsg("PowerMethod", "array must be square")
	}

	n := shape[0]
	if maxIter <= 0 {
		maxIter = 1000
	}
	if tolerance <= 0 {
		tolerance = 1e-10
	}

	// Initialize random vector
	v := array.Ones(internal.Shape{n}, arr.DType())
	for i := 0; i < n; i++ {
		v.Set(gomath.Sin(float64(i+1)), i) // Simple pseudo-random initialization
	}
	// Normalize initial vector
	initialNorm := vectorNorm(v)
	for i := 0; i < n; i++ {
		val := convertToFloat64(v.At(i)) / initialNorm
		v.Set(val, i)
	}

	var eigenvalue float64
	for iter := 0; iter < maxIter; iter++ {
		// v_new = A * v
		vNew, err := matrixVectorDot(arr, v)
		if err != nil {
			return nil, err
		}

		// Normalize v_new and get the norm (this approximates the eigenvalue)
		norm := vectorNorm(vNew)
		if norm < 1e-14 {
			return nil, internal.NewValidationErrorWithMsg("PowerMethod", "vector became zero during iteration")
		}

		// The eigenvalue is approximated by the Rayleigh quotient
		vTAv := vectorDotProduct(v, vNew)
		vTv := vectorDotProduct(v, v)
		newEigenvalue := vTAv / vTv

		// Debug: Print first few iterations
		if iter < 5 {
			internal.DebugVerbose("Power iteration %d: eigenvalue=%.6f, norm=%.6f, vTAv=%.6f, vTv=%.6f",
				iter, newEigenvalue, norm, vTAv, vTv)
		}

		// Check convergence
		if iter > 0 && gomath.Abs(newEigenvalue-eigenvalue) < tolerance {
			internal.DebugVerbose("Power method converged after %d iterations", iter+1)
			break
		}

		eigenvalue = newEigenvalue

		// Normalize v_new for next iteration
		for i := 0; i < n; i++ {
			val := convertToFloat64(vNew.At(i)) / norm
			v.Set(val, i)
		}
	}

	// Create result arrays
	eigenvalues := array.Empty(internal.Shape{1}, arr.DType())
	eigenvalues.Set(eigenvalue, 0)

	eigenvectors := array.Empty(internal.Shape{n, 1}, arr.DType())
	for i := 0; i < n; i++ {
		eigenvectors.Set(v.At(i), i, 0)
	}

	internal.DebugVerbose("Power method found dominant eigenvalue: %f", eigenvalue)
	return &EigenResult{Values: eigenvalues, Vectors: eigenvectors}, nil
}

// Helper functions

// swapRows swaps two rows in a matrix
func swapRows(mat *array.Array, row1, row2 int) {
	shape := mat.Shape()
	cols := shape[1]

	for j := 0; j < cols; j++ {
		val1 := mat.At(row1, j)
		val2 := mat.At(row2, j)
		mat.Set(val2, row1, j)
		mat.Set(val1, row2, j)
	}
}

// swapRowsPartial swaps two rows in a matrix up to a given column
func swapRowsPartial(mat *array.Array, row1, row2, maxCol int) {
	for j := 0; j < maxCol; j++ {
		val1 := mat.At(row1, j)
		val2 := mat.At(row2, j)
		mat.Set(val2, row1, j)
		mat.Set(val1, row2, j)
	}
}

// extractColumn extracts a column from a matrix as a vector
func extractColumn(mat *array.Array, col int) *array.Array {
	shape := mat.Shape()
	rows := shape[0]

	result := array.Empty(internal.Shape{rows}, mat.DType())
	for i := 0; i < rows; i++ {
		result.Set(mat.At(i, col), i)
	}
	return result
}

// vectorDotProduct computes dot product of two vectors
func vectorDotProduct(a, b *array.Array) float64 {
	size := a.Shape()[0]
	sum := 0.0

	for i := 0; i < size; i++ {
		aVal := convertToFloat64(a.At(i))
		bVal := convertToFloat64(b.At(i))
		sum += aVal * bVal
	}
	return sum
}

// vectorNorm computes the L2 norm of a vector
func vectorNorm(v *array.Array) float64 {
	size := v.Shape()[0]
	sum := 0.0

	for i := 0; i < size; i++ {
		val := convertToFloat64(v.At(i))
		sum += val * val
	}
	return gomath.Sqrt(sum)
}

// SVD computes the Singular Value Decomposition A = U * S * V^T
// Uses the Jacobi SVD method which is stable for small to medium matrices
func SVD(arr *array.Array) (*SVDResult, error) {
	ctx := internal.StartProfiler("Math.SVD")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("SVD", "array cannot be nil")
	}

	shape := arr.Shape()
	if shape.Ndim() != 2 {
		return nil, internal.NewValidationErrorWithMsg("SVD", "array must be 2-dimensional")
	}

	m, n := shape[0], shape[1]
	minDim := m
	if n < minDim {
		minDim = n
	}

	// For rectangular matrices, we'll use the approach:
	// If m >= n: work with A^T * A (n x n) to find V and singular values,
	//           then compute U = A * V / sigma
	// If m < n: work with A * A^T (m x m) to find U and singular values,
	//          then compute V = A^T * U / sigma

	var U, V *array.Array
	var singularValues []float64

	if m >= n {
		// Compute A^T * A
		AT, err := Transpose(arr)
		if err != nil {
			return nil, err
		}
		ATA, err := Dot(AT, arr)
		if err != nil {
			return nil, err
		}

		// Find eigenvalues and eigenvectors of A^T * A
		V, singularValues, err = jacobiEigenSVD(ATA, n)
		if err != nil {
			return nil, err
		}

		// Sort eigenvalues and eigenvectors in descending order
		sortEigenPairs(singularValues, V)

		// Convert eigenvalues to singular values (sqrt of eigenvalues)
		for i := range singularValues {
			if singularValues[i] < 0 {
				singularValues[i] = 0 // Handle numerical errors
			}
			singularValues[i] = gomath.Sqrt(singularValues[i])
		}

		// Compute U = A * V / sigma for non-zero singular values
		U = array.Zeros(internal.Shape{m, minDim}, arr.DType())
		for j := 0; j < minDim; j++ {
			if singularValues[j] > 1e-14 {
				// Extract j-th column of V
				vj := extractColumn(V, j)
				// Compute A * vj
				Avj, err := matrixVectorDot(arr, vj)
				if err != nil {
					return nil, err
				}
				// Normalize by singular value: uj = (A * vj) / sigma_j
				for i := 0; i < m; i++ {
					val := convertToFloat64(Avj.At(i)) / singularValues[j]
					U.Set(val, i, j)
				}
			} else {
				// For zero singular values, use Gram-Schmidt to orthogonalize
				// This is a simplified approach - in practice, more sophisticated methods are used
				for i := 0; i < m; i++ {
					U.Set(0.0, i, j)
				}
			}
		}
	} else {
		// Compute A * A^T
		AT, err := Transpose(arr)
		if err != nil {
			return nil, err
		}
		AAT, err := Dot(arr, AT)
		if err != nil {
			return nil, err
		}

		// Find eigenvalues and eigenvectors of A * A^T
		U, singularValues, err = jacobiEigenSVD(AAT, m)
		if err != nil {
			return nil, err
		}

		// Sort eigenvalues and eigenvectors in descending order
		sortEigenPairs(singularValues, U)

		// Convert eigenvalues to singular values
		for i := range singularValues {
			if singularValues[i] < 0 {
				singularValues[i] = 0
			}
			singularValues[i] = gomath.Sqrt(singularValues[i])
		}

		// Compute V = A^T * U / sigma for non-zero singular values
		V = array.Zeros(internal.Shape{n, minDim}, arr.DType())
		for j := 0; j < minDim; j++ {
			if singularValues[j] > 1e-14 {
				// Extract j-th column of U
				uj := extractColumn(U, j)
				// Compute A^T * uj
				ATuj, err := matrixVectorDot(AT, uj)
				if err != nil {
					return nil, err
				}
				// Normalize by singular value: vj = (A^T * uj) / sigma_j
				for i := 0; i < n; i++ {
					val := convertToFloat64(ATuj.At(i)) / singularValues[j]
					V.Set(val, i, j)
				}
			} else {
				// For zero singular values, use zeros (simplified)
				for i := 0; i < n; i++ {
					V.Set(0.0, i, j)
				}
			}
		}
	}

	// Create singular values array
	S := array.Empty(internal.Shape{minDim}, arr.DType())
	for i := 0; i < minDim; i++ {
		S.Set(singularValues[i], i)
	}

	internal.DebugVerbose("SVD completed for %dx%d matrix, rank=%d", m, n, countNonZeroSingularValues(singularValues, 1e-14))
	return &SVDResult{U: U, S: S, V: V}, nil
}

// jacobiEigenSVD computes eigenvalues and eigenvectors using Jacobi method
// Returns eigenvectors as columns and eigenvalues
func jacobiEigenSVD(A *array.Array, n int) (*array.Array, []float64, error) {
	// Create working copy
	work := A.Copy()

	// Initialize V as identity matrix (will contain eigenvectors)
	V := array.Zeros(internal.Shape{n, n}, A.DType())
	for i := 0; i < n; i++ {
		V.Set(1.0, i, i)
	}

	maxIter := 50
	tolerance := 1e-12

	// Jacobi iteration
	for iter := 0; iter < maxIter; iter++ {
		// Find the largest off-diagonal element
		maxOffDiag := 0.0
		p, q := 0, 1

		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				val := gomath.Abs(convertToFloat64(work.At(i, j)))
				if val > maxOffDiag {
					maxOffDiag = val
					p, q = i, j
				}
			}
		}

		// Check convergence
		if maxOffDiag < tolerance {
			internal.DebugVerbose("Jacobi method converged after %d iterations", iter+1)
			break
		}

		// Compute rotation angle
		App := convertToFloat64(work.At(p, p))
		Aqq := convertToFloat64(work.At(q, q))
		Apq := convertToFloat64(work.At(p, q))

		var theta, c, s float64
		if gomath.Abs(Apq) < tolerance {
			c, s = 1.0, 0.0
		} else {
			tau := (Aqq - App) / (2.0 * Apq)
			if tau >= 0 {
				theta = 1.0 / (tau + gomath.Sqrt(1.0+tau*tau))
			} else {
				theta = -1.0 / (-tau + gomath.Sqrt(1.0+tau*tau))
			}
			c = 1.0 / gomath.Sqrt(1.0+theta*theta)
			s = theta * c
		}

		// Apply Jacobi rotation to A
		for i := 0; i < n; i++ {
			if i != p && i != q {
				Aip := convertToFloat64(work.At(i, p))
				Aiq := convertToFloat64(work.At(i, q))
				work.Set(c*Aip-s*Aiq, i, p)
				work.Set(s*Aip+c*Aiq, i, q)
				work.Set(c*Aip-s*Aiq, p, i) // Symmetric
				work.Set(s*Aip+c*Aiq, q, i)
			}
		}

		// Update diagonal elements
		newApp := c*c*App + s*s*Aqq - 2.0*c*s*Apq
		newAqq := s*s*App + c*c*Aqq + 2.0*c*s*Apq
		work.Set(newApp, p, p)
		work.Set(newAqq, q, q)
		work.Set(0.0, p, q)
		work.Set(0.0, q, p)

		// Update eigenvectors
		for i := 0; i < n; i++ {
			Vip := convertToFloat64(V.At(i, p))
			Viq := convertToFloat64(V.At(i, q))
			V.Set(c*Vip-s*Viq, i, p)
			V.Set(s*Vip+c*Viq, i, q)
		}
	}

	// Extract eigenvalues from diagonal
	eigenvalues := make([]float64, n)
	for i := 0; i < n; i++ {
		eigenvalues[i] = convertToFloat64(work.At(i, i))
	}

	return V, eigenvalues, nil
}

// sortEigenPairs sorts eigenvalues and corresponding eigenvectors in descending order
func sortEigenPairs(eigenvalues []float64, eigenvectors *array.Array) {
	n := len(eigenvalues)

	// Simple selection sort
	for i := 0; i < n-1; i++ {
		maxIdx := i
		for j := i + 1; j < n; j++ {
			if eigenvalues[j] > eigenvalues[maxIdx] {
				maxIdx = j
			}
		}

		if maxIdx != i {
			// Swap eigenvalues
			eigenvalues[i], eigenvalues[maxIdx] = eigenvalues[maxIdx], eigenvalues[i]

			// Swap corresponding eigenvector columns
			rows := eigenvectors.Shape()[0]
			for row := 0; row < rows; row++ {
				val_i := eigenvectors.At(row, i)
				val_max := eigenvectors.At(row, maxIdx)
				eigenvectors.Set(val_max, row, i)
				eigenvectors.Set(val_i, row, maxIdx)
			}
		}
	}
}

// countNonZeroSingularValues counts singular values above tolerance
func countNonZeroSingularValues(singularValues []float64, tolerance float64) int {
	count := 0
	for _, s := range singularValues {
		if s > tolerance {
			count++
		}
	}
	return count
}

// PCAResult represents the result of Principal Component Analysis
type PCAResult struct {
	Components             *array.Array // Principal components (eigenvectors)
	ExplainedVariance      *array.Array // Variance explained by each component
	ExplainedVarianceRatio *array.Array // Proportion of variance explained
	SingularValues         *array.Array // Singular values from SVD
	Mean                   *array.Array // Mean of original data (for centering)
}

// PCA performs Principal Component Analysis using SVD
// data should be organized as samples x features
func PCA(data *array.Array, nComponents int) (*PCAResult, error) {
	ctx := internal.StartProfiler("Math.PCA")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if data == nil {
		return nil, internal.NewValidationErrorWithMsg("PCA", "data cannot be nil")
	}

	shape := data.Shape()
	if shape.Ndim() != 2 {
		return nil, internal.NewValidationErrorWithMsg("PCA", "data must be 2-dimensional")
	}

	nSamples, nFeatures := shape[0], shape[1]
	if nComponents <= 0 {
		nComponents = nFeatures
	}
	if nComponents > nFeatures {
		nComponents = nFeatures
	}

	// Step 1: Center the data (subtract mean from each feature)
	means := array.Empty(internal.Shape{nFeatures}, data.DType())
	for j := 0; j < nFeatures; j++ {
		sum := 0.0
		for i := 0; i < nSamples; i++ {
			sum += convertToFloat64(data.At(i, j))
		}
		mean := sum / float64(nSamples)
		means.Set(mean, j)
	}

	// Create centered data matrix
	centeredData := array.Zeros(shape, data.DType())
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			val := convertToFloat64(data.At(i, j))
			mean := convertToFloat64(means.At(j))
			centeredData.Set(val-mean, i, j)
		}
	}

	// Step 2: Perform SVD on centered data
	// For PCA, we typically use the transpose: features x samples
	centeredDataT, err := Transpose(centeredData)
	if err != nil {
		return nil, err
	}

	svdResult, err := SVD(centeredDataT)
	if err != nil {
		return nil, err
	}

	// Step 3: Extract principal components (first nComponents columns of U)
	// Note: In our SVD, U contains the eigenvectors of the covariance matrix
	components := array.Empty(internal.Shape{nFeatures, nComponents}, data.DType())
	for i := 0; i < nFeatures; i++ {
		for j := 0; j < nComponents; j++ {
			val := svdResult.U.At(i, j)
			components.Set(val, i, j)
		}
	}

	// Step 4: Calculate explained variance
	// Singular values are related to eigenvalues by: eigenvalue = (singular_value^2) / (n_samples - 1)
	explainedVariance := array.Empty(internal.Shape{nComponents}, data.DType())
	totalVariance := 0.0

	for i := 0; i < nComponents; i++ {
		s := convertToFloat64(svdResult.S.At(i))
		eigenvalue := (s * s) / float64(nSamples-1)
		explainedVariance.Set(eigenvalue, i)
		totalVariance += eigenvalue
	}

	// Calculate total variance from all components for ratio calculation
	fullTotalVariance := 0.0
	minDim := svdResult.S.Shape()[0]
	for i := 0; i < minDim; i++ {
		s := convertToFloat64(svdResult.S.At(i))
		eigenvalue := (s * s) / float64(nSamples-1)
		fullTotalVariance += eigenvalue
	}

	// Step 5: Calculate explained variance ratio
	explainedVarianceRatio := array.Empty(internal.Shape{nComponents}, data.DType())
	for i := 0; i < nComponents; i++ {
		variance := convertToFloat64(explainedVariance.At(i))
		ratio := variance / fullTotalVariance
		explainedVarianceRatio.Set(ratio, i)
	}

	// Extract singular values for the selected components
	singularValues := array.Empty(internal.Shape{nComponents}, data.DType())
	for i := 0; i < nComponents; i++ {
		singularValues.Set(svdResult.S.At(i), i)
	}

	internal.DebugVerbose("PCA completed: %d components explain %.2f%% of variance",
		nComponents, sumArray(explainedVarianceRatio)*100)

	return &PCAResult{
		Components:             components,
		ExplainedVariance:      explainedVariance,
		ExplainedVarianceRatio: explainedVarianceRatio,
		SingularValues:         singularValues,
		Mean:                   means,
	}, nil
}

// Transform projects data onto the principal components
func (pca *PCAResult) Transform(data *array.Array) (*array.Array, error) {
	if data == nil {
		return nil, internal.NewValidationErrorWithMsg("PCA.Transform", "data cannot be nil")
	}

	shape := data.Shape()
	if shape.Ndim() != 2 {
		return nil, internal.NewValidationErrorWithMsg("PCA.Transform", "data must be 2-dimensional")
	}

	nSamples, nFeatures := shape[0], shape[1]
	if nFeatures != pca.Mean.Shape()[0] {
		return nil, internal.NewValidationErrorWithMsg("PCA.Transform",
			fmt.Sprintf("data features (%d) don't match PCA features (%d)", nFeatures, pca.Mean.Shape()[0]))
	}

	// Center the data using stored means
	centeredData := array.Zeros(shape, data.DType())
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			val := convertToFloat64(data.At(i, j))
			mean := convertToFloat64(pca.Mean.At(j))
			centeredData.Set(val-mean, i, j)
		}
	}

	// Project onto principal components: transformed = centered_data @ components
	transformed, err := Dot(centeredData, pca.Components)
	if err != nil {
		return nil, err
	}

	return transformed, nil
}

// InverseTransform reconstructs data from principal component representation
func (pca *PCAResult) InverseTransform(transformedData *array.Array) (*array.Array, error) {
	if transformedData == nil {
		return nil, internal.NewValidationErrorWithMsg("PCA.InverseTransform", "data cannot be nil")
	}

	shape := transformedData.Shape()
	if shape.Ndim() != 2 {
		return nil, internal.NewValidationErrorWithMsg("PCA.InverseTransform", "data must be 2-dimensional")
	}

	nSamples, nComponents := shape[0], shape[1]
	if nComponents != pca.Components.Shape()[1] {
		return nil, internal.NewValidationErrorWithMsg("PCA.InverseTransform",
			fmt.Sprintf("data components (%d) don't match PCA components (%d)", nComponents, pca.Components.Shape()[1]))
	}

	nFeatures := pca.Components.Shape()[0]

	// Reconstruct: reconstructed = transformed @ components^T
	componentsT, err := Transpose(pca.Components)
	if err != nil {
		return nil, err
	}

	reconstructed, err := Dot(transformedData, componentsT)
	if err != nil {
		return nil, err
	}

	// Add back the means
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			val := convertToFloat64(reconstructed.At(i, j))
			mean := convertToFloat64(pca.Mean.At(j))
			reconstructed.Set(val+mean, i, j)
		}
	}

	return reconstructed, nil
}

// Rank computes the numerical rank of a matrix using SVD
func Rank(arr *array.Array, tolerance float64) (int, error) {
	if tolerance <= 0 {
		tolerance = 1e-12
	}

	svdResult, err := SVD(arr)
	if err != nil {
		return 0, err
	}

	rank := 0
	for i := 0; i < svdResult.S.Shape()[0]; i++ {
		s := convertToFloat64(svdResult.S.At(i))
		if s > tolerance {
			rank++
		}
	}

	return rank, nil
}

// ConditionNumber computes the condition number of a matrix using SVD
// Returns the ratio of largest to smallest singular value
func ConditionNumber(arr *array.Array) (float64, error) {
	svdResult, err := SVD(arr)
	if err != nil {
		return 0, err
	}

	sMax := convertToFloat64(svdResult.S.At(0))                          // Largest singular value
	sMin := convertToFloat64(svdResult.S.At(svdResult.S.Shape()[0] - 1)) // Smallest

	if sMin == 0 {
		return gomath.Inf(1), nil // Singular matrix has infinite condition number
	}

	return sMax / sMin, nil
}

// Helper function to sum array elements
func sumArray(arr *array.Array) float64 {
	sum := 0.0
	for i := 0; i < arr.Shape()[0]; i++ {
		sum += convertToFloat64(arr.At(i))
	}
	return sum
}
