package math

import (
	"fmt"
	gomath "math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Solve solves the linear system Ax = b using appropriate decomposition
func Solve(A, b *array.Array) (*array.Array, error) {
	ctx := internal.StartProfiler("Math.Solve")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if A == nil {
		return nil, internal.NewValidationErrorWithMsg("Solve", "matrix A cannot be nil")
	}
	if b == nil {
		return nil, internal.NewValidationErrorWithMsg("Solve", "vector b cannot be nil")
	}

	aShape := A.Shape()
	bShape := b.Shape()

	if aShape.Ndim() != 2 {
		return nil, internal.NewValidationErrorWithMsg("Solve", "matrix A must be 2-dimensional")
	}

	if bShape.Ndim() != 1 && bShape.Ndim() != 2 {
		return nil, internal.NewValidationErrorWithMsg("Solve", "b must be 1-D or 2-D")
	}

	m, n := aShape[0], aShape[1]

	// Check dimensions
	expectedRows := m
	if bShape.Ndim() == 1 && bShape[0] != expectedRows {
		return nil, internal.NewValidationErrorWithMsg("Solve",
			fmt.Sprintf("incompatible dimensions: A is %dx%d, b is %d", m, n, bShape[0]))
	}
	if bShape.Ndim() == 2 && bShape[0] != expectedRows {
		return nil, internal.NewValidationErrorWithMsg("Solve",
			fmt.Sprintf("incompatible dimensions: A is %dx%d, b is %dx%d", m, n, bShape[0], bShape[1]))
	}

	// Choose solving method based on matrix properties
	if m == n {
		// Square matrix - try different methods based on properties
		return solveLU(A, b)
	} else if m > n {
		// Overdetermined system - use least squares
		return lstsq(A, b)
	} else {
		// Underdetermined system - use minimum norm solution
		return solveUnderdetermined(A, b)
	}
}

// solveLU solves Ax = b using LU decomposition (when it works properly)
func solveLU(A, b *array.Array) (*array.Array, error) {
	// For now, fall back to QR method since LU has issues
	return solveQR(A, b)
}

// solveQR solves Ax = b using QR decomposition
func solveQR(A, b *array.Array) (*array.Array, error) {
	qr, err := QR(A)
	if err != nil {
		return nil, err
	}

	// Solve R*x = Q^T*b
	// First compute Q^T * b
	QT, err := Transpose(qr.Q)
	if err != nil {
		return nil, err
	}

	var QTb *array.Array
	if b.Shape().Ndim() == 1 {
		QTb, err = matrixVectorDot(QT, b)
	} else {
		QTb, err = Dot(QT, b)
	}
	if err != nil {
		return nil, err
	}

	// Then solve R*x = QTb using back substitution
	return solveUpperTriangular(qr.R, QTb)
}

// solveCholesky solves Ax = b using Cholesky decomposition for positive definite matrices
func solveCholesky(A, b *array.Array) (*array.Array, error) {
	L, err := Chol(A)
	if err != nil {
		return nil, err
	}

	// Solve L*y = b using forward substitution
	y, err := solveLowerTriangular(L, b)
	if err != nil {
		return nil, err
	}

	// Solve L^T*x = y using back substitution
	LT, err := Transpose(L)
	if err != nil {
		return nil, err
	}

	return solveUpperTriangular(LT, y)
}

// solveLowerTriangular solves L*x = b where L is lower triangular
func solveLowerTriangular(L, b *array.Array) (*array.Array, error) {
	shape := L.Shape()
	n := shape[0]

	var x *array.Array
	if b.Shape().Ndim() == 1 {
		x = array.Zeros(internal.Shape{n}, b.DType())
	} else {
		x = array.Zeros(b.Shape(), b.DType())
	}

	// Forward substitution
	for i := 0; i < n; i++ {
		var sum interface{}
		sum = 0.0

		for j := 0; j < i; j++ {
			Lij := L.At(i, j)
			var xj interface{}
			if b.Shape().Ndim() == 1 {
				xj = x.At(j)
			} else {
				xj = x.At(j, 0) // Assume single column for now
			}
			product := multiplyValues(Lij, xj)
			sum = addValues(sum, product)
		}

		Lii := L.At(i, i)
		var bi interface{}
		if b.Shape().Ndim() == 1 {
			bi = b.At(i)
		} else {
			bi = b.At(i, 0)
		}

		diff := subtractValues(bi, sum)
		result := divideValues(diff, Lii)

		if b.Shape().Ndim() == 1 {
			x.Set(result, i)
		} else {
			x.Set(result, i, 0)
		}
	}

	return x, nil
}

// solveUpperTriangular solves U*x = b where U is upper triangular
func solveUpperTriangular(U, b *array.Array) (*array.Array, error) {
	shape := U.Shape()
	n := shape[1] // Use column count for potentially rectangular R from QR

	var x *array.Array
	if b.Shape().Ndim() == 1 {
		x = array.Zeros(internal.Shape{n}, b.DType())
	} else {
		x = array.Zeros(internal.Shape{n, b.Shape()[1]}, b.DType())
	}

	// Back substitution
	for i := n - 1; i >= 0; i-- {
		var sum interface{}
		sum = 0.0

		for j := i + 1; j < n; j++ {
			Uij := U.At(i, j)
			var xj interface{}
			if b.Shape().Ndim() == 1 {
				xj = x.At(j)
			} else {
				xj = x.At(j, 0) // Assume single column for now
			}
			product := multiplyValues(Uij, xj)
			sum = addValues(sum, product)
		}

		Uii := U.At(i, i)
		if gomath.Abs(convertToFloat64(Uii)) < 1e-14 {
			return nil, internal.NewValidationErrorWithMsg("solveUpperTriangular",
				fmt.Sprintf("matrix is singular at diagonal element %d", i))
		}

		var bi interface{}
		if b.Shape().Ndim() == 1 {
			bi = b.At(i)
		} else {
			bi = b.At(i, 0)
		}

		diff := subtractValues(bi, sum)
		result := divideValues(diff, Uii)

		if b.Shape().Ndim() == 1 {
			x.Set(result, i)
		} else {
			x.Set(result, i, 0)
		}
	}

	return x, nil
}

// lstsq computes the least-squares solution of Ax = b
func lstsq(A, b *array.Array) (*array.Array, error) {
	ctx := internal.StartProfiler("Math.Lstsq")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	// Use QR decomposition for least squares
	// The solution is x = R^(-1) * Q^T * b (for the first n components)
	qr, err := QR(A)
	if err != nil {
		return nil, err
	}

	// Compute Q^T * b
	QT, err := Transpose(qr.Q)
	if err != nil {
		return nil, err
	}

	var QTb *array.Array
	if b.Shape().Ndim() == 1 {
		QTb, err = matrixVectorDot(QT, b)
	} else {
		QTb, err = Dot(QT, b)
	}
	if err != nil {
		return nil, err
	}

	// Extract the first n components (corresponding to R's size)
	n := qr.R.Shape()[1]
	var QTb_truncated *array.Array
	if QTb.Shape().Ndim() == 1 {
		QTb_truncated = array.Zeros(internal.Shape{n}, QTb.DType())
		for i := 0; i < n; i++ {
			QTb_truncated.Set(QTb.At(i), i)
		}
	} else {
		QTb_truncated = array.Zeros(internal.Shape{n, QTb.Shape()[1]}, QTb.DType())
		for i := 0; i < n; i++ {
			for j := 0; j < QTb.Shape()[1]; j++ {
				QTb_truncated.Set(QTb.At(i, j), i, j)
			}
		}
	}

	// Solve R*x = QTb_truncated
	return solveUpperTriangular(qr.R, QTb_truncated)
}

// solveUnderdetermined solves underdetermined systems (more unknowns than equations)
func solveUnderdetermined(A, b *array.Array) (*array.Array, error) {
	// Use the minimum norm solution: x = A^T * (A * A^T)^(-1) * b
	AT, err := Transpose(A)
	if err != nil {
		return nil, err
	}

	// Compute A * A^T
	AAT, err := Dot(A, AT)
	if err != nil {
		return nil, err
	}

	// Solve (A * A^T) * y = b
	y, err := Solve(AAT, b)
	if err != nil {
		return nil, err
	}

	// Compute x = A^T * y
	if y.Shape().Ndim() == 1 {
		return matrixVectorDot(AT, y)
	} else {
		return Dot(AT, y)
	}
}

// Additional utility functions for arithmetic operations

func subtractValues(a, b interface{}) interface{} {
	switch va := a.(type) {
	case float64:
		return va - convertToFloat64(b)
	case float32:
		return va - float32(convertToFloat64(b))
	case int64:
		return va - convertToInt64(b)
	case int32:
		return va - int32(convertToInt64(b))
	case complex128:
		return va - convertToComplex128(b)
	case complex64:
		return va - convertToComplex64(b)
	default:
		return 0.0
	}
}

func divideValues(a, b interface{}) interface{} {
	switch va := a.(type) {
	case float64:
		return va / convertToFloat64(b)
	case float32:
		return va / float32(convertToFloat64(b))
	case int64:
		return va / convertToInt64(b)
	case int32:
		return va / int32(convertToInt64(b))
	case complex128:
		return va / convertToComplex128(b)
	case complex64:
		return va / convertToComplex64(b)
	default:
		return 0.0
	}
}

// Note: convertToComplex128 and convertToComplex64 functions are defined in power.go
