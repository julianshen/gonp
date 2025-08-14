package math

import (
	"fmt"
	gomath "math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Matrix Operations

// Dot computes the dot product of two arrays
// For 1-D arrays, it computes the inner product
// For 2-D arrays, it computes matrix multiplication
func Dot(a, b *array.Array) (*array.Array, error) {
	ctx := internal.StartProfiler("Math.Dot")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if a == nil {
		return nil, internal.NewValidationErrorWithMsg("Dot", "first array cannot be nil")
	}
	if b == nil {
		return nil, internal.NewValidationErrorWithMsg("Dot", "second array cannot be nil")
	}

	aShape := a.Shape()
	bShape := b.Shape()

	// Handle different dimensionalities
	switch {
	case aShape.Ndim() == 1 && bShape.Ndim() == 1:
		// Vector dot product (inner product)
		return vectorDot(a, b)
	case aShape.Ndim() == 2 && bShape.Ndim() == 1:
		// Matrix-vector multiplication
		return matrixVectorDot(a, b)
	case aShape.Ndim() == 1 && bShape.Ndim() == 2:
		// Vector-matrix multiplication
		return vectorMatrixDot(a, b)
	case aShape.Ndim() == 2 && bShape.Ndim() == 2:
		// Matrix-matrix multiplication
		return matrixMatrixDot(a, b)
	default:
		return nil, internal.NewValidationErrorWithMsg("Dot",
			fmt.Sprintf("unsupported array dimensions: %d-D and %d-D",
				aShape.Ndim(), bShape.Ndim()))
	}
}

// MatMul performs matrix multiplication with broadcasting support
func MatMul(a, b *array.Array) (*array.Array, error) {
	ctx := internal.StartProfiler("Math.MatMul")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if a == nil {
		return nil, internal.NewValidationErrorWithMsg("MatMul", "first array cannot be nil")
	}
	if b == nil {
		return nil, internal.NewValidationErrorWithMsg("MatMul", "second array cannot be nil")
	}

	// For now, delegate to Dot - can be enhanced later for more complex broadcasting
	return Dot(a, b)
}

// Transpose returns the transpose of a 2-D array or swaps the last two axes for higher dimensions
func Transpose(arr *array.Array) (*array.Array, error) {
	ctx := internal.StartProfiler("Math.Transpose")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("Transpose", "array cannot be nil")
	}

	shape := arr.Shape()
	if shape.Ndim() < 2 {
		return nil, internal.NewValidationErrorWithMsg("Transpose",
			"array must be at least 2-dimensional")
	}

	// For 2-D arrays, simple transpose
	if shape.Ndim() == 2 {
		return transpose2D(arr)
	}

	// For higher dimensions, swap last two axes
	return transposeLastTwoAxes(arr)
}

// Trace computes the sum of the diagonal elements of a 2-D array
func Trace(arr *array.Array) (interface{}, error) {
	ctx := internal.StartProfiler("Math.Trace")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("Trace", "array cannot be nil")
	}

	shape := arr.Shape()
	if shape.Ndim() != 2 {
		return nil, internal.NewValidationErrorWithMsg("Trace",
			"array must be 2-dimensional")
	}

	rows, cols := shape[0], shape[1]
	minDim := rows
	if cols < minDim {
		minDim = cols
	}

	// Sum diagonal elements
	var sum interface{}
	for i := 0; i < minDim; i++ {
		val := arr.At(i, i)
		if sum == nil {
			sum = val
		} else {
			sum = addValues(sum, val)
		}
	}

	return sum, nil
}

// Det computes the determinant of a square matrix
func Det(arr *array.Array) (float64, error) {
	ctx := internal.StartProfiler("Math.Det")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return 0, internal.NewValidationErrorWithMsg("Det", "array cannot be nil")
	}

	shape := arr.Shape()
	if shape.Ndim() != 2 {
		return 0, internal.NewValidationErrorWithMsg("Det",
			"array must be 2-dimensional")
	}

	if shape[0] != shape[1] {
		return 0, internal.NewValidationErrorWithMsg("Det",
			"array must be square")
	}

	n := shape[0]

	// Handle small matrices efficiently
	switch n {
	case 1:
		val := arr.At(0, 0)
		return convertToFloat64(val), nil
	case 2:
		return det2x2(arr), nil
	case 3:
		return det3x3(arr), nil
	default:
		// Use LU decomposition for larger matrices
		return detLU(arr)
	}
}

// Norm computes various norms of an array
func Norm(arr *array.Array, ord interface{}) (float64, error) {
	ctx := internal.StartProfiler("Math.Norm")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return 0, internal.NewValidationErrorWithMsg("Norm", "array cannot be nil")
	}

	// Default to Frobenius norm (2-norm)
	if ord == nil {
		ord = 2
	}

	shape := arr.Shape()

	switch ordVal := ord.(type) {
	case int:
		return computeNormInt(arr, ordVal, shape)
	case float64:
		return computeNormFloat(arr, ordVal, shape)
	case string:
		return computeNormString(arr, ordVal, shape)
	default:
		return 0, internal.NewValidationErrorWithMsg("Norm",
			"unsupported norm order type")
	}
}

// Helper functions for matrix operations

// vectorDot computes dot product of two 1-D arrays
func vectorDot(a, b *array.Array) (*array.Array, error) {
	aShape := a.Shape()
	bShape := b.Shape()

	if aShape[0] != bShape[0] {
		return nil, internal.NewShapeError("vectorDot", aShape, bShape)
	}

	// Try SIMD optimization for float64 vectors
	if canUseSIMDDot(a, b) {
		return simdVectorDot(a, b)
	}

	// Fallback to scalar implementation
	return scalarVectorDot(a, b)
}

// matrixVectorDot computes matrix-vector multiplication
func matrixVectorDot(mat, vec *array.Array) (*array.Array, error) {
	matShape := mat.Shape()
	vecShape := vec.Shape()

	if matShape[1] != vecShape[0] {
		return nil, internal.NewShapeError("matrixVectorDot", matShape, vecShape)
	}

	rows, cols := matShape[0], matShape[1]
	resultShape := internal.Shape{rows}
	result := array.Empty(resultShape, mat.DType())

	// Compute matrix-vector multiplication
	for i := 0; i < rows; i++ {
		var sum interface{}
		for j := 0; j < cols; j++ {
			matVal := mat.At(i, j)
			vecVal := vec.At(j)
			product := multiplyValues(matVal, vecVal)

			if sum == nil {
				sum = product
			} else {
				sum = addValues(sum, product)
			}
		}
		result.Set(sum, i)
	}

	return result, nil
}

// vectorMatrixDot computes vector-matrix multiplication
func vectorMatrixDot(vec, mat *array.Array) (*array.Array, error) {
	vecShape := vec.Shape()
	matShape := mat.Shape()

	if vecShape[0] != matShape[0] {
		return nil, internal.NewShapeError("vectorMatrixDot", vecShape, matShape)
	}

	cols := matShape[1]
	resultShape := internal.Shape{cols}
	result := array.Empty(resultShape, vec.DType())

	// Compute vector-matrix multiplication
	for j := 0; j < cols; j++ {
		var sum interface{}
		for i := 0; i < vecShape[0]; i++ {
			vecVal := vec.At(i)
			matVal := mat.At(i, j)
			product := multiplyValues(vecVal, matVal)

			if sum == nil {
				sum = product
			} else {
				sum = addValues(sum, product)
			}
		}
		result.Set(sum, j)
	}

	return result, nil
}

// matrixMatrixDot computes matrix-matrix multiplication
func matrixMatrixDot(a, b *array.Array) (*array.Array, error) {
	aShape := a.Shape()
	bShape := b.Shape()

	if aShape[1] != bShape[0] {
		return nil, internal.NewShapeError("matrixMatrixDot", aShape, bShape)
	}

	rows, inner, cols := aShape[0], aShape[1], bShape[1]
	resultShape := internal.Shape{rows, cols}
	result := array.Empty(resultShape, a.DType())

	// Basic matrix multiplication algorithm
	// TODO: Optimize with blocking, SIMD, or BLAS
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			var sum interface{}
			for k := 0; k < inner; k++ {
				aVal := a.At(i, k)
				bVal := b.At(k, j)
				product := multiplyValues(aVal, bVal)

				if sum == nil {
					sum = product
				} else {
					sum = addValues(sum, product)
				}
			}
			result.Set(sum, i, j)
		}
	}

	return result, nil
}

// transpose2D transposes a 2-D array
func transpose2D(arr *array.Array) (*array.Array, error) {
	shape := arr.Shape()
	rows, cols := shape[0], shape[1]
	transposedShape := internal.Shape{cols, rows}
	result := array.Empty(transposedShape, arr.DType())

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := arr.At(i, j)
			result.Set(val, j, i)
		}
	}

	return result, nil
}

// transposeLastTwoAxes swaps the last two axes of an array
func transposeLastTwoAxes(arr *array.Array) (*array.Array, error) {
	// Simplified implementation - full n-dimensional transpose is complex
	return nil, internal.NewValidationErrorWithMsg("transposeLastTwoAxes",
		"high-dimensional transpose not yet implemented")
}

// Determinant calculations

// det2x2 computes determinant of 2x2 matrix
func det2x2(arr *array.Array) float64 {
	a := convertToFloat64(arr.At(0, 0))
	b := convertToFloat64(arr.At(0, 1))
	c := convertToFloat64(arr.At(1, 0))
	d := convertToFloat64(arr.At(1, 1))

	return a*d - b*c
}

// det3x3 computes determinant of 3x3 matrix using rule of Sarrus
func det3x3(arr *array.Array) float64 {
	// Extract all elements
	a00 := convertToFloat64(arr.At(0, 0))
	a01 := convertToFloat64(arr.At(0, 1))
	a02 := convertToFloat64(arr.At(0, 2))
	a10 := convertToFloat64(arr.At(1, 0))
	a11 := convertToFloat64(arr.At(1, 1))
	a12 := convertToFloat64(arr.At(1, 2))
	a20 := convertToFloat64(arr.At(2, 0))
	a21 := convertToFloat64(arr.At(2, 1))
	a22 := convertToFloat64(arr.At(2, 2))

	// Rule of Sarrus
	return a00*(a11*a22-a12*a21) - a01*(a10*a22-a12*a20) + a02*(a10*a21-a11*a20)
}

// detLU computes determinant using LU decomposition
func detLU(arr *array.Array) (float64, error) {
	// Simplified implementation - would need full LU decomposition
	return 0, internal.NewValidationErrorWithMsg("detLU",
		"LU decomposition not yet implemented")
}

// SIMD optimization helpers

// canUseSIMDDot checks if SIMD can be used for dot product
func canUseSIMDDot(a, b *array.Array) bool {
	return a.DType() == internal.Float64 &&
		b.DType() == internal.Float64 &&
		a.Shape()[0] >= internal.SIMDThreshold
}

// simdVectorDot computes dot product using SIMD
func simdVectorDot(a, b *array.Array) (*array.Array, error) {
	size := a.Shape()[0]

	// Extract data for SIMD operation
	aData := make([]float64, size)
	bData := make([]float64, size)

	for i := 0; i < size; i++ {
		aData[i] = convertToFloat64(a.At(i))
		bData[i] = convertToFloat64(b.At(i))
	}

	// Element-wise multiplication then sum
	product := make([]float64, size)
	internal.SIMDMulFloat64(aData, bData, product)
	sum := internal.SIMDSumFloat64(product)

	// Create scalar result
	result := array.Empty(internal.Shape{}, internal.Float64)
	result.Set(sum)

	internal.DebugVerbose("Used SIMD for vector dot product, size=%d", size)
	return result, nil
}

// scalarVectorDot computes dot product using scalar operations
func scalarVectorDot(a, b *array.Array) (*array.Array, error) {
	size := a.Shape()[0]
	var sum interface{}

	for i := 0; i < size; i++ {
		aVal := a.At(i)
		bVal := b.At(i)
		product := multiplyValues(aVal, bVal)

		if sum == nil {
			sum = product
		} else {
			sum = addValues(sum, product)
		}
	}

	// Create scalar result
	result := array.Empty(internal.Shape{}, a.DType())
	result.Set(sum)

	return result, nil
}

// Norm computation helpers

func computeNormInt(arr *array.Array, ord int, shape internal.Shape) (float64, error) {
	switch ord {
	case 1:
		return computeL1Norm(arr), nil
	case 2:
		return computeL2Norm(arr), nil
	case -1:
		return computeMinNorm(arr), nil
	case -2:
		return computeMaxNorm(arr), nil
	default:
		return computeLpNorm(arr, float64(ord)), nil
	}
}

func computeNormFloat(arr *array.Array, ord float64, shape internal.Shape) (float64, error) {
	if ord == gomath.Inf(1) {
		return computeMaxNorm(arr), nil
	} else if ord == gomath.Inf(-1) {
		return computeMinNorm(arr), nil
	} else {
		return computeLpNorm(arr, ord), nil
	}
}

func computeNormString(arr *array.Array, ord string, shape internal.Shape) (float64, error) {
	switch ord {
	case "fro":
		return computeFrobeniusNorm(arr), nil
	case "nuc":
		return 0, internal.NewValidationErrorWithMsg("Norm",
			"nuclear norm not yet implemented")
	default:
		return 0, internal.NewValidationErrorWithMsg("Norm",
			"unsupported norm order: "+ord)
	}
}

func computeL1Norm(arr *array.Array) float64 {
	var sum float64
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		sum += gomath.Abs(val)
	}
	return sum
}

func computeL2Norm(arr *array.Array) float64 {
	var sumSquares float64
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		sumSquares += val * val
	}
	return gomath.Sqrt(sumSquares)
}

func computeLpNorm(arr *array.Array, p float64) float64 {
	var sum float64
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		sum += gomath.Pow(gomath.Abs(val), p)
	}
	return gomath.Pow(sum, 1.0/p)
}

func computeMaxNorm(arr *array.Array) float64 {
	max := 0.0
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := gomath.Abs(convertToFloat64(flatArr.At(i)))
		if val > max {
			max = val
		}
	}
	return max
}

func computeMinNorm(arr *array.Array) float64 {
	min := gomath.Inf(1)
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := gomath.Abs(convertToFloat64(flatArr.At(i)))
		if val < min {
			min = val
		}
	}
	return min
}

func computeFrobeniusNorm(arr *array.Array) float64 {
	return computeL2Norm(arr) // Frobenius norm is the same as L2 norm for matrices
}

// Utility functions for arithmetic operations on interface{} values

func addValues(a, b interface{}) interface{} {
	switch va := a.(type) {
	case float64:
		return va + convertToFloat64(b)
	case float32:
		return va + float32(convertToFloat64(b))
	case int64:
		return va + convertToInt64(b)
	case int32:
		return va + int32(convertToInt64(b))
	case complex128:
		return va + convertToComplex128(b)
	case complex64:
		return va + convertToComplex64(b)
	default:
		return 0.0
	}
}

func multiplyValues(a, b interface{}) interface{} {
	switch va := a.(type) {
	case float64:
		return va * convertToFloat64(b)
	case float32:
		return va * float32(convertToFloat64(b))
	case int64:
		return va * convertToInt64(b)
	case int32:
		return va * int32(convertToInt64(b))
	case complex128:
		return va * convertToComplex128(b)
	case complex64:
		return va * convertToComplex64(b)
	default:
		return 0.0
	}
}

func convertToInt64(val interface{}) int64 {
	switch v := val.(type) {
	case int64:
		return v
	case int32:
		return int64(v)
	case int16:
		return int64(v)
	case int8:
		return int64(v)
	case int:
		return int64(v)
	case uint64:
		return int64(v)
	case uint32:
		return int64(v)
	case uint16:
		return int64(v)
	case uint8:
		return int64(v)
	case uint:
		return int64(v)
	case float64:
		return int64(v)
	case float32:
		return int64(v)
	default:
		return 0
	}
}
