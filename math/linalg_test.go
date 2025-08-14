package math

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestVectorDot(t *testing.T) {
	// Test 1-D vector dot product
	a, _ := array.FromSlice([]float64{1, 2, 3})
	b, _ := array.FromSlice([]float64{4, 5, 6})

	result, err := Dot(a, b)
	if err != nil {
		t.Fatal(err)
	}

	expected := float64(1*4 + 2*5 + 3*6) // 32
	actual := convertToFloat64(result.At())
	if math.Abs(actual-32.0) > 1e-10 {
		t.Errorf("Vector dot product failed: got %.6f, expected %.6f", actual, expected)
	}
}

func TestMatrixVectorDot(t *testing.T) {
	// Test matrix-vector multiplication - create 3x2 matrix manually
	mat := array.Zeros(internal.Shape{3, 2}, internal.Float64)
	mat.Set(1.0, 0, 0)
	mat.Set(2.0, 0, 1)
	mat.Set(3.0, 1, 0)
	mat.Set(4.0, 1, 1)
	mat.Set(5.0, 2, 0)
	mat.Set(6.0, 2, 1)

	vec, _ := array.FromSlice([]float64{2, 3})

	result, err := Dot(mat, vec)
	if err != nil {
		t.Fatal(err)
	}

	// Expected: [1*2+2*3, 3*2+4*3, 5*2+6*3] = [8, 18, 28]
	expected := []float64{8, 18, 28}
	for i, exp := range expected {
		actual := convertToFloat64(result.At(i))
		if math.Abs(actual-exp) > 1e-10 {
			t.Errorf("Matrix-vector dot at index %d: got %.6f, expected %.6f", i, actual, exp)
		}
	}
}

func TestMatrixMatrixDot(t *testing.T) {
	// Test matrix-matrix multiplication - create 2x2 matrices manually
	a := array.Zeros(internal.Shape{2, 2}, internal.Float64)
	a.Set(1.0, 0, 0)
	a.Set(2.0, 0, 1)
	a.Set(3.0, 1, 0)
	a.Set(4.0, 1, 1)

	b := array.Zeros(internal.Shape{2, 2}, internal.Float64)
	b.Set(5.0, 0, 0)
	b.Set(6.0, 0, 1)
	b.Set(7.0, 1, 0)
	b.Set(8.0, 1, 1)

	result, err := Dot(a, b)
	if err != nil {
		t.Fatal(err)
	}

	// Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
	expected := [][]float64{
		{19, 22},
		{43, 50},
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			actual := convertToFloat64(result.At(i, j))
			if math.Abs(actual-expected[i][j]) > 1e-10 {
				t.Errorf("Matrix multiplication at [%d,%d]: got %.6f, expected %.6f",
					i, j, actual, expected[i][j])
			}
		}
	}
}

func TestTranspose(t *testing.T) {
	// Test 2-D transpose - create 2x3 matrix manually
	arr := array.Zeros(internal.Shape{2, 3}, internal.Float64)
	arr.Set(1.0, 0, 0)
	arr.Set(2.0, 0, 1)
	arr.Set(3.0, 0, 2)
	arr.Set(4.0, 1, 0)
	arr.Set(5.0, 1, 1)
	arr.Set(6.0, 1, 2)

	result, err := Transpose(arr)
	if err != nil {
		t.Fatal(err)
	}

	// Expected transpose: [[1, 4], [2, 5], [3, 6]]
	expected := [][]float64{
		{1, 4},
		{2, 5},
		{3, 6},
	}

	shape := result.Shape()
	if shape[0] != 3 || shape[1] != 2 {
		t.Errorf("Transpose shape incorrect: got %v, expected [3, 2]", shape)
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			actual := convertToFloat64(result.At(i, j))
			if math.Abs(actual-expected[i][j]) > 1e-10 {
				t.Errorf("Transpose at [%d,%d]: got %.6f, expected %.6f",
					i, j, actual, expected[i][j])
			}
		}
	}
}

func TestTrace(t *testing.T) {
	// Test trace of square matrix - create 3x3 matrix manually
	arr := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	arr.Set(1.0, 0, 0)
	arr.Set(2.0, 0, 1)
	arr.Set(3.0, 0, 2)
	arr.Set(4.0, 1, 0)
	arr.Set(5.0, 1, 1)
	arr.Set(6.0, 1, 2)
	arr.Set(7.0, 2, 0)
	arr.Set(8.0, 2, 1)
	arr.Set(9.0, 2, 2)

	result, err := Trace(arr)
	if err != nil {
		t.Fatal(err)
	}

	expected := 1.0 + 5.0 + 9.0 // 15
	actual := convertToFloat64(result)
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("Trace failed: got %.6f, expected %.6f", actual, expected)
	}

	// Test trace of non-square matrix - create 2x3 matrix manually
	rectArr := array.Zeros(internal.Shape{2, 3}, internal.Float64)
	rectArr.Set(1.0, 0, 0)
	rectArr.Set(2.0, 0, 1)
	rectArr.Set(3.0, 0, 2)
	rectArr.Set(4.0, 1, 0)
	rectArr.Set(5.0, 1, 1)
	rectArr.Set(6.0, 1, 2)

	result2, err := Trace(rectArr)
	if err != nil {
		t.Fatal(err)
	}

	expected2 := 1.0 + 5.0 // 6 (min(2,3) = 2 elements)
	actual2 := convertToFloat64(result2)
	if math.Abs(actual2-expected2) > 1e-10 {
		t.Errorf("Rectangular trace failed: got %.6f, expected %.6f", actual2, expected2)
	}
}

func TestDet2x2(t *testing.T) {
	// Test 2x2 determinant - create matrix manually
	arr := array.Zeros(internal.Shape{2, 2}, internal.Float64)
	arr.Set(1.0, 0, 0)
	arr.Set(2.0, 0, 1)
	arr.Set(3.0, 1, 0)
	arr.Set(4.0, 1, 1)

	result, err := Det(arr)
	if err != nil {
		t.Fatal(err)
	}

	expected := float64(1*4 - 2*3) // -2
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("2x2 determinant failed: got %.6f, expected %.6f", result, expected)
	}
}

func TestDet3x3(t *testing.T) {
	// Test 3x3 determinant - create singular matrix manually
	arr := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	arr.Set(1.0, 0, 0)
	arr.Set(2.0, 0, 1)
	arr.Set(3.0, 0, 2)
	arr.Set(4.0, 1, 0)
	arr.Set(5.0, 1, 1)
	arr.Set(6.0, 1, 2)
	arr.Set(7.0, 2, 0)
	arr.Set(8.0, 2, 1)
	arr.Set(9.0, 2, 2)

	result, err := Det(arr)
	if err != nil {
		t.Fatal(err)
	}

	// This matrix is singular (determinant = 0)
	expected := 0.0
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("3x3 determinant failed: got %.6f, expected %.6f", result, expected)
	}

	// Test non-singular 3x3 matrix - create manually
	arr2 := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	arr2.Set(1.0, 0, 0)
	arr2.Set(0.0, 0, 1)
	arr2.Set(2.0, 0, 2)
	arr2.Set(-1.0, 1, 0)
	arr2.Set(3.0, 1, 1)
	arr2.Set(1.0, 1, 2)
	arr2.Set(2.0, 2, 0)
	arr2.Set(4.0, 2, 1)
	arr2.Set(-2.0, 2, 2)

	result2, err := Det(arr2)
	if err != nil {
		t.Fatal(err)
	}

	// Manual calculation: 1*(3*(-2) - 1*4) - 0*(...) + 2*(-1*4 - 3*2) = 1*(-10) + 2*(-10) = -30
	expected2 := -30.0
	if math.Abs(result2-expected2) > 1e-10 {
		t.Errorf("3x3 non-singular determinant failed: got %.6f, expected %.6f", result2, expected2)
	}
}

func TestNorms(t *testing.T) {
	// Test various norms
	data := []float64{3, 4} // A 3-4-5 right triangle
	arr, _ := array.FromSlice(data)

	// L2 norm (default)
	l2Norm, err := Norm(arr, nil)
	if err != nil {
		t.Fatal(err)
	}
	expected := 5.0 // sqrt(3^2 + 4^2)
	if math.Abs(l2Norm-expected) > 1e-10 {
		t.Errorf("L2 norm failed: got %.6f, expected %.6f", l2Norm, expected)
	}

	// L1 norm
	l1Norm, err := Norm(arr, 1)
	if err != nil {
		t.Fatal(err)
	}
	expected = 7.0 // |3| + |4|
	if math.Abs(l1Norm-expected) > 1e-10 {
		t.Errorf("L1 norm failed: got %.6f, expected %.6f", l1Norm, expected)
	}

	// Max norm (infinity norm)
	maxNorm, err := Norm(arr, math.Inf(1))
	if err != nil {
		t.Fatal(err)
	}
	expected = 4.0 // max(|3|, |4|)
	if math.Abs(maxNorm-expected) > 1e-10 {
		t.Errorf("Max norm failed: got %.6f, expected %.6f", maxNorm, expected)
	}

	// Frobenius norm
	frobNorm, err := Norm(arr, "fro")
	if err != nil {
		t.Fatal(err)
	}
	expected = 5.0 // Same as L2 for vectors
	if math.Abs(frobNorm-expected) > 1e-10 {
		t.Errorf("Frobenius norm failed: got %.6f, expected %.6f", frobNorm, expected)
	}
}

func TestSIMDVectorDot(t *testing.T) {
	// Test SIMD optimization with large vectors
	size := 1000
	aData := make([]float64, size)
	bData := make([]float64, size)

	for i := 0; i < size; i++ {
		aData[i] = float64(i + 1)
		bData[i] = 1.0
	}

	a, _ := array.FromSlice(aData)
	b, _ := array.FromSlice(bData)

	// Enable debug to see SIMD usage
	originalConfig := internal.GetDebugConfig()
	defer internal.SetDebugConfig(&originalConfig)
	internal.EnableDebugMode()

	result, err := Dot(a, b)
	if err != nil {
		t.Fatal(err)
	}

	// Sum of 1 to 1000 = 1000*1001/2 = 500500
	expected := float64(size * (size + 1) / 2)
	actual := convertToFloat64(result.At())
	if math.Abs(actual-expected) > 1e-10 {
		t.Errorf("SIMD vector dot failed: got %.6f, expected %.6f", actual, expected)
	}

	t.Logf("SIMD vector dot result: %.0f", actual)
}

func TestErrorHandling(t *testing.T) {
	// Test nil array validation
	_, err := Dot(nil, nil)
	if err == nil {
		t.Error("Expected error for nil arrays")
	}

	// Test shape mismatch
	a, _ := array.FromSlice([]float64{1, 2, 3})
	b, _ := array.FromSlice([]float64{1, 2}) // Different size

	_, err = Dot(a, b)
	if err == nil {
		t.Error("Expected error for shape mismatch")
	}

	// Test determinant of non-square matrix
	rectArr := array.Zeros(internal.Shape{2, 3}, internal.Float64)
	rectArr.Set(1.0, 0, 0)
	rectArr.Set(2.0, 0, 1)
	rectArr.Set(3.0, 0, 2)
	rectArr.Set(4.0, 1, 0)
	rectArr.Set(5.0, 1, 1)
	rectArr.Set(6.0, 1, 2)

	_, err = Det(rectArr)
	if err == nil {
		t.Error("Expected error for determinant of non-square matrix")
	}

	// Test transpose of 1-D array
	vec, _ := array.FromSlice([]float64{1, 2, 3})
	_, err = Transpose(vec)
	if err == nil {
		t.Error("Expected error for transpose of 1-D array")
	}
}

// Benchmark tests
func BenchmarkVectorDot(b *testing.B) {
	size := 1000
	aData := make([]float64, size)
	bData := make([]float64, size)

	for i := 0; i < size; i++ {
		aData[i] = float64(i)
		bData[i] = float64(i)
	}

	a, _ := array.FromSlice(aData)
	vec, _ := array.FromSlice(bData)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Dot(a, vec)
	}
}

func BenchmarkMatrixMultiplication(b *testing.B) {
	size := 100

	// Create matrices manually
	a := array.Zeros(internal.Shape{size, size}, internal.Float64)
	bMat := array.Zeros(internal.Shape{size, size}, internal.Float64)

	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			a.Set(float64(i*size+j), i, j)
			bMat.Set(float64(j*size+i), i, j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Dot(a, bMat)
	}
}

func BenchmarkTranspose(b *testing.B) {
	size := 500

	// Create matrix manually
	arr := array.Zeros(internal.Shape{size, size}, internal.Float64)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			arr.Set(float64(i*size+j), i, j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Transpose(arr)
	}
}
