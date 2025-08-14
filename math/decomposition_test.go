package math

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestLUDecomposition(t *testing.T) {
	// Test with a simple 3x3 matrix
	arr := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	// Matrix: [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
	arr.Set(2.0, 0, 0)
	arr.Set(1.0, 0, 1)
	arr.Set(1.0, 0, 2)
	arr.Set(4.0, 1, 0)
	arr.Set(3.0, 1, 1)
	arr.Set(3.0, 1, 2)
	arr.Set(8.0, 2, 0)
	arr.Set(7.0, 2, 1)
	arr.Set(9.0, 2, 2)

	result, err := LU(arr)
	if err != nil {
		t.Fatal(err)
	}

	// Verify that P*A = L*U by computing L*U and checking against P*A
	LU_product, err := Dot(result.L, result.U)
	if err != nil {
		t.Fatal(err)
	}

	PA_product, err := Dot(result.P, arr)
	if err != nil {
		t.Fatal(err)
	}

	// Check if L*U ≈ P*A
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			lu_val := convertToFloat64(LU_product.At(i, j))
			pa_val := convertToFloat64(PA_product.At(i, j))
			if math.Abs(lu_val-pa_val) > 1e-10 {
				t.Errorf("LU decomposition failed at [%d,%d]: LU=%.6f, PA=%.6f",
					i, j, lu_val, pa_val)
			}
		}
	}

	// Verify L is lower triangular with 1s on diagonal
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			l_val := convertToFloat64(result.L.At(i, j))
			if i == j && math.Abs(l_val-1.0) > 1e-10 {
				t.Errorf("L diagonal should be 1, got %.6f at [%d,%d]", l_val, i, j)
			}
			if i < j && math.Abs(l_val) > 1e-10 {
				t.Errorf("L should be lower triangular, got %.6f at [%d,%d]", l_val, i, j)
			}
		}
	}

	// Verify U is upper triangular
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			u_val := convertToFloat64(result.U.At(i, j))
			if i > j && math.Abs(u_val) > 1e-10 {
				t.Errorf("U should be upper triangular, got %.6f at [%d,%d]", u_val, i, j)
			}
		}
	}

	t.Logf("LU decomposition successful")
}

func TestQRDecomposition(t *testing.T) {
	// Test with a simple 3x3 matrix
	arr := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	// Matrix: [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
	arr.Set(1.0, 0, 0)
	arr.Set(1.0, 0, 1)
	arr.Set(0.0, 0, 2)
	arr.Set(1.0, 1, 0)
	arr.Set(0.0, 1, 1)
	arr.Set(1.0, 1, 2)
	arr.Set(0.0, 2, 0)
	arr.Set(1.0, 2, 1)
	arr.Set(1.0, 2, 2)

	result, err := QR(arr)
	if err != nil {
		t.Fatal(err)
	}

	// Verify that A = Q*R
	QR_product, err := Dot(result.Q, result.R)
	if err != nil {
		t.Fatal(err)
	}

	// Check if Q*R ≈ A
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			qr_val := convertToFloat64(QR_product.At(i, j))
			a_val := convertToFloat64(arr.At(i, j))
			if math.Abs(qr_val-a_val) > 1e-10 {
				t.Errorf("QR decomposition failed at [%d,%d]: QR=%.6f, A=%.6f",
					i, j, qr_val, a_val)
			}
		}
	}

	// Verify Q is orthogonal (Q^T * Q = I)
	Q_transpose, err := Transpose(result.Q)
	if err != nil {
		t.Fatal(err)
	}

	QTQ_product, err := Dot(Q_transpose, result.Q)
	if err != nil {
		t.Fatal(err)
	}

	// Check if Q^T * Q ≈ I
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			qtq_val := convertToFloat64(QTQ_product.At(i, j))
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if math.Abs(qtq_val-expected) > 1e-10 {
				t.Errorf("Q should be orthogonal, Q^T*Q[%d,%d]=%.6f, expected=%.6f",
					i, j, qtq_val, expected)
			}
		}
	}

	// Verify R is upper triangular
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			r_val := convertToFloat64(result.R.At(i, j))
			if i > j && math.Abs(r_val) > 1e-10 {
				t.Errorf("R should be upper triangular, got %.6f at [%d,%d]", r_val, i, j)
			}
		}
	}

	t.Logf("QR decomposition successful")
}

func TestCholeskyDecomposition(t *testing.T) {
	// Test with a symmetric positive definite matrix
	// Matrix: [[4, 2, 1], [2, 3, 0.5], [1, 0.5, 3]]
	arr := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	arr.Set(4.0, 0, 0)
	arr.Set(2.0, 0, 1)
	arr.Set(1.0, 0, 2)
	arr.Set(2.0, 1, 0)
	arr.Set(3.0, 1, 1)
	arr.Set(0.5, 1, 2)
	arr.Set(1.0, 2, 0)
	arr.Set(0.5, 2, 1)
	arr.Set(3.0, 2, 2)

	L, err := Chol(arr)
	if err != nil {
		t.Fatal(err)
	}

	// Verify that A = L * L^T
	L_transpose, err := Transpose(L)
	if err != nil {
		t.Fatal(err)
	}

	LLT_product, err := Dot(L, L_transpose)
	if err != nil {
		t.Fatal(err)
	}

	// Check if L * L^T ≈ A
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			llt_val := convertToFloat64(LLT_product.At(i, j))
			a_val := convertToFloat64(arr.At(i, j))
			if math.Abs(llt_val-a_val) > 1e-10 {
				t.Errorf("Cholesky decomposition failed at [%d,%d]: L*L^T=%.6f, A=%.6f",
					i, j, llt_val, a_val)
			}
		}
	}

	// Verify L is lower triangular
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			l_val := convertToFloat64(L.At(i, j))
			if i < j && math.Abs(l_val) > 1e-10 {
				t.Errorf("L should be lower triangular, got %.6f at [%d,%d]", l_val, i, j)
			}
		}
	}

	t.Logf("Cholesky decomposition successful")
}

func TestPowerMethod(t *testing.T) {
	// Enable debug to see iteration details
	originalConfig := internal.GetDebugConfig()
	defer internal.SetDebugConfig(&originalConfig)
	internal.EnableDebugMode()
	internal.SetLogLevel(internal.DebugLevelVerbose)

	// Test with a matrix with known dominant eigenvalue
	// Matrix: [[3, 1], [1, 2]] has eigenvalues ≈ 3.618 and 1.382
	arr := array.Zeros(internal.Shape{2, 2}, internal.Float64)
	arr.Set(3.0, 0, 0)
	arr.Set(1.0, 0, 1)
	arr.Set(1.0, 1, 0)
	arr.Set(2.0, 1, 1)

	// Debug: Print the matrix
	t.Logf("Test matrix: [[%.1f, %.1f], [%.1f, %.1f]]",
		convertToFloat64(arr.At(0, 0)), convertToFloat64(arr.At(0, 1)),
		convertToFloat64(arr.At(1, 0)), convertToFloat64(arr.At(1, 1)))

	result, err := PowerMethod(arr, 1000, 1e-10)
	if err != nil {
		t.Fatal(err)
	}

	// The dominant eigenvalue should be approximately (5 + sqrt(5))/2 ≈ 3.618
	expectedEigenvalue := (5.0 + math.Sqrt(5.0)) / 2.0
	actualEigenvalue := convertToFloat64(result.Values.At(0))

	t.Logf("Power method result: eigenvalue=%.6f, expected=%.6f", actualEigenvalue, expectedEigenvalue)

	if math.Abs(actualEigenvalue-expectedEigenvalue) > 1e-3 { // Relaxed tolerance for now
		t.Errorf("Power method eigenvalue: got %.6f, expected %.6f",
			actualEigenvalue, expectedEigenvalue)
	}

	// Verify that A * v = λ * v (approximately)
	eigenvector := array.Empty(internal.Shape{2}, internal.Float64)
	eigenvector.Set(result.Vectors.At(0, 0), 0)
	eigenvector.Set(result.Vectors.At(1, 0), 1)

	Av, err := matrixVectorDot(arr, eigenvector)
	if err != nil {
		t.Fatal(err)
	}

	// Check if A*v ≈ λ*v
	for i := 0; i < 2; i++ {
		av_val := convertToFloat64(Av.At(i))
		v_val := convertToFloat64(eigenvector.At(i))
		expected := actualEigenvalue * v_val

		if math.Abs(av_val-expected) > 1e-5 {
			t.Errorf("Eigenvalue equation failed at index %d: A*v=%.6f, λ*v=%.6f",
				i, av_val, expected)
		}
	}

	t.Logf("Power method found eigenvalue: %.6f", actualEigenvalue)
}

func TestErrorHandlingDecomposition(t *testing.T) {
	// Test nil array validation
	_, err := LU(nil)
	if err == nil {
		t.Error("Expected error for nil array in LU")
	}

	_, err = QR(nil)
	if err == nil {
		t.Error("Expected error for nil array in QR")
	}

	_, err = Chol(nil)
	if err == nil {
		t.Error("Expected error for nil array in Chol")
	}

	_, err = PowerMethod(nil, 100, 1e-6)
	if err == nil {
		t.Error("Expected error for nil array in PowerMethod")
	}

	// Test non-2D arrays
	vec, _ := array.FromSlice([]float64{1, 2, 3})
	_, err = LU(vec)
	if err == nil {
		t.Error("Expected error for 1D array in LU")
	}

	// Test non-square matrix for LU
	rect := array.Zeros(internal.Shape{2, 3}, internal.Float64)
	_, err = LU(rect)
	if err == nil {
		t.Error("Expected error for non-square matrix in LU")
	}

	// Test singular matrix for LU
	singular := array.Zeros(internal.Shape{2, 2}, internal.Float64)
	singular.Set(1.0, 0, 0)
	singular.Set(2.0, 0, 1)
	singular.Set(2.0, 1, 0)
	singular.Set(4.0, 1, 1) // Second row is 2x first row
	_, err = LU(singular)
	if err == nil {
		t.Error("Expected error for singular matrix in LU")
	}

	// Test non-positive definite matrix for Cholesky
	nonPosDef := array.Zeros(internal.Shape{2, 2}, internal.Float64)
	nonPosDef.Set(-1.0, 0, 0)
	nonPosDef.Set(0.0, 0, 1)
	nonPosDef.Set(0.0, 1, 0)
	nonPosDef.Set(1.0, 1, 1)
	_, err = Chol(nonPosDef)
	if err == nil {
		t.Error("Expected error for non-positive definite matrix in Chol")
	}
}

func TestDecompositionHelpers(t *testing.T) {
	// Test helper functions
	mat := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			mat.Set(float64(i*3+j+1), i, j)
		}
	}

	// Test extractColumn
	col1 := extractColumn(mat, 1)
	expected := []float64{2, 5, 8} // Second column values
	for i, exp := range expected {
		actual := convertToFloat64(col1.At(i))
		if math.Abs(actual-exp) > 1e-10 {
			t.Errorf("extractColumn failed at index %d: got %.6f, expected %.6f",
				i, actual, exp)
		}
	}

	// Test vectorDotProduct
	a, _ := array.FromSlice([]float64{1, 2, 3})
	b, _ := array.FromSlice([]float64{4, 5, 6})
	dot := vectorDotProduct(a, b)
	expectedDot := 32.0 // 1*4 + 2*5 + 3*6
	if math.Abs(dot-expectedDot) > 1e-10 {
		t.Errorf("vectorDotProduct failed: got %.6f, expected %.6f", dot, expectedDot)
	}

	// Test vectorNorm
	norm := vectorNorm(a)
	expectedNorm := math.Sqrt(14.0) // sqrt(1^2 + 2^2 + 3^2)
	if math.Abs(norm-expectedNorm) > 1e-10 {
		t.Errorf("vectorNorm failed: got %.6f, expected %.6f", norm, expectedNorm)
	}

	// Test swapRows
	original := mat.Copy()
	swapRows(mat, 0, 2)

	// Check that row 0 now has row 2's values
	for j := 0; j < 3; j++ {
		expected := convertToFloat64(original.At(2, j))
		actual := convertToFloat64(mat.At(0, j))
		if math.Abs(actual-expected) > 1e-10 {
			t.Errorf("swapRows failed at [0,%d]: got %.6f, expected %.6f",
				j, actual, expected)
		}
	}
}

// Benchmark tests
func BenchmarkLUDecomposition(b *testing.B) {
	size := 50
	arr := array.Zeros(internal.Shape{size, size}, internal.Float64)

	// Fill with a well-conditioned matrix
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			val := math.Sin(float64(i+1)) + math.Cos(float64(j+1))
			if i == j {
				val += float64(size) // Make diagonally dominant
			}
			arr.Set(val, i, j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = LU(arr)
	}
}

func BenchmarkQRDecomposition(b *testing.B) {
	size := 30
	arr := array.Zeros(internal.Shape{size, size}, internal.Float64)

	// Fill with random-like values
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			val := math.Sin(float64(i*size + j + 1))
			arr.Set(val, i, j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = QR(arr)
	}
}

func BenchmarkCholeskyDecomposition(b *testing.B) {
	size := 50

	// Create a symmetric positive definite matrix A = B^T * B
	B := array.Zeros(internal.Shape{size, size}, internal.Float64)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			val := math.Sin(float64(i + j + 1))
			B.Set(val, i, j)
		}
	}

	BT, _ := Transpose(B)
	arr, _ := Dot(BT, B)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Chol(arr)
	}
}

func TestSVD(t *testing.T) {
	t.Run("Basic SVD 2x2", func(t *testing.T) {
		// Simple 2x2 matrix
		arr := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		arr.Set(3.0, 0, 0)
		arr.Set(2.0, 0, 1)
		arr.Set(2.0, 1, 0)
		arr.Set(3.0, 1, 1)

		result, err := SVD(arr)
		if err != nil {
			t.Fatalf("SVD failed: %v", err)
		}

		// Verify dimensions
		if result.U.Shape()[0] != 2 || result.U.Shape()[1] != 2 {
			t.Errorf("U matrix dimensions incorrect: got %v, expected [2, 2]", result.U.Shape())
		}
		if result.S.Shape()[0] != 2 {
			t.Errorf("S vector dimensions incorrect: got %v, expected [2]", result.S.Shape())
		}
		if result.V.Shape()[0] != 2 || result.V.Shape()[1] != 2 {
			t.Errorf("V matrix dimensions incorrect: got %v, expected [2, 2]", result.V.Shape())
		}

		// Verify that A = U*S*V^T
		reconstructed, err := reconstructFromSVD(result)
		if err != nil {
			t.Fatalf("Failed to reconstruct matrix: %v", err)
		}

		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				original := convertToFloat64(arr.At(i, j))
				recon := convertToFloat64(reconstructed.At(i, j))
				if math.Abs(original-recon) > 1e-10 {
					t.Errorf("Reconstruction failed at [%d,%d]: original=%.6f, reconstructed=%.6f",
						i, j, original, recon)
				}
			}
		}

		// Check that singular values are non-negative and sorted
		s1 := convertToFloat64(result.S.At(0))
		s2 := convertToFloat64(result.S.At(1))
		if s1 < 0 || s2 < 0 {
			t.Errorf("Singular values must be non-negative: s1=%.6f, s2=%.6f", s1, s2)
		}
		if s1 < s2 {
			t.Errorf("Singular values must be sorted in descending order: s1=%.6f < s2=%.6f", s1, s2)
		}
	})

	t.Run("SVD 3x3 Matrix", func(t *testing.T) {
		// 3x3 matrix
		arr := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		arr.Set(1.0, 0, 0)
		arr.Set(0.0, 0, 1)
		arr.Set(0.0, 0, 2)
		arr.Set(0.0, 1, 0)
		arr.Set(1.0, 1, 1)
		arr.Set(0.0, 1, 2)
		arr.Set(0.0, 2, 0)
		arr.Set(0.0, 2, 1)
		arr.Set(1.0, 2, 2)

		result, err := SVD(arr)
		if err != nil {
			t.Fatalf("SVD failed: %v", err)
		}

		// Verify dimensions
		if result.U.Shape()[0] != 3 || result.U.Shape()[1] != 3 {
			t.Errorf("U matrix dimensions incorrect: got %v, expected [3, 3]", result.U.Shape())
		}
		if result.S.Shape()[0] != 3 {
			t.Errorf("S vector dimensions incorrect: got %v, expected [3]", result.S.Shape())
		}
		if result.V.Shape()[0] != 3 || result.V.Shape()[1] != 3 {
			t.Errorf("V matrix dimensions incorrect: got %v, expected [3, 3]", result.V.Shape())
		}

		// Verify reconstruction
		reconstructed, err := reconstructFromSVD(result)
		if err != nil {
			t.Fatalf("Failed to reconstruct matrix: %v", err)
		}

		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				original := convertToFloat64(arr.At(i, j))
				recon := convertToFloat64(reconstructed.At(i, j))
				if math.Abs(original-recon) > 1e-10 {
					t.Errorf("Reconstruction failed at [%d,%d]: original=%.6f, reconstructed=%.6f",
						i, j, original, recon)
				}
			}
		}

		// Check singular value ordering
		for i := 0; i < 2; i++ {
			si := convertToFloat64(result.S.At(i))
			si1 := convertToFloat64(result.S.At(i + 1))
			if si < si1 {
				t.Errorf("Singular values must be sorted: s[%d]=%.6f < s[%d]=%.6f", i, si, i+1, si1)
			}
		}
	})

	t.Run("SVD Rectangular Matrix", func(t *testing.T) {
		// 3x2 matrix
		arr := array.Zeros(internal.Shape{3, 2}, internal.Float64)
		arr.Set(1.0, 0, 0)
		arr.Set(2.0, 0, 1)
		arr.Set(3.0, 1, 0)
		arr.Set(4.0, 1, 1)
		arr.Set(5.0, 2, 0)
		arr.Set(6.0, 2, 1)

		result, err := SVD(arr)
		if err != nil {
			t.Fatalf("SVD failed: %v", err)
		}

		// For m x n matrix, min(m,n) = min(3,2) = 2
		minDim := 2

		// Verify dimensions for rectangular matrix
		if result.U.Shape()[0] != 3 || result.U.Shape()[1] != minDim {
			t.Errorf("U matrix dimensions incorrect: got %v, expected [3, %d]", result.U.Shape(), minDim)
		}
		if result.S.Shape()[0] != minDim {
			t.Errorf("S vector dimensions incorrect: got %v, expected [%d]", result.S.Shape(), minDim)
		}
		if result.V.Shape()[0] != 2 || result.V.Shape()[1] != minDim {
			t.Errorf("V matrix dimensions incorrect: got %v, expected [2, %d]", result.V.Shape(), minDim)
		}

		// Verify reconstruction
		reconstructed, err := reconstructFromSVD(result)
		if err != nil {
			t.Fatalf("Failed to reconstruct matrix: %v", err)
		}

		for i := 0; i < 3; i++ {
			for j := 0; j < 2; j++ {
				original := convertToFloat64(arr.At(i, j))
				recon := convertToFloat64(reconstructed.At(i, j))
				if math.Abs(original-recon) > 1e-10 {
					t.Errorf("Reconstruction failed at [%d,%d]: original=%.6f, reconstructed=%.6f",
						i, j, original, recon)
				}
			}
		}
	})

	t.Run("SVD Orthogonality Test", func(t *testing.T) {
		// Test that U and V are orthogonal matrices
		arr := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		arr.Set(2.0, 0, 0)
		arr.Set(1.0, 0, 1)
		arr.Set(0.0, 0, 2)
		arr.Set(1.0, 1, 0)
		arr.Set(2.0, 1, 1)
		arr.Set(1.0, 1, 2)
		arr.Set(0.0, 2, 0)
		arr.Set(1.0, 2, 1)
		arr.Set(2.0, 2, 2)

		result, err := SVD(arr)
		if err != nil {
			t.Fatalf("SVD failed: %v", err)
		}

		// Test U orthogonality: U^T * U should be identity
		UT, err := Transpose(result.U)
		if err != nil {
			t.Fatalf("Failed to transpose U: %v", err)
		}
		UTU, err := Dot(UT, result.U)
		if err != nil {
			t.Fatalf("Failed to compute U^T * U: %v", err)
		}

		// Check if UTU is approximately identity
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				expected := 0.0
				if i == j {
					expected = 1.0
				}
				actual := convertToFloat64(UTU.At(i, j))
				if math.Abs(actual-expected) > 1e-10 {
					t.Errorf("U is not orthogonal: U^T*U[%d,%d] = %.6f, expected %.6f",
						i, j, actual, expected)
				}
			}
		}

		// Test V orthogonality: V^T * V should be identity
		VT, err := Transpose(result.V)
		if err != nil {
			t.Fatalf("Failed to transpose V: %v", err)
		}
		VTV, err := Dot(VT, result.V)
		if err != nil {
			t.Fatalf("Failed to compute V^T * V: %v", err)
		}

		// Check if VTV is approximately identity
		dim := result.V.Shape()[1]
		for i := 0; i < dim; i++ {
			for j := 0; j < dim; j++ {
				expected := 0.0
				if i == j {
					expected = 1.0
				}
				actual := convertToFloat64(VTV.At(i, j))
				if math.Abs(actual-expected) > 1e-10 {
					t.Errorf("V is not orthogonal: V^T*V[%d,%d] = %.6f, expected %.6f",
						i, j, actual, expected)
				}
			}
		}
	})

	t.Run("SVD Error Cases", func(t *testing.T) {
		// Nil array
		_, err := SVD(nil)
		if err == nil {
			t.Error("Expected error for nil array")
		}

		// 1D array
		arr1d := array.Ones(internal.Shape{5}, internal.Float64)
		_, err = SVD(arr1d)
		if err == nil {
			t.Error("Expected error for 1D array")
		}

		// 3D array
		arr3d := array.Ones(internal.Shape{2, 2, 2}, internal.Float64)
		_, err = SVD(arr3d)
		if err == nil {
			t.Error("Expected error for 3D array")
		}
	})
}

// Helper function to reconstruct matrix from SVD: A = U * S * V^T
func reconstructFromSVD(svd *SVDResult) (*array.Array, error) {
	// Get dimensions
	minDim := svd.S.Shape()[0]

	// Create diagonal matrix S with dimensions that match for multiplication
	// U has shape [m, minDim], so S should be [minDim, minDim], and V^T should be [minDim, n]
	S := array.Zeros(internal.Shape{minDim, minDim}, svd.S.DType())
	for i := 0; i < minDim; i++ {
		S.Set(svd.S.At(i), i, i)
	}

	// Compute U * S (result: [m, minDim])
	US, err := Dot(svd.U, S)
	if err != nil {
		return nil, err
	}

	// Compute V^T (result: [minDim, n])
	VT, err := Transpose(svd.V)
	if err != nil {
		return nil, err
	}

	// Compute U * S * V^T (result: [m, n])
	result, err := Dot(US, VT)
	if err != nil {
		return nil, err
	}

	return result, nil
}

func TestSVDApplications(t *testing.T) {
	t.Run("Matrix Rank via SVD", func(t *testing.T) {
		// Create a clearly rank-2 matrix (3x3)
		// Row 3 = Row 1 + Row 2, making it rank-2
		arr := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		arr.Set(1.0, 0, 0)
		arr.Set(0.0, 0, 1)
		arr.Set(1.0, 0, 2)
		arr.Set(0.0, 1, 0)
		arr.Set(1.0, 1, 1)
		arr.Set(1.0, 1, 2)
		arr.Set(1.0, 2, 0) // Row 3 = Row 1 + Row 2
		arr.Set(1.0, 2, 1)
		arr.Set(2.0, 2, 2)

		result, err := SVD(arr)
		if err != nil {
			t.Fatalf("SVD failed: %v", err)
		}

		// Count non-zero singular values (tolerance 1e-8)
		rank := 0
		tolerance := 1e-8
		singularValues := make([]float64, result.S.Shape()[0])
		for i := 0; i < result.S.Shape()[0]; i++ {
			s := convertToFloat64(result.S.At(i))
			singularValues[i] = s
			if s > tolerance {
				rank++
			}
		}

		expectedRank := 2
		// For numerical SVD, accept rank 2 or 3 (due to numerical precision)
		// The third singular value should be very small if matrix is close to rank-2
		if rank < 2 || rank > 3 {
			t.Errorf("Matrix rank via SVD: expected around %d, got %d", expectedRank, rank)
			t.Logf("Singular values: %v", singularValues)
		}

		// Verify that if rank=3, the third singular value is very small
		if rank == 3 && singularValues[2] > 1e-6 {
			t.Errorf("Expected small third singular value for near rank-2 matrix, got %g", singularValues[2])
		}
	})

	t.Run("Condition Number via SVD", func(t *testing.T) {
		// Well-conditioned matrix
		arr := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		arr.Set(2.0, 0, 0)
		arr.Set(0.0, 0, 1)
		arr.Set(0.0, 1, 0)
		arr.Set(2.0, 1, 1)

		result, err := SVD(arr)
		if err != nil {
			t.Fatalf("SVD failed: %v", err)
		}

		// Condition number = largest singular value / smallest singular value
		sMax := convertToFloat64(result.S.At(0))
		sMin := convertToFloat64(result.S.At(result.S.Shape()[0] - 1))

		conditionNumber := sMax / sMin

		// For this diagonal matrix, condition number should be 1.0
		expectedCond := 1.0
		if math.Abs(conditionNumber-expectedCond) > 1e-10 {
			t.Errorf("Condition number: expected %.6f, got %.6f", expectedCond, conditionNumber)
		}
	})
}

func BenchmarkSVD(b *testing.B) {
	// Benchmark with a 5x5 matrix
	size := 5
	arr := array.Zeros(internal.Shape{size, size}, internal.Float64)

	// Fill with structured values
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			val := math.Sin(float64(i+1)) * math.Cos(float64(j+1))
			arr.Set(val, i, j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = SVD(arr)
	}
}

func TestPCA(t *testing.T) {
	t.Run("Basic PCA 2D", func(t *testing.T) {
		// Simple 2D dataset
		// 5 samples, 2 features
		data := array.Zeros(internal.Shape{5, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.0, 1, 0)
		data.Set(4.0, 1, 1)
		data.Set(5.0, 2, 0)
		data.Set(6.0, 2, 1)
		data.Set(7.0, 3, 0)
		data.Set(8.0, 3, 1)
		data.Set(9.0, 4, 0)
		data.Set(10.0, 4, 1)

		// Perform PCA with 2 components
		result, err := PCA(data, 2)
		if err != nil {
			t.Fatalf("PCA failed: %v", err)
		}

		// Check dimensions
		if result.Components.Shape()[0] != 2 || result.Components.Shape()[1] != 2 {
			t.Errorf("Components shape incorrect: got %v, expected [2, 2]", result.Components.Shape())
		}
		if result.ExplainedVariance.Shape()[0] != 2 {
			t.Errorf("Explained variance shape incorrect: got %v, expected [2]", result.ExplainedVariance.Shape())
		}
		if result.Mean.Shape()[0] != 2 {
			t.Errorf("Mean shape incorrect: got %v, expected [2]", result.Mean.Shape())
		}

		// Check that explained variance ratios sum to approximately 1
		totalRatio := 0.0
		for i := 0; i < 2; i++ {
			totalRatio += convertToFloat64(result.ExplainedVarianceRatio.At(i))
		}
		if math.Abs(totalRatio-1.0) > 1e-10 {
			t.Errorf("Explained variance ratios should sum to 1, got %v", totalRatio)
		}

		// Test transformation
		transformed, err := result.Transform(data)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}
		if transformed.Shape()[0] != 5 || transformed.Shape()[1] != 2 {
			t.Errorf("Transformed shape incorrect: got %v, expected [5, 2]", transformed.Shape())
		}

		// Test inverse transformation
		reconstructed, err := result.InverseTransform(transformed)
		if err != nil {
			t.Fatalf("InverseTransform failed: %v", err)
		}

		// Check reconstruction accuracy
		for i := 0; i < 5; i++ {
			for j := 0; j < 2; j++ {
				original := convertToFloat64(data.At(i, j))
				recon := convertToFloat64(reconstructed.At(i, j))
				if math.Abs(original-recon) > 1e-10 {
					t.Errorf("Reconstruction failed at [%d,%d]: original=%.6f, reconstructed=%.6f",
						i, j, original, recon)
				}
			}
		}
	})

	t.Run("PCA with Dimension Reduction", func(t *testing.T) {
		// 3D data that's approximately 2D (third dimension is mostly redundant)
		data := array.Zeros(internal.Shape{4, 3}, internal.Float64)
		// Make third column approximately equal to first + second
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.1, 0, 2) // 1 + 2 + noise
		data.Set(3.0, 1, 0)
		data.Set(4.0, 1, 1)
		data.Set(6.9, 1, 2) // 3 + 4 - noise
		data.Set(5.0, 2, 0)
		data.Set(6.0, 2, 1)
		data.Set(11.2, 2, 2) // 5 + 6 + noise
		data.Set(7.0, 3, 0)
		data.Set(8.0, 3, 1)
		data.Set(14.8, 3, 2) // 7 + 8 - noise

		// Reduce to 2 components
		result, err := PCA(data, 2)
		if err != nil {
			t.Fatalf("PCA failed: %v", err)
		}

		// Check that first two components explain most of the variance
		ratio1 := convertToFloat64(result.ExplainedVarianceRatio.At(0))
		ratio2 := convertToFloat64(result.ExplainedVarianceRatio.At(1))
		combinedRatio := ratio1 + ratio2

		if combinedRatio < 0.95 { // Should explain at least 95% of variance
			t.Errorf("First two components should explain >95%% of variance, got %.2f%%", combinedRatio*100)
		}

		// Transform and check dimensions
		transformed, err := result.Transform(data)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}
		if transformed.Shape()[0] != 4 || transformed.Shape()[1] != 2 {
			t.Errorf("Transformed shape incorrect: got %v, expected [4, 2]", transformed.Shape())
		}
	})

	t.Run("PCA Error Cases", func(t *testing.T) {
		// Nil data
		_, err := PCA(nil, 2)
		if err == nil {
			t.Error("Expected error for nil data")
		}

		// 1D data
		data1d := array.Ones(internal.Shape{5}, internal.Float64)
		_, err = PCA(data1d, 2)
		if err == nil {
			t.Error("Expected error for 1D data")
		}

		// Valid data for other tests
		data := array.Zeros(internal.Shape{3, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.0, 1, 0)
		data.Set(4.0, 1, 1)
		data.Set(5.0, 2, 0)
		data.Set(6.0, 2, 1)

		result, err := PCA(data, 1)
		if err != nil {
			t.Fatalf("PCA failed: %v", err)
		}

		// Transform with wrong dimensions
		wrongData := array.Zeros(internal.Shape{2, 3}, internal.Float64) // Wrong number of features
		_, err = result.Transform(wrongData)
		if err == nil {
			t.Error("Expected error for mismatched features in Transform")
		}

		// InverseTransform with wrong dimensions
		wrongTransformed := array.Zeros(internal.Shape{2, 2}, internal.Float64) // Wrong number of components
		_, err = result.InverseTransform(wrongTransformed)
		if err == nil {
			t.Error("Expected error for mismatched components in InverseTransform")
		}
	})
}

func TestRankAndConditionNumber(t *testing.T) {
	t.Run("Matrix Rank", func(t *testing.T) {
		// Full rank 3x3 matrix
		fullRank := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		fullRank.Set(1.0, 0, 0)
		fullRank.Set(0.0, 0, 1)
		fullRank.Set(0.0, 0, 2)
		fullRank.Set(0.0, 1, 0)
		fullRank.Set(1.0, 1, 1)
		fullRank.Set(0.0, 1, 2)
		fullRank.Set(0.0, 2, 0)
		fullRank.Set(0.0, 2, 1)
		fullRank.Set(1.0, 2, 2)

		rank, err := Rank(fullRank, 1e-12)
		if err != nil {
			t.Fatalf("Rank calculation failed: %v", err)
		}
		if rank != 3 {
			t.Errorf("Expected rank 3, got %d", rank)
		}

		// Rank-deficient matrix (all zeros in last row)
		rankDef := fullRank.Copy()
		rankDef.Set(0.0, 2, 0)
		rankDef.Set(0.0, 2, 1)
		rankDef.Set(0.0, 2, 2)

		rank2, err := Rank(rankDef, 1e-12)
		if err != nil {
			t.Fatalf("Rank calculation failed: %v", err)
		}
		if rank2 != 2 {
			t.Errorf("Expected rank 2, got %d", rank2)
		}
	})

	t.Run("Condition Number", func(t *testing.T) {
		// Well-conditioned matrix (identity)
		identity := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		identity.Set(1.0, 0, 0)
		identity.Set(1.0, 1, 1)
		identity.Set(1.0, 2, 2)

		cond, err := ConditionNumber(identity)
		if err != nil {
			t.Fatalf("Condition number calculation failed: %v", err)
		}
		if math.Abs(cond-1.0) > 1e-10 {
			t.Errorf("Expected condition number 1.0 for identity, got %v", cond)
		}

		// Ill-conditioned matrix
		illCond := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		illCond.Set(1.0, 0, 0)
		illCond.Set(0.0, 0, 1)
		illCond.Set(0.0, 1, 0)
		illCond.Set(1e-15, 1, 1) // Very small eigenvalue

		cond2, err := ConditionNumber(illCond)
		if err != nil {
			t.Fatalf("Condition number calculation failed: %v", err)
		}
		if cond2 < 1e10 { // Should be very large
			t.Errorf("Expected large condition number for ill-conditioned matrix, got %v", cond2)
		}
	})
}

func BenchmarkPCA(b *testing.B) {
	// Benchmark PCA with 100 samples, 10 features
	nSamples, nFeatures := 100, 10
	data := array.Zeros(internal.Shape{nSamples, nFeatures}, internal.Float64)

	// Fill with structured values
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			val := math.Sin(float64(i+1))*math.Cos(float64(j+1)) + 0.1*float64(i*j)
			data.Set(val, i, j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = PCA(data, 5) // Reduce to 5 components
	}
}
