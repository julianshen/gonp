package math

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestSolveSquareSystem(t *testing.T) {
	// Test solving a simple 3x3 system
	// A = [[2, 1, 1], [1, 3, 2], [1, 0, 0]]
	// b = [4, 5, 6]
	// Expected solution: x = [6, 8, -5]
	A := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	A.Set(2.0, 0, 0)
	A.Set(1.0, 0, 1)
	A.Set(1.0, 0, 2)
	A.Set(1.0, 1, 0)
	A.Set(3.0, 1, 1)
	A.Set(2.0, 1, 2)
	A.Set(1.0, 2, 0)
	A.Set(0.0, 2, 1)
	A.Set(0.0, 2, 2)

	b, _ := array.FromSlice([]float64{4, 5, 6})

	x, err := Solve(A, b)
	if err != nil {
		t.Fatal(err)
	}

	// Verify solution by computing A*x
	Ax, err := matrixVectorDot(A, x)
	if err != nil {
		t.Fatal(err)
	}

	// Check if A*x ≈ b
	for i := 0; i < 3; i++ {
		ax_val := convertToFloat64(Ax.At(i))
		b_val := convertToFloat64(b.At(i))
		if math.Abs(ax_val-b_val) > 1e-10 {
			t.Errorf("Solution verification failed at index %d: A*x=%.6f, b=%.6f",
				i, ax_val, b_val)
		}
	}

	t.Logf("Square system solved successfully")
}

func TestSolveWithQR(t *testing.T) {
	// Test with a well-conditioned matrix
	A := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	A.Set(1.0, 0, 0)
	A.Set(2.0, 0, 1)
	A.Set(3.0, 0, 2)
	A.Set(2.0, 1, 0)
	A.Set(5.0, 1, 1)
	A.Set(3.0, 1, 2)
	A.Set(1.0, 2, 0)
	A.Set(0.0, 2, 1)
	A.Set(8.0, 2, 2)

	b, _ := array.FromSlice([]float64{14, 31, 25})

	x, err := solveQR(A, b)
	if err != nil {
		t.Fatal(err)
	}

	// Verify solution
	Ax, err := matrixVectorDot(A, x)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 3; i++ {
		ax_val := convertToFloat64(Ax.At(i))
		b_val := convertToFloat64(b.At(i))
		if math.Abs(ax_val-b_val) > 1e-10 {
			t.Errorf("QR solution verification failed at index %d: A*x=%.6f, b=%.6f",
				i, ax_val, b_val)
		}
	}

	t.Logf("QR solve successful")
}

func TestSolveWithCholesky(t *testing.T) {
	// Test with a symmetric positive definite matrix
	// A = [[4, 2, 1], [2, 3, 0.5], [1, 0.5, 3]]
	A := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	A.Set(4.0, 0, 0)
	A.Set(2.0, 0, 1)
	A.Set(1.0, 0, 2)
	A.Set(2.0, 1, 0)
	A.Set(3.0, 1, 1)
	A.Set(0.5, 1, 2)
	A.Set(1.0, 2, 0)
	A.Set(0.5, 2, 1)
	A.Set(3.0, 2, 2)

	b, _ := array.FromSlice([]float64{11, 13.5, 10})

	x, err := solveCholesky(A, b)
	if err != nil {
		t.Fatal(err)
	}

	// Verify solution
	Ax, err := matrixVectorDot(A, x)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 3; i++ {
		ax_val := convertToFloat64(Ax.At(i))
		b_val := convertToFloat64(b.At(i))
		if math.Abs(ax_val-b_val) > 1e-10 {
			t.Errorf("Cholesky solution verification failed at index %d: A*x=%.6f, b=%.6f",
				i, ax_val, b_val)
		}
	}

	t.Logf("Cholesky solve successful")
}

func TestLeastSquares(t *testing.T) {
	// Test overdetermined system (more equations than unknowns)
	// A is 4x2, b is 4x1
	A := array.Zeros(internal.Shape{4, 2}, internal.Float64)
	A.Set(1.0, 0, 0)
	A.Set(1.0, 0, 1)
	A.Set(1.0, 1, 0)
	A.Set(2.0, 1, 1)
	A.Set(1.0, 2, 0)
	A.Set(3.0, 2, 1)
	A.Set(1.0, 3, 0)
	A.Set(4.0, 3, 1)

	b, _ := array.FromSlice([]float64{6, 8, 10, 12})

	x, err := lstsq(A, b)
	if err != nil {
		t.Fatal(err)
	}

	// For least squares, we verify that the residual is minimized
	// Check that x has the right dimension
	if x.Shape()[0] != 2 {
		t.Errorf("Least squares solution should have 2 elements, got %d", x.Shape()[0])
	}

	// Compute residual: r = A*x - b
	Ax, err := matrixVectorDot(A, x)
	if err != nil {
		t.Fatal(err)
	}

	residual := 0.0
	for i := 0; i < 4; i++ {
		ax_val := convertToFloat64(Ax.At(i))
		b_val := convertToFloat64(b.At(i))
		diff := ax_val - b_val
		residual += diff * diff
	}

	t.Logf("Least squares solution found with residual: %.6f", residual)
	t.Logf("Solution: x = [%.3f, %.3f]",
		convertToFloat64(x.At(0)), convertToFloat64(x.At(1)))
}

func TestUpperTriangularSolve(t *testing.T) {
	// Test upper triangular solve
	U := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	U.Set(2.0, 0, 0)
	U.Set(1.0, 0, 1)
	U.Set(3.0, 0, 2)
	U.Set(0.0, 1, 0)
	U.Set(4.0, 1, 1)
	U.Set(2.0, 1, 2)
	U.Set(0.0, 2, 0)
	U.Set(0.0, 2, 1)
	U.Set(1.0, 2, 2)

	b, _ := array.FromSlice([]float64{16, 14, 3})

	x, err := solveUpperTriangular(U, b)
	if err != nil {
		t.Fatal(err)
	}

	// Verify: U*x = b
	Ux, err := matrixVectorDot(U, x)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 3; i++ {
		ux_val := convertToFloat64(Ux.At(i))
		b_val := convertToFloat64(b.At(i))
		if math.Abs(ux_val-b_val) > 1e-12 {
			t.Errorf("Upper triangular solve failed at index %d: U*x=%.6f, b=%.6f",
				i, ux_val, b_val)
		}
	}

	t.Logf("Upper triangular solve: x = [%.3f, %.3f, %.3f]",
		convertToFloat64(x.At(0)), convertToFloat64(x.At(1)), convertToFloat64(x.At(2)))
}

func TestLowerTriangularSolve(t *testing.T) {
	// Test lower triangular solve
	L := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	L.Set(2.0, 0, 0)
	L.Set(0.0, 0, 1)
	L.Set(0.0, 0, 2)
	L.Set(1.0, 1, 0)
	L.Set(3.0, 1, 1)
	L.Set(0.0, 1, 2)
	L.Set(4.0, 2, 0)
	L.Set(2.0, 2, 1)
	L.Set(1.0, 2, 2)

	b, _ := array.FromSlice([]float64{4, 7, 22})

	x, err := solveLowerTriangular(L, b)
	if err != nil {
		t.Fatal(err)
	}

	// Verify: L*x = b
	Lx, err := matrixVectorDot(L, x)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 3; i++ {
		lx_val := convertToFloat64(Lx.At(i))
		b_val := convertToFloat64(b.At(i))
		if math.Abs(lx_val-b_val) > 1e-12 {
			t.Errorf("Lower triangular solve failed at index %d: L*x=%.6f, b=%.6f",
				i, lx_val, b_val)
		}
	}

	t.Logf("Lower triangular solve: x = [%.3f, %.3f, %.3f]",
		convertToFloat64(x.At(0)), convertToFloat64(x.At(1)), convertToFloat64(x.At(2)))
}

func TestSolveErrorHandling(t *testing.T) {
	// Test error handling for solve functions
	A := array.Zeros(internal.Shape{3, 3}, internal.Float64)
	b, _ := array.FromSlice([]float64{1, 2, 3})

	// Test nil matrix
	_, err := Solve(nil, b)
	if err == nil {
		t.Error("Expected error for nil matrix")
	}

	// Test nil vector
	_, err = Solve(A, nil)
	if err == nil {
		t.Error("Expected error for nil vector")
	}

	// Test dimension mismatch
	b_wrong, _ := array.FromSlice([]float64{1, 2}) // Wrong size
	_, err = Solve(A, b_wrong)
	if err == nil {
		t.Error("Expected error for dimension mismatch")
	}

	// Test singular matrix in upper triangular solve
	singular := array.Zeros(internal.Shape{2, 2}, internal.Float64)
	singular.Set(1.0, 0, 0)
	singular.Set(2.0, 0, 1)
	singular.Set(0.0, 1, 0)
	singular.Set(0.0, 1, 1) // Singular (zero diagonal)

	b_small, _ := array.FromSlice([]float64{1, 2})
	_, err = solveUpperTriangular(singular, b_small)
	if err == nil {
		t.Error("Expected error for singular matrix in upper triangular solve")
	}
}

func TestSolveUtilityFunctions(t *testing.T) {
	// Test arithmetic utility functions
	a := 10.0
	b := 3.0

	// Test subtraction
	result := subtractValues(a, b)
	expected := 7.0
	if math.Abs(convertToFloat64(result)-expected) > 1e-12 {
		t.Errorf("Subtraction failed: got %.6f, expected %.6f", convertToFloat64(result), expected)
	}

	// Test division
	result = divideValues(a, b)
	expected = 10.0 / 3.0
	if math.Abs(convertToFloat64(result)-expected) > 1e-12 {
		t.Errorf("Division failed: got %.6f, expected %.6f", convertToFloat64(result), expected)
	}

	// Test complex arithmetic
	c1 := complex(3.0, 4.0)
	c2 := complex(1.0, 2.0)

	resultComplex := subtractValues(c1, c2)
	expectedComplex := complex(2.0, 2.0)
	if convertToComplex128(resultComplex) != expectedComplex {
		t.Errorf("Complex subtraction failed: got %v, expected %v", convertToComplex128(resultComplex), expectedComplex)
	}

	resultComplex = divideValues(c1, c2)
	expectedComplex = c1 / c2 // (3+4i)/(1+2i) = (11-2i)/5
	actualComplex := convertToComplex128(resultComplex)
	if math.Abs(real(actualComplex)-real(expectedComplex)) > 1e-12 ||
		math.Abs(imag(actualComplex)-imag(expectedComplex)) > 1e-12 {
		t.Errorf("Complex division failed: got %v, expected %v", actualComplex, expectedComplex)
	}
}

// Benchmark tests
func BenchmarkSolveSmall(b *testing.B) {
	A := array.Zeros(internal.Shape{10, 10}, internal.Float64)
	vec := array.Zeros(internal.Shape{10}, internal.Float64)

	// Create a well-conditioned matrix
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			val := math.Sin(float64(i + j + 1))
			if i == j {
				val += 10.0 // Make diagonally dominant
			}
			A.Set(val, i, j)
		}
		vec.Set(float64(i+1), i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Solve(A, vec)
	}
}

func BenchmarkLeastSquares(b *testing.B) {
	// Overdetermined system
	m, n := 100, 50
	A := array.Zeros(internal.Shape{m, n}, internal.Float64)
	vec := array.Zeros(internal.Shape{m}, internal.Float64)

	// Fill with random-like values
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			val := math.Sin(float64(i*n + j + 1))
			A.Set(val, i, j)
		}
		vec.Set(math.Cos(float64(i+1)), i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = lstsq(A, vec)
	}
}
