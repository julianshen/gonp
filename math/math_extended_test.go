package math

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Extended tests to increase coverage for math package

// Test exponential functions that weren't covered
func TestExponentialExtended(t *testing.T) {
	data := []float64{0, 1, 2, 3, -1}
	arr := createTestArray(data)

	t.Run("Exp10", func(t *testing.T) {
		result, err := Exp10(arr)
		if err != nil {
			t.Fatalf("Exp10 failed: %v", err)
		}

		expected := []float64{1, 10, 100, 1000, 0.1}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Expm1", func(t *testing.T) {
		result, err := Expm1(arr)
		if err != nil {
			t.Fatalf("Expm1 failed: %v", err)
		}

		expected := []float64{0, math.E - 1, math.E*math.E - 1, math.E*math.E*math.E - 1, 1/math.E - 1}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Log2", func(t *testing.T) {
		positiveData := []float64{1, 2, 4, 8, 16}
		positiveArr := createTestArray(positiveData)

		result, err := Log2(positiveArr)
		if err != nil {
			t.Fatalf("Log2 failed: %v", err)
		}

		expected := []float64{0, 1, 2, 3, 4}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Log1p", func(t *testing.T) {
		positiveData := []float64{0, math.E - 1, math.E*math.E - 1}
		positiveArr := createTestArray(positiveData)

		result, err := Log1p(positiveArr)
		if err != nil {
			t.Fatalf("Log1p failed: %v", err)
		}

		expected := []float64{0, 1, 2}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Logb", func(t *testing.T) {
		positiveData := []float64{1, 2, 4, 8}
		positiveArr := createTestArray(positiveData)

		result, err := Logb(positiveArr)
		if err != nil {
			t.Fatalf("Logb failed: %v", err)
		}

		// Logb returns the binary exponent
		expected := []float64{0, 1, 2, 3}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})
}

// Test inverse hyperbolic functions
func TestInverseHyperbolic(t *testing.T) {
	t.Run("Asinh", func(t *testing.T) {
		data := []float64{0, 1, 2, 3}
		arr := createTestArray(data)

		result, err := Asinh(arr)
		if err != nil {
			t.Fatalf("Asinh failed: %v", err)
		}

		expected := []float64{0, math.Asinh(1), math.Asinh(2), math.Asinh(3)}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Acosh", func(t *testing.T) {
		data := []float64{1, 2, 3, 4} // Acosh requires x >= 1
		arr := createTestArray(data)

		result, err := Acosh(arr)
		if err != nil {
			t.Fatalf("Acosh failed: %v", err)
		}

		expected := []float64{0, math.Acosh(2), math.Acosh(3), math.Acosh(4)}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Atanh", func(t *testing.T) {
		data := []float64{0, 0.5, 0.8, 0.9} // Atanh requires |x| < 1
		arr := createTestArray(data)

		result, err := Atanh(arr)
		if err != nil {
			t.Fatalf("Atanh failed: %v", err)
		}

		expected := []float64{0, math.Atanh(0.5), math.Atanh(0.8), math.Atanh(0.9)}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})
}

// Test additional trigonometric functions
func TestTrigonometricExtended(t *testing.T) {
	t.Run("Atan2", func(t *testing.T) {
		y := createTestArray([]float64{1, 1, -1, -1})
		x := createTestArray([]float64{1, -1, 1, -1})

		result, err := Atan2(y, x)
		if err != nil {
			t.Fatalf("Atan2 failed: %v", err)
		}

		expected := []float64{math.Pi / 4, 3 * math.Pi / 4, -math.Pi / 4, -3 * math.Pi / 4}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Deg2Rad", func(t *testing.T) {
		degrees := createTestArray([]float64{0, 90, 180, 270, 360})

		result, err := Deg2Rad(degrees)
		if err != nil {
			t.Fatalf("Deg2Rad failed: %v", err)
		}

		expected := []float64{0, math.Pi / 2, math.Pi, 3 * math.Pi / 2, 2 * math.Pi}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Rad2Deg", func(t *testing.T) {
		radians := createTestArray([]float64{0, math.Pi / 2, math.Pi, 3 * math.Pi / 2, 2 * math.Pi})

		result, err := Rad2Deg(radians)
		if err != nil {
			t.Fatalf("Rad2Deg failed: %v", err)
		}

		expected := []float64{0, 90, 180, 270, 360}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Hypot", func(t *testing.T) {
		x := createTestArray([]float64{3, 5, 8, 0})
		y := createTestArray([]float64{4, 12, 15, 1})

		result, err := Hypot(x, y)
		if err != nil {
			t.Fatalf("Hypot failed: %v", err)
		}

		expected := []float64{5, 13, 17, 1}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})
}

// Test power functions that weren't covered
func TestPowerExtended(t *testing.T) {
	t.Run("PowerScalar", func(t *testing.T) {
		data := []float64{1, 2, 3, 4}
		arr := createTestArray(data)

		result, err := PowerScalar(arr, 3.0)
		if err != nil {
			t.Fatalf("PowerScalar failed: %v", err)
		}

		expected := []float64{1, 8, 27, 64}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Reciprocal", func(t *testing.T) {
		data := []float64{1, 2, 4, 0.5}
		arr := createTestArray(data)

		result, err := Reciprocal(arr)
		if err != nil {
			t.Fatalf("Reciprocal failed: %v", err)
		}

		expected := []float64{1, 0.5, 0.25, 2}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("ReciprocalWithZero", func(t *testing.T) {
		data := []float64{0, 1, -1}
		arr := createTestArray(data)

		result, err := Reciprocal(arr)
		if err != nil {
			t.Fatalf("Reciprocal failed: %v", err)
		}

		// Check that 1/0 gives infinity
		if !math.IsInf(result.At(0).(float64), 1) {
			t.Errorf("Expected +Inf for 1/0, got %v", result.At(0))
		}

		if result.At(1) != 1.0 {
			t.Errorf("Expected 1.0 for 1/1, got %v", result.At(1))
		}

		if result.At(2) != -1.0 {
			t.Errorf("Expected -1.0 for 1/(-1), got %v", result.At(2))
		}
	})
}

// Test rounding functions that weren't covered
func TestRoundingExtended(t *testing.T) {
	data := []float64{1.234, 1.567, -1.234, -1.567, 2.5, -2.5}
	arr := createTestArray(data)

	t.Run("RoundN", func(t *testing.T) {
		result, err := RoundN(arr, 1)
		if err != nil {
			t.Fatalf("RoundN failed: %v", err)
		}

		expected := []float64{1.2, 1.6, -1.2, -1.6, 2.5, -2.5}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Fix", func(t *testing.T) {
		result, err := Fix(arr)
		if err != nil {
			t.Fatalf("Fix failed: %v", err)
		}

		expected := []float64{1, 1, -1, -1, 2, -2}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Rint", func(t *testing.T) {
		result, err := Rint(arr)
		if err != nil {
			t.Fatalf("Rint failed: %v", err)
		}

		// Rint uses round-to-even
		expected := []float64{1, 2, -1, -2, 2, -2}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})
}

// Test utility functions that weren't covered
func TestUtilityExtended(t *testing.T) {
	t.Run("Fmod", func(t *testing.T) {
		x := createTestArray([]float64{7, 10, 15, 20})
		y := createTestArray([]float64{3, 3, 4, 6})

		result, err := Fmod(x, y)
		if err != nil {
			t.Fatalf("Fmod failed: %v", err)
		}

		expected := []float64{1, 1, 3, 2}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Remainder", func(t *testing.T) {
		x := createTestArray([]float64{7, 10, 15, 20})
		y := createTestArray([]float64{3, 3, 4, 6})

		result, err := Remainder(x, y)
		if err != nil {
			t.Fatalf("Remainder failed: %v", err)
		}

		expected := []float64{math.Remainder(7, 3), math.Remainder(10, 3), math.Remainder(15, 4), math.Remainder(20, 6)}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Copysign", func(t *testing.T) {
		mag := createTestArray([]float64{1, 2, 3, 4})
		sign := createTestArray([]float64{1, -1, 1, -1})

		result, err := Copysign(mag, sign)
		if err != nil {
			t.Fatalf("Copysign failed: %v", err)
		}

		expected := []float64{1, -2, 3, -4}
		for i, exp := range expected {
			actualVal := result.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})
}

// Test complex number handling in Square function
func TestSquareComplexNumbers(t *testing.T) {
	complexData := []complex128{1 + 1i, 2 - 2i, -1 + 3i}
	arr, _ := array.FromSlice(complexData)

	result, err := Square(arr)
	if err != nil {
		t.Fatalf("Square with complex numbers failed: %v", err)
	}

	expected := []complex128{(1 + 1i) * (1 + 1i), (2 - 2i) * (2 - 2i), (-1 + 3i) * (-1 + 3i)}
	for i, exp := range expected {
		actualVal := result.At(i)
		if actualVal != exp {
			t.Errorf("Expected %v at position %d, got %v", exp, i, actualVal)
		}
	}
}

// Test all missing in-place operations to boost coverage
func TestMissingInPlaceOperations(t *testing.T) {
	// Exponential in-place operations
	t.Run("Exp2InPlace", func(t *testing.T) {
		data := []float64{0, 1, 2, 3}
		arr := createTestArray(data)

		err := Exp2InPlace(arr)
		if err != nil {
			t.Fatalf("Exp2InPlace failed: %v", err)
		}

		expected := []float64{1, 2, 4, 8}
		for i, exp := range expected {
			actualVal := arr.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Exp10InPlace", func(t *testing.T) {
		data := []float64{0, 1, 2}
		arr := createTestArray(data)

		err := Exp10InPlace(arr)
		if err != nil {
			t.Fatalf("Exp10InPlace failed: %v", err)
		}

		expected := []float64{1, 10, 100}
		for i, exp := range expected {
			actualVal := arr.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("Log2InPlace", func(t *testing.T) {
		data := []float64{1, 2, 4, 8}
		arr := createTestArray(data)

		err := Log2InPlace(arr)
		if err != nil {
			t.Fatalf("Log2InPlace failed: %v", err)
		}

		expected := []float64{0, 1, 2, 3}
		for i, exp := range expected {
			actualVal := arr.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("TanhInPlace", func(t *testing.T) {
		data := []float64{0, 1, -1}
		arr := createTestArray(data)

		err := TanhInPlace(arr)
		if err != nil {
			t.Fatalf("TanhInPlace failed: %v", err)
		}

		expected := []float64{0, math.Tanh(1), math.Tanh(-1)}
		for i, exp := range expected {
			actualVal := arr.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("CosInPlace", func(t *testing.T) {
		data := []float64{0, math.Pi / 2, math.Pi}
		arr := createTestArray(data)

		err := CosInPlace(arr)
		if err != nil {
			t.Fatalf("CosInPlace failed: %v", err)
		}

		expected := []float64{1, 0, -1}
		for i, exp := range expected {
			actualVal := arr.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("SignInPlace", func(t *testing.T) {
		data := []float64{-5, 0, 3}
		arr := createTestArray(data)

		err := SignInPlace(arr)
		if err != nil {
			t.Fatalf("SignInPlace failed: %v", err)
		}

		expected := []float64{-1, 0, 1}
		for i, exp := range expected {
			actualVal := arr.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})

	t.Run("ClipInPlace", func(t *testing.T) {
		data := []float64{-2, -1, 0, 1, 2, 3, 4, 5}
		arr := createTestArray(data)

		err := ClipInPlace(arr, 0.0, 3.0)
		if err != nil {
			t.Fatalf("ClipInPlace failed: %v", err)
		}

		expected := []float64{0, 0, 0, 1, 2, 3, 3, 3}
		for i, exp := range expected {
			actualVal := arr.At(i)
			if !valuesEqual(actualVal, exp, 1e-10) {
				t.Errorf("Expected %f at position %d, got %v", exp, i, actualVal)
			}
		}
	})
}

// Test edge cases and error conditions
func TestMathEdgeCases(t *testing.T) {
	t.Run("EmptyArray", func(t *testing.T) {
		// Create empty array using Empty function
		arr := array.Empty([]int{0}, internal.Float64)

		result, err := Sin(arr)
		if err != nil {
			t.Fatalf("Sin with empty array failed: %v", err)
		}

		if result.Size() != 0 {
			t.Errorf("Expected empty result, got size %d", result.Size())
		}
	})

	t.Run("SingleElement", func(t *testing.T) {
		arr := createTestArray([]float64{math.Pi})

		result, err := Sin(arr)
		if err != nil {
			t.Fatalf("Sin with single element failed: %v", err)
		}

		if result.Size() != 1 {
			t.Errorf("Expected single element result, got size %d", result.Size())
		}

		// sin(π) should be approximately 0
		actualVal := result.At(0)
		if !valuesEqual(actualVal, 0.0, 1e-10) {
			t.Errorf("Expected sin(π) ≈ 0, got %v", actualVal)
		}
	})

	t.Run("NaNAndInf", func(t *testing.T) {
		data := []float64{math.NaN(), math.Inf(1), math.Inf(-1)}
		arr := createTestArray(data)

		// Test that functions handle NaN and Inf gracefully
		_, err := Sin(arr)
		if err != nil {
			t.Fatalf("Sin with NaN/Inf failed: %v", err)
		}

		_, err = Exp(arr)
		if err != nil {
			t.Fatalf("Exp with NaN/Inf failed: %v", err)
		}

		_, err = Sqrt(arr)
		if err != nil {
			t.Fatalf("Sqrt with NaN/Inf failed: %v", err)
		}
	})
}
