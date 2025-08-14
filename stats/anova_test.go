package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
)

func TestOneWayANOVA(t *testing.T) {
	t.Run("Basic One-Way ANOVA", func(t *testing.T) {
		// Three groups with different means
		group1 := []float64{1, 2, 3, 4, 5} // mean = 3
		group2 := []float64{3, 4, 5, 6, 7} // mean = 5
		group3 := []float64{5, 6, 7, 8, 9} // mean = 7

		g1, _ := array.FromSlice(group1)
		g2, _ := array.FromSlice(group2)
		g3, _ := array.FromSlice(group3)

		groups := []*array.Array{g1, g2, g3}

		result, err := OneWayANOVA(groups)
		if err != nil {
			t.Fatalf("OneWayANOVA failed: %v", err)
		}

		// Check that F-statistic is positive and reasonable
		if result.FStatistic <= 0 {
			t.Errorf("Expected positive F-statistic, got %v", result.FStatistic)
		}

		// With clear separation between groups, F should be large
		if result.FStatistic < 5 {
			t.Errorf("Expected large F-statistic for separated groups, got %v", result.FStatistic)
		}

		// P-value should be small (significant difference)
		if result.PValue > 0.05 {
			t.Errorf("Expected significant p-value < 0.05, got %v", result.PValue)
		}

		// Degrees of freedom
		expectedDFBetween := 2 // 3 groups - 1
		expectedDFWithin := 12 // 15 total - 3
		if result.DFBetween != expectedDFBetween {
			t.Errorf("Expected DFBetween = %d, got %d", expectedDFBetween, result.DFBetween)
		}
		if result.DFWithin != expectedDFWithin {
			t.Errorf("Expected DFWithin = %d, got %d", expectedDFWithin, result.DFWithin)
		}
	})

	t.Run("ANOVA with No Difference", func(t *testing.T) {
		// Three groups with same mean
		group1 := []float64{5, 5, 5, 5, 5}
		group2 := []float64{5, 5, 5, 5, 5}
		group3 := []float64{5, 5, 5, 5, 5}

		g1, _ := array.FromSlice(group1)
		g2, _ := array.FromSlice(group2)
		g3, _ := array.FromSlice(group3)

		groups := []*array.Array{g1, g2, g3}

		result, err := OneWayANOVA(groups)
		if err != nil {
			t.Fatalf("OneWayANOVA failed: %v", err)
		}

		// F-statistic should be close to 0 (no between-group variance)
		if result.FStatistic > 1e-10 {
			t.Errorf("Expected F-statistic near 0 for identical groups, got %v", result.FStatistic)
		}

		// P-value should be high (non-significant)
		if result.PValue < 0.5 {
			t.Errorf("Expected high p-value for identical groups, got %v", result.PValue)
		}
	})

	t.Run("ANOVA with Two Groups", func(t *testing.T) {
		// Two groups (should be equivalent to t-test)
		group1 := []float64{1, 2, 3, 4, 5}
		group2 := []float64{6, 7, 8, 9, 10}

		g1, _ := array.FromSlice(group1)
		g2, _ := array.FromSlice(group2)

		groups := []*array.Array{g1, g2}

		result, err := OneWayANOVA(groups)
		if err != nil {
			t.Fatalf("OneWayANOVA failed: %v", err)
		}

		// Compare with t-test result
		tResult, err := TwoSampleTTest(g1, g2)
		if err != nil {
			t.Fatalf("TwoSampleTTest failed: %v", err)
		}

		// F should be approximately t²
		expectedF := tResult.Statistic * tResult.Statistic
		if math.Abs(result.FStatistic-expectedF) > 0.5 {
			t.Errorf("F-statistic should be t²: expected ~%v, got %v", expectedF, result.FStatistic)
		}
	})

	t.Run("ANOVA Error Cases", func(t *testing.T) {
		// Empty groups
		_, err := OneWayANOVA([]*array.Array{})
		if err == nil {
			t.Error("Expected error for empty groups")
		}

		// Single group
		group1 := []float64{1, 2, 3}
		g1, _ := array.FromSlice(group1)
		_, err = OneWayANOVA([]*array.Array{g1})
		if err == nil {
			t.Error("Expected error for single group")
		}

		// Group with single observation
		group2 := []float64{5}
		g2, _ := array.FromSlice(group2)
		_, err = OneWayANOVA([]*array.Array{g1, g2})
		if err == nil {
			t.Error("Expected error for group with single observation")
		}
	})
}

func TestTwoWayANOVA(t *testing.T) {
	t.Run("Basic Two-Way ANOVA", func(t *testing.T) {
		// Simple 2x2 design with interaction
		// Factor A: 2 levels, Factor B: 2 levels
		// Data arranged as: [A1B1, A1B2, A2B1, A2B2]
		values := []float64{
			1, 2, 3, // A1, B1
			4, 5, 6, // A1, B2
			2, 3, 4, // A2, B1
			7, 8, 9, // A2, B2
		}

		factorA := []float64{1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2}
		factorB := []float64{1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2}

		valArr, _ := array.FromSlice(values)
		factorAArr, _ := array.FromSlice(factorA)
		factorBArr, _ := array.FromSlice(factorB)

		result, err := TwoWayANOVA(valArr, factorAArr, factorBArr)
		if err != nil {
			t.Fatalf("TwoWayANOVA failed: %v", err)
		}

		// Check that F-statistics are calculated
		if result.FStatisticA <= 0 {
			t.Errorf("Expected positive F-statistic for factor A, got %v", result.FStatisticA)
		}
		if result.FStatisticB <= 0 {
			t.Errorf("Expected positive F-statistic for factor B, got %v", result.FStatisticB)
		}
		if result.FStatisticInteraction <= 0 {
			t.Errorf("Expected positive F-statistic for interaction, got %v", result.FStatisticInteraction)
		}

		// Check degrees of freedom
		if result.DFFactorA != 1 {
			t.Errorf("Expected DF for factor A = 1, got %d", result.DFFactorA)
		}
		if result.DFFactorB != 1 {
			t.Errorf("Expected DF for factor B = 1, got %d", result.DFFactorB)
		}
		if result.DFInteraction != 1 {
			t.Errorf("Expected DF for interaction = 1, got %d", result.DFInteraction)
		}
		if result.DFError != 8 {
			t.Errorf("Expected DF for error = 8, got %d", result.DFError)
		}
	})

	t.Run("Two-Way ANOVA Error Cases", func(t *testing.T) {
		values := []float64{1, 2, 3}
		factorA := []float64{1, 1} // Mismatched length
		factorB := []float64{1, 2, 1}

		valArr, _ := array.FromSlice(values)
		factorAArr, _ := array.FromSlice(factorA)
		factorBArr, _ := array.FromSlice(factorB)

		_, err := TwoWayANOVA(valArr, factorAArr, factorBArr)
		if err == nil {
			t.Error("Expected error for mismatched array lengths")
		}
	})
}

func TestRepeatedMeasuresANOVA(t *testing.T) {
	t.Run("Basic Repeated Measures ANOVA", func(t *testing.T) {
		// 3 subjects, 4 time points
		// Each row is a subject, each column is a time point
		// Strong time effect with some realistic variation
		data := []float64{
			1.1, 4.0, 6.9, 10.2, // Subject 1: clear time trend + noise
			1.9, 5.1, 7.8, 11.1, // Subject 2: same trend + noise
			3.2, 5.8, 9.1, 11.8, // Subject 3: same trend + noise
		}

		dataArr, _ := array.NewArrayWithShape(data, []int{3, 4})

		result, err := RepeatedMeasuresANOVA(dataArr)
		if err != nil {
			t.Fatalf("RepeatedMeasuresANOVA failed: %v", err)
		}

		// Check F-statistic for time effect
		if result.FStatisticTime <= 0 {
			t.Errorf("Expected positive F-statistic for time, got %v", result.FStatisticTime)
		}

		// With linear trend, should be significant
		if result.PValueTime > 0.05 {
			t.Errorf("Expected significant time effect, got p = %v", result.PValueTime)
		}

		// Check degrees of freedom
		expectedDFTime := 3  // 4 time points - 1
		expectedDFError := 6 // (subjects-1) * (time-1) = 2*3
		if result.DFTime != expectedDFTime {
			t.Errorf("Expected DF for time = %d, got %d", expectedDFTime, result.DFTime)
		}
		if result.DFError != expectedDFError {
			t.Errorf("Expected DF for error = %d, got %d", expectedDFError, result.DFError)
		}
	})

	t.Run("Repeated Measures ANOVA Error Cases", func(t *testing.T) {
		// 1D array (should be 2D)
		data := []float64{1, 2, 3, 4}
		dataArr, _ := array.FromSlice(data)

		_, err := RepeatedMeasuresANOVA(dataArr)
		if err == nil {
			t.Error("Expected error for 1D array")
		}

		// Single subject
		data2 := []float64{1, 2, 3, 4}
		dataArr2, _ := array.NewArrayWithShape(data2, []int{1, 4})
		_, err = RepeatedMeasuresANOVA(dataArr2)
		if err == nil {
			t.Error("Expected error for single subject")
		}
	})
}

func TestMANOVA(t *testing.T) {
	t.Run("Basic MANOVA", func(t *testing.T) {
		// Two groups, two dependent variables
		// Group 1: lower values on both variables
		// Group 2: higher values on both variables
		group1Data := []float64{
			1, 2, // Subject 1: var1=1, var2=2
			2, 3, // Subject 2: var1=2, var2=3
			1, 3, // Subject 3: var1=1, var2=3
		}
		group2Data := []float64{
			4, 5, // Subject 1: var1=4, var2=5
			5, 6, // Subject 2: var1=5, var2=6
			4, 6, // Subject 3: var1=4, var2=6
		}

		group1, _ := array.NewArrayWithShape(group1Data, []int{3, 2})
		group2, _ := array.NewArrayWithShape(group2Data, []int{3, 2})

		groups := []*array.Array{group1, group2}

		result, err := MANOVA(groups)
		if err != nil {
			t.Fatalf("MANOVA failed: %v", err)
		}

		// Check that test statistics are calculated
		if result.WilksLambda <= 0 || result.WilksLambda >= 1 {
			t.Errorf("Wilks' Lambda should be between 0 and 1, got %v", result.WilksLambda)
		}

		if result.PillaiTrace < 0 {
			t.Errorf("Pillai's trace should be non-negative, got %v", result.PillaiTrace)
		}

		if result.FStatistic <= 0 {
			t.Errorf("Expected positive F-statistic, got %v", result.FStatistic)
		}

		// With clear group separation, should be significant
		if result.PValue > 0.05 {
			t.Errorf("Expected significant result, got p = %v", result.PValue)
		}
	})

	t.Run("MANOVA Error Cases", func(t *testing.T) {
		// Single group
		data := []float64{1, 2, 3, 4}
		group1, _ := array.NewArrayWithShape(data, []int{2, 2})
		_, err := MANOVA([]*array.Array{group1})
		if err == nil {
			t.Error("Expected error for single group")
		}

		// 1D data (should be 2D)
		data1d := []float64{1, 2, 3}
		group1d, _ := array.FromSlice(data1d)
		group2d, _ := array.FromSlice(data1d)
		_, err = MANOVA([]*array.Array{group1d, group2d})
		if err == nil {
			t.Error("Expected error for 1D arrays")
		}
	})
}

func TestKruskalWallisTest(t *testing.T) {
	t.Run("Basic Kruskal-Wallis Test", func(t *testing.T) {
		// Three groups with different medians
		group1 := []float64{1, 2, 3, 4, 5}
		group2 := []float64{3, 4, 5, 6, 7}
		group3 := []float64{6, 7, 8, 9, 10}

		g1, _ := array.FromSlice(group1)
		g2, _ := array.FromSlice(group2)
		g3, _ := array.FromSlice(group3)

		groups := []*array.Array{g1, g2, g3}

		result, err := KruskalWallisTest(groups)
		if err != nil {
			t.Fatalf("KruskalWallisTest failed: %v", err)
		}

		// Check H-statistic
		if result.HStatistic <= 0 {
			t.Errorf("Expected positive H-statistic, got %v", result.HStatistic)
		}

		// With clear separation, should be significant
		if result.PValue > 0.05 {
			t.Errorf("Expected significant result, got p = %v", result.PValue)
		}

		// Check degrees of freedom
		expectedDF := 2 // 3 groups - 1
		if result.DegreesOfFreedom != expectedDF {
			t.Errorf("Expected DF = %d, got %d", expectedDF, result.DegreesOfFreedom)
		}
	})

	t.Run("Kruskal-Wallis with Identical Groups", func(t *testing.T) {
		// Three identical groups
		group1 := []float64{5, 5, 5}
		group2 := []float64{5, 5, 5}
		group3 := []float64{5, 5, 5}

		g1, _ := array.FromSlice(group1)
		g2, _ := array.FromSlice(group2)
		g3, _ := array.FromSlice(group3)

		groups := []*array.Array{g1, g2, g3}

		result, err := KruskalWallisTest(groups)
		if err != nil {
			t.Fatalf("KruskalWallisTest failed: %v", err)
		}

		// H-statistic should be close to 0
		if result.HStatistic > 1e-10 {
			t.Errorf("Expected H-statistic near 0 for identical groups, got %v", result.HStatistic)
		}

		// P-value should be high
		if result.PValue < 0.5 {
			t.Errorf("Expected high p-value for identical groups, got %v", result.PValue)
		}
	})
}

func TestFriedmanTest(t *testing.T) {
	t.Run("Basic Friedman Test", func(t *testing.T) {
		// 4 subjects, 3 treatments
		// Each row is a subject, each column is a treatment
		data := []float64{
			1, 3, 2, // Subject 1: Treatment 1=1, 2=3, 3=2
			2, 4, 3, // Subject 2: Treatment 1=2, 2=4, 3=3
			1, 4, 2, // Subject 3: Treatment 1=1, 2=4, 3=2
			2, 5, 3, // Subject 4: Treatment 1=2, 2=5, 3=3
		}

		dataArr, _ := array.NewArrayWithShape(data, []int{4, 3})

		result, err := FriedmanTest(dataArr)
		if err != nil {
			t.Fatalf("FriedmanTest failed: %v", err)
		}

		// Check Chi-square statistic
		if result.ChiSquareStatistic <= 0 {
			t.Errorf("Expected positive chi-square statistic, got %v", result.ChiSquareStatistic)
		}

		// Treatment 2 consistently highest, should be significant
		if result.PValue > 0.05 {
			t.Errorf("Expected significant result, got p = %v", result.PValue)
		}

		// Check degrees of freedom
		expectedDF := 2 // 3 treatments - 1
		if result.DegreesOfFreedom != expectedDF {
			t.Errorf("Expected DF = %d, got %d", expectedDF, result.DegreesOfFreedom)
		}
	})

	t.Run("Friedman Test Error Cases", func(t *testing.T) {
		// 1D array (should be 2D)
		data := []float64{1, 2, 3}
		dataArr, _ := array.FromSlice(data)

		_, err := FriedmanTest(dataArr)
		if err == nil {
			t.Error("Expected error for 1D array")
		}

		// Single subject
		data2 := []float64{1, 2, 3}
		dataArr2, _ := array.NewArrayWithShape(data2, []int{1, 3})
		_, err = FriedmanTest(dataArr2)
		if err == nil {
			t.Error("Expected error for single subject")
		}
	})
}
