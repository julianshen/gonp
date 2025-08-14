package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TestMICEImputer tests the Multiple Imputation by Chained Equations implementation using TDD
func TestMICEImputer(t *testing.T) {
	t.Run("Basic MICE imputation with multiple datasets", func(t *testing.T) {
		// Red phase: Write a failing test first
		// Create data with missing values where features have relationships
		data := array.Empty(internal.Shape{6, 3}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.0, 0, 2) // Complete sample
		data.Set(math.NaN(), 1, 0)
		data.Set(4.0, 1, 1)
		data.Set(6.0, 1, 2) // Missing x0, x0 ≈ x1/2
		data.Set(3.0, 2, 0)
		data.Set(math.NaN(), 2, 1)
		data.Set(9.0, 2, 2) // Missing x1, x1 ≈ x2/3
		data.Set(4.0, 3, 0)
		data.Set(8.0, 3, 1)
		data.Set(math.NaN(), 3, 2) // Missing x2, x2 ≈ 2*x1
		data.Set(5.0, 4, 0)
		data.Set(10.0, 4, 1)
		data.Set(15.0, 4, 2) // Complete sample
		data.Set(6.0, 5, 0)
		data.Set(12.0, 5, 1)
		data.Set(18.0, 5, 2) // Complete sample

		// MICE should generate multiple imputed datasets (e.g., 3 datasets)
		mice := NewMICEImputer(3) // 3 multiple imputations
		datasets, err := mice.FitTransform(data)
		if err != nil {
			t.Fatalf("MICE imputation failed: %v", err)
		}

		// Should return multiple datasets
		if len(datasets) != 3 {
			t.Errorf("Expected 3 imputed datasets, got %d", len(datasets))
		}

		// Each dataset should have the same shape as input
		for i, dataset := range datasets {
			if dataset.Shape()[0] != 6 || dataset.Shape()[1] != 3 {
				t.Errorf("Dataset %d has wrong shape: expected [6,3], got %v", i, dataset.Shape())
			}

			// Check that missing values were imputed (no NaN values)
			for row := 0; row < 6; row++ {
				for col := 0; col < 3; col++ {
					value := dataset.At(row, col).(float64)
					if math.IsNaN(value) {
						t.Errorf("Dataset %d still has NaN at [%d,%d]", i, row, col)
					}
				}
			}
		}

		// Different imputations should produce different results (uncertainty)
		dataset1Value := datasets[0].At(1, 0).(float64) // First imputation of missing x0
		dataset2Value := datasets[1].At(1, 0).(float64) // Second imputation of missing x0

		// Results might be the same or different depending on randomness and relationships
		t.Logf("MICE imputation variations: %.3f, %.3f", dataset1Value, dataset2Value)
		t.Logf("Multiple datasets generated successfully")
	})

	t.Run("MICE imputation with custom estimator", func(t *testing.T) {
		// Test MICE with different regression estimators
		data := array.Empty(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(10.0, 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(20.0, 1, 1) // Missing, should be ~2
		data.Set(3.0, 2, 0)
		data.Set(30.0, 2, 1)
		data.Set(4.0, 3, 0)
		data.Set(math.NaN(), 3, 1) // Missing, should be ~40

		// Test with linear regression estimator
		miceLinear := NewMICEImputerWithEstimator("linear_regression")
		datasetsLinear, err := miceLinear.FitTransform(data)
		if err != nil {
			t.Fatalf("MICE with linear regression failed: %v", err)
		}

		// Test with ridge regression estimator
		miceRidge := NewMICEImputerWithEstimator("ridge_regression")
		datasetsRidge, err := miceRidge.FitTransform(data)
		if err != nil {
			t.Fatalf("MICE with ridge regression failed: %v", err)
		}

		// Both should produce valid results
		if len(datasetsLinear) == 0 || len(datasetsRidge) == 0 {
			t.Error("Expected non-empty datasets from both estimators")
		}

		linearValue := datasetsLinear[0].At(1, 0).(float64)
		ridgeValue := datasetsRidge[0].At(1, 0).(float64)

		t.Logf("Linear regression imputation: %.3f", linearValue)
		t.Logf("Ridge regression imputation: %.3f", ridgeValue)
	})

	t.Run("MICE convergence and iterations", func(t *testing.T) {
		// Test MICE iteration control and convergence
		data := array.Empty(internal.Shape{5, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(4.0, 1, 1)
		data.Set(3.0, 2, 0)
		data.Set(math.NaN(), 2, 1)
		data.Set(4.0, 3, 0)
		data.Set(8.0, 3, 1)
		data.Set(5.0, 4, 0)
		data.Set(10.0, 4, 1)

		mice := NewMICEImputer(2)
		mice.SetMaxIterations(5)           // Limit iterations
		mice.SetConvergenceTolerance(1e-3) // Convergence threshold

		datasets, err := mice.FitTransform(data)
		if err != nil {
			t.Fatalf("MICE with convergence control failed: %v", err)
		}

		// Check convergence information
		converged := mice.HasConverged()
		iterations := mice.GetActualIterations()

		t.Logf("MICE converged: %v after %d iterations", converged, iterations)

		// Should have performed some iterations
		if iterations == 0 {
			t.Error("Expected at least one iteration")
		}

		// Should not exceed max iterations
		if iterations > 5 {
			t.Errorf("Expected at most 5 iterations, got %d", iterations)
		}

		if len(datasets) != 2 {
			t.Errorf("Expected 2 datasets, got %d", len(datasets))
		}
	})

	t.Run("MICE parameter validation", func(t *testing.T) {
		// Test invalid parameters

		// Invalid number of imputations
		mice := NewMICEImputer(0)
		data := array.Ones(internal.Shape{3, 2}, internal.Float64)
		_, err := mice.FitTransform(data)
		if err == nil {
			t.Error("Expected error for 0 imputations")
		}

		// Nil data
		mice = NewMICEImputer(2)
		_, err = mice.FitTransform(nil)
		if err == nil {
			t.Error("Expected error for nil data")
		}

		// Invalid estimator
		mice = NewMICEImputerWithEstimator("invalid_estimator")
		_, err = mice.FitTransform(data)
		if err == nil {
			t.Error("Expected error for invalid estimator")
		}

		// Invalid max iterations
		mice = NewMICEImputer(2)
		mice.SetMaxIterations(0)
		_, err = mice.FitTransform(data)
		if err == nil {
			t.Error("Expected error for 0 max iterations")
		}

		// Invalid convergence tolerance
		mice = NewMICEImputer(2)
		mice.SetConvergenceTolerance(-1.0)
		_, err = mice.FitTransform(data)
		if err == nil {
			t.Error("Expected error for negative convergence tolerance")
		}
	})

	t.Run("MICE with all features missing in some samples", func(t *testing.T) {
		// Test edge case where some samples have all features missing
		data := array.Empty(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(math.NaN(), 1, 1) // All missing
		data.Set(3.0, 2, 0)
		data.Set(6.0, 2, 1)
		data.Set(4.0, 3, 0)
		data.Set(8.0, 3, 1)

		mice := NewMICEImputer(2)
		datasets, err := mice.FitTransform(data)

		// This might be handled or might return an error
		if err != nil {
			t.Logf("All-missing sample handled with error: %v", err)
		} else {
			t.Logf("All-missing sample imputed successfully")
			if len(datasets) != 2 {
				t.Errorf("Expected 2 datasets, got %d", len(datasets))
			}
		}
	})
}

// TestMICEImputerEdgeCases tests edge cases and boundary conditions
func TestMICEImputerEdgeCases(t *testing.T) {
	t.Run("MICE with no missing values", func(t *testing.T) {
		// Complete data - MICE should return original data
		data := array.Empty(internal.Shape{3, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(3.0, 1, 0)
		data.Set(4.0, 1, 1)
		data.Set(5.0, 2, 0)
		data.Set(6.0, 2, 1)

		mice := NewMICEImputer(2)
		datasets, err := mice.FitTransform(data)
		if err != nil {
			t.Fatalf("MICE on complete data failed: %v", err)
		}

		// Should still return multiple datasets even for complete data
		if len(datasets) != 2 {
			t.Errorf("Expected 2 datasets, got %d", len(datasets))
		}

		// Data should be unchanged in each dataset
		for i, dataset := range datasets {
			for row := 0; row < 3; row++ {
				for col := 0; col < 2; col++ {
					original := data.At(row, col).(float64)
					result := dataset.At(row, col).(float64)
					if math.Abs(original-result) > 1e-15 {
						t.Errorf("Dataset %d data changed at [%d,%d]: %.6f -> %.6f",
							i, row, col, original, result)
					}
				}
			}
		}

		t.Logf("Complete data preserved in all imputations")
	})

	t.Run("MICE with single feature", func(t *testing.T) {
		// Single feature case
		data := array.Empty(internal.Shape{4, 1}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(math.NaN(), 1, 0) // Missing
		data.Set(3.0, 2, 0)
		data.Set(5.0, 3, 0)

		mice := NewMICEImputer(2)
		datasets, err := mice.FitTransform(data)

		// Single feature MICE might fall back to simple imputation
		if err != nil {
			t.Logf("Single feature MICE handled with error: %v", err)
		} else {
			// Should have imputed the missing value
			for i, dataset := range datasets {
				value := dataset.At(1, 0).(float64)
				if math.IsNaN(value) {
					t.Errorf("Dataset %d still has NaN in single feature", i)
				}
				t.Logf("Dataset %d single feature imputed: %.3f", i, value)
			}
		}
	})

	t.Run("MICE diagnostic information", func(t *testing.T) {
		// Test diagnostic and monitoring features
		data := array.Empty(internal.Shape{4, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(10.0, 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(20.0, 1, 1)
		data.Set(3.0, 2, 0)
		data.Set(math.NaN(), 2, 1)
		data.Set(4.0, 3, 0)
		data.Set(40.0, 3, 1)

		mice := NewMICEImputer(2)
		mice.SetMaxIterations(3)
		mice.SetEnableDiagnostics(true) // Enable detailed diagnostics

		datasets, err := mice.FitTransform(data)
		if err != nil {
			t.Fatalf("MICE with diagnostics failed: %v", err)
		}

		// Get diagnostic information
		diagnostics := mice.GetDiagnostics()
		if diagnostics == nil {
			t.Error("Expected diagnostic information")
		} else {
			t.Logf("MICE diagnostics available: iterations=%v", diagnostics.Iterations)
			t.Logf("Convergence history: %v", diagnostics.ConvergenceHistory)
		}

		// Check iteration information
		actualIterations := mice.GetActualIterations()
		converged := mice.HasConverged()

		t.Logf("Actual iterations: %d, Converged: %v", actualIterations, converged)

		if len(datasets) != 2 {
			t.Errorf("Expected 2 datasets, got %d", len(datasets))
		}
	})
}

// TestMICEImputerAPI tests the API consistency and usability
func TestMICEImputerAPI(t *testing.T) {
	t.Run("MICE API consistency", func(t *testing.T) {
		// Test that MICE follows the same patterns as other imputers
		data := array.Empty(internal.Shape{3, 2}, internal.Float64)
		data.Set(1.0, 0, 0)
		data.Set(2.0, 0, 1)
		data.Set(math.NaN(), 1, 0)
		data.Set(4.0, 1, 1)
		data.Set(3.0, 2, 0)
		data.Set(6.0, 2, 1)

		// Test separate Fit and Transform
		mice := NewMICEImputer(2)
		err := mice.Fit(data)
		if err != nil {
			t.Fatalf("MICE Fit failed: %v", err)
		}

		datasets, err := mice.Transform(data)
		if err != nil {
			t.Fatalf("MICE Transform failed: %v", err)
		}

		// Test FitTransform
		mice2 := NewMICEImputer(2)
		datasets2, err := mice2.FitTransform(data)
		if err != nil {
			t.Fatalf("MICE FitTransform failed: %v", err)
		}

		// Both methods should produce valid results
		if len(datasets) != 2 || len(datasets2) != 2 {
			t.Error("Expected 2 datasets from both methods")
		}

		t.Logf("MICE API consistency verified")
	})

	t.Run("MICE getter methods", func(t *testing.T) {
		// Test getter methods for MICE configuration
		mice := NewMICEImputer(3)
		mice.SetMaxIterations(10)
		mice.SetConvergenceTolerance(1e-4)

		if mice.GetNImputations() != 3 {
			t.Errorf("Expected 3 imputations, got %d", mice.GetNImputations())
		}

		if mice.GetMaxIterations() != 10 {
			t.Errorf("Expected 10 max iterations, got %d", mice.GetMaxIterations())
		}

		if math.Abs(mice.GetConvergenceTolerance()-1e-4) > 1e-15 {
			t.Errorf("Expected 1e-4 tolerance, got %v", mice.GetConvergenceTolerance())
		}

		estimator := mice.GetEstimator()
		if estimator == "" {
			t.Error("Expected non-empty estimator")
		}

		t.Logf("MICE getters: imputations=%d, iterations=%d, tolerance=%v, estimator=%s",
			mice.GetNImputations(), mice.GetMaxIterations(),
			mice.GetConvergenceTolerance(), estimator)
	})
}
