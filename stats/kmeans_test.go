package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestKMeans(t *testing.T) {
	t.Run("Simple 2D clustering", func(t *testing.T) {
		// Create clearly separable clusters
		data := [][]float64{
			// Cluster 1: around (1, 1)
			{1, 1}, {1.2, 0.8}, {0.8, 1.2}, {1.1, 0.9},
			// Cluster 2: around (5, 5)
			{5, 5}, {5.2, 4.8}, {4.8, 5.2}, {5.1, 4.9},
			// Cluster 3: around (1, 5)
			{1, 5}, {1.2, 4.8}, {0.8, 5.2}, {1.1, 5.1},
		}

		dataArray := array.Empty(internal.Shape{len(data), len(data[0])}, internal.Float64)
		for i, row := range data {
			for j, val := range row {
				dataArray.Set(val, i, j)
			}
		}

		result, err := KMeans(dataArray, 3, nil) // 3 clusters, default options
		if err != nil {
			t.Fatalf("KMeans failed: %v", err)
		}

		// Check basic result structure
		if result.NClusters != 3 {
			t.Errorf("Expected 3 clusters, got %d", result.NClusters)
		}

		if len(result.Labels) != len(data) {
			t.Errorf("Expected %d labels, got %d", len(data), len(result.Labels))
		}

		if result.Centers == nil || result.Centers.Shape()[0] != 3 {
			t.Error("Expected 3 cluster centers")
		}

		// Check that points within the same group have the same label
		// First 4 points should be in one cluster
		firstClusterLabel := result.Labels[0]
		for i := 1; i < 4; i++ {
			if result.Labels[i] != firstClusterLabel {
				t.Errorf("Points 0-%d should be in same cluster", i)
			}
		}

		// Points 4-7 should be in another cluster
		secondClusterLabel := result.Labels[4]
		for i := 5; i < 8; i++ {
			if result.Labels[i] != secondClusterLabel {
				t.Errorf("Points 4-%d should be in same cluster", i)
			}
		}

		// Points 8-11 should be in third cluster
		thirdClusterLabel := result.Labels[8]
		for i := 9; i < 12; i++ {
			if result.Labels[i] != thirdClusterLabel {
				t.Errorf("Points 8-%d should be in same cluster", i)
			}
		}

		// All three cluster labels should be different
		if firstClusterLabel == secondClusterLabel ||
			firstClusterLabel == thirdClusterLabel ||
			secondClusterLabel == thirdClusterLabel {
			t.Error("All three clusters should have different labels")
		}

		// Inertia should be relatively low for well-separated clusters
		if result.Inertia > 5.0 {
			t.Errorf("Inertia too high for well-separated clusters: %f", result.Inertia)
		}

		t.Logf("KMeans completed in %d iterations with inertia %.3f",
			result.NIterations, result.Inertia)
		t.Logf("Cluster labels: %v", result.Labels)
	})

	t.Run("Single cluster", func(t *testing.T) {
		// All points close together
		data := [][]float64{
			{2, 2}, {2.1, 2.1}, {1.9, 1.9}, {2.05, 1.95},
		}

		dataArray := array.Empty(internal.Shape{len(data), len(data[0])}, internal.Float64)
		for i, row := range data {
			for j, val := range row {
				dataArray.Set(val, i, j)
			}
		}

		result, err := KMeans(dataArray, 1, nil)
		if err != nil {
			t.Fatalf("KMeans failed: %v", err)
		}

		// All points should be in cluster 0
		for i, label := range result.Labels {
			if label != 0 {
				t.Errorf("Point %d should be in cluster 0, got %d", i, label)
			}
		}

		// Center should be close to (2, 2)
		centerX := result.Centers.At(0, 0).(float64)
		centerY := result.Centers.At(0, 1).(float64)

		if math.Abs(centerX-2.0) > 0.1 || math.Abs(centerY-2.0) > 0.1 {
			t.Errorf("Center should be near (2, 2), got (%.3f, %.3f)", centerX, centerY)
		}
	})

	t.Run("Different initialization methods", func(t *testing.T) {
		// Simple data for testing different initializations
		data := [][]float64{
			{0, 0}, {0, 1}, {1, 0}, {1, 1}, // Cluster 1
			{5, 5}, {5, 6}, {6, 5}, {6, 6}, // Cluster 2
		}

		dataArray := array.Empty(internal.Shape{len(data), len(data[0])}, internal.Float64)
		for i, row := range data {
			for j, val := range row {
				dataArray.Set(val, i, j)
			}
		}

		// Test different initialization methods
		initMethods := []string{"random", "kmeans++", "manual"}

		for _, initMethod := range initMethods {
			t.Run("Init_"+initMethod, func(t *testing.T) {
				options := &KMeansOptions{
					InitMethod:    initMethod,
					MaxIterations: 100,
					Tolerance:     1e-4,
					RandomSeed:    42,
				}

				if initMethod == "manual" {
					// Provide manual initial centers
					initialCenters := array.Empty(internal.Shape{2, 2}, internal.Float64)
					initialCenters.Set(0.0, 0, 0) // Center 1 at (0, 0)
					initialCenters.Set(0.0, 0, 1)
					initialCenters.Set(6.0, 1, 0) // Center 2 at (6, 6)
					initialCenters.Set(6.0, 1, 1)
					options.InitialCenters = initialCenters
				}

				result, err := KMeans(dataArray, 2, options)
				if err != nil {
					t.Fatalf("KMeans with %s init failed: %v", initMethod, err)
				}

				// Should still find 2 good clusters
				if result.NClusters != 2 {
					t.Errorf("Expected 2 clusters, got %d", result.NClusters)
				}

				if result.Inertia > 10.0 {
					t.Errorf("Inertia too high for %s init: %f", initMethod, result.Inertia)
				}

				t.Logf("%s init: %d iterations, inertia=%.3f",
					initMethod, result.NIterations, result.Inertia)
			})
		}
	})

	t.Run("Parameter validation", func(t *testing.T) {
		data := array.Empty(internal.Shape{4, 2}, internal.Float64)
		for i := 0; i < 4; i++ {
			data.Set(float64(i), i, 0)
			data.Set(float64(i), i, 1)
		}

		// Test nil data
		_, err := KMeans(nil, 2, nil)
		if err == nil {
			t.Error("Expected error for nil data")
		}

		// Test empty data
		empty := array.Empty(internal.Shape{0, 2}, internal.Float64)
		_, err = KMeans(empty, 2, nil)
		if err == nil {
			t.Error("Expected error for empty data")
		}

		// Test k <= 0
		_, err = KMeans(data, 0, nil)
		if err == nil {
			t.Error("Expected error for k=0")
		}

		// Test k > n_samples
		_, err = KMeans(data, 5, nil)
		if err == nil {
			t.Error("Expected error for k > n_samples")
		}

		// Test 1D data
		data1D := array.Empty(internal.Shape{5}, internal.Float64)
		_, err = KMeans(data1D, 2, nil)
		if err == nil {
			t.Error("Expected error for 1D data")
		}

		// Test invalid initial centers shape
		invalidCenters := array.Empty(internal.Shape{2, 3}, internal.Float64) // Wrong features
		options := &KMeansOptions{
			InitMethod:     "manual",
			InitialCenters: invalidCenters,
		}
		_, err = KMeans(data, 2, options)
		if err == nil {
			t.Error("Expected error for mismatched initial centers")
		}
	})

	t.Run("Convergence and iterations", func(t *testing.T) {
		// Create data that should converge quickly
		data := [][]float64{
			{0, 0}, {0, 0}, // Duplicate points to test edge cases
			{10, 10}, {10, 10},
		}

		dataArray := array.Empty(internal.Shape{len(data), len(data[0])}, internal.Float64)
		for i, row := range data {
			for j, val := range row {
				dataArray.Set(val, i, j)
			}
		}

		options := &KMeansOptions{
			MaxIterations: 5, // Very limited iterations
			Tolerance:     1e-8,
			RandomSeed:    123,
		}

		result, err := KMeans(dataArray, 2, options)
		if err != nil {
			t.Fatalf("KMeans failed: %v", err)
		}

		// Should converge quickly due to clear separation
		if result.NIterations > 5 {
			t.Errorf("Should converge within 5 iterations, took %d", result.NIterations)
		}

		// Test tight tolerance
		tightOptions := &KMeansOptions{
			MaxIterations: 100,
			Tolerance:     1e-12, // Very tight tolerance
			RandomSeed:    123,
		}

		result2, err := KMeans(dataArray, 2, tightOptions)
		if err != nil {
			t.Fatalf("KMeans with tight tolerance failed: %v", err)
		}

		t.Logf("Tight tolerance: %d iterations, inertia=%.6f",
			result2.NIterations, result2.Inertia)
	})
}

func TestKMeansEdgeCases(t *testing.T) {
	t.Run("Identical points", func(t *testing.T) {
		// All points are identical
		data := [][]float64{
			{3, 3}, {3, 3}, {3, 3}, {3, 3},
		}

		dataArray := array.Empty(internal.Shape{len(data), len(data[0])}, internal.Float64)
		for i, row := range data {
			for j, val := range row {
				dataArray.Set(val, i, j)
			}
		}

		result, err := KMeans(dataArray, 2, nil)
		if err != nil {
			t.Fatalf("KMeans failed on identical points: %v", err)
		}

		// Inertia should be 0 (or very close to 0)
		if result.Inertia > 1e-10 {
			t.Errorf("Inertia should be ~0 for identical points, got %f", result.Inertia)
		}

		// All centers should be at (3, 3)
		for i := 0; i < result.NClusters; i++ {
			centerX := result.Centers.At(i, 0).(float64)
			centerY := result.Centers.At(i, 1).(float64)

			if math.Abs(centerX-3.0) > 1e-10 || math.Abs(centerY-3.0) > 1e-10 {
				t.Errorf("Center %d should be at (3, 3), got (%.6f, %.6f)",
					i, centerX, centerY)
			}
		}
	})

	t.Run("High dimensional data", func(t *testing.T) {
		// 5D data with 2 clusters
		data := make([][]float64, 8)
		// First cluster: all features around 1
		for i := 0; i < 4; i++ {
			data[i] = []float64{1, 1, 1, 1, 1}
			// Add small noise
			for j := range data[i] {
				data[i][j] += 0.1 * float64(i%2-1) // -0.1 or 0.1
			}
		}
		// Second cluster: all features around 5
		for i := 4; i < 8; i++ {
			data[i] = []float64{5, 5, 5, 5, 5}
			// Add small noise
			for j := range data[i] {
				data[i][j] += 0.1 * float64(i%2-1)
			}
		}

		dataArray := array.Empty(internal.Shape{len(data), len(data[0])}, internal.Float64)
		for i, row := range data {
			for j, val := range row {
				dataArray.Set(val, i, j)
			}
		}

		result, err := KMeans(dataArray, 2, nil)
		if err != nil {
			t.Fatalf("KMeans failed on 5D data: %v", err)
		}

		// Should separate into 2 clusters
		if result.NClusters != 2 {
			t.Errorf("Expected 2 clusters, got %d", result.NClusters)
		}

		// Verify cluster separation: first 4 points should be in one cluster
		firstLabel := result.Labels[0]
		for i := 1; i < 4; i++ {
			if result.Labels[i] != firstLabel {
				t.Errorf("Points 0-3 should be in same cluster")
			}
		}

		t.Logf("5D clustering: %d iterations, inertia=%.3f",
			result.NIterations, result.Inertia)
	})
}
