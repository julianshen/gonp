package dataframe

import (
	"fmt"
	"testing"

	"github.com/julianshen/gonp/series"
)

func TestDataFramePivotEdgeCases(t *testing.T) {
	t.Run("Single Value Pivot", func(t *testing.T) {
		// DataFrame with only one row
		regionSeries, _ := series.FromSlice([]interface{}{"North"}, nil, "region")
		productSeries, _ := series.FromSlice([]interface{}{"A"}, nil, "product")
		salesSeries, _ := series.FromSlice([]float64{100}, nil, "sales")

		df, _ := FromSeries([]*series.Series{regionSeries, productSeries, salesSeries})

		pivot, err := df.PivotTable("region", "product", "sales", "sum")
		if err != nil {
			t.Fatalf("PivotTable failed: %v", err)
		}

		if pivot.Len() != 1 {
			t.Errorf("Expected 1 row, got %d", pivot.Len())
		}

		if len(pivot.Columns()) != 2 { // region + A
			t.Errorf("Expected 2 columns, got %d", len(pivot.Columns()))
		}

		value, _ := pivot.IAt(0, 1) // Product A column
		if value.(float64) != 100.0 {
			t.Errorf("Expected value 100.0, got %v", value)
		}
	})

	t.Run("Pivot with Nil Values", func(t *testing.T) {
		regionSeries, _ := series.FromSlice([]interface{}{"North", "South"}, nil, "region")
		productSeries, _ := series.FromSlice([]interface{}{"A", "B"}, nil, "product")
		salesSeries, _ := series.FromSlice([]interface{}{100.0, nil}, nil, "sales")

		df, _ := FromSeries([]*series.Series{regionSeries, productSeries, salesSeries})

		pivot, err := df.PivotTable("region", "product", "sales", "sum")
		if err != nil {
			t.Fatalf("PivotTable failed: %v", err)
		}

		// Should handle nil values properly
		if pivot.Len() != 2 {
			t.Errorf("Expected 2 rows, got %d", pivot.Len())
		}
	})

	t.Run("Large Pivot Table", func(t *testing.T) {
		// Create larger test data
		regions := []interface{}{}
		products := []interface{}{}
		sales := []float64{}

		for i := 0; i < 100; i++ {
			regions = append(regions, fmt.Sprintf("Region_%d", i%5))
			products = append(products, fmt.Sprintf("Product_%d", i%3))
			sales = append(sales, float64(i*10))
		}

		regionSeries, _ := series.FromSlice(regions, nil, "region")
		productSeries, _ := series.FromSlice(products, nil, "product")
		salesSeries, _ := series.FromSlice(sales, nil, "sales")

		df, _ := FromSeries([]*series.Series{regionSeries, productSeries, salesSeries})

		pivot, err := df.PivotTable("region", "product", "sales", "sum")
		if err != nil {
			t.Fatalf("PivotTable failed: %v", err)
		}

		// Should have 5 regions and 4 columns (region + 3 products)
		if pivot.Len() != 5 {
			t.Errorf("Expected 5 rows, got %d", pivot.Len())
		}

		if len(pivot.Columns()) != 4 {
			t.Errorf("Expected 4 columns, got %d", len(pivot.Columns()))
		}
	})
}

func TestDataFrameReshapeEdgeCases(t *testing.T) {
	t.Run("Stack Single Column", func(t *testing.T) {
		nameSeries, _ := series.FromSlice([]interface{}{"Alice", "Bob"}, nil, "name")
		valueSeries, _ := series.FromSlice([]float64{10, 20}, nil, "value")

		df, _ := FromSeries([]*series.Series{nameSeries, valueSeries})

		// Stack only the value column
		stacked, err := df.Stack([]string{"value"}, "variable", "measurement")
		if err != nil {
			t.Fatalf("Stack failed: %v", err)
		}

		// Should have 2 rows (one for each original row)
		if stacked.Len() != 2 {
			t.Errorf("Expected 2 rows, got %d", stacked.Len())
		}

		// Should have 3 columns: name, variable, measurement
		if len(stacked.Columns()) != 3 {
			t.Errorf("Expected 3 columns, got %d", len(stacked.Columns()))
		}

		// Check values
		name0, _ := stacked.IAt(0, 0)
		var0, _ := stacked.IAt(0, 1)
		val0, _ := stacked.IAt(0, 2)

		if name0.(string) != "Alice" || var0.(string) != "value" || val0.(float64) != 10.0 {
			t.Errorf("Unexpected stacked values: %v, %v, %v", name0, var0, val0)
		}
	})

	t.Run("Stack All Columns", func(t *testing.T) {
		col1Series, _ := series.FromSlice([]float64{1, 2}, nil, "A")
		col2Series, _ := series.FromSlice([]float64{3, 4}, nil, "B")

		df, _ := FromSeries([]*series.Series{col1Series, col2Series})

		// Stack all columns (no identity columns)
		stacked, err := df.Stack([]string{"A", "B"}, "variable", "value")
		if err != nil {
			t.Fatalf("Stack failed: %v", err)
		}

		// Should have 4 rows (2 original rows * 2 columns)
		if stacked.Len() != 4 {
			t.Errorf("Expected 4 rows, got %d", stacked.Len())
		}

		// Should have 2 columns: variable, value
		if len(stacked.Columns()) != 2 {
			t.Errorf("Expected 2 columns, got %d", len(stacked.Columns()))
		}
	})

	t.Run("Unstack with Duplicates", func(t *testing.T) {
		// Data with duplicate index-column combinations (should use last value)
		nameSeries, _ := series.FromSlice([]interface{}{"Alice", "Alice", "Bob"}, nil, "name")
		subjectSeries, _ := series.FromSlice([]interface{}{"math", "math", "math"}, nil, "subject")
		scoreSeries, _ := series.FromSlice([]float64{85, 90, 95}, nil, "score") // Alice has two math scores

		df, _ := FromSeries([]*series.Series{nameSeries, subjectSeries, scoreSeries})

		unstacked, err := df.Unstack("name", "subject", "score")
		if err != nil {
			t.Fatalf("Unstack failed: %v", err)
		}

		// Should have 2 rows (Alice, Bob)
		if unstacked.Len() != 2 {
			t.Errorf("Expected 2 rows, got %d", unstacked.Len())
		}

		// Find Alice's math score (should be the last one: 90)
		var aliceIdx = -1
		for i := 0; i < unstacked.Len(); i++ {
			name, _ := unstacked.IAt(i, 0)
			if name.(string) == "Alice" {
				aliceIdx = i
				break
			}
		}

		if aliceIdx == -1 {
			t.Fatal("Alice not found")
		}

		aliceMath, _ := unstacked.IAt(aliceIdx, 1)
		// Note: This behavior depends on implementation - it could be first or last value
		// Our implementation should use the last encountered value
		if aliceMath.(float64) != 90.0 {
			t.Errorf("Expected Alice's math score to be 90.0 (last value), got %v", aliceMath)
		}
	})

	t.Run("Transpose Non-Square Matrix", func(t *testing.T) {
		// 2x3 matrix
		col1Series, _ := series.FromSlice([]float64{1, 2}, nil, "A")
		col2Series, _ := series.FromSlice([]float64{3, 4}, nil, "B")
		col3Series, _ := series.FromSlice([]float64{5, 6}, nil, "C")

		df, _ := FromSeries([]*series.Series{col1Series, col2Series, col3Series})

		transposed, err := df.Transpose()
		if err != nil {
			t.Fatalf("Transpose failed: %v", err)
		}

		// Original: 2 rows x 3 columns
		// Transposed: 3 rows x 2 columns (rows become columns)
		if transposed.Len() != 3 {
			t.Errorf("Expected 3 rows, got %d", transposed.Len())
		}

		if len(transposed.Columns()) != 2 {
			t.Errorf("Expected 2 columns, got %d", len(transposed.Columns()))
		}

		// Check transpose: original[1][0] (row 1, col A = 2) should equal transposed[0][1] (row A, col 1 = 2)
		originalVal, _ := df.IAt(1, 0)           // df[1][A] = 2
		transposedVal, _ := transposed.IAt(0, 1) // transposed[A][1] = 2

		if originalVal.(float64) != transposedVal.(float64) {
			t.Errorf("Transpose incorrect: original[1][0]=%v != transposed[0][1]=%v", originalVal, transposedVal)
		}
	})

	t.Run("Transpose Single Row", func(t *testing.T) {
		col1Series, _ := series.FromSlice([]float64{1}, nil, "A")
		col2Series, _ := series.FromSlice([]float64{2}, nil, "B")

		df, _ := FromSeries([]*series.Series{col1Series, col2Series})

		transposed, err := df.Transpose()
		if err != nil {
			t.Fatalf("Transpose failed: %v", err)
		}

		// Original: 1 row x 2 columns
		// Transposed: 2 rows x 1 column
		if transposed.Len() != 2 {
			t.Errorf("Expected 2 rows, got %d", transposed.Len())
		}

		if len(transposed.Columns()) != 1 {
			t.Errorf("Expected 1 column, got %d", len(transposed.Columns()))
		}
	})

	t.Run("Transpose Single Column", func(t *testing.T) {
		colSeries, _ := series.FromSlice([]float64{1, 2, 3}, nil, "A")

		df, _ := FromSeries([]*series.Series{colSeries})

		transposed, err := df.Transpose()
		if err != nil {
			t.Fatalf("Transpose failed: %v", err)
		}

		// Original: 3 rows x 1 column
		// Transposed: 1 row x 3 columns
		if transposed.Len() != 1 {
			t.Errorf("Expected 1 row, got %d", transposed.Len())
		}

		if len(transposed.Columns()) != 3 {
			t.Errorf("Expected 3 columns, got %d", len(transposed.Columns()))
		}

		// Values should be transposed
		originalVal0, _ := df.IAt(0, 0) // df[0][A] = 1
		originalVal1, _ := df.IAt(1, 0) // df[1][A] = 2
		originalVal2, _ := df.IAt(2, 0) // df[2][A] = 3

		transposedVal0, _ := transposed.IAt(0, 0) // transposed[A][0] = 1
		transposedVal1, _ := transposed.IAt(0, 1) // transposed[A][1] = 2
		transposedVal2, _ := transposed.IAt(0, 2) // transposed[A][2] = 3

		if originalVal0 != transposedVal0 || originalVal1 != transposedVal1 || originalVal2 != transposedVal2 {
			t.Errorf("Transpose values incorrect: orig=(%v,%v,%v) trans=(%v,%v,%v)",
				originalVal0, originalVal1, originalVal2, transposedVal0, transposedVal1, transposedVal2)
		}
	})
}
