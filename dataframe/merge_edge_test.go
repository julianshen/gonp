package dataframe

import (
	"fmt"
	"testing"

	"github.com/julianshen/gonp/series"
)

func TestDataFrameJoinEdgeCases(t *testing.T) {
	t.Run("Empty DataFrames", func(t *testing.T) {
		// Create empty DataFrames
		idSeries1, _ := series.FromValues([]interface{}{}, nil, "id")
		df1, _ := FromSeries([]*series.Series{idSeries1})

		idSeries2, _ := series.FromValues([]interface{}{}, nil, "id")
		df2, _ := FromSeries([]*series.Series{idSeries2})

		result, err := InnerJoin(df1, df2, "id", "id")
		if err != nil {
			t.Fatalf("InnerJoin on empty DataFrames failed: %v", err)
		}

		if result.Len() != 0 {
			t.Errorf("Expected 0 rows, got %d", result.Len())
		}
	})

	t.Run("One Empty DataFrame", func(t *testing.T) {
		// Left DataFrame empty, right has data
		idSeries1, _ := series.FromValues([]interface{}{}, nil, "id")
		df1, _ := FromSeries([]*series.Series{idSeries1})

		idSeries2, _ := series.FromSlice([]int64{1, 2}, nil, "id")
		df2, _ := FromSeries([]*series.Series{idSeries2})

		// Inner join should return empty
		result, err := InnerJoin(df1, df2, "id", "id")
		if err != nil {
			t.Fatalf("InnerJoin failed: %v", err)
		}
		if result.Len() != 0 {
			t.Errorf("Expected 0 rows for inner join, got %d", result.Len())
		}

		// Left join should return empty (no left rows)
		result, err = LeftJoin(df1, df2, "id", "id")
		if err != nil {
			t.Fatalf("LeftJoin failed: %v", err)
		}
		if result.Len() != 0 {
			t.Errorf("Expected 0 rows for left join, got %d", result.Len())
		}

		// Right join should return all right rows with nil left values
		result, err = RightJoin(df1, df2, "id", "id")
		if err != nil {
			t.Fatalf("RightJoin failed: %v", err)
		}
		if result.Len() != 2 {
			t.Errorf("Expected 2 rows for right join, got %d", result.Len())
		}
	})

	t.Run("Duplicate Keys", func(t *testing.T) {
		// Left DataFrame with duplicate keys
		idSeries1, _ := series.FromSlice([]int64{1, 1, 2}, nil, "id")
		nameSeries1, _ := series.FromSlice([]interface{}{"Alice1", "Alice2", "Bob"}, nil, "name")
		df1, _ := FromSeries([]*series.Series{idSeries1, nameSeries1})

		// Right DataFrame with duplicate keys
		idSeries2, _ := series.FromSlice([]int64{1, 1}, nil, "id")
		salarySeries2, _ := series.FromSlice([]float64{50000.0, 55000.0}, nil, "salary")
		df2, _ := FromSeries([]*series.Series{idSeries2, salarySeries2})

		result, err := InnerJoin(df1, df2, "id", "id")
		if err != nil {
			t.Fatalf("InnerJoin with duplicate keys failed: %v", err)
		}

		// Should have 4 rows (2x2 cartesian product for id=1)
		if result.Len() != 4 {
			t.Errorf("Expected 4 rows (cartesian product), got %d", result.Len())
		}

		// Verify all combinations exist
		expectedCombinations := map[string]bool{
			"Alice1_50000": false,
			"Alice1_55000": false,
			"Alice2_50000": false,
			"Alice2_55000": false,
		}

		for i := 0; i < result.Len(); i++ {
			name, _ := result.IAt(i, 1)
			salary, _ := result.IAt(i, 2)
			key := name.(string) + "_" + fmt.Sprintf("%.0f", salary.(float64))
			expectedCombinations[key] = true
		}

		for combo, found := range expectedCombinations {
			if !found {
				t.Errorf("Expected combination %s not found", combo)
			}
		}
	})

	t.Run("No Matching Keys", func(t *testing.T) {
		idSeries1, _ := series.FromSlice([]int64{1, 2}, nil, "id")
		df1, _ := FromSeries([]*series.Series{idSeries1})

		idSeries2, _ := series.FromSlice([]int64{3, 4}, nil, "id")
		df2, _ := FromSeries([]*series.Series{idSeries2})

		// Inner join should return empty
		result, err := InnerJoin(df1, df2, "id", "id")
		if err != nil {
			t.Fatalf("InnerJoin failed: %v", err)
		}
		if result.Len() != 0 {
			t.Errorf("Expected 0 rows for inner join with no matches, got %d", result.Len())
		}

		// Outer join should return all rows with null values
		result, err = OuterJoin(df1, df2, "id", "id")
		if err != nil {
			t.Fatalf("OuterJoin failed: %v", err)
		}
		if result.Len() != 4 {
			t.Errorf("Expected 4 rows for outer join, got %d", result.Len())
		}
	})

	t.Run("Single Row DataFrames", func(t *testing.T) {
		idSeries1, _ := series.FromSlice([]int64{1}, nil, "id")
		nameSeries1, _ := series.FromSlice([]interface{}{"Alice"}, nil, "name")
		df1, _ := FromSeries([]*series.Series{idSeries1, nameSeries1})

		idSeries2, _ := series.FromSlice([]int64{1}, nil, "id")
		salarySeries2, _ := series.FromSlice([]float64{50000.0}, nil, "salary")
		df2, _ := FromSeries([]*series.Series{idSeries2, salarySeries2})

		result, err := InnerJoin(df1, df2, "id", "id")
		if err != nil {
			t.Fatalf("InnerJoin on single row DataFrames failed: %v", err)
		}

		if result.Len() != 1 {
			t.Errorf("Expected 1 row, got %d", result.Len())
		}

		// Verify data
		name, _ := result.IAt(0, 1)
		salary, _ := result.IAt(0, 2)
		if name.(string) != "Alice" || salary.(float64) != 50000.0 {
			t.Errorf("Incorrect join result: name=%v, salary=%v", name, salary)
		}
	})

	t.Run("Mixed Data Types in Join Keys", func(t *testing.T) {
		// Test with string keys
		idSeries1, _ := series.FromSlice([]interface{}{"A", "B", "C"}, nil, "id")
		valueSeries1, _ := series.FromSlice([]int64{1, 2, 3}, nil, "value")
		df1, _ := FromSeries([]*series.Series{idSeries1, valueSeries1})

		idSeries2, _ := series.FromSlice([]interface{}{"B", "C", "D"}, nil, "id")
		valueSeries2, _ := series.FromSlice([]float64{10.0, 20.0, 30.0}, nil, "price")
		df2, _ := FromSeries([]*series.Series{idSeries2, valueSeries2})

		result, err := InnerJoin(df1, df2, "id", "id")
		if err != nil {
			t.Fatalf("InnerJoin with string keys failed: %v", err)
		}

		if result.Len() != 2 {
			t.Errorf("Expected 2 rows, got %d", result.Len())
		}

		// Check that B and C are matched
		id0, _ := result.IAt(0, 0)
		id1, _ := result.IAt(1, 0)

		matchedIds := []string{id0.(string), id1.(string)}
		expectedIds := []string{"B", "C"}

		for _, expected := range expectedIds {
			found := false
			for _, matched := range matchedIds {
				if matched == expected {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("Expected ID '%s' not found in results", expected)
			}
		}
	})
}

func TestDataFrameJoinColumnNaming(t *testing.T) {
	t.Run("Same Column Names", func(t *testing.T) {
		// Test that duplicate column names are handled correctly
		idSeries1, _ := series.FromSlice([]int64{1, 2}, nil, "id")
		valueSeries1, _ := series.FromSlice([]int64{10, 20}, nil, "value")
		df1, _ := FromSeries([]*series.Series{idSeries1, valueSeries1})

		idSeries2, _ := series.FromSlice([]int64{1, 2}, nil, "id")
		valueSeries2, _ := series.FromSlice([]int64{100, 200}, nil, "value") // Same column name
		df2, _ := FromSeries([]*series.Series{idSeries2, valueSeries2})

		result, err := InnerJoin(df1, df2, "id", "id")
		if err != nil {
			t.Fatalf("InnerJoin with same column names failed: %v", err)
		}

		columns := result.Columns()

		// Should have: id, value (from left), value (from right but may be renamed)
		if len(columns) != 3 {
			t.Errorf("Expected 3 columns, got %d: %v", len(columns), columns)
		}

		// The join key (id) should appear only once
		idCount := 0
		for _, col := range columns {
			if col == "id" {
				idCount++
			}
		}
		if idCount != 1 {
			t.Errorf("Expected id column to appear exactly once, got %d times", idCount)
		}
	})
}
