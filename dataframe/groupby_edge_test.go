package dataframe

import (
	"fmt"
	"testing"

	"github.com/julianshen/gonp/series"
)

func TestDataFrameGroupByEdgeCases(t *testing.T) {
	t.Run("Single Group", func(t *testing.T) {
		// All rows have the same grouping value
		categorySeries, _ := series.FromSlice([]interface{}{"A", "A", "A"}, nil, "category")
		valueSeries, _ := series.FromSlice([]float64{10.0, 20.0, 30.0}, nil, "value")

		df, _ := FromSeries([]*series.Series{categorySeries, valueSeries})

		grouped, err := df.GroupBy("category")
		if err != nil {
			t.Fatalf("GroupBy failed: %v", err)
		}

		keys := grouped.Keys()
		if len(keys) != 1 {
			t.Errorf("Expected 1 group, got %d", len(keys))
		}

		result, err := grouped.Sum()
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}

		if result.Len() != 1 {
			t.Errorf("Expected 1 row in result, got %d", result.Len())
		}

		// Check sum: 10 + 20 + 30 = 60
		valueSum, _ := result.IAt(0, 1)
		if valueSum.(float64) != 60.0 {
			t.Errorf("Expected sum 60.0, got %v", valueSum)
		}
	})

	t.Run("Each Row Different Group", func(t *testing.T) {
		// Each row is its own group
		categorySeries, _ := series.FromSlice([]interface{}{"A", "B", "C"}, nil, "category")
		valueSeries, _ := series.FromSlice([]float64{10.0, 20.0, 30.0}, nil, "value")

		df, _ := FromSeries([]*series.Series{categorySeries, valueSeries})

		grouped, err := df.GroupBy("category")
		if err != nil {
			t.Fatalf("GroupBy failed: %v", err)
		}

		keys := grouped.Keys()
		if len(keys) != 3 {
			t.Errorf("Expected 3 groups, got %d", len(keys))
		}

		result, err := grouped.Sum()
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}

		if result.Len() != 3 {
			t.Errorf("Expected 3 rows in result, got %d", result.Len())
		}

		// Each group should have the original value (no aggregation)
		expectedValues := map[string]float64{"A": 10.0, "B": 20.0, "C": 30.0}
		for i := 0; i < result.Len(); i++ {
			cat, _ := result.IAt(i, 0)
			val, _ := result.IAt(i, 1)

			category := cat.(string)
			expected := expectedValues[category]
			if val.(float64) != expected {
				t.Errorf("Expected value %v for category %s, got %v", expected, category, val)
			}
		}
	})

	t.Run("Mixed Data Types", func(t *testing.T) {
		categorySeries, _ := series.FromSlice([]interface{}{"A", "B", "A"}, nil, "category")
		intSeries, _ := series.FromSlice([]int64{1, 2, 3}, nil, "int_col")
		floatSeries, _ := series.FromSlice([]float64{1.5, 2.5, 3.5}, nil, "float_col")

		df, _ := FromSeries([]*series.Series{categorySeries, intSeries, floatSeries})

		grouped, _ := df.GroupBy("category")
		result, err := grouped.Sum()
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}

		// Find group A sums: int_col = 1+3=4, float_col = 1.5+3.5=5.0
		var groupAIdx = -1
		for i := 0; i < result.Len(); i++ {
			cat, _ := result.IAt(i, 0)
			if cat.(string) == "A" {
				groupAIdx = i
				break
			}
		}

		if groupAIdx == -1 {
			t.Fatal("Group A not found")
		}

		intSum, _ := result.IAt(groupAIdx, 1)
		floatSum, _ := result.IAt(groupAIdx, 2)

		if intSum.(int64) != 4 {
			t.Errorf("Expected int sum 4 for group A, got %v", intSum)
		}
		if floatSum.(float64) != 5.0 {
			t.Errorf("Expected float sum 5.0 for group A, got %v", floatSum)
		}
	})

	t.Run("With Nil Values", func(t *testing.T) {
		categorySeries, _ := series.FromSlice([]interface{}{"A", "B", "A"}, nil, "category")
		valueSeries, _ := series.FromSlice([]interface{}{10.0, nil, 20.0}, nil, "value")

		df, _ := FromSeries([]*series.Series{categorySeries, valueSeries})

		grouped, _ := df.GroupBy("category")

		// Test count (should ignore nils)
		countResult, err := grouped.Count()
		if err != nil {
			t.Fatalf("Count failed: %v", err)
		}

		// Find group A count: should be 2 (10.0 and 20.0, ignoring nil)
		var groupAIdx = -1
		for i := 0; i < countResult.Len(); i++ {
			cat, _ := countResult.IAt(i, 0)
			if cat.(string) == "A" {
				groupAIdx = i
				break
			}
		}

		if groupAIdx == -1 {
			t.Fatal("Group A not found")
		}

		count, _ := countResult.IAt(groupAIdx, 1)
		if count.(int64) != 2 {
			t.Errorf("Expected count 2 for group A, got %v", count)
		}
	})

	t.Run("Non-Numeric Aggregation Error", func(t *testing.T) {
		categorySeries, _ := series.FromSlice([]interface{}{"A", "B", "A"}, nil, "category")
		stringSeries, _ := series.FromSlice([]interface{}{"hello", "world", "test"}, nil, "text")

		df, _ := FromSeries([]*series.Series{categorySeries, stringSeries})

		grouped, _ := df.GroupBy("category")

		// Sum should fail on string column
		_, err := grouped.Sum()
		if err == nil {
			t.Error("Expected error when summing non-numeric column")
		}

		// Mean should fail on string column
		_, err = grouped.Mean()
		if err == nil {
			t.Error("Expected error when computing mean of non-numeric column")
		}

		// Count should work on any column type
		_, err = grouped.Count()
		if err != nil {
			t.Errorf("Count should work on string columns: %v", err)
		}
	})

	t.Run("Large Number of Groups", func(t *testing.T) {
		// Create 1000 different groups
		categories := make([]interface{}, 1000)
		values := make([]float64, 1000)

		for i := 0; i < 1000; i++ {
			categories[i] = fmt.Sprintf("group_%d", i)
			values[i] = float64(i)
		}

		categorySeries, _ := series.FromSlice(categories, nil, "category")
		valueSeries, _ := series.FromSlice(values, nil, "value")

		df, _ := FromSeries([]*series.Series{categorySeries, valueSeries})

		grouped, err := df.GroupBy("category")
		if err != nil {
			t.Fatalf("GroupBy failed: %v", err)
		}

		keys := grouped.Keys()
		if len(keys) != 1000 {
			t.Errorf("Expected 1000 groups, got %d", len(keys))
		}

		result, err := grouped.Sum()
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}

		if result.Len() != 1000 {
			t.Errorf("Expected 1000 rows in result, got %d", result.Len())
		}
	})

	t.Run("Three Column Grouping", func(t *testing.T) {
		cat1Series, _ := series.FromSlice([]interface{}{"A", "A", "B", "B"}, nil, "cat1")
		cat2Series, _ := series.FromSlice([]interface{}{"X", "Y", "X", "Y"}, nil, "cat2")
		cat3Series, _ := series.FromSlice([]interface{}{"P", "P", "Q", "Q"}, nil, "cat3")
		valueSeries, _ := series.FromSlice([]float64{10.0, 20.0, 30.0, 40.0}, nil, "value")

		df, _ := FromSeries([]*series.Series{cat1Series, cat2Series, cat3Series, valueSeries})

		grouped, err := df.GroupByMultiple([]string{"cat1", "cat2", "cat3"})
		if err != nil {
			t.Fatalf("GroupByMultiple failed: %v", err)
		}

		keys := grouped.Keys()
		if len(keys) != 4 {
			t.Errorf("Expected 4 groups, got %d", len(keys))
		}

		result, err := grouped.Sum()
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}

		if result.Len() != 4 {
			t.Errorf("Expected 4 rows in result, got %d", result.Len())
		}

		// Each combination should appear once with its original value
		for i := 0; i < result.Len(); i++ {
			value, _ := result.IAt(i, 3) // value column is 4th (index 3)
			expectedValues := []float64{10.0, 20.0, 30.0, 40.0}

			found := false
			for _, expected := range expectedValues {
				if value.(float64) == expected {
					found = true
					break
				}
			}

			if !found {
				t.Errorf("Unexpected value in result: %v", value)
			}
		}
	})
}
