package dataframe

import (
	"testing"

	"github.com/julianshen/gonp/series"
)

func TestDataFrameInnerJoin(t *testing.T) {
	// Create first DataFrame
	idSeries1, _ := series.FromSlice([]int64{1, 2, 3}, nil, "id")
	nameSeries1, _ := series.FromSlice([]interface{}{"Alice", "Bob", "Charlie"}, nil, "name")
	ageSeries1, _ := series.FromSlice([]int64{25, 30, 35}, nil, "age")

	df1, err := FromSeries([]*series.Series{idSeries1, nameSeries1, ageSeries1})
	if err != nil {
		t.Fatalf("Failed to create first DataFrame: %v", err)
	}

	// Create second DataFrame
	idSeries2, _ := series.FromSlice([]int64{2, 3, 4}, nil, "id")
	salarySeries2, _ := series.FromSlice([]float64{50000.0, 60000.0, 70000.0}, nil, "salary")
	deptSeries2, _ := series.FromSlice([]interface{}{"Engineering", "Sales", "Marketing"}, nil, "department")

	df2, err := FromSeries([]*series.Series{idSeries2, salarySeries2, deptSeries2})
	if err != nil {
		t.Fatalf("Failed to create second DataFrame: %v", err)
	}

	// Perform inner join on 'id' column
	result, err := InnerJoin(df1, df2, "id", "id")
	if err != nil {
		t.Fatalf("InnerJoin failed: %v", err)
	}

	// Verify result structure
	expectedColumns := []string{"id", "name", "age", "salary", "department"}
	if len(result.Columns()) != len(expectedColumns) {
		t.Errorf("Expected %d columns, got %d", len(expectedColumns), len(result.Columns()))
	}

	for _, col := range expectedColumns {
		if _, err := result.GetColumn(col); err != nil {
			t.Errorf("Expected column '%s' not found: %v", col, err)
		}
	}

	// Verify result data (should contain rows where id = 2, 3)
	if result.Len() != 2 {
		t.Errorf("Expected 2 rows, got %d", result.Len())
	}

	// Check that both expected rows are present (order doesn't matter)
	foundRows := make(map[int64]string)
	for i := 0; i < result.Len(); i++ {
		id, _ := result.IAt(i, 0)
		name, _ := result.IAt(i, 1)
		foundRows[id.(int64)] = name.(string)
	}

	expectedRows := map[int64]string{2: "Bob", 3: "Charlie"}
	for expectedId, expectedName := range expectedRows {
		if foundName, ok := foundRows[expectedId]; !ok {
			t.Errorf("Expected row with id=%d not found", expectedId)
		} else if foundName != expectedName {
			t.Errorf("Expected name '%s' for id=%d, got '%s'", expectedName, expectedId, foundName)
		}
	}
}

func TestDataFrameLeftJoin(t *testing.T) {
	// Create test DataFrames (same as inner join test)
	idSeries1, _ := series.FromSlice([]int64{1, 2, 3}, nil, "id")
	nameSeries1, _ := series.FromSlice([]interface{}{"Alice", "Bob", "Charlie"}, nil, "name")

	df1, _ := FromSeries([]*series.Series{idSeries1, nameSeries1})

	idSeries2, _ := series.FromSlice([]int64{2, 4}, nil, "id")
	salarySeries2, _ := series.FromSlice([]float64{50000.0, 70000.0}, nil, "salary")

	df2, _ := FromSeries([]*series.Series{idSeries2, salarySeries2})

	// Perform left join
	result, err := LeftJoin(df1, df2, "id", "id")
	if err != nil {
		t.Fatalf("LeftJoin failed: %v", err)
	}

	// Should have all rows from left DataFrame (3 rows)
	if result.Len() != 3 {
		t.Errorf("Expected 3 rows, got %d", result.Len())
	}

	// Check salary values for each name (order-independent)
	aliceSalary, bobSalary := interface{}(nil), interface{}(nil)
	charleSalary := interface{}(nil)

	for i := 0; i < result.Len(); i++ {
		name, _ := result.IAt(i, 1)
		salary, _ := result.IAt(i, 2)

		switch name.(string) {
		case "Alice":
			aliceSalary = salary
		case "Bob":
			bobSalary = salary
		case "Charlie":
			charleSalary = salary
		}
	}

	// Alice should have nil salary (no match in right)
	if aliceSalary != nil {
		t.Errorf("Expected nil salary for Alice, got %v", aliceSalary)
	}

	// Bob should have 50000.0 salary
	if bobSalary == nil {
		t.Error("Expected Bob to have a salary, got nil")
	} else if bobSalary.(float64) != 50000.0 {
		t.Errorf("Expected salary 50000.0 for Bob, got %v", bobSalary)
	}

	// Charlie should have nil salary (no match in right)
	if charleSalary != nil {
		t.Errorf("Expected nil salary for Charlie, got %v", charleSalary)
	}
}

func TestDataFrameRightJoin(t *testing.T) {
	// Similar to left join but reversed
	idSeries1, _ := series.FromSlice([]int64{1, 2}, nil, "id")
	nameSeries1, _ := series.FromSlice([]interface{}{"Alice", "Bob"}, nil, "name")

	df1, _ := FromSeries([]*series.Series{idSeries1, nameSeries1})

	idSeries2, _ := series.FromSlice([]int64{2, 3, 4}, nil, "id")
	salarySeries2, _ := series.FromSlice([]float64{50000.0, 60000.0, 70000.0}, nil, "salary")

	df2, _ := FromSeries([]*series.Series{idSeries2, salarySeries2})

	// Perform right join
	result, err := RightJoin(df1, df2, "id", "id")
	if err != nil {
		t.Fatalf("RightJoin failed: %v", err)
	}

	// Should have all rows from right DataFrame (3 rows)
	if result.Len() != 3 {
		t.Errorf("Expected 3 rows, got %d", result.Len())
	}

	// Check that unmatched rows have nil values for left DataFrame columns
	name1, _ := result.IAt(1, 1) // name for id=3 (should be nil)
	if name1 != nil {
		t.Errorf("Expected nil name for id=3, got %v", name1)
	}
}

func TestDataFrameOuterJoin(t *testing.T) {
	idSeries1, _ := series.FromSlice([]int64{1, 2}, nil, "id")
	nameSeries1, _ := series.FromSlice([]interface{}{"Alice", "Bob"}, nil, "name")

	df1, _ := FromSeries([]*series.Series{idSeries1, nameSeries1})

	idSeries2, _ := series.FromSlice([]int64{2, 3}, nil, "id")
	salarySeries2, _ := series.FromSlice([]float64{50000.0, 60000.0}, nil, "salary")

	df2, _ := FromSeries([]*series.Series{idSeries2, salarySeries2})

	// Perform outer join
	result, err := OuterJoin(df1, df2, "id", "id")
	if err != nil {
		t.Fatalf("OuterJoin failed: %v", err)
	}

	// Should have union of all unique keys (3 rows: 1, 2, 3)
	if result.Len() != 3 {
		t.Errorf("Expected 3 rows, got %d", result.Len())
	}
}

func TestDataFrameMergeMultipleKeys(t *testing.T) {
	// Test merging on multiple columns
	yearSeries1, _ := series.FromSlice([]int64{2020, 2020, 2021}, nil, "year")
	monthSeries1, _ := series.FromSlice([]int64{1, 2, 1}, nil, "month")
	salesSeries1, _ := series.FromSlice([]float64{100, 200, 150}, nil, "sales")

	df1, _ := FromSeries([]*series.Series{yearSeries1, monthSeries1, salesSeries1})

	yearSeries2, _ := series.FromSlice([]int64{2020, 2021}, nil, "year")
	monthSeries2, _ := series.FromSlice([]int64{1, 1}, nil, "month")
	costsSeries2, _ := series.FromSlice([]float64{80, 120}, nil, "costs")

	df2, _ := FromSeries([]*series.Series{yearSeries2, monthSeries2, costsSeries2})

	// Join on multiple keys
	result, err := InnerJoinMultiple(df1, df2, []string{"year", "month"}, []string{"year", "month"})
	if err != nil {
		t.Fatalf("InnerJoinMultiple failed: %v", err)
	}

	// Should have 2 rows (2020-1 and 2021-1)
	if result.Len() != 2 {
		t.Errorf("Expected 2 rows, got %d", result.Len())
	}
}

func TestDataFrameMergeErrors(t *testing.T) {
	// Test error cases
	nameSeries1, _ := series.FromSlice([]int64{1, 2}, nil, "id")
	df1, _ := FromSeries([]*series.Series{nameSeries1})

	nameSeries2, _ := series.FromSlice([]int64{2, 3}, nil, "id")
	df2, _ := FromSeries([]*series.Series{nameSeries2})

	// Test join on non-existent column
	_, err := InnerJoin(df1, df2, "nonexistent", "id")
	if err == nil {
		t.Error("Expected error for non-existent left column")
	}

	_, err = InnerJoin(df1, df2, "id", "nonexistent")
	if err == nil {
		t.Error("Expected error for non-existent right column")
	}

	// Test join with nil DataFrames
	_, err = InnerJoin(nil, df2, "id", "id")
	if err == nil {
		t.Error("Expected error for nil left DataFrame")
	}

	_, err = InnerJoin(df1, nil, "id", "id")
	if err == nil {
		t.Error("Expected error for nil right DataFrame")
	}
}
