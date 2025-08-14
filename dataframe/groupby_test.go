package dataframe

import (
	"testing"

	"github.com/julianshen/gonp/series"
)

func TestDataFrameGroupBy(t *testing.T) {
	// Create test DataFrame
	categorySeries, _ := series.FromSlice([]interface{}{"A", "B", "A", "B", "A"}, nil, "category")
	valueSeries, _ := series.FromSlice([]float64{10.0, 20.0, 30.0, 40.0, 50.0}, nil, "value")
	quantitySeries, _ := series.FromSlice([]int64{1, 2, 3, 4, 5}, nil, "quantity")

	df, err := FromSeries([]*series.Series{categorySeries, valueSeries, quantitySeries})
	if err != nil {
		t.Fatalf("Failed to create DataFrame: %v", err)
	}

	// Group by category
	grouped, err := df.GroupBy("category")
	if err != nil {
		t.Fatalf("GroupBy failed: %v", err)
	}

	// Test group keys
	keys := grouped.Keys()
	expectedKeys := []interface{}{"A", "B"}
	if len(keys) != len(expectedKeys) {
		t.Errorf("Expected %d groups, got %d", len(expectedKeys), len(keys))
	}

	// Verify keys are present (order doesn't matter)
	keyMap := make(map[interface{}]bool)
	for _, key := range keys {
		keyMap[key] = true
	}
	for _, expected := range expectedKeys {
		if !keyMap[expected] {
			t.Errorf("Expected group key '%v' not found", expected)
		}
	}
}

func TestDataFrameGroupBySum(t *testing.T) {
	// Create test DataFrame
	categorySeries, _ := series.FromSlice([]interface{}{"A", "B", "A", "B", "A"}, nil, "category")
	valueSeries, _ := series.FromSlice([]float64{10.0, 20.0, 30.0, 40.0, 50.0}, nil, "value")
	quantitySeries, _ := series.FromSlice([]int64{1, 2, 3, 4, 5}, nil, "quantity")

	df, _ := FromSeries([]*series.Series{categorySeries, valueSeries, quantitySeries})

	// Group by category and sum
	grouped, _ := df.GroupBy("category")
	result, err := grouped.Sum()
	if err != nil {
		t.Fatalf("GroupBy Sum failed: %v", err)
	}

	// Should have 2 rows (A and B)
	if result.Len() != 2 {
		t.Errorf("Expected 2 rows, got %d", result.Len())
	}

	// Check that category column exists
	_, err = result.GetColumn("category")
	if err != nil {
		t.Errorf("Category column missing in result: %v", err)
	}

	// Verify sum calculations for group A: value=90 (10+30+50), quantity=9 (1+3+5)
	// Find group A row
	var groupAIdx = -1
	for i := 0; i < result.Len(); i++ {
		cat, _ := result.IAt(i, 0)
		if cat.(string) == "A" {
			groupAIdx = i
			break
		}
	}

	if groupAIdx == -1 {
		t.Fatal("Group A not found in result")
	}

	valueSum, _ := result.IAt(groupAIdx, 1)    // value column
	quantitySum, _ := result.IAt(groupAIdx, 2) // quantity column

	if valueSum.(float64) != 90.0 {
		t.Errorf("Expected value sum 90.0 for group A, got %v", valueSum)
	}
	if quantitySum.(int64) != 9 {
		t.Errorf("Expected quantity sum 9 for group A, got %v", quantitySum)
	}
}

func TestDataFrameGroupByMean(t *testing.T) {
	categorySeries, _ := series.FromSlice([]interface{}{"A", "B", "A", "B"}, nil, "category")
	valueSeries, _ := series.FromSlice([]float64{10.0, 20.0, 30.0, 40.0}, nil, "value")

	df, _ := FromSeries([]*series.Series{categorySeries, valueSeries})

	grouped, _ := df.GroupBy("category")
	result, err := grouped.Mean()
	if err != nil {
		t.Fatalf("GroupBy Mean failed: %v", err)
	}

	// Find group A mean: (10+30)/2 = 20
	var groupAIdx = -1
	for i := 0; i < result.Len(); i++ {
		cat, _ := result.IAt(i, 0)
		if cat.(string) == "A" {
			groupAIdx = i
			break
		}
	}

	if groupAIdx == -1 {
		t.Fatal("Group A not found in result")
	}

	meanValue, _ := result.IAt(groupAIdx, 1)
	if meanValue.(float64) != 20.0 {
		t.Errorf("Expected mean 20.0 for group A, got %v", meanValue)
	}
}

func TestDataFrameGroupByCount(t *testing.T) {
	categorySeries, _ := series.FromSlice([]interface{}{"A", "B", "A", "B", "A"}, nil, "category")
	valueSeries, _ := series.FromSlice([]float64{10.0, 20.0, 30.0, 40.0, 50.0}, nil, "value")

	df, _ := FromSeries([]*series.Series{categorySeries, valueSeries})

	grouped, _ := df.GroupBy("category")
	result, err := grouped.Count()
	if err != nil {
		t.Fatalf("GroupBy Count failed: %v", err)
	}

	// Find group A count: should be 3
	var groupAIdx = -1
	for i := 0; i < result.Len(); i++ {
		cat, _ := result.IAt(i, 0)
		if cat.(string) == "A" {
			groupAIdx = i
			break
		}
	}

	if groupAIdx == -1 {
		t.Fatal("Group A not found in result")
	}

	countValue, _ := result.IAt(groupAIdx, 1)
	if countValue.(int64) != 3 {
		t.Errorf("Expected count 3 for group A, got %v", countValue)
	}
}

func TestDataFrameGroupByMin(t *testing.T) {
	categorySeries, _ := series.FromSlice([]interface{}{"A", "B", "A"}, nil, "category")
	valueSeries, _ := series.FromSlice([]float64{30.0, 20.0, 10.0}, nil, "value")

	df, _ := FromSeries([]*series.Series{categorySeries, valueSeries})

	grouped, _ := df.GroupBy("category")
	result, err := grouped.Min()
	if err != nil {
		t.Fatalf("GroupBy Min failed: %v", err)
	}

	// Find group A min: should be 10.0
	var groupAIdx = -1
	for i := 0; i < result.Len(); i++ {
		cat, _ := result.IAt(i, 0)
		if cat.(string) == "A" {
			groupAIdx = i
			break
		}
	}

	if groupAIdx == -1 {
		t.Fatal("Group A not found in result")
	}

	minValue, _ := result.IAt(groupAIdx, 1)
	if minValue.(float64) != 10.0 {
		t.Errorf("Expected min 10.0 for group A, got %v", minValue)
	}
}

func TestDataFrameGroupByMax(t *testing.T) {
	categorySeries, _ := series.FromSlice([]interface{}{"A", "B", "A"}, nil, "category")
	valueSeries, _ := series.FromSlice([]float64{10.0, 20.0, 30.0}, nil, "value")

	df, _ := FromSeries([]*series.Series{categorySeries, valueSeries})

	grouped, _ := df.GroupBy("category")
	result, err := grouped.Max()
	if err != nil {
		t.Fatalf("GroupBy Max failed: %v", err)
	}

	// Find group A max: should be 30.0
	var groupAIdx = -1
	for i := 0; i < result.Len(); i++ {
		cat, _ := result.IAt(i, 0)
		if cat.(string) == "A" {
			groupAIdx = i
			break
		}
	}

	if groupAIdx == -1 {
		t.Fatal("Group A not found in result")
	}

	maxValue, _ := result.IAt(groupAIdx, 1)
	if maxValue.(float64) != 30.0 {
		t.Errorf("Expected max 30.0 for group A, got %v", maxValue)
	}
}

func TestDataFrameGroupByMultipleColumns(t *testing.T) {
	// Group by multiple columns
	cat1Series, _ := series.FromSlice([]interface{}{"A", "A", "B", "B"}, nil, "cat1")
	cat2Series, _ := series.FromSlice([]interface{}{"X", "Y", "X", "Y"}, nil, "cat2")
	valueSeries, _ := series.FromSlice([]float64{10.0, 20.0, 30.0, 40.0}, nil, "value")

	df, _ := FromSeries([]*series.Series{cat1Series, cat2Series, valueSeries})

	grouped, err := df.GroupByMultiple([]string{"cat1", "cat2"})
	if err != nil {
		t.Fatalf("GroupByMultiple failed: %v", err)
	}

	result, err := grouped.Sum()
	if err != nil {
		t.Fatalf("GroupBy Sum failed: %v", err)
	}

	// Should have 4 groups: (A,X), (A,Y), (B,X), (B,Y)
	if result.Len() != 4 {
		t.Errorf("Expected 4 groups, got %d", result.Len())
	}
}

func TestDataFrameGroupByGetGroup(t *testing.T) {
	categorySeries, _ := series.FromSlice([]interface{}{"A", "B", "A", "B", "A"}, nil, "category")
	valueSeries, _ := series.FromSlice([]float64{10.0, 20.0, 30.0, 40.0, 50.0}, nil, "value")

	df, _ := FromSeries([]*series.Series{categorySeries, valueSeries})

	grouped, _ := df.GroupBy("category")

	// Get group A
	groupA, err := grouped.GetGroup("A")
	if err != nil {
		t.Fatalf("GetGroup failed: %v", err)
	}

	// Group A should have 3 rows
	if groupA.Len() != 3 {
		t.Errorf("Expected 3 rows in group A, got %d", groupA.Len())
	}

	// Verify all rows in group A have category "A"
	for i := 0; i < groupA.Len(); i++ {
		cat, _ := groupA.IAt(i, 0)
		if cat.(string) != "A" {
			t.Errorf("Expected category A in group A, got %v", cat)
		}
	}
}

func TestDataFrameGroupByErrors(t *testing.T) {
	categorySeries, _ := series.FromSlice([]interface{}{"A", "B"}, nil, "category")
	df, _ := FromSeries([]*series.Series{categorySeries})

	// Test grouping by non-existent column
	_, err := df.GroupBy("nonexistent")
	if err == nil {
		t.Error("Expected error when grouping by non-existent column")
	}

	// Test grouping empty DataFrame
	emptySeries, _ := series.FromValues([]interface{}{}, nil, "empty")
	emptyDF, _ := FromSeries([]*series.Series{emptySeries})

	_, err = emptyDF.GroupBy("empty")
	if err == nil {
		t.Error("Expected error when grouping empty DataFrame")
	}
}
