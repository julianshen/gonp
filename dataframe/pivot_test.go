package dataframe

import (
	"testing"

	"github.com/julianshen/gonp/series"
)

func TestDataFramePivotTable(t *testing.T) {
	// Create sample sales data
	regionSeries, _ := series.FromSlice([]interface{}{"North", "South", "North", "South", "North", "South"}, nil, "region")
	productSeries, _ := series.FromSlice([]interface{}{"A", "A", "B", "B", "A", "B"}, nil, "product")
	salesSeries, _ := series.FromSlice([]float64{100, 150, 200, 250, 120, 300}, nil, "sales")

	df, err := FromSeries([]*series.Series{regionSeries, productSeries, salesSeries})
	if err != nil {
		t.Fatalf("Failed to create DataFrame: %v", err)
	}

	// Create pivot table: regions as rows, products as columns, sales as values
	pivot, err := df.PivotTable("region", "product", "sales", "sum")
	if err != nil {
		t.Fatalf("PivotTable failed: %v", err)
	}

	// Should have 2 rows (North, South) and 3 columns (region + A + B)
	if pivot.Len() != 2 {
		t.Errorf("Expected 2 rows, got %d", pivot.Len())
	}

	expectedColumns := []string{"region", "A", "B"}
	actualColumns := pivot.Columns()
	if len(actualColumns) != len(expectedColumns) {
		t.Errorf("Expected %d columns, got %d", len(expectedColumns), len(actualColumns))
	}

	// Verify pivot values
	// North: A=220 (100+120), B=200
	// South: A=150, B=550 (250+300)

	// Find North row
	var northIdx = -1
	for i := 0; i < pivot.Len(); i++ {
		region, _ := pivot.IAt(i, 0)
		if region.(string) == "North" {
			northIdx = i
			break
		}
	}

	if northIdx == -1 {
		t.Fatal("North region not found in pivot table")
	}

	northA, _ := pivot.IAt(northIdx, 1) // Product A column
	northB, _ := pivot.IAt(northIdx, 2) // Product B column

	if northA.(float64) != 220.0 {
		t.Errorf("Expected North-A value 220.0, got %v", northA)
	}
	if northB.(float64) != 200.0 {
		t.Errorf("Expected North-B value 200.0, got %v", northB)
	}
}

func TestDataFramePivotTableMean(t *testing.T) {
	regionSeries, _ := series.FromSlice([]interface{}{"North", "North", "South", "South"}, nil, "region")
	productSeries, _ := series.FromSlice([]interface{}{"A", "A", "B", "B"}, nil, "product")
	salesSeries, _ := series.FromSlice([]float64{100, 200, 150, 250}, nil, "sales")

	df, _ := FromSeries([]*series.Series{regionSeries, productSeries, salesSeries})

	// Create pivot table with mean aggregation
	pivot, err := df.PivotTable("region", "product", "sales", "mean")
	if err != nil {
		t.Fatalf("PivotTable with mean failed: %v", err)
	}

	// Find North row - should have mean of A = (100+200)/2 = 150
	var northIdx = -1
	for i := 0; i < pivot.Len(); i++ {
		region, _ := pivot.IAt(i, 0)
		if region.(string) == "North" {
			northIdx = i
			break
		}
	}

	if northIdx == -1 {
		t.Fatal("North region not found")
	}

	northA, _ := pivot.IAt(northIdx, 1)
	if northA.(float64) != 150.0 {
		t.Errorf("Expected North-A mean 150.0, got %v", northA)
	}
}

func TestDataFramePivotTableCount(t *testing.T) {
	regionSeries, _ := series.FromSlice([]interface{}{"North", "South", "North", "South", "North"}, nil, "region")
	productSeries, _ := series.FromSlice([]interface{}{"A", "A", "B", "B", "A"}, nil, "product")
	salesSeries, _ := series.FromSlice([]float64{100, 150, 200, 250, 120}, nil, "sales")

	df, _ := FromSeries([]*series.Series{regionSeries, productSeries, salesSeries})

	pivot, err := df.PivotTable("region", "product", "sales", "count")
	if err != nil {
		t.Fatalf("PivotTable with count failed: %v", err)
	}

	// North should have 2 entries for A, 1 for B
	var northIdx = -1
	for i := 0; i < pivot.Len(); i++ {
		region, _ := pivot.IAt(i, 0)
		if region.(string) == "North" {
			northIdx = i
			break
		}
	}

	if northIdx == -1 {
		t.Fatal("North region not found")
	}

	northA, _ := pivot.IAt(northIdx, 1)
	northB, _ := pivot.IAt(northIdx, 2)

	if northA.(int64) != 2 {
		t.Errorf("Expected North-A count 2, got %v", northA)
	}
	if northB.(int64) != 1 {
		t.Errorf("Expected North-B count 1, got %v", northB)
	}
}

func TestDataFrameReshapeStack(t *testing.T) {
	// Create wide format data
	nameSeries, _ := series.FromSlice([]interface{}{"Alice", "Bob", "Charlie"}, nil, "name")
	math2020Series, _ := series.FromSlice([]float64{85, 90, 95}, nil, "math_2020")
	math2021Series, _ := series.FromSlice([]float64{88, 92, 97}, nil, "math_2021")
	eng2020Series, _ := series.FromSlice([]float64{82, 85, 90}, nil, "eng_2020")
	eng2021Series, _ := series.FromSlice([]float64{84, 87, 92}, nil, "eng_2021")

	df, _ := FromSeries([]*series.Series{nameSeries, math2020Series, math2021Series, eng2020Series, eng2021Series})

	// Stack to long format
	stacked, err := df.Stack([]string{"math_2020", "math_2021", "eng_2020", "eng_2021"}, "subject_year", "score")
	if err != nil {
		t.Fatalf("Stack failed: %v", err)
	}

	// Should have 3 * 4 = 12 rows (3 people * 4 subject-year combinations)
	if stacked.Len() != 12 {
		t.Errorf("Expected 12 rows, got %d", stacked.Len())
	}

	// Should have 3 columns: name, subject_year, score
	expectedColumns := []string{"name", "subject_year", "score"}
	actualColumns := stacked.Columns()
	if len(actualColumns) != len(expectedColumns) {
		t.Errorf("Expected %d columns, got %d", len(expectedColumns), len(actualColumns))
	}

	for _, col := range expectedColumns {
		if _, err := stacked.GetColumn(col); err != nil {
			t.Errorf("Expected column '%s' not found: %v", col, err)
		}
	}
}

func TestDataFrameReshapeUnstack(t *testing.T) {
	// Create long format data
	nameSeries, _ := series.FromSlice([]interface{}{"Alice", "Alice", "Bob", "Bob"}, nil, "name")
	subjectSeries, _ := series.FromSlice([]interface{}{"math", "eng", "math", "eng"}, nil, "subject")
	scoreSeries, _ := series.FromSlice([]float64{85, 82, 90, 85}, nil, "score")

	df, _ := FromSeries([]*series.Series{nameSeries, subjectSeries, scoreSeries})

	// Unstack to wide format
	unstacked, err := df.Unstack("name", "subject", "score")
	if err != nil {
		t.Fatalf("Unstack failed: %v", err)
	}

	// Should have 2 rows (Alice, Bob) and 3 columns (name + math + eng)
	if unstacked.Len() != 2 {
		t.Errorf("Expected 2 rows, got %d", unstacked.Len())
	}

	expectedColumns := []string{"name", "eng", "math"}
	actualColumns := unstacked.Columns()
	if len(actualColumns) != len(expectedColumns) {
		t.Errorf("Expected %d columns, got %d", len(expectedColumns), len(actualColumns))
	}

	// Find Alice row
	var aliceIdx = -1
	for i := 0; i < unstacked.Len(); i++ {
		name, _ := unstacked.IAt(i, 0)
		if name.(string) == "Alice" {
			aliceIdx = i
			break
		}
	}

	if aliceIdx == -1 {
		t.Fatal("Alice not found in unstacked data")
	}

	// Check Alice's scores
	aliceMath, _ := unstacked.IAt(aliceIdx, 2) // math column
	aliceEng, _ := unstacked.IAt(aliceIdx, 1)  // eng column

	if aliceMath.(float64) != 85.0 {
		t.Errorf("Expected Alice math score 85.0, got %v", aliceMath)
	}
	if aliceEng.(float64) != 82.0 {
		t.Errorf("Expected Alice eng score 82.0, got %v", aliceEng)
	}
}

func TestDataFrameTranspose(t *testing.T) {
	// Create a simple DataFrame
	col1Series, _ := series.FromSlice([]float64{1, 2, 3}, nil, "A")
	col2Series, _ := series.FromSlice([]float64{4, 5, 6}, nil, "B")
	col3Series, _ := series.FromSlice([]float64{7, 8, 9}, nil, "C")

	df, _ := FromSeries([]*series.Series{col1Series, col2Series, col3Series})

	// Transpose
	transposed, err := df.Transpose()
	if err != nil {
		t.Fatalf("Transpose failed: %v", err)
	}

	// Original: 3 rows x 3 columns
	// Transposed: 3 rows x 3 columns (columns become rows)
	if transposed.Len() != 3 {
		t.Errorf("Expected 3 rows, got %d", transposed.Len())
	}

	if len(transposed.Columns()) != 3 {
		t.Errorf("Expected 3 columns, got %d", len(transposed.Columns()))
	}

	// Check that transposition worked: original[i][j] == transposed[j][i]
	// Original A column [1,2,3] should become first row [1,2,3]
	val00, _ := transposed.IAt(0, 0) // Should be 1 (original A[0])
	val01, _ := transposed.IAt(0, 1) // Should be 2 (original A[1])
	val02, _ := transposed.IAt(0, 2) // Should be 3 (original A[2])

	if val00.(float64) != 1.0 || val01.(float64) != 2.0 || val02.(float64) != 3.0 {
		t.Errorf("Transpose values incorrect: got %v, %v, %v", val00, val01, val02)
	}
}

func TestDataFrameReshapeErrors(t *testing.T) {
	nameSeries, _ := series.FromSlice([]interface{}{"Alice", "Bob"}, nil, "name")
	df, _ := FromSeries([]*series.Series{nameSeries})

	// Test pivot table with non-existent columns
	_, err := df.PivotTable("nonexistent", "name", "name", "sum")
	if err == nil {
		t.Error("Expected error for non-existent row column")
	}

	_, err = df.PivotTable("name", "nonexistent", "name", "sum")
	if err == nil {
		t.Error("Expected error for non-existent column column")
	}

	_, err = df.PivotTable("name", "name", "nonexistent", "sum")
	if err == nil {
		t.Error("Expected error for non-existent value column")
	}

	// Test invalid aggregation function
	_, err = df.PivotTable("name", "name", "name", "invalid")
	if err == nil {
		t.Error("Expected error for invalid aggregation function")
	}

	// Test stack with non-existent columns
	_, err = df.Stack([]string{"nonexistent"}, "var", "val")
	if err == nil {
		t.Error("Expected error for non-existent columns in stack")
	}

	// Test unstack with non-existent columns
	_, err = df.Unstack("nonexistent", "name", "name")
	if err == nil {
		t.Error("Expected error for non-existent index column in unstack")
	}
}

func TestDataFramePivotTableWithMissingValues(t *testing.T) {
	// Test pivot table where some combinations don't exist
	regionSeries, _ := series.FromSlice([]interface{}{"North", "South", "North"}, nil, "region")
	productSeries, _ := series.FromSlice([]interface{}{"A", "B", "A"}, nil, "product")
	salesSeries, _ := series.FromSlice([]float64{100, 250, 120}, nil, "sales")

	df, _ := FromSeries([]*series.Series{regionSeries, productSeries, salesSeries})

	// Create pivot table - South will have no A values, North will have no B values
	pivot, err := df.PivotTable("region", "product", "sales", "sum")
	if err != nil {
		t.Fatalf("PivotTable failed: %v", err)
	}

	// Should still have all combinations with nil for missing values
	if pivot.Len() != 2 {
		t.Errorf("Expected 2 rows, got %d", pivot.Len())
	}

	// Find South row
	var southIdx = -1
	for i := 0; i < pivot.Len(); i++ {
		region, _ := pivot.IAt(i, 0)
		if region.(string) == "South" {
			southIdx = i
			break
		}
	}

	if southIdx == -1 {
		t.Fatal("South region not found")
	}

	// South should have nil for product A
	southA, _ := pivot.IAt(southIdx, 1) // Product A column
	if southA != nil {
		t.Errorf("Expected nil for South-A, got %v", southA)
	}
}
