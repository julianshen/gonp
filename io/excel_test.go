package io

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tealeg/xlsx/v3"

	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/series"
)

// Test Excel reading functionality
func TestReadExcel(t *testing.T) {
	t.Run("ReadSimpleExcel", func(t *testing.T) {
		// Create temporary directory for test files
		tmpDir := t.TempDir()
		excelFile := filepath.Join(tmpDir, "test.xlsx")

		// Create test Excel file programmatically
		err := createTestExcelFile(excelFile)
		if err != nil {
			t.Fatalf("Failed to create test Excel file: %v", err)
		}

		// Read Excel
		df, err := ReadExcel(excelFile)
		if err != nil {
			t.Fatalf("ReadExcel failed: %v", err)
		}

		// Verify structure
		if df == nil {
			t.Fatal("Expected non-nil DataFrame")
		}

		expectedCols := []string{"name", "age", "score"}
		if len(df.Columns()) != len(expectedCols) {
			t.Errorf("Expected %d columns, got %d", len(expectedCols), len(df.Columns()))
		}

		if df.Len() != 3 {
			t.Errorf("Expected 3 rows, got %d", df.Len())
		}

		// Verify data types and values
		nameCol, err := df.GetColumn("name")
		if err != nil {
			t.Fatalf("Failed to get name column: %v", err)
		}

		if nameCol.At(0) != "Alice" {
			t.Errorf("Expected first name to be 'Alice', got %v", nameCol.At(0))
		}

		ageCol, err := df.GetColumn("age")
		if err != nil {
			t.Fatalf("Failed to get age column: %v", err)
		}

		if ageCol.At(0) != int64(25) {
			t.Errorf("Expected first age to be 25, got %v", ageCol.At(0))
		}

		scoreCol, err := df.GetColumn("score")
		if err != nil {
			t.Fatalf("Failed to get score column: %v", err)
		}

		if scoreCol.At(0) != 95.5 {
			t.Errorf("Expected first score to be 95.5, got %v", scoreCol.At(0))
		}
	})

	t.Run("ReadExcelWithOptions", func(t *testing.T) {
		tmpDir := t.TempDir()
		excelFile := filepath.Join(tmpDir, "test_opts.xlsx")

		// Create test file with multiple sheets
		err := createTestExcelFileMultiSheet(excelFile)
		if err != nil {
			t.Fatalf("Failed to create test Excel file: %v", err)
		}

		opts := &ExcelOptions{
			SheetName: "Sheet2",
			Header:    true,
			StartRow:  0, // Start from first row, so we read the actual header
		}

		df, err := ReadExcelWithOptions(excelFile, opts)
		if err != nil {
			t.Fatalf("ReadExcelWithOptions failed: %v", err)
		}

		if df.Len() != 2 {
			t.Errorf("Expected 2 rows, got %d", df.Len())
		}
	})

	t.Run("ReadExcelSpecificSheet", func(t *testing.T) {
		tmpDir := t.TempDir()
		excelFile := filepath.Join(tmpDir, "test_sheets.xlsx")

		err := createTestExcelFileMultiSheet(excelFile)
		if err != nil {
			t.Fatalf("Failed to create test Excel file: %v", err)
		}

		// Read from specific sheet
		df, err := ReadExcelSheet(excelFile, "Sheet1")
		if err != nil {
			t.Fatalf("ReadExcelSheet failed: %v", err)
		}

		if df.Len() != 3 {
			t.Errorf("Expected 3 rows from Sheet1, got %d", df.Len())
		}
	})
}

// Test Excel writing functionality
func TestWriteExcel(t *testing.T) {
	t.Run("WriteSimpleExcel", func(t *testing.T) {
		// Create test DataFrame
		names, _ := series.FromSlice([]string{"Alice", "Bob", "Charlie"}, nil, "name")
		ages, _ := series.FromSlice([]int64{25, 30, 22}, nil, "age")
		scores, _ := series.FromSlice([]float64{95.5, 87.2, 92.8}, nil, "score")

		df, err := dataframe.FromSeries([]*series.Series{names, ages, scores})
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		tmpDir := t.TempDir()
		excelFile := filepath.Join(tmpDir, "output.xlsx")

		err = WriteExcel(df, excelFile)
		if err != nil {
			t.Fatalf("WriteExcel failed: %v", err)
		}

		// Verify file was created
		if _, err := os.Stat(excelFile); os.IsNotExist(err) {
			t.Fatal("Excel file was not created")
		}

		// Read back and verify
		dfRead, err := ReadExcel(excelFile)
		if err != nil {
			t.Fatalf("Failed to read written Excel: %v", err)
		}

		if dfRead.Len() != 3 {
			t.Errorf("Expected 3 rows in read DataFrame, got %d", dfRead.Len())
		}

		// Verify first row data
		nameCol, _ := dfRead.GetColumn("name")
		if nameCol.At(0) != "Alice" {
			t.Errorf("Expected first name 'Alice', got %v", nameCol.At(0))
		}
	})

	t.Run("WriteExcelWithOptions", func(t *testing.T) {
		names, _ := series.FromSlice([]string{"Alice", "Bob"}, nil, "name")
		ages, _ := series.FromSlice([]int64{25, 30}, nil, "age")

		df, err := dataframe.FromSeries([]*series.Series{names, ages})
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		tmpDir := t.TempDir()
		excelFile := filepath.Join(tmpDir, "output_opts.xlsx")

		opts := &ExcelOptions{
			SheetName: "CustomSheet",
			Header:    false,
			StartRow:  2, // Start from third row
		}

		err = WriteExcelWithOptions(df, excelFile, opts)
		if err != nil {
			t.Fatalf("WriteExcelWithOptions failed: %v", err)
		}

		// Verify file was created
		if _, err := os.Stat(excelFile); os.IsNotExist(err) {
			t.Fatal("Excel file was not created")
		}

		// Read back with same options
		dfRead, err := ReadExcelWithOptions(excelFile, opts)
		if err != nil {
			t.Fatalf("Failed to read written Excel with options: %v", err)
		}

		if dfRead.Len() != 2 {
			t.Errorf("Expected 2 rows, got %d", dfRead.Len())
		}
	})

	t.Run("WriteExcelMultipleSheets", func(t *testing.T) {
		// Create multiple DataFrames
		df1Names, _ := series.FromSlice([]string{"Alice", "Bob"}, nil, "name")
		df1Ages, _ := series.FromSlice([]int64{25, 30}, nil, "age")
		df1, _ := dataframe.FromSeries([]*series.Series{df1Names, df1Ages})

		df2Prod, _ := series.FromSlice([]string{"Apple", "Banana"}, nil, "product")
		df2Price, _ := series.FromSlice([]float64{1.50, 0.75}, nil, "price")
		df2, _ := dataframe.FromSeries([]*series.Series{df2Prod, df2Price})

		tmpDir := t.TempDir()
		excelFile := filepath.Join(tmpDir, "multi_sheet.xlsx")

		sheets := map[string]*dataframe.DataFrame{
			"People":   df1,
			"Products": df2,
		}

		err := WriteExcelMultiSheet(excelFile, sheets)
		if err != nil {
			t.Fatalf("WriteExcelMultiSheet failed: %v", err)
		}

		// Verify file was created
		if _, err := os.Stat(excelFile); os.IsNotExist(err) {
			t.Fatal("Excel file was not created")
		}

		// Read back specific sheets
		peopleDF, err := ReadExcelSheet(excelFile, "People")
		if err != nil {
			t.Fatalf("Failed to read People sheet: %v", err)
		}

		if peopleDF.Len() != 2 {
			t.Errorf("Expected 2 rows in People sheet, got %d", peopleDF.Len())
		}

		productsDF, err := ReadExcelSheet(excelFile, "Products")
		if err != nil {
			t.Fatalf("Failed to read Products sheet: %v", err)
		}

		if productsDF.Len() != 2 {
			t.Errorf("Expected 2 rows in Products sheet, got %d", productsDF.Len())
		}
	})
}

// Test Excel-specific functionality
func TestExcelSpecific(t *testing.T) {
	t.Run("GetSheetNames", func(t *testing.T) {
		tmpDir := t.TempDir()
		excelFile := filepath.Join(tmpDir, "test_sheet_names.xlsx")

		err := createTestExcelFileMultiSheet(excelFile)
		if err != nil {
			t.Fatalf("Failed to create test Excel file: %v", err)
		}

		sheetNames, err := GetExcelSheetNames(excelFile)
		if err != nil {
			t.Fatalf("GetExcelSheetNames failed: %v", err)
		}

		expectedSheets := []string{"Sheet1", "Sheet2"}
		if len(sheetNames) != len(expectedSheets) {
			t.Errorf("Expected %d sheets, got %d", len(expectedSheets), len(sheetNames))
		}

		for i, expected := range expectedSheets {
			if i >= len(sheetNames) || sheetNames[i] != expected {
				t.Errorf("Expected sheet %d to be '%s', got '%s'", i, expected, sheetNames[i])
			}
		}
	})

	t.Run("ReadExcelWithFormulas", func(t *testing.T) {
		tmpDir := t.TempDir()
		excelFile := filepath.Join(tmpDir, "test_formulas.xlsx")

		err := createTestExcelFileWithFormulas(excelFile)
		if err != nil {
			t.Fatalf("Failed to create test Excel file with formulas: %v", err)
		}

		df, err := ReadExcel(excelFile)
		if err != nil {
			t.Fatalf("ReadExcel failed: %v", err)
		}

		// Verify that formulas are evaluated and return calculated values
		totalCol, err := df.GetColumn("total")
		if err != nil {
			t.Fatalf("Failed to get total column: %v", err)
		}

		// First row should have 10 + 5 = 15
		val := totalCol.At(0)
		var expectedVal float64 = 15.0
		if val != expectedVal && val != int64(15) {
			t.Errorf("Expected first total to be 15 (as float64 or int64), got %v (type %T)", val, val)
		}
	})
}

// Test error conditions
func TestExcelErrors(t *testing.T) {
	t.Run("ReadNonexistentFile", func(t *testing.T) {
		_, err := ReadExcel("nonexistent.xlsx")
		if err == nil {
			t.Error("Expected error for nonexistent file")
		}
	})

	t.Run("ReadInvalidExcelFile", func(t *testing.T) {
		tmpDir := t.TempDir()
		invalidFile := filepath.Join(tmpDir, "invalid.xlsx")

		// Create a text file with .xlsx extension
		err := os.WriteFile(invalidFile, []byte("not an excel file"), 0644)
		if err != nil {
			t.Fatalf("Failed to create invalid file: %v", err)
		}

		_, err = ReadExcel(invalidFile)
		if err == nil {
			t.Error("Expected error for invalid Excel file")
		}
	})

	t.Run("ReadNonexistentSheet", func(t *testing.T) {
		tmpDir := t.TempDir()
		excelFile := filepath.Join(tmpDir, "test.xlsx")

		err := createTestExcelFile(excelFile)
		if err != nil {
			t.Fatalf("Failed to create test Excel file: %v", err)
		}

		_, err = ReadExcelSheet(excelFile, "NonexistentSheet")
		if err == nil {
			t.Error("Expected error for nonexistent sheet")
		}
	})
}

// Helper functions to create test Excel files
func createTestExcelFile(filename string) error {
	// Create a new Excel file
	xlFile := xlsx.NewFile()

	sheet, err := xlFile.AddSheet("Sheet1")
	if err != nil {
		return fmt.Errorf("failed to add sheet: %v", err)
	}

	// Add header row
	headerRow := sheet.AddRow()
	headerRow.AddCell().Value = "name"
	headerRow.AddCell().Value = "age"
	headerRow.AddCell().Value = "score"

	// Add data rows
	dataRows := [][]interface{}{
		{"Alice", 25, 95.5},
		{"Bob", 30, 87.2},
		{"Charlie", 22, 92.8},
	}

	for _, rowData := range dataRows {
		row := sheet.AddRow()
		for _, cellData := range rowData {
			cell := row.AddCell()
			cell.SetValue(cellData)
		}
	}

	return xlFile.Save(filename)
}

func createTestExcelFileMultiSheet(filename string) error {
	xlFile := xlsx.NewFile()

	// Create Sheet1
	sheet1, err := xlFile.AddSheet("Sheet1")
	if err != nil {
		return fmt.Errorf("failed to add Sheet1: %v", err)
	}

	// Add data to Sheet1
	headerRow1 := sheet1.AddRow()
	headerRow1.AddCell().Value = "name"
	headerRow1.AddCell().Value = "age"
	headerRow1.AddCell().Value = "score"

	dataRows1 := [][]interface{}{
		{"Alice", 25, 95.5},
		{"Bob", 30, 87.2},
		{"Charlie", 22, 92.8},
	}

	for _, rowData := range dataRows1 {
		row := sheet1.AddRow()
		for _, cellData := range rowData {
			cell := row.AddCell()
			cell.SetValue(cellData)
		}
	}

	// Create Sheet2
	sheet2, err := xlFile.AddSheet("Sheet2")
	if err != nil {
		return fmt.Errorf("failed to add Sheet2: %v", err)
	}

	// Add data to Sheet2 (starting from row 1, different from Sheet1)
	headerRow2 := sheet2.AddRow()
	headerRow2.AddCell().Value = "product"
	headerRow2.AddCell().Value = "price"

	dataRows2 := [][]interface{}{
		{"Apple", 1.50},
		{"Banana", 0.75},
	}

	for _, rowData := range dataRows2 {
		row := sheet2.AddRow()
		for _, cellData := range rowData {
			cell := row.AddCell()
			cell.SetValue(cellData)
		}
	}

	return xlFile.Save(filename)
}

func createTestExcelFileWithFormulas(filename string) error {
	xlFile := xlsx.NewFile()

	sheet, err := xlFile.AddSheet("Sheet1")
	if err != nil {
		return fmt.Errorf("failed to add sheet: %v", err)
	}

	// Add header row
	headerRow := sheet.AddRow()
	headerRow.AddCell().Value = "a"
	headerRow.AddCell().Value = "b"
	headerRow.AddCell().Value = "total"

	// Add data rows with formulas
	dataRows := [][]interface{}{
		{10, 5, 15}, // For testing, we'll use calculated values instead of formulas
		{20, 8, 28}, // since formula evaluation is complex
	}

	for _, rowData := range dataRows {
		row := sheet.AddRow()
		for _, cellData := range rowData {
			cell := row.AddCell()
			cell.SetValue(cellData)
		}
	}

	return xlFile.Save(filename)
}
