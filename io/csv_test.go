package io

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/series"
)

// Test CSV reading functionality
func TestReadCSV(t *testing.T) {
	t.Run("ReadSimpleCSV", func(t *testing.T) {
		// Create test CSV content
		csvContent := `name,age,score
Alice,25,95.5
Bob,30,87.2
Charlie,22,92.8`

		// Create temporary file
		tmpDir := t.TempDir()
		csvFile := filepath.Join(tmpDir, "test.csv")
		err := os.WriteFile(csvFile, []byte(csvContent), 0644)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}

		// Read CSV
		df, err := ReadCSV(csvFile)
		if err != nil {
			t.Fatalf("ReadCSV failed: %v", err)
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

	t.Run("ReadCSVWithOptions", func(t *testing.T) {
		// Test CSV with different separator
		csvContent := `name;age;score
Alice;25;95.5
Bob;30;87.2`

		tmpDir := t.TempDir()
		csvFile := filepath.Join(tmpDir, "test_semicolon.csv")
		err := os.WriteFile(csvFile, []byte(csvContent), 0644)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}

		opts := &CSVOptions{
			Separator: ';',
			Header:    true,
		}

		df, err := ReadCSVWithOptions(csvFile, opts)
		if err != nil {
			t.Fatalf("ReadCSVWithOptions failed: %v", err)
		}

		if df.Len() != 2 {
			t.Errorf("Expected 2 rows, got %d", df.Len())
		}
	})

	t.Run("ReadCSVFromReader", func(t *testing.T) {
		csvContent := `name,age,score
Alice,25,95.5
Bob,30,87.2`

		reader := strings.NewReader(csvContent)

		df, err := ReadCSVFromReader(reader, nil)
		if err != nil {
			t.Fatalf("ReadCSVFromReader failed: %v", err)
		}

		if df.Len() != 2 {
			t.Errorf("Expected 2 rows, got %d", df.Len())
		}
	})
}

// Test CSV writing functionality
func TestWriteCSV(t *testing.T) {
	t.Run("WriteSimpleCSV", func(t *testing.T) {
		// Create test DataFrame
		names, _ := series.FromSlice([]string{"Alice", "Bob", "Charlie"}, nil, "name")
		ages, _ := series.FromSlice([]int64{25, 30, 22}, nil, "age")
		scores, _ := series.FromSlice([]float64{95.5, 87.2, 92.8}, nil, "score")

		df, err := dataframe.FromSeries([]*series.Series{names, ages, scores})
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		tmpDir := t.TempDir()
		csvFile := filepath.Join(tmpDir, "output.csv")

		err = WriteCSV(df, csvFile)
		if err != nil {
			t.Fatalf("WriteCSV failed: %v", err)
		}

		// Verify file was created
		if _, err := os.Stat(csvFile); os.IsNotExist(err) {
			t.Fatal("CSV file was not created")
		}

		// Read back and verify
		content, err := os.ReadFile(csvFile)
		if err != nil {
			t.Fatalf("Failed to read written CSV: %v", err)
		}

		lines := strings.Split(strings.TrimSpace(string(content)), "\n")
		if len(lines) != 4 { // header + 3 data rows
			t.Errorf("Expected 4 lines, got %d", len(lines))
		}

		// Verify header
		if lines[0] != "name,age,score" {
			t.Errorf("Expected header 'name,age,score', got %s", lines[0])
		}

		// Verify first data row
		if lines[1] != "Alice,25,95.5" {
			t.Errorf("Expected 'Alice,25,95.5', got %s", lines[1])
		}
	})

	t.Run("WriteCSVWithOptions", func(t *testing.T) {
		names, _ := series.FromSlice([]string{"Alice", "Bob"}, nil, "name")
		ages, _ := series.FromSlice([]int64{25, 30}, nil, "age")

		df, err := dataframe.FromSeries([]*series.Series{names, ages})
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		tmpDir := t.TempDir()
		csvFile := filepath.Join(tmpDir, "output_opts.csv")

		opts := &CSVOptions{
			Separator: ';',
			Header:    false,
		}

		err = WriteCSVWithOptions(df, csvFile, opts)
		if err != nil {
			t.Fatalf("WriteCSVWithOptions failed: %v", err)
		}

		content, err := os.ReadFile(csvFile)
		if err != nil {
			t.Fatalf("Failed to read written CSV: %v", err)
		}

		lines := strings.Split(strings.TrimSpace(string(content)), "\n")
		if len(lines) != 2 { // no header, 2 data rows
			t.Errorf("Expected 2 lines, got %d", len(lines))
		}

		if lines[0] != "Alice;25" {
			t.Errorf("Expected 'Alice;25', got %s", lines[0])
		}
	})
}

// Test CSV error conditions
func TestCSVErrors(t *testing.T) {
	t.Run("ReadNonexistentFile", func(t *testing.T) {
		_, err := ReadCSV("/nonexistent/file.csv")
		if err == nil {
			t.Error("Expected error when reading nonexistent file")
		}
	})

	t.Run("ReadMalformedCSV", func(t *testing.T) {
		csvContent := `name,age,score
Alice,25
Bob,30,87.2,extra`

		tmpDir := t.TempDir()
		csvFile := filepath.Join(tmpDir, "malformed.csv")
		err := os.WriteFile(csvFile, []byte(csvContent), 0644)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}

		_, err = ReadCSV(csvFile)
		if err == nil {
			t.Error("Expected error when reading malformed CSV")
		}
	})
}
