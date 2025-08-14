package io

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/series"
)

func TestParquetSimple(t *testing.T) {
	t.Run("Basic Write and Read", func(t *testing.T) {
		// Create simple test data (use float64 which is supported)
		intSer, err := series.FromSlice([]float64{1, 2, 3}, nil, "col1")
		if err != nil {
			t.Fatalf("Failed to create series: %v", err)
		}

		df, err := dataframe.FromSeries([]*series.Series{intSer})
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		// Create temporary file
		tmpFile := filepath.Join(t.TempDir(), "simple.parquet")

		// Write to Parquet
		err = WriteParquet(tmpFile, df)
		if err != nil {
			t.Fatalf("Failed to write Parquet: %v", err)
		}

		// Check file exists
		if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
			t.Fatal("Parquet file was not created")
		}

		// Read from Parquet
		readDF, err := ReadParquet(tmpFile)
		if err != nil {
			t.Fatalf("Failed to read Parquet: %v", err)
		}

		// Basic validation
		if readDF.Len() != df.Len() {
			t.Errorf("Row count mismatch: expected %d, got %d", df.Len(), readDF.Len())
		}

		if len(readDF.Columns()) != len(df.Columns()) {
			t.Errorf("Column count mismatch: expected %d, got %d", len(df.Columns()), len(readDF.Columns()))
		}
	})

	t.Run("Array to Parquet", func(t *testing.T) {
		// Create test array
		arr, err := array.FromSlice([]float64{1.1, 2.2, 3.3})
		if err != nil {
			t.Fatalf("Failed to create array: %v", err)
		}

		tmpFile := filepath.Join(t.TempDir(), "array.parquet")

		// Write array as Parquet
		err = WriteArrayAsParquet(tmpFile, arr, "values")
		if err != nil {
			t.Fatalf("Failed to write array as Parquet: %v", err)
		}

		// Read back as DataFrame
		df, err := ReadParquet(tmpFile)
		if err != nil {
			t.Fatalf("Failed to read array Parquet: %v", err)
		}

		if df.Len() != 3 {
			t.Errorf("Expected 3 rows, got %d", df.Len())
		}

		if len(df.Columns()) != 1 {
			t.Errorf("Expected 1 column, got %d", len(df.Columns()))
		}
	})

	t.Run("Error Handling", func(t *testing.T) {
		// Test nil DataFrame
		err := WriteParquet("test.parquet", nil)
		if err == nil {
			t.Error("Expected error for nil DataFrame")
		}

		// Test non-existent file
		_, err = ReadParquet("non_existent.parquet")
		if err == nil {
			t.Error("Expected error for non-existent file")
		}
	})
}
