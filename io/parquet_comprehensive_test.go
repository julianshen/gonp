package io

import (
	"path/filepath"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/series"
)

func TestParquetComprehensive(t *testing.T) {
	t.Run("Multiple Data Types", func(t *testing.T) {
		// Create test data with different types
		floatSer, err := series.FromSlice([]float64{1.1, 2.2, 3.3, 4.4, 5.5}, nil, "float_col")
		if err != nil {
			t.Fatalf("Failed to create float series: %v", err)
		}

		intSer, err := series.FromSlice([]float64{10, 20, 30, 40, 50}, nil, "int_col")
		if err != nil {
			t.Fatalf("Failed to create int series: %v", err)
		}

		df, err := dataframe.FromSeries([]*series.Series{floatSer, intSer})
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		// Create temporary file
		tmpFile := filepath.Join(t.TempDir(), "comprehensive.parquet")

		// Write to Parquet
		err = WriteParquet(tmpFile, df)
		if err != nil {
			t.Fatalf("Failed to write Parquet: %v", err)
		}

		// Read from Parquet
		readDF, err := ReadParquet(tmpFile)
		if err != nil {
			t.Fatalf("Failed to read Parquet: %v", err)
		}

		// Verify data integrity
		if readDF.Len() != df.Len() {
			t.Errorf("Row count mismatch: expected %d, got %d", df.Len(), readDF.Len())
		}

		if len(readDF.Columns()) != len(df.Columns()) {
			t.Errorf("Column count mismatch: expected %d, got %d", len(df.Columns()), len(readDF.Columns()))
		}

		// Verify first few values
		for i := 0; i < 3; i++ {
			originalFloat, _ := df.IAt(i, 0)
			readFloat, _ := readDF.IAt(i, 0)
			if originalFloat != readFloat {
				t.Errorf("Float value mismatch at row %d: expected %v, got %v", i, originalFloat, readFloat)
			}

			originalInt, _ := df.IAt(i, 1)
			readInt, _ := readDF.IAt(i, 1)
			if originalInt != readInt {
				t.Errorf("Int value mismatch at row %d: expected %v, got %v", i, originalInt, readInt)
			}
		}
	})

	t.Run("Parquet Options", func(t *testing.T) {
		// Create test data
		ser, err := series.FromSlice([]float64{1, 2, 3, 4, 5}, nil, "values")
		if err != nil {
			t.Fatalf("Failed to create series: %v", err)
		}

		df, err := dataframe.FromSeries([]*series.Series{ser})
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		// Test with custom options
		options := &ParquetWriteOptions{
			Compression:  CompressionGzip,
			RowGroupSize: 2,
			PageSize:     512,
			EnableStats:  true,
		}

		tmpFile := filepath.Join(t.TempDir(), "options.parquet")

		err = WriteParquetWithOptions(tmpFile, df, options)
		if err != nil {
			t.Fatalf("Failed to write Parquet with options: %v", err)
		}

		// Read back and verify
		readDF, err := ReadParquet(tmpFile)
		if err != nil {
			t.Fatalf("Failed to read Parquet: %v", err)
		}

		if readDF.Len() != 5 {
			t.Errorf("Expected 5 rows, got %d", readDF.Len())
		}
	})

	t.Run("Metadata Reading", func(t *testing.T) {
		// Create test data
		ser, err := series.FromSlice([]float64{1, 2, 3}, nil, "test_col")
		if err != nil {
			t.Fatalf("Failed to create series: %v", err)
		}

		df, err := dataframe.FromSeries([]*series.Series{ser})
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		tmpFile := filepath.Join(t.TempDir(), "metadata.parquet")

		// Write file
		err = WriteParquet(tmpFile, df)
		if err != nil {
			t.Fatalf("Failed to write Parquet: %v", err)
		}

		// Read metadata
		metadata, err := ReadParquetMetadata(tmpFile)
		if err != nil {
			t.Fatalf("Failed to read metadata: %v", err)
		}

		if metadata.NumRows != 3 {
			t.Errorf("Expected 3 rows in metadata, got %d", metadata.NumRows)
		}

		if metadata.NumColumns != 1 {
			t.Errorf("Expected 1 column in metadata, got %d", metadata.NumColumns)
		}

		if len(metadata.Columns) != 1 {
			t.Errorf("Expected 1 column schema, got %d", len(metadata.Columns))
		}

		if metadata.Columns[0].Name != "test_col" {
			t.Errorf("Expected column name 'test_col', got '%s'", metadata.Columns[0].Name)
		}
	})

	t.Run("File Info", func(t *testing.T) {
		// Create test data
		ser, err := series.FromSlice([]float64{1, 2, 3}, nil, "info_col")
		if err != nil {
			t.Fatalf("Failed to create series: %v", err)
		}

		df, err := dataframe.FromSeries([]*series.Series{ser})
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		tmpFile := filepath.Join(t.TempDir(), "info.parquet")

		// Write file
		err = WriteParquet(tmpFile, df)
		if err != nil {
			t.Fatalf("Failed to write Parquet: %v", err)
		}

		// Get file info
		info, err := GetParquetFileInfo(tmpFile)
		if err != nil {
			t.Fatalf("Failed to get file info: %v", err)
		}

		if info.Path != tmpFile {
			t.Errorf("Expected path '%s', got '%s'", tmpFile, info.Path)
		}

		if info.Size <= 0 {
			t.Error("Expected positive file size")
		}

		if info.NumRowGroups != 1 {
			t.Errorf("Expected 1 row group, got %d", info.NumRowGroups)
		}
	})

	t.Run("Streaming Reader", func(t *testing.T) {
		// Create test data
		ser, err := series.FromSlice([]float64{1, 2, 3, 4, 5}, nil, "stream_col")
		if err != nil {
			t.Fatalf("Failed to create series: %v", err)
		}

		df, err := dataframe.FromSeries([]*series.Series{ser})
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		tmpFile := filepath.Join(t.TempDir(), "stream.parquet")

		// Write file
		err = WriteParquet(tmpFile, df)
		if err != nil {
			t.Fatalf("Failed to write Parquet: %v", err)
		}

		// Create reader
		reader, err := NewParquetReader(tmpFile)
		if err != nil {
			t.Fatalf("Failed to create reader: %v", err)
		}
		defer reader.Close()

		// Check if has data
		if !reader.HasNext() {
			t.Error("Reader should have data")
		}

		// Read batch
		batch, err := reader.ReadBatch(100)
		if err != nil {
			t.Fatalf("Failed to read batch: %v", err)
		}

		if batch.Len() != 5 {
			t.Errorf("Expected 5 rows in batch, got %d", batch.Len())
		}

		// Should be finished now
		if reader.HasNext() {
			t.Error("Reader should not have more data")
		}
	})

	t.Run("Streaming Writer", func(t *testing.T) {
		// Create writer
		tmpFile := filepath.Join(t.TempDir(), "stream_write.parquet")
		writer, err := NewParquetWriter(tmpFile, nil)
		if err != nil {
			t.Fatalf("Failed to create writer: %v", err)
		}

		// Write batches
		for i := 0; i < 3; i++ {
			ser, err := series.FromSlice([]float64{float64(i*10 + 1), float64(i*10 + 2)}, nil, "batch_col")
			if err != nil {
				t.Fatalf("Failed to create series for batch %d: %v", i, err)
			}

			df, err := dataframe.FromSeries([]*series.Series{ser})
			if err != nil {
				t.Fatalf("Failed to create DataFrame for batch %d: %v", i, err)
			}

			err = writer.WriteBatch(df)
			if err != nil {
				t.Fatalf("Failed to write batch %d: %v", i, err)
			}
		}

		// Close writer
		err = writer.Close()
		if err != nil {
			t.Fatalf("Failed to close writer: %v", err)
		}

		// Read back and verify
		readDF, err := ReadParquet(tmpFile)
		if err != nil {
			t.Fatalf("Failed to read streamed file: %v", err)
		}

		if readDF.Len() != 6 { // 3 batches * 2 rows each
			t.Errorf("Expected 6 rows from streamed writes, got %d", readDF.Len())
		}
	})

	t.Run("Large Dataset", func(t *testing.T) {
		// Create larger dataset to test performance
		size := 1000
		values := make([]float64, size)
		for i := 0; i < size; i++ {
			values[i] = float64(i) * 0.1
		}

		ser, err := series.FromSlice(values, nil, "large_col")
		if err != nil {
			t.Fatalf("Failed to create large series: %v", err)
		}

		df, err := dataframe.FromSeries([]*series.Series{ser})
		if err != nil {
			t.Fatalf("Failed to create large DataFrame: %v", err)
		}

		tmpFile := filepath.Join(t.TempDir(), "large.parquet")

		// Write with row group optimization
		options := &ParquetWriteOptions{
			Compression:  CompressionSnappy,
			RowGroupSize: 100, // Smaller row groups
			EnableStats:  true,
		}

		err = WriteParquetWithOptions(tmpFile, df, options)
		if err != nil {
			t.Fatalf("Failed to write large Parquet: %v", err)
		}

		// Read back and verify size
		readDF, err := ReadParquet(tmpFile)
		if err != nil {
			t.Fatalf("Failed to read large Parquet: %v", err)
		}

		if readDF.Len() != size {
			t.Errorf("Expected %d rows, got %d", size, readDF.Len())
		}

		// Verify a few sample values
		for i := 0; i < 10; i++ {
			original, _ := df.IAt(i, 0)
			read, _ := readDF.IAt(i, 0)
			if original != read {
				t.Errorf("Value mismatch at row %d: expected %v, got %v", i, original, read)
			}
		}
	})

	t.Run("Array Integration", func(t *testing.T) {
		// Test different array types
		arrays := []struct {
			name   string
			data   interface{}
			column string
		}{
			{"float64", []float64{1.1, 2.2, 3.3}, "float_vals"},
			{"int_as_float", []float64{10, 20, 30}, "int_vals"}, // Use float64 since array.FromSlice supports it
		}

		for _, test := range arrays {
			t.Run(test.name, func(t *testing.T) {
				arr, err := array.FromSlice(test.data)
				if err != nil {
					t.Fatalf("Failed to create array: %v", err)
				}

				tmpFile := filepath.Join(t.TempDir(), test.name+".parquet")

				err = WriteArrayAsParquet(tmpFile, arr, test.column)
				if err != nil {
					t.Fatalf("Failed to write array as Parquet: %v", err)
				}

				// Read back as array
				readArr, err := ReadParquetAsArray(tmpFile, test.column)
				if err != nil {
					t.Fatalf("Failed to read Parquet as array: %v", err)
				}

				if readArr.Size() != arr.Size() {
					t.Errorf("Array size mismatch: expected %d, got %d", arr.Size(), readArr.Size())
				}

				// Verify values
				for i := 0; i < arr.Size(); i++ {
					original := arr.At(i)
					read := readArr.At(i)
					if original != read {
						t.Errorf("Array value mismatch at index %d: expected %v, got %v", i, original, read)
					}
				}
			})
		}
	})
}
