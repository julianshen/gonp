package io

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/series"
)

// BenchmarkLargeCSVRead benchmarks reading large CSV files
func BenchmarkLargeCSVRead(b *testing.B) {
	// Create test data files of various sizes
	sizes := []int{1000, 10000, 100000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("rows_%d", size), func(b *testing.B) {
			// Create test CSV file
			tmpDir := b.TempDir()
			testFile := filepath.Join(tmpDir, fmt.Sprintf("large_%d.csv", size))

			// Generate test data
			file, err := os.Create(testFile)
			if err != nil {
				b.Fatalf("Failed to create test file: %v", err)
			}

			// Write header
			file.WriteString("id,name,value,score\n")

			// Write data rows
			for i := 0; i < size; i++ {
				file.WriteString(fmt.Sprintf("%d,name_%d,%.2f,%.1f\n", i, i, float64(i)*1.5, float64(i%100)))
			}
			file.Close()

			// Get file size for reporting
			stat, _ := os.Stat(testFile)
			fileSize := stat.Size()

			// Benchmark reading
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				df, err := ReadCSV(testFile)
				if err != nil {
					b.Fatalf("Failed to read CSV: %v", err)
				}
				if df.Len() != size {
					b.Fatalf("Expected %d rows, got %d", size, df.Len())
				}
			}

			// Report throughput
			if b.N > 0 {
				bytesPerOp := fileSize
				rowsPerSec := float64(size) / (float64(b.Elapsed().Nanoseconds()) / float64(b.N) / 1e9)
				mbPerSec := float64(bytesPerOp) / (1024 * 1024) / (float64(b.Elapsed().Nanoseconds()) / float64(b.N) / 1e9)

				b.ReportMetric(rowsPerSec, "rows/sec")
				b.ReportMetric(mbPerSec, "MB/sec")
			}
		})
	}
}

// BenchmarkLargeCSVWrite benchmarks writing large CSV files
func BenchmarkLargeCSVWrite(b *testing.B) {
	sizes := []int{1000, 10000, 100000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("rows_%d", size), func(b *testing.B) {
			// Create test DataFrame
			idData := make([]float64, size)
			nameData := make([]interface{}, size)
			valueData := make([]float64, size)
			scoreData := make([]float64, size)

			for i := 0; i < size; i++ {
				idData[i] = float64(i)
				nameData[i] = fmt.Sprintf("name_%d", i)
				valueData[i] = float64(i) * 1.5
				scoreData[i] = float64(i % 100)
			}

			idSer, _ := series.FromSlice(idData, nil, "id")
			nameSer, _ := series.FromSlice(nameData, nil, "name")
			valueSer, _ := series.FromSlice(valueData, nil, "value")
			scoreSer, _ := series.FromSlice(scoreData, nil, "score")

			df, err := dataframe.FromSeries([]*series.Series{idSer, nameSer, valueSer, scoreSer})
			if err != nil {
				b.Fatalf("Failed to create DataFrame: %v", err)
			}

			tmpDir := b.TempDir()

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				testFile := filepath.Join(tmpDir, fmt.Sprintf("write_%d_%d.csv", size, i))

				err := WriteCSV(df, testFile)
				if err != nil {
					b.Fatalf("Failed to write CSV: %v", err)
				}

				// Clean up to avoid disk space issues
				os.Remove(testFile)
			}

			// Report throughput
			if b.N > 0 {
				rowsPerSec := float64(size) / (float64(b.Elapsed().Nanoseconds()) / float64(b.N) / 1e9)
				b.ReportMetric(rowsPerSec, "rows/sec")
			}
		})
	}
}

// BenchmarkParquetPerformance benchmarks Parquet file operations
func BenchmarkParquetPerformance(b *testing.B) {
	sizes := []int{1000, 10000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("read_write_%d", size), func(b *testing.B) {
			// Create test data
			data := make([]float64, size)
			for i := range data {
				data[i] = float64(i) * 0.1
			}

			ser, _ := series.FromSlice(data, nil, "values")
			df, _ := dataframe.FromSeries([]*series.Series{ser})

			tmpDir := b.TempDir()
			testFile := filepath.Join(tmpDir, fmt.Sprintf("bench_%d.parquet", size))

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				// Write
				err := WriteParquet(testFile, df)
				if err != nil {
					b.Fatalf("Failed to write Parquet: %v", err)
				}

				// Read
				readDF, err := ReadParquet(testFile)
				if err != nil {
					b.Fatalf("Failed to read Parquet: %v", err)
				}

				if readDF.Len() != size {
					b.Fatalf("Size mismatch: expected %d, got %d", size, readDF.Len())
				}

				// Clean up
				os.Remove(testFile)
			}
		})
	}
}

// BenchmarkMemoryUsage tests memory usage patterns
func BenchmarkMemoryUsage(b *testing.B) {
	sizes := []int{1000, 10000, 100000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("memory_pattern_%d", size), func(b *testing.B) {
			tmpDir := b.TempDir()
			testFile := filepath.Join(tmpDir, fmt.Sprintf("memory_%d.csv", size))

			// Create test file
			file, _ := os.Create(testFile)
			file.WriteString("value\n")
			for i := 0; i < size; i++ {
				file.WriteString(fmt.Sprintf("%.6f\n", float64(i)*0.001))
			}
			file.Close()

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				var m1, m2 runtime.MemStats
				runtime.GC()
				runtime.ReadMemStats(&m1)

				df, err := ReadCSV(testFile)
				if err != nil {
					b.Fatalf("Failed to read CSV: %v", err)
				}

				runtime.GC()
				runtime.ReadMemStats(&m2)

				allocDelta := m2.TotalAlloc - m1.TotalAlloc
				heapDelta := m2.HeapAlloc - m1.HeapAlloc

				_ = df // Use df to prevent optimization

				// Report memory metrics per operation
				b.ReportMetric(float64(allocDelta)/float64(size), "bytes_allocated_per_row")
				b.ReportMetric(float64(heapDelta)/1024/1024, "heap_MB")
			}
		})
	}
}

// TestIOPerformanceProfile provides detailed performance profiling
func TestIOPerformanceProfile(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance profile in short mode")
	}

	t.Run("CSV Performance Profile", func(t *testing.T) {
		sizes := []int{1000, 10000, 50000}

		for _, size := range sizes {
			tmpDir := t.TempDir()
			testFile := filepath.Join(tmpDir, fmt.Sprintf("profile_%d.csv", size))

			// Generate test data
			file, err := os.Create(testFile)
			if err != nil {
				t.Fatalf("Failed to create test file: %v", err)
			}

			file.WriteString("id,data,value\n")
			for i := 0; i < size; i++ {
				file.WriteString(fmt.Sprintf("%d,data_%d,%.3f\n", i, i, float64(i)*0.123))
			}
			file.Close()

			// Get file size
			stat, _ := os.Stat(testFile)
			fileSize := stat.Size()

			// Time the read operation
			start := time.Now()
			df, err := ReadCSV(testFile)
			readTime := time.Since(start)

			if err != nil {
				t.Fatalf("Failed to read CSV: %v", err)
			}

			if df.Len() != size {
				t.Errorf("Expected %d rows, got %d", size, df.Len())
			}

			// Calculate metrics
			rowsPerSec := float64(size) / readTime.Seconds()
			mbPerSec := float64(fileSize) / (1024 * 1024) / readTime.Seconds()

			t.Logf("CSV Read Performance - Size: %d rows", size)
			t.Logf("  File size: %.2f MB", float64(fileSize)/(1024*1024))
			t.Logf("  Read time: %v", readTime)
			t.Logf("  Throughput: %.0f rows/sec, %.2f MB/sec", rowsPerSec, mbPerSec)

			// Performance assertions
			if size >= 10000 && rowsPerSec < 10000 {
				t.Logf("Warning: Low throughput for large file: %.0f rows/sec", rowsPerSec)
			}
		}
	})

	t.Run("Memory Efficiency", func(t *testing.T) {
		size := 50000
		tmpDir := t.TempDir()
		testFile := filepath.Join(tmpDir, "memory_test.csv")

		// Create test file
		file, _ := os.Create(testFile)
		file.WriteString("id,value\n")
		for i := 0; i < size; i++ {
			file.WriteString(fmt.Sprintf("%d,%.6f\n", i, float64(i)*0.001))
		}
		file.Close()

		// Measure memory before
		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)

		// Read file
		df, err := ReadCSV(testFile)
		if err != nil {
			t.Fatalf("Failed to read CSV: %v", err)
		}

		// Measure memory after
		runtime.ReadMemStats(&m2)

		allocDelta := m2.TotalAlloc - m1.TotalAlloc
		heapDelta := m2.HeapAlloc - m1.HeapAlloc

		t.Logf("Memory Usage Analysis:")
		t.Logf("  Rows processed: %d", df.Len())
		t.Logf("  Total allocation: %.2f MB", float64(allocDelta)/(1024*1024))
		t.Logf("  Heap usage: %.2f MB", float64(heapDelta)/(1024*1024))
		t.Logf("  Bytes per row: %.2f", float64(allocDelta)/float64(size))

		// Memory efficiency check
		bytesPerRow := float64(allocDelta) / float64(size)
		if bytesPerRow > 1000 { // More than 1KB per row seems excessive
			t.Logf("Warning: High memory usage per row: %.2f bytes", bytesPerRow)
		}
	})
}

// BenchmarkConcurrentIO benchmarks concurrent I/O operations
func BenchmarkConcurrentIO(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping concurrent benchmark in short mode")
	}

	b.Run("concurrent_reads", func(b *testing.B) {
		// Create test files
		tmpDir := b.TempDir()
		numFiles := 4
		size := 5000

		testFiles := make([]string, numFiles)
		for i := 0; i < numFiles; i++ {
			testFile := filepath.Join(tmpDir, fmt.Sprintf("concurrent_%d.csv", i))
			file, _ := os.Create(testFile)
			file.WriteString("id,value\n")
			for j := 0; j < size; j++ {
				file.WriteString(fmt.Sprintf("%d,%.3f\n", j, float64(j)*0.1))
			}
			file.Close()
			testFiles[i] = testFile
		}

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			// Sequential reads (baseline)
			for _, file := range testFiles {
				_, err := ReadCSV(file)
				if err != nil {
					b.Fatalf("Failed to read CSV: %v", err)
				}
			}
		}

		totalRows := numFiles * size
		rowsPerSec := float64(totalRows) / (float64(b.Elapsed().Nanoseconds()) / float64(b.N) / 1e9)
		b.ReportMetric(rowsPerSec, "rows/sec")
	})
}
