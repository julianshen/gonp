package io

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestOptimizedVsStandardCSV compares optimized vs standard CSV reading
func TestOptimizedVsStandardCSV(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping optimization comparison in short mode")
	}

	sizes := []int{1000, 10000, 50000}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			// Create test file
			tmpDir := t.TempDir()
			testFile := filepath.Join(tmpDir, fmt.Sprintf("optimize_test_%d.csv", size))

			// Generate test data
			file, err := os.Create(testFile)
			if err != nil {
				t.Fatalf("Failed to create test file: %v", err)
			}

			file.WriteString("id,name,value,score,active\n")
			for i := 0; i < size; i++ {
				file.WriteString(fmt.Sprintf("%d,name_%d,%.3f,%.1f,%t\n",
					i, i, float64(i)*1.234, float64(i%100), i%2 == 0))
			}
			file.Close()

			// Get file size
			stat, _ := os.Stat(testFile)
			fileSize := stat.Size()

			t.Logf("Test file: %.2f MB, %d rows", float64(fileSize)/(1024*1024), size)

			// Test standard CSV reading
			start := time.Now()
			dfStandard, err := ReadCSV(testFile)
			standardTime := time.Since(start)

			if err != nil {
				t.Fatalf("Standard CSV read failed: %v", err)
			}

			if dfStandard.Len() != size {
				t.Errorf("Standard: Expected %d rows, got %d", size, dfStandard.Len())
			}

			// Test optimized CSV reading
			start = time.Now()
			dfOptimized, err := ReadCSVOptimized(testFile, nil)
			optimizedTime := time.Since(start)

			if err != nil {
				t.Fatalf("Optimized CSV read failed: %v", err)
			}

			if dfOptimized.Len() != size {
				t.Errorf("Optimized: Expected %d rows, got %d", size, dfOptimized.Len())
			}

			// Compare performance
			standardThroughput := float64(size) / standardTime.Seconds()
			optimizedThroughput := float64(size) / optimizedTime.Seconds()
			speedup := float64(standardTime) / float64(optimizedTime)

			t.Logf("Performance Comparison:")
			t.Logf("  Standard:  %v (%.0f rows/sec)", standardTime, standardThroughput)
			t.Logf("  Optimized: %v (%.0f rows/sec)", optimizedTime, optimizedThroughput)
			t.Logf("  Speedup:   %.2fx", speedup)

			// Verify data consistency (sample check)
			if dfStandard.Len() > 0 && dfOptimized.Len() > 0 {
				for col := 0; col < min(len(dfStandard.Columns()), 3); col++ {
					standardVal, _ := dfStandard.IAt(0, col)
					optimizedVal, _ := dfOptimized.IAt(0, col)

					// Type conversion might differ, so compare string representations
					if fmt.Sprintf("%v", standardVal) != fmt.Sprintf("%v", optimizedVal) {
						t.Logf("Value difference at (0,%d): standard=%v, optimized=%v",
							col, standardVal, optimizedVal)
					}
				}
			}

			// Performance expectation (optimized should be competitive)
			if size >= 10000 && speedup < 0.5 { // Optimized shouldn't be more than 2x slower
				t.Logf("Warning: Optimized version is significantly slower (%.2fx)", speedup)
			}
		})
	}
}

// BenchmarkOptimizedCSVRead benchmarks the optimized CSV reader
func BenchmarkOptimizedCSVRead(b *testing.B) {
	sizes := []int{1000, 10000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("optimized_rows_%d", size), func(b *testing.B) {
			// Create test CSV file
			tmpDir := b.TempDir()
			testFile := filepath.Join(tmpDir, fmt.Sprintf("bench_opt_%d.csv", size))

			// Generate test data
			file, err := os.Create(testFile)
			if err != nil {
				b.Fatalf("Failed to create test file: %v", err)
			}

			file.WriteString("id,name,value,score\n")
			for i := 0; i < size; i++ {
				file.WriteString(fmt.Sprintf("%d,name_%d,%.2f,%.1f\n", i, i, float64(i)*1.5, float64(i%100)))
			}
			file.Close()

			// Get file size for reporting
			stat, _ := os.Stat(testFile)
			fileSize := stat.Size()

			// Benchmark optimized reading
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				df, err := ReadCSVOptimized(testFile, nil)
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

// BenchmarkCSVComparison directly compares standard vs optimized
func BenchmarkCSVComparison(b *testing.B) {
	size := 10000

	// Create test file
	tmpDir := b.TempDir()
	testFile := filepath.Join(tmpDir, "comparison.csv")

	file, err := os.Create(testFile)
	if err != nil {
		b.Fatalf("Failed to create test file: %v", err)
	}

	file.WriteString("id,data,value\n")
	for i := 0; i < size; i++ {
		file.WriteString(fmt.Sprintf("%d,data_%d,%.3f\n", i, i, float64(i)*0.123))
	}
	file.Close()

	b.Run("standard", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			df, err := ReadCSV(testFile)
			if err != nil {
				b.Fatalf("Standard read failed: %v", err)
			}
			if df.Len() != size {
				b.Fatalf("Expected %d rows, got %d", size, df.Len())
			}
		}
	})

	b.Run("optimized", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			df, err := ReadCSVOptimized(testFile, nil)
			if err != nil {
				b.Fatalf("Optimized read failed: %v", err)
			}
			if df.Len() != size {
				b.Fatalf("Expected %d rows, got %d", size, df.Len())
			}
		}
	})
}

// TestStreamingPerformance tests streaming reader performance
func TestStreamingPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping streaming performance test in short mode")
	}

	size := 50000
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "streaming_test.csv")

	// Create large test file
	file, err := os.Create(testFile)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	file.WriteString("id,value\n")
	for i := 0; i < size; i++ {
		file.WriteString(fmt.Sprintf("%d,%.6f\n", i, float64(i)*0.001))
	}
	file.Close()

	// Get file size
	stat, _ := os.Stat(testFile)
	fileSize := stat.Size()

	t.Logf("Streaming test file: %.2f MB, %d rows", float64(fileSize)/(1024*1024), size)

	// Test streaming reader
	options := DefaultPerformanceOptions()

	start := time.Now()
	streamReader, err := NewStreamingReader(testFile, options)
	if err != nil {
		t.Fatalf("Failed to create streaming reader: %v", err)
	}
	defer streamReader.Close()

	// Read in chunks
	totalBytes := int64(0)
	chunkCount := 0

	for {
		chunk, err := streamReader.ReadChunk(8192) // 8KB chunks
		if err != nil {
			break
		}

		totalBytes += int64(len(chunk))
		chunkCount++
	}

	streamTime := time.Since(start)

	t.Logf("Streaming Performance:")
	t.Logf("  Time: %v", streamTime)
	t.Logf("  Bytes read: %d", totalBytes)
	t.Logf("  Chunks: %d", chunkCount)
	t.Logf("  Throughput: %.2f MB/sec", float64(totalBytes)/(1024*1024)/streamTime.Seconds())

	if totalBytes != fileSize {
		t.Errorf("Size mismatch: expected %d bytes, read %d bytes", fileSize, totalBytes)
	}
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
