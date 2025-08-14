package io

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
)

// CompressionType represents different compression algorithms for Parquet
type CompressionType string

const (
	CompressionUncompressed CompressionType = "UNCOMPRESSED"
	CompressionSnappy       CompressionType = "SNAPPY"
	CompressionGzip         CompressionType = "GZIP"
	CompressionLZ4          CompressionType = "LZ4"
	CompressionZstd         CompressionType = "ZSTD"
)

// ParquetWriteOptions contains options for writing Parquet files
type ParquetWriteOptions struct {
	Compression    CompressionType // Compression algorithm to use
	RowGroupSize   int             // Number of rows per row group
	PageSize       int             // Page size in bytes
	DictionarySize int             // Dictionary page size in bytes
	EnableStats    bool            // Whether to write column statistics
}

// DefaultParquetWriteOptions returns default options for writing Parquet files
func DefaultParquetWriteOptions() *ParquetWriteOptions {
	return &ParquetWriteOptions{
		Compression:    CompressionSnappy,
		RowGroupSize:   8192,
		PageSize:       1024 * 1024, // 1MB
		DictionarySize: 1024 * 1024, // 1MB
		EnableStats:    true,
	}
}

// ParquetMetadata contains metadata information about a Parquet file
type ParquetMetadata struct {
	NumRows       int64                  // Total number of rows
	NumColumns    int                    // Number of columns
	NumRowGroups  int                    // Number of row groups
	Columns       []*ParquetColumnSchema // Column schema information
	CreatedBy     string                 // Created by information
	FormatVersion int                    // Parquet format version
}

// ParquetColumnSchema contains schema information for a column
type ParquetColumnSchema struct {
	Name          string // Column name
	Type          string // Physical type (INT32, INT64, FLOAT, DOUBLE, BYTE_ARRAY, etc.)
	LogicalType   string // Logical type (STRING, TIMESTAMP, etc.)
	Repetition    string // REQUIRED, OPTIONAL, REPEATED
	MaxDefinition int    // Maximum definition level
	MaxRepetition int    // Maximum repetition level
	Compression   string // Compression used for this column
}

// ParquetFileInfo contains file-level information about a Parquet file
type ParquetFileInfo struct {
	Path         string    // File path
	Size         int64     // File size in bytes
	NumRowGroups int       // Number of row groups
	ModTime      time.Time // Last modification time
	Created      time.Time // Creation time (if available)
}

// ParquetReader provides streaming read access to Parquet files
type ParquetReader struct {
	path         string
	currentBatch int
	totalBatches int
	closed       bool
}

// ParquetWriter provides streaming write access to Parquet files
type ParquetWriter struct {
	path    string
	options *ParquetWriteOptions
	closed  bool
	batches []*dataframe.DataFrame // Store batches until close
}

// WriteParquet writes a DataFrame to a Parquet file using default options
func WriteParquet(path string, df *dataframe.DataFrame) error {
	return WriteParquetWithOptions(path, df, DefaultParquetWriteOptions())
}

// WriteParquetWithOptions writes a DataFrame to a Parquet file with specified options
func WriteParquetWithOptions(path string, df *dataframe.DataFrame, options *ParquetWriteOptions) error {
	ctx := internal.StartProfiler("IO.WriteParquet")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if df == nil {
		return internal.NewValidationErrorWithMsg("WriteParquet", "DataFrame cannot be nil")
	}

	if path == "" {
		return internal.NewValidationErrorWithMsg("WriteParquet", "file path cannot be empty")
	}

	if options == nil {
		options = DefaultParquetWriteOptions()
	}

	// Check if DataFrame has data
	numRows := df.Len()
	numCols := len(df.Columns())
	if numRows == 0 || numCols == 0 {
		return errors.New("cannot write empty DataFrame to Parquet")
	}

	// For now, this is a simplified implementation that converts to a basic format
	// In a production implementation, you would use a library like github.com/apache/arrow/go
	// or implement the full Parquet specification

	return writeParquetSimulated(path, df, options)
}

// writeParquetSimulated provides a simulated implementation for testing purposes
// In production, this would use the full Parquet specification
func writeParquetSimulated(path string, df *dataframe.DataFrame, options *ParquetWriteOptions) error {
	// Create directory if needed
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create directory: %v", err)
	}

	// Create a simple binary format that simulates Parquet structure
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Write magic bytes (simplified Parquet signature)
	if _, err := file.Write([]byte("PAR1")); err != nil {
		return fmt.Errorf("failed to write magic bytes: %v", err)
	}

	// Write metadata header
	numRows := df.Len()
	columns := df.Columns()
	metadata := &simulatedMetadata{
		NumRows:    int64(numRows),
		NumColumns: int64(len(columns)),
		Columns:    make([]*simulatedColumn, len(columns)),
	}

	// Write column metadata
	for i, colName := range columns {
		col, err := df.GetColumn(colName)
		if err != nil {
			return fmt.Errorf("column %s not found: %v", colName, err)
		}

		metadata.Columns[i] = &simulatedColumn{
			Name: colName,
			Type: inferColumnType(col),
			Size: int64(col.Len()),
		}
	}

	// Serialize metadata (simplified)
	metadataBytes, err := serializeSimulatedMetadata(metadata)
	if err != nil {
		return fmt.Errorf("failed to serialize metadata: %v", err)
	}

	// Write metadata length
	metadataLen := int64(len(metadataBytes))
	if err := writeBinaryInt64(file, metadataLen); err != nil {
		return fmt.Errorf("failed to write metadata length: %v", err)
	}

	// Write metadata
	if _, err := file.Write(metadataBytes); err != nil {
		return fmt.Errorf("failed to write metadata: %v", err)
	}

	// Write row data in row groups
	rowGroupSize := options.RowGroupSize
	if rowGroupSize <= 0 {
		rowGroupSize = numRows // Single row group
	}

	// Write total number of rows first (for reading)
	if err := writeBinaryInt64(file, int64(numRows)); err != nil {
		return fmt.Errorf("failed to write total row count: %v", err)
	}

	// Write number of row groups
	numRowGroups := (numRows + rowGroupSize - 1) / rowGroupSize // Ceiling division
	if err := writeBinaryInt64(file, int64(numRowGroups)); err != nil {
		return fmt.Errorf("failed to write row group count: %v", err)
	}

	for startRow := 0; startRow < numRows; startRow += rowGroupSize {
		endRow := startRow + rowGroupSize
		if endRow > numRows {
			endRow = numRows
		}

		if err := writeRowGroup(file, df, startRow, endRow, options); err != nil {
			return fmt.Errorf("failed to write row group: %v", err)
		}
	}

	// Write footer magic bytes
	if _, err := file.Write([]byte("PAR1")); err != nil {
		return fmt.Errorf("failed to write footer magic: %v", err)
	}

	internal.DebugVerbose("Parquet file written: %s (%d rows, %d columns)", path, numRows, len(columns))
	return nil
}

// ReadParquet reads a DataFrame from a Parquet file
func ReadParquet(path string) (*dataframe.DataFrame, error) {
	ctx := internal.StartProfiler("IO.ReadParquet")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if path == "" {
		return nil, internal.NewValidationErrorWithMsg("ReadParquet", "file path cannot be empty")
	}

	// Check if file exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, fmt.Errorf("file does not exist: %s", path)
	}

	return readParquetSimulated(path)
}

// readParquetSimulated provides a simulated implementation for testing
func readParquetSimulated(path string) (*dataframe.DataFrame, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Read and verify magic bytes
	magic := make([]byte, 4)
	if _, err := file.Read(magic); err != nil {
		return nil, fmt.Errorf("failed to read magic bytes: %v", err)
	}
	if string(magic) != "PAR1" {
		return nil, errors.New("invalid Parquet file: wrong magic bytes")
	}

	// Read metadata length
	metadataLen, err := readBinaryInt64(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata length: %v", err)
	}

	// Read metadata
	metadataBytes := make([]byte, metadataLen)
	if _, err := file.Read(metadataBytes); err != nil {
		return nil, fmt.Errorf("failed to read metadata: %v", err)
	}

	metadata, err := deserializeSimulatedMetadata(metadataBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize metadata: %v", err)
	}

	// Read total number of rows
	totalRows, err := readBinaryInt64(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read total row count: %v", err)
	}

	// Read number of row groups
	numRowGroups, err := readBinaryInt64(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read row group count: %v", err)
	}

	// Read row groups and reconstruct DataFrame
	data := make(map[string]*series.Series)

	for _, colMeta := range metadata.Columns {
		// Initialize series based on column type
		var ser *series.Series
		var err error

		switch colMeta.Type {
		case "int":
			ser, err = series.FromSlice(make([]int, totalRows), nil, colMeta.Name)
		case "float64":
			ser, err = series.FromSlice(make([]float64, totalRows), nil, colMeta.Name)
		case "string":
			ser, err = series.FromSlice(make([]string, totalRows), nil, colMeta.Name)
		case "bool":
			ser, err = series.FromSlice(make([]bool, totalRows), nil, colMeta.Name)
		default:
			return nil, fmt.Errorf("unsupported column type: %s", colMeta.Type)
		}

		if err != nil {
			return nil, fmt.Errorf("failed to create series for column %s: %v", colMeta.Name, err)
		}

		data[colMeta.Name] = ser
	}

	// Read row data (simplified format)
	if err := readRowGroups(file, data, int(totalRows), int(numRowGroups), metadata); err != nil {
		return nil, fmt.Errorf("failed to read row data: %v", err)
	}

	// Convert map to series list
	seriesList := make([]*series.Series, 0, len(data))
	for _, ser := range data {
		seriesList = append(seriesList, ser)
	}

	df, err := dataframe.FromSeries(seriesList)
	if err != nil {
		return nil, fmt.Errorf("failed to create DataFrame: %v", err)
	}

	internal.DebugVerbose("Parquet file read: %s (%d rows, %d columns)", path, metadata.NumRows, metadata.NumColumns)
	return df, nil
}

// ReadParquetMetadata reads only the metadata from a Parquet file
func ReadParquetMetadata(path string) (*ParquetMetadata, error) {
	if path == "" {
		return nil, internal.NewValidationErrorWithMsg("ReadParquetMetadata", "file path cannot be empty")
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, fmt.Errorf("file does not exist: %s", path)
	}

	// This is a simplified implementation
	// Read the simulated metadata from our format
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Skip magic bytes
	if _, err := file.Seek(4, 0); err != nil {
		return nil, fmt.Errorf("failed to seek: %v", err)
	}

	// Read metadata length
	metadataLen, err := readBinaryInt64(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata length: %v", err)
	}

	// Read metadata
	metadataBytes := make([]byte, metadataLen)
	if _, err := file.Read(metadataBytes); err != nil {
		return nil, fmt.Errorf("failed to read metadata: %v", err)
	}

	simMeta, err := deserializeSimulatedMetadata(metadataBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize metadata: %v", err)
	}

	// Convert to public metadata format
	metadata := &ParquetMetadata{
		NumRows:       simMeta.NumRows,
		NumColumns:    int(simMeta.NumColumns),
		NumRowGroups:  1, // Simplified
		Columns:       make([]*ParquetColumnSchema, len(simMeta.Columns)),
		CreatedBy:     "gonp",
		FormatVersion: 1,
	}

	for i, col := range simMeta.Columns {
		metadata.Columns[i] = &ParquetColumnSchema{
			Name:          col.Name,
			Type:          strings.ToUpper(col.Type),
			LogicalType:   inferLogicalType(col.Type),
			Repetition:    "REQUIRED",
			MaxDefinition: 0,
			MaxRepetition: 0,
			Compression:   "SNAPPY",
		}
	}

	return metadata, nil
}

// GetParquetFileInfo returns file information for a Parquet file
func GetParquetFileInfo(path string) (*ParquetFileInfo, error) {
	if path == "" {
		return nil, internal.NewValidationErrorWithMsg("GetParquetFileInfo", "file path cannot be empty")
	}

	stat, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %v", err)
	}

	// Read metadata to get additional info
	metadata, err := ReadParquetMetadata(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata: %v", err)
	}

	info := &ParquetFileInfo{
		Path:         path,
		Size:         stat.Size(),
		NumRowGroups: metadata.NumRowGroups,
		ModTime:      stat.ModTime(),
		Created:      stat.ModTime(), // Use ModTime as Created (simplified)
	}

	return info, nil
}

// WriteArrayAsParquet writes a single array as a Parquet file with the specified column name
func WriteArrayAsParquet(path string, arr *array.Array, columnName string) error {
	if arr == nil {
		return internal.NewValidationErrorWithMsg("WriteArrayAsParquet", "array cannot be nil")
	}

	if columnName == "" {
		columnName = "values"
	}

	// Convert array to series
	var ser *series.Series
	flatArr := arr.Flatten()

	// Detect type and create appropriate series
	if flatArr.Size() > 0 {
		firstVal := flatArr.At(0)
		switch firstVal.(type) {
		case int, int32, int64:
			values := make([]int, flatArr.Size())
			for i := 0; i < flatArr.Size(); i++ {
				values[i] = convertToInt(flatArr.At(i))
			}
			var err error
			ser, err = series.FromSlice(values, nil, columnName)
			if err != nil {
				return fmt.Errorf("failed to create int series: %v", err)
			}
		case float32, float64:
			values := make([]float64, flatArr.Size())
			for i := 0; i < flatArr.Size(); i++ {
				values[i] = convertToFloat64(flatArr.At(i))
			}
			var err error
			ser, err = series.FromSlice(values, nil, columnName)
			if err != nil {
				return fmt.Errorf("failed to create float series: %v", err)
			}
		case bool:
			values := make([]bool, flatArr.Size())
			for i := 0; i < flatArr.Size(); i++ {
				values[i] = convertToBool(flatArr.At(i))
			}
			var err error
			ser, err = series.FromSlice(values, nil, columnName)
			if err != nil {
				return fmt.Errorf("failed to create bool series: %v", err)
			}
		case string:
			values := make([]string, flatArr.Size())
			for i := 0; i < flatArr.Size(); i++ {
				values[i] = convertToString(flatArr.At(i))
			}
			var err error
			ser, err = series.FromSlice(values, nil, columnName)
			if err != nil {
				return fmt.Errorf("failed to create string series: %v", err)
			}
		default:
			return fmt.Errorf("unsupported array type for Parquet: %T", firstVal)
		}
	} else {
		return errors.New("cannot write empty array to Parquet")
	}

	// Create DataFrame with single column
	df, err := dataframe.FromSeries([]*series.Series{ser})
	if err != nil {
		return fmt.Errorf("failed to create DataFrame: %v", err)
	}

	return WriteParquet(path, df)
}

// ReadParquetAsArray reads a single column from a Parquet file as an array
func ReadParquetAsArray(path string, columnName string) (*array.Array, error) {
	df, err := ReadParquet(path)
	if err != nil {
		return nil, err
	}

	column, err := df.GetColumn(columnName)
	if err != nil {
		return nil, fmt.Errorf("column %s not found in Parquet file: %v", columnName, err)
	}

	// Convert series to array
	values := make([]interface{}, column.Len())
	for i := 0; i < column.Len(); i++ {
		values[i] = column.At(i)
	}

	arr, err := array.FromSlice(values)
	if err != nil {
		return nil, fmt.Errorf("failed to create array from column: %v", err)
	}

	return arr, nil
}

// NewParquetReader creates a new streaming reader for Parquet files
func NewParquetReader(path string) (*ParquetReader, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, fmt.Errorf("file does not exist: %s", path)
	}

	// Get metadata to determine batch count
	metadata, err := ReadParquetMetadata(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata: %v", err)
	}

	reader := &ParquetReader{
		path:         path,
		currentBatch: 0,
		totalBatches: metadata.NumRowGroups,
		closed:       false,
	}

	return reader, nil
}

// HasNext returns true if there are more batches to read
func (r *ParquetReader) HasNext() bool {
	return !r.closed && r.currentBatch < r.totalBatches
}

// ReadBatch reads the next batch of data
func (r *ParquetReader) ReadBatch(batchSize int) (*dataframe.DataFrame, error) {
	if r.closed {
		return nil, errors.New("reader is closed")
	}

	if !r.HasNext() {
		return nil, nil
	}

	// For this simplified implementation, just read the entire file
	// In production, this would read specific row groups
	df, err := ReadParquet(r.path)
	if err != nil {
		return nil, err
	}

	r.currentBatch = r.totalBatches // Mark as complete
	return df, nil
}

// Close closes the reader
func (r *ParquetReader) Close() error {
	r.closed = true
	return nil
}

// NewParquetWriter creates a new streaming writer for Parquet files
func NewParquetWriter(path string, options *ParquetWriteOptions) (*ParquetWriter, error) {
	if options == nil {
		options = DefaultParquetWriteOptions()
	}

	writer := &ParquetWriter{
		path:    path,
		options: options,
		closed:  false,
		batches: make([]*dataframe.DataFrame, 0),
	}

	return writer, nil
}

// WriteBatch writes a batch of data
func (w *ParquetWriter) WriteBatch(df *dataframe.DataFrame) error {
	if w.closed {
		return errors.New("writer is closed")
	}

	if df == nil {
		return errors.New("DataFrame cannot be nil")
	}

	// Store batch for later writing
	w.batches = append(w.batches, df)
	return nil
}

// Close finalizes the file and closes the writer
func (w *ParquetWriter) Close() error {
	if w.closed {
		return nil
	}

	// Combine all batches and write as single file
	if len(w.batches) == 0 {
		return errors.New("no data to write")
	}

	// For simplicity, concatenate all batches
	// In production, this would write row groups incrementally
	combinedData := make(map[string]*series.Series)

	// Get column names from first batch
	firstBatch := w.batches[0]
	columns := firstBatch.Columns()

	// Initialize combined series
	for _, colName := range columns {
		col, err := firstBatch.GetColumn(colName)
		if err != nil {
			continue
		}

		// Calculate total size
		totalSize := 0
		for _, batch := range w.batches {
			totalSize += batch.Len()
		}

		// Create combined series based on type
		if totalSize > 0 {
			firstVal := col.At(0)
			switch firstVal.(type) {
			case int:
				values := make([]int, 0, totalSize)
				for _, batch := range w.batches {
					batchCol, err := batch.GetColumn(colName)
					if err != nil {
						continue
					}
					for i := 0; i < batchCol.Len(); i++ {
						values = append(values, convertToInt(batchCol.At(i)))
					}
				}
				ser, err := series.FromSlice(values, nil, colName)
				if err != nil {
					return fmt.Errorf("failed to create combined int series: %v", err)
				}
				combinedData[colName] = ser
			case float64:
				values := make([]float64, 0, totalSize)
				for _, batch := range w.batches {
					batchCol, err := batch.GetColumn(colName)
					if err != nil {
						continue
					}
					for i := 0; i < batchCol.Len(); i++ {
						values = append(values, convertToFloat64(batchCol.At(i)))
					}
				}
				ser, err := series.FromSlice(values, nil, colName)
				if err != nil {
					return fmt.Errorf("failed to create combined float series: %v", err)
				}
				combinedData[colName] = ser
			case string:
				values := make([]string, 0, totalSize)
				for _, batch := range w.batches {
					batchCol, err := batch.GetColumn(colName)
					if err != nil {
						continue
					}
					for i := 0; i < batchCol.Len(); i++ {
						values = append(values, convertToString(batchCol.At(i)))
					}
				}
				ser, err := series.FromSlice(values, nil, colName)
				if err != nil {
					return fmt.Errorf("failed to create combined string series: %v", err)
				}
				combinedData[colName] = ser
			}
		}
	}

	// Create combined DataFrame
	seriesList := make([]*series.Series, 0, len(combinedData))
	for _, ser := range combinedData {
		seriesList = append(seriesList, ser)
	}

	combinedDF, err := dataframe.FromSeries(seriesList)
	if err != nil {
		return fmt.Errorf("failed to create combined DataFrame: %v", err)
	}

	// Write to file
	err = WriteParquetWithOptions(w.path, combinedDF, w.options)
	if err != nil {
		return fmt.Errorf("failed to write Parquet file: %v", err)
	}

	w.closed = true
	return nil
}

// Helper functions for the simulated implementation
// In production, these would be replaced with proper Parquet encoding/decoding

type simulatedMetadata struct {
	NumRows    int64              `json:"num_rows"`
	NumColumns int64              `json:"num_columns"`
	Columns    []*simulatedColumn `json:"columns"`
}

type simulatedColumn struct {
	Name string `json:"name"`
	Type string `json:"type"`
	Size int64  `json:"size"`
}

func inferColumnType(series *series.Series) string {
	if series.Len() == 0 {
		return "string"
	}

	firstVal := series.At(0)
	switch firstVal.(type) {
	case int, int32, int64:
		return "int"
	case float32, float64:
		return "float64"
	case bool:
		return "bool"
	case string:
		return "string"
	default:
		return "string"
	}
}

func inferLogicalType(physicalType string) string {
	switch physicalType {
	case "int":
		return "INTEGER"
	case "float64":
		return "DOUBLE"
	case "bool":
		return "BOOLEAN"
	case "string":
		return "STRING"
	default:
		return "STRING"
	}
}

func inferSeriesType(series *series.Series) string {
	if series.Len() == 0 {
		return "string"
	}

	firstVal := series.At(0)
	switch firstVal.(type) {
	case int, int32, int64:
		return "int"
	case float32, float64:
		return "float64"
	case bool:
		return "bool"
	case string:
		return "string"
	default:
		return "string"
	}
}

func serializeSimulatedMetadata(metadata *simulatedMetadata) ([]byte, error) {
	return json.Marshal(metadata)
}

func deserializeSimulatedMetadata(data []byte) (*simulatedMetadata, error) {
	var metadata simulatedMetadata
	err := json.Unmarshal(data, &metadata)
	return &metadata, err
}

func writeBinaryInt64(w io.Writer, value int64) error {
	return binary.Write(w, binary.LittleEndian, value)
}

func readBinaryInt64(r io.Reader) (int64, error) {
	var value int64
	err := binary.Read(r, binary.LittleEndian, &value)
	return value, err
}

func writeRowGroup(file *os.File, df *dataframe.DataFrame, startRow, endRow int, options *ParquetWriteOptions) error {
	// Simplified row group writing
	// Write row count for this group
	rowCount := int64(endRow - startRow)
	if err := writeBinaryInt64(file, rowCount); err != nil {
		return err
	}

	// Write data for each column
	for _, colName := range df.Columns() {
		col, err := df.GetColumn(colName)
		if err != nil {
			continue
		}

		// Write column data
		for i := startRow; i < endRow; i++ {
			value := col.At(i)

			switch v := value.(type) {
			case int:
				if err := writeBinaryInt64(file, int64(v)); err != nil {
					return err
				}
			case float64:
				if err := binary.Write(file, binary.LittleEndian, v); err != nil {
					return err
				}
			case string:
				// Write string length then string data
				if err := writeBinaryInt64(file, int64(len(v))); err != nil {
					return err
				}
				if _, err := file.Write([]byte(v)); err != nil {
					return err
				}
			case bool:
				var b byte
				if v {
					b = 1
				}
				if err := binary.Write(file, binary.LittleEndian, b); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func readRowGroups(file *os.File, data map[string]*series.Series, totalRows int, numRowGroups int, metadata *simulatedMetadata) error {
	// Read all row groups
	currentRow := 0

	for groupIdx := 0; groupIdx < numRowGroups; groupIdx++ {
		// Read row count for this group
		rowCount, err := readBinaryInt64(file)
		if err != nil {
			return err
		}

		// Read data for this row group, using metadata column order
		for _, colMeta := range metadata.Columns {
			series := data[colMeta.Name]
			if series == nil {
				continue
			}

			for i := 0; i < int(rowCount); i++ {
				if currentRow+i >= totalRows {
					return fmt.Errorf("row index out of bounds: %d >= %d", currentRow+i, totalRows)
				}

				switch colMeta.Type {
				case "int":
					value, err := readBinaryInt64(file)
					if err != nil {
						return err
					}
					series.Set(currentRow+i, int(value))
				case "float64":
					var value float64
					if err := binary.Read(file, binary.LittleEndian, &value); err != nil {
						return err
					}
					series.Set(currentRow+i, value)
				case "string":
					// Read string length
					length, err := readBinaryInt64(file)
					if err != nil {
						return err
					}
					// Read string data
					strBytes := make([]byte, length)
					if _, err := file.Read(strBytes); err != nil {
						return err
					}
					series.Set(currentRow+i, string(strBytes))
				case "bool":
					var b byte
					if err := binary.Read(file, binary.LittleEndian, &b); err != nil {
						return err
					}
					series.Set(currentRow+i, b == 1)
				}
			}
		}

		currentRow += int(rowCount)
	}

	if currentRow != totalRows {
		return fmt.Errorf("total row count mismatch: expected %d, got %d", totalRows, currentRow)
	}

	return nil
}

// Conversion helper functions
func convertToBool(value interface{}) bool {
	switch v := value.(type) {
	case bool:
		return v
	case int:
		return v != 0
	case float64:
		return v != 0.0
	case string:
		return v == "true" || v == "True" || v == "TRUE"
	default:
		return false
	}
}

func convertToString(value interface{}) string {
	return fmt.Sprintf("%v", value)
}

func convertToInt(value interface{}) int {
	switch v := value.(type) {
	case int:
		return v
	case int32:
		return int(v)
	case int64:
		return int(v)
	case float32:
		return int(v)
	case float64:
		return int(v)
	default:
		return 0
	}
}

func convertToFloat64(value interface{}) float64 {
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int32:
		return float64(v)
	case int64:
		return float64(v)
	default:
		return 0.0
	}
}
