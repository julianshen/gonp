package io

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"

	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
)

// OptimizedCSVOptions contains options for optimized CSV reading
type OptimizedCSVOptions struct {
	BufferSize   int     // Buffer size for file reading
	EstimateRows int     // Estimated number of rows for pre-allocation
	Workers      int     // Number of parallel workers (future use)
	MemoryFactor float64 // Memory estimation factor
}

// DefaultOptimizedCSVOptions returns default optimized CSV options
func DefaultOptimizedCSVOptions() *OptimizedCSVOptions {
	return &OptimizedCSVOptions{
		BufferSize:   128 * 1024, // 128KB buffer
		EstimateRows: 10000,      // Default row estimate
		Workers:      runtime.NumCPU(),
		MemoryFactor: 3.0, // 3x file size for memory estimation
	}
}

// ReadCSVOptimized reads a CSV file with performance optimizations
func ReadCSVOptimized(filename string, options *OptimizedCSVOptions) (*dataframe.DataFrame, error) {
	ctx := internal.StartProfiler("IO.ReadCSVOptimized")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if options == nil {
		options = DefaultOptimizedCSVOptions()
	}

	// Check file size and memory requirements
	fileSize, err := GetFileSize(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to get file size: %v", err)
	}

	estimatedMemory, err := EstimateMemoryUsage(filename, options.MemoryFactor)
	if err != nil {
		return nil, fmt.Errorf("failed to estimate memory usage: %v", err)
	}

	internal.DebugVerbose("Reading CSV file: %.2f MB, estimated memory: %.2f MB",
		float64(fileSize)/(1024*1024), float64(estimatedMemory)/(1024*1024))

	// Check available memory
	if !CheckAvailableMemory(estimatedMemory) {
		internal.DebugVerbose("Warning: Potentially insufficient memory for file")
	}

	// Open file with optimized buffer
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Use optimized buffered reader
	reader := bufio.NewReaderSize(file, options.BufferSize)

	// Read and parse with optimizations
	return readCSVOptimized(reader, options)
}

// readCSVOptimized performs optimized CSV parsing
func readCSVOptimized(reader *bufio.Reader, options *OptimizedCSVOptions) (*dataframe.DataFrame, error) {
	// Read header line
	headerBytes, err := reader.ReadSlice('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read header: %v", err)
	}

	// Parse header efficiently
	headerLine := strings.TrimSpace(string(headerBytes))
	columnNames := parseCSVLineOptimized(headerLine)
	numCols := len(columnNames)

	if numCols == 0 {
		return nil, fmt.Errorf("no columns found in header")
	}

	// Pre-allocate column data with estimates
	columnData := make([][]interface{}, numCols)
	for i := range columnData {
		columnData[i] = make([]interface{}, 0, options.EstimateRows)
	}

	// Create reusable buffers for performance
	fieldBuffer := make([]string, numCols)

	rowCount := 0

	// Read data lines with optimizations
	for {
		line, err := reader.ReadSlice('\n')
		if err != nil {
			if len(line) == 0 {
				break // EOF
			}
		}

		// Trim whitespace efficiently
		line = trimSpaceBytes(line)
		if len(line) == 0 {
			continue // Skip empty lines
		}

		// Parse line efficiently
		fields := parseCSVLineOptimizedBytes(line, fieldBuffer[:0])

		if len(fields) != numCols {
			continue // Skip malformed rows
		}

		// Convert and store values with type inference
		for i, field := range fields {
			value := inferAndConvertValue(field)
			columnData[i] = append(columnData[i], value)
		}

		rowCount++

		// Memory management: trigger GC periodically for very large files
		if rowCount%100000 == 0 {
			runtime.GC()
			internal.DebugVerbose("Processed %d rows", rowCount)
		}
	}

	internal.DebugVerbose("Parsed %d rows, %d columns", rowCount, numCols)

	// Create series with optimized type detection
	seriesList := make([]*series.Series, numCols)
	for i, colName := range columnNames {
		ser, err := createOptimizedSeries(columnData[i], colName)
		if err != nil {
			return nil, fmt.Errorf("failed to create series for column %s: %v", colName, err)
		}
		seriesList[i] = ser
	}

	// Create DataFrame
	df, err := dataframe.FromSeries(seriesList)
	if err != nil {
		return nil, fmt.Errorf("failed to create DataFrame: %v", err)
	}

	return df, nil
}

// parseCSVLineOptimized parses a CSV line with optimizations
func parseCSVLineOptimized(line string) []string {
	if len(line) == 0 {
		return nil
	}

	// Fast path for simple cases (no quotes, no embedded commas)
	if !strings.Contains(line, "\"") && !strings.Contains(line, "\\") {
		return strings.Split(line, ",")
	}

	// Slower path for complex CSV
	return parseCSVLineComplex(line)
}

// parseCSVLineOptimizedBytes parses CSV line from bytes with field reuse
func parseCSVLineOptimizedBytes(line []byte, fields []string) []string {
	if len(line) == 0 {
		return fields
	}

	// Convert to string for parsing (could be optimized further)
	lineStr := string(line)
	return parseCSVLineOptimized(lineStr)
}

// parseCSVLineComplex handles complex CSV parsing (quotes, escapes)
func parseCSVLineComplex(line string) []string {
	var fields []string
	var currentField strings.Builder
	inQuotes := false

	for i, char := range line {
		switch char {
		case '"':
			if inQuotes && i < len(line)-1 && rune(line[i+1]) == '"' {
				// Escaped quote
				currentField.WriteRune('"')
				// Skip next character (we'll handle it in next iteration)
			} else {
				inQuotes = !inQuotes
			}
		case ',':
			if !inQuotes {
				fields = append(fields, currentField.String())
				currentField.Reset()
			} else {
				currentField.WriteRune(char)
			}
		default:
			currentField.WriteRune(char)
		}
	}

	// Add final field
	fields = append(fields, currentField.String())

	return fields
}

// inferAndConvertValue infers and converts a string value efficiently
func inferAndConvertValue(value string) interface{} {
	if len(value) == 0 {
		return ""
	}

	// Fast boolean check
	if len(value) <= 5 {
		switch value {
		case "true", "True", "TRUE", "1":
			return true
		case "false", "False", "FALSE", "0":
			return false
		}
	}

	// Fast integer check
	if isIntegerString(value) {
		if val, err := strconv.ParseInt(value, 10, 64); err == nil {
			return val
		}
	}

	// Fast float check
	if isNumericString(value) {
		if val, err := strconv.ParseFloat(value, 64); err == nil {
			return val
		}
	}

	// Default to string
	return value
}

// isIntegerString checks if string represents an integer efficiently
func isIntegerString(s string) bool {
	if len(s) == 0 {
		return false
	}

	start := 0
	if s[0] == '-' || s[0] == '+' {
		if len(s) == 1 {
			return false
		}
		start = 1
	}

	for i := start; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// isNumericString checks if string represents a number efficiently
func isNumericString(s string) bool {
	if len(s) == 0 {
		return false
	}

	hasDecimal := false
	hasE := false
	start := 0

	if s[0] == '-' || s[0] == '+' {
		if len(s) == 1 {
			return false
		}
		start = 1
	}

	for i := start; i < len(s); i++ {
		char := s[i]
		if char >= '0' && char <= '9' {
			continue
		} else if char == '.' && !hasDecimal && !hasE {
			hasDecimal = true
		} else if (char == 'e' || char == 'E') && !hasE && i > start {
			hasE = true
			// Next character can be +/- or digit
			if i+1 < len(s) && (s[i+1] == '+' || s[i+1] == '-') {
				i++ // Skip the +/- after e/E
			}
		} else {
			return false
		}
	}
	return true
}

// createOptimizedSeries creates a series with type optimization
func createOptimizedSeries(data []interface{}, name string) (*series.Series, error) {
	if len(data) == 0 {
		return series.FromSlice([]string{}, nil, name)
	}

	// Analyze data types for optimization
	typeCount := make(map[string]int)
	for i := 0; i < len(data) && i < 100; i++ { // Sample first 100 values
		switch data[i].(type) {
		case int64:
			typeCount["int64"]++
		case float64:
			typeCount["float64"]++
		case bool:
			typeCount["bool"]++
		case string:
			typeCount["string"]++
		}
	}

	// Determine optimal type
	maxCount := 0
	dominantType := "string"
	for typ, count := range typeCount {
		if count > maxCount {
			maxCount = count
			dominantType = typ
		}
	}

	// Create typed slice based on dominant type
	switch dominantType {
	case "int64":
		intSlice := make([]int64, len(data))
		for i, val := range data {
			if intVal, ok := val.(int64); ok {
				intSlice[i] = intVal
			} else {
				// Fallback conversion
				intSlice[i] = convertToInt64Safe(val)
			}
		}
		return series.FromSlice(intSlice, nil, name)

	case "float64":
		floatSlice := make([]float64, len(data))
		for i, val := range data {
			if floatVal, ok := val.(float64); ok {
				floatSlice[i] = floatVal
			} else {
				floatSlice[i] = convertToFloat64Safe(val)
			}
		}
		return series.FromSlice(floatSlice, nil, name)

	case "bool":
		boolSlice := make([]bool, len(data))
		for i, val := range data {
			if boolVal, ok := val.(bool); ok {
				boolSlice[i] = boolVal
			} else {
				boolSlice[i] = convertToBoolSafe(val)
			}
		}
		return series.FromSlice(boolSlice, nil, name)

	default:
		// Use interface{} slice for mixed types
		return series.FromSlice(data, nil, name)
	}
}

// Safe conversion functions that don't return errors
func convertToInt64Safe(val interface{}) int64 {
	switch v := val.(type) {
	case int64:
		return v
	case int:
		return int64(v)
	case float64:
		return int64(v)
	case string:
		if result, err := strconv.ParseInt(v, 10, 64); err == nil {
			return result
		}
	case bool:
		if v {
			return 1
		}
		return 0
	}
	return 0
}

func convertToFloat64Safe(val interface{}) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case int64:
		return float64(v)
	case int:
		return float64(v)
	case string:
		if result, err := strconv.ParseFloat(v, 64); err == nil {
			return result
		}
	case bool:
		if v {
			return 1.0
		}
		return 0.0
	}
	return 0.0
}

func convertToBoolSafe(val interface{}) bool {
	switch v := val.(type) {
	case bool:
		return v
	case int64:
		return v != 0
	case float64:
		return v != 0.0
	case string:
		return v == "true" || v == "True" || v == "TRUE" || v == "1"
	}
	return false
}

// trimSpaceBytes trims whitespace from byte slice efficiently
func trimSpaceBytes(b []byte) []byte {
	// Trim leading space
	for len(b) > 0 && isSpace(b[0]) {
		b = b[1:]
	}

	// Trim trailing space
	for len(b) > 0 && isSpace(b[len(b)-1]) {
		b = b[:len(b)-1]
	}

	return b
}

// isSpace checks if byte is a space character
func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

// WriteCSVOptimized writes a DataFrame to CSV with performance optimizations
func WriteCSVOptimized(df *dataframe.DataFrame, filename string, options *OptimizedCSVOptions) error {
	ctx := internal.StartProfiler("IO.WriteCSVOptimized")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if options == nil {
		options = DefaultOptimizedCSVOptions()
	}

	if df == nil {
		return internal.NewValidationErrorWithMsg("WriteCSVOptimized", "DataFrame cannot be nil")
	}

	// Create file with optimized writer
	writer, err := NewStreamingWriter(filename, &PerformanceOptions{
		BufferSize: options.BufferSize,
	})
	if err != nil {
		return fmt.Errorf("failed to create streaming writer: %v", err)
	}
	defer writer.Close()

	// Write header
	columns := df.Columns()
	headerLine := strings.Join(columns, ",") + "\n"
	err = writer.WriteChunk([]byte(headerLine))
	if err != nil {
		return fmt.Errorf("failed to write header: %v", err)
	}

	// Write data rows
	numRows := df.Len()
	var line strings.Builder

	for row := 0; row < numRows; row++ {
		line.Reset()

		for col := range columns {
			if col > 0 {
				line.WriteByte(',')
			}

			val, err := df.IAt(row, col)
			if err != nil {
				return fmt.Errorf("failed to get value at (%d, %d): %v", row, col, err)
			}

			// Format value efficiently
			switch v := val.(type) {
			case string:
				// Escape if necessary
				if strings.Contains(v, ",") || strings.Contains(v, "\"") || strings.Contains(v, "\n") {
					line.WriteByte('"')
					line.WriteString(strings.ReplaceAll(v, "\"", "\"\""))
					line.WriteByte('"')
				} else {
					line.WriteString(v)
				}
			case nil:
				// Empty field
			default:
				line.WriteString(fmt.Sprintf("%v", v))
			}
		}

		line.WriteByte('\n')

		err = writer.WriteChunk([]byte(line.String()))
		if err != nil {
			return fmt.Errorf("failed to write row %d: %v", row, err)
		}

		// Periodic flushing for large files
		if row%10000 == 0 {
			err = writer.Flush()
			if err != nil {
				return fmt.Errorf("failed to flush at row %d: %v", row, err)
			}
		}
	}

	internal.DebugVerbose("Wrote %d rows to %s", numRows, filename)
	return nil
}
