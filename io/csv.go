package io

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
)

// CSVOptions represents options for CSV reading and writing
type CSVOptions struct {
	Separator rune // field separator (default: ',')
	Header    bool // whether the file has a header row (default: true)
}

// Default CSV options
var DefaultCSVOptions = &CSVOptions{
	Separator: ',',
	Header:    true,
}

// ReadCSV reads a CSV file and returns a DataFrame
func ReadCSV(filename string) (*dataframe.DataFrame, error) {
	if err := internal.QuickValidateNotNil(filename, "ReadCSV", "filename"); err != nil {
		return nil, err
	}
	if filename == "" {
		return nil, internal.NewValidationErrorWithMsg("ReadCSV", "filename cannot be empty")
	}
	return ReadCSVWithOptions(filename, DefaultCSVOptions)
}

// ReadCSVWithOptions reads a CSV file with specified options
func ReadCSVWithOptions(filename string, opts *CSVOptions) (*dataframe.DataFrame, error) {
	if filename == "" {
		return nil, internal.NewValidationErrorWithMsg("ReadCSVWithOptions", "filename cannot be empty")
	}
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	return ReadCSVFromReader(file, opts)
}

// ReadCSVFromReader reads CSV data from an io.Reader
func ReadCSVFromReader(reader io.Reader, opts *CSVOptions) (*dataframe.DataFrame, error) {
	if err := internal.QuickValidateNotNil(reader, "ReadCSVFromReader", "reader"); err != nil {
		return nil, err
	}
	if opts == nil {
		opts = DefaultCSVOptions
	}

	csvReader := csv.NewReader(reader)
	csvReader.Comma = opts.Separator

	// Read all records
	records, err := csvReader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV data: %v", err)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("CSV file is empty")
	}

	var headers []string
	var dataStartIdx int

	if opts.Header {
		headers = records[0]
		dataStartIdx = 1
	} else {
		// Generate column names
		headers = make([]string, len(records[0]))
		for i := range headers {
			headers[i] = fmt.Sprintf("col_%d", i)
		}
		dataStartIdx = 0
	}

	if len(records) <= dataStartIdx {
		return nil, fmt.Errorf("CSV file has no data rows")
	}

	numCols := len(headers)
	numRows := len(records) - dataStartIdx

	// Collect all values for each column to determine types
	columnData := make([][]string, numCols)
	for i := range columnData {
		columnData[i] = make([]string, numRows)
	}

	// Extract data
	for rowIdx := dataStartIdx; rowIdx < len(records); rowIdx++ {
		record := records[rowIdx]
		if len(record) != numCols {
			return nil, fmt.Errorf("row %d has %d columns, expected %d", rowIdx, len(record), numCols)
		}

		for colIdx, value := range record {
			columnData[colIdx][rowIdx-dataStartIdx] = value
		}
	}

	// Create series for each column
	seriesList := make([]*series.Series, numCols)
	for i, colName := range headers {
		values := columnData[i]

		// Determine the type for this column and create appropriate typed slice
		if isAllIntegers(values) {
			// Create int64 slice
			intValues := make([]int64, len(values))
			for j, strVal := range values {
				if val, err := strconv.ParseInt(strings.TrimSpace(strVal), 10, 64); err == nil {
					intValues[j] = val
				} else {
					intValues[j] = 0 // default for empty/invalid values
				}
			}
			s, err := series.FromSlice(intValues, nil, colName)
			if err != nil {
				return nil, fmt.Errorf("failed to create series for column %s: %v", colName, err)
			}
			seriesList[i] = s
		} else if isAllNumbers(values) {
			// Create float64 slice
			floatValues := make([]float64, len(values))
			for j, strVal := range values {
				if val, err := strconv.ParseFloat(strings.TrimSpace(strVal), 64); err == nil {
					floatValues[j] = val
				} else {
					floatValues[j] = 0.0 // default for empty/invalid values
				}
			}
			s, err := series.FromSlice(floatValues, nil, colName)
			if err != nil {
				return nil, fmt.Errorf("failed to create series for column %s: %v", colName, err)
			}
			seriesList[i] = s
		} else {
			// For mixed types including strings, create interface{} slice
			interfaceValues := make([]interface{}, len(values))
			for j, strVal := range values {
				strVal = strings.TrimSpace(strVal)
				if strVal == "" {
					interfaceValues[j] = nil
				} else {
					interfaceValues[j] = strVal
				}
			}
			s, err := series.FromSlice(interfaceValues, nil, colName)
			if err != nil {
				return nil, fmt.Errorf("failed to create series for column %s: %v", colName, err)
			}
			seriesList[i] = s
		}
	}

	return dataframe.FromSeries(seriesList)
}

// WriteCSV writes a DataFrame to a CSV file
func WriteCSV(df *dataframe.DataFrame, filename string) error {
	if err := internal.QuickValidateNotNil(df, "WriteCSV", "dataframe"); err != nil {
		return err
	}
	if filename == "" {
		return internal.NewValidationErrorWithMsg("WriteCSV", "filename cannot be empty")
	}
	return WriteCSVWithOptions(df, filename, DefaultCSVOptions)
}

// WriteCSVWithOptions writes a DataFrame to a CSV file with specified options
func WriteCSVWithOptions(df *dataframe.DataFrame, filename string, opts *CSVOptions) error {
	if err := internal.QuickValidateNotNil(df, "WriteCSVWithOptions", "dataframe"); err != nil {
		return err
	}
	if filename == "" {
		return internal.NewValidationErrorWithMsg("WriteCSVWithOptions", "filename cannot be empty")
	}
	if opts == nil {
		opts = DefaultCSVOptions
	}

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	writer.Comma = opts.Separator
	defer writer.Flush()

	// Write header if requested
	if opts.Header {
		err = writer.Write(df.Columns())
		if err != nil {
			return fmt.Errorf("failed to write header: %v", err)
		}
	}

	// Write data rows
	seriesList := df.GetSeries()
	numRows := df.Len()

	for rowIdx := 0; rowIdx < numRows; rowIdx++ {
		record := make([]string, len(seriesList))
		for colIdx, s := range seriesList {
			value := s.At(rowIdx)
			record[colIdx] = formatValue(value)
		}

		err = writer.Write(record)
		if err != nil {
			return fmt.Errorf("failed to write row %d: %v", rowIdx, err)
		}
	}

	return nil
}

// isAllIntegers checks if all values in a string slice can be parsed as integers
func isAllIntegers(values []string) bool {
	for _, val := range values {
		val = strings.TrimSpace(val)
		if val == "" {
			continue
		}
		if _, err := strconv.ParseInt(val, 10, 64); err != nil {
			return false
		}
	}
	return true
}

// isAllNumbers checks if all values in a string slice can be parsed as numbers
func isAllNumbers(values []string) bool {
	for _, val := range values {
		val = strings.TrimSpace(val)
		if val == "" {
			continue
		}
		if _, err := strconv.ParseFloat(val, 64); err != nil {
			return false
		}
	}
	return true
}

// formatValue converts a value to its string representation for CSV
func formatValue(value interface{}) string {
	if value == nil {
		return ""
	}

	switch v := value.(type) {
	case string:
		return v
	case int64:
		return strconv.FormatInt(v, 10)
	case float64:
		return strconv.FormatFloat(v, 'g', -1, 64)
	case bool:
		return strconv.FormatBool(v)
	default:
		return fmt.Sprintf("%v", v)
	}
}
