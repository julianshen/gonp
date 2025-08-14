package io

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/tealeg/xlsx/v3"

	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/series"
)

// ExcelOptions represents options for Excel reading and writing
type ExcelOptions struct {
	SheetName string // sheet name to read/write (default: first sheet)
	Header    bool   // whether the sheet has a header row (default: true)
	StartRow  int    // starting row index (0-based, default: 0)
	StartCol  int    // starting column index (0-based, default: 0)
}

// Default Excel options
var DefaultExcelOptions = &ExcelOptions{
	SheetName: "",
	Header:    true,
	StartRow:  0,
	StartCol:  0,
}

// ReadExcel reads an Excel file and returns a DataFrame from the first sheet
func ReadExcel(filename string) (*dataframe.DataFrame, error) {
	return ReadExcelWithOptions(filename, DefaultExcelOptions)
}

// ReadExcelSheet reads a specific sheet from an Excel file
func ReadExcelSheet(filename, sheetName string) (*dataframe.DataFrame, error) {
	opts := *DefaultExcelOptions
	opts.SheetName = sheetName
	return ReadExcelWithOptions(filename, &opts)
}

// ReadExcelWithOptions reads an Excel file with specified options
func ReadExcelWithOptions(filename string, opts *ExcelOptions) (*dataframe.DataFrame, error) {
	if opts == nil {
		opts = DefaultExcelOptions
	}

	// Open Excel file
	xlFile, err := xlsx.OpenFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open Excel file: %v", err)
	}

	// Select sheet
	var sheet *xlsx.Sheet
	if opts.SheetName != "" {
		var ok bool
		sheet, ok = xlFile.Sheet[opts.SheetName]
		if !ok {
			return nil, fmt.Errorf("sheet '%s' not found", opts.SheetName)
		}
	} else {
		// Use first sheet
		if len(xlFile.Sheets) == 0 {
			return nil, fmt.Errorf("Excel file has no sheets")
		}
		sheet = xlFile.Sheets[0]
	}

	// Read data from sheet using the correct xlsx API
	if sheet.MaxRow == 0 {
		return nil, fmt.Errorf("sheet has no data")
	}

	// Apply start row offset
	startRow := opts.StartRow
	if startRow >= sheet.MaxRow {
		return nil, fmt.Errorf("start row %d exceeds sheet length %d", startRow, sheet.MaxRow)
	}

	// Read all rows into string slices
	var rows [][]string
	for rowIdx := startRow; rowIdx < sheet.MaxRow; rowIdx++ {
		row, err := sheet.Row(rowIdx)
		if err != nil {
			continue // Skip invalid rows
		}

		var rowData []string
		// Read all cells in this row up to MaxCol
		for colIdx := 0; colIdx < sheet.MaxCol; colIdx++ {
			cell := row.GetCell(colIdx)
			if cell != nil {
				rowData = append(rowData, cell.Value)
			} else {
				rowData = append(rowData, "") // Empty cell
			}
		}
		rows = append(rows, rowData)
	}

	if len(rows) == 0 {
		return nil, fmt.Errorf("no data rows available")
	}

	// Determine headers and data start index
	var headers []string
	var dataStartIdx int

	if opts.Header {
		if len(rows) == 0 {
			return nil, fmt.Errorf("cannot read header from empty sheet")
		}
		headerRow := rows[0]

		// Apply start column offset to header
		if opts.StartCol >= len(headerRow) {
			return nil, fmt.Errorf("start column %d exceeds header length %d", opts.StartCol, len(headerRow))
		}
		headerRow = headerRow[opts.StartCol:]

		headers = make([]string, len(headerRow))
		for i, h := range headerRow {
			if h == "" {
				headers[i] = fmt.Sprintf("col_%d", i)
			} else {
				headers[i] = h
			}
		}
		dataStartIdx = 1
	} else {
		// Generate column names based on first row width
		if len(rows) == 0 {
			return nil, fmt.Errorf("cannot determine columns from empty sheet")
		}
		firstRow := rows[0]

		// Apply start column offset
		if opts.StartCol >= len(firstRow) {
			return nil, fmt.Errorf("start column %d exceeds row length %d", opts.StartCol, len(firstRow))
		}
		firstRow = firstRow[opts.StartCol:]

		headers = make([]string, len(firstRow))
		for i := range headers {
			headers[i] = fmt.Sprintf("col_%d", i)
		}
		dataStartIdx = 0
	}

	if len(rows) <= dataStartIdx {
		return nil, fmt.Errorf("no data rows available")
	}

	numCols := len(headers)
	numRows := len(rows) - dataStartIdx

	// Collect all values for each column
	columnData := make([][]string, numCols)
	for i := range columnData {
		columnData[i] = make([]string, numRows)
	}

	// Extract data
	for rowIdx := dataStartIdx; rowIdx < len(rows); rowIdx++ {
		row := rows[rowIdx]

		// Apply start column offset
		if opts.StartCol < len(row) {
			row = row[opts.StartCol:]
		} else {
			row = []string{} // Empty row if start column exceeds row length
		}

		for colIdx := 0; colIdx < numCols; colIdx++ {
			if colIdx < len(row) {
				columnData[colIdx][rowIdx-dataStartIdx] = row[colIdx]
			} else {
				columnData[colIdx][rowIdx-dataStartIdx] = "" // Empty cell
			}
		}
	}

	// Create series for each column using the same logic as CSV
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

// WriteExcel writes a DataFrame to an Excel file
func WriteExcel(df *dataframe.DataFrame, filename string) error {
	return WriteExcelWithOptions(df, filename, DefaultExcelOptions)
}

// WriteExcelWithOptions writes a DataFrame to an Excel file with specified options
func WriteExcelWithOptions(df *dataframe.DataFrame, filename string, opts *ExcelOptions) error {
	if opts == nil {
		opts = DefaultExcelOptions
	}

	// Create new Excel file
	xlFile := xlsx.NewFile()

	sheetName := opts.SheetName
	if sheetName == "" {
		sheetName = "Sheet1"
	}

	sheet, err := xlFile.AddSheet(sheetName)
	if err != nil {
		return fmt.Errorf("failed to create sheet: %v", err)
	}

	// Add empty rows for start row offset
	for i := 0; i < opts.StartRow; i++ {
		sheet.AddRow()
	}

	// Write header if requested
	if opts.Header {
		headerRow := sheet.AddRow()
		columns := df.Columns()

		// Add empty cells for start column offset
		for i := 0; i < opts.StartCol; i++ {
			headerRow.AddCell()
		}

		for _, colName := range columns {
			cell := headerRow.AddCell()
			cell.Value = colName
		}
	}

	// Write data rows
	seriesList := df.GetSeries()
	numRows := df.Len()

	for rowIdx := 0; rowIdx < numRows; rowIdx++ {
		row := sheet.AddRow()

		// Add empty cells for start column offset
		for i := 0; i < opts.StartCol; i++ {
			row.AddCell()
		}

		for _, s := range seriesList {
			cell := row.AddCell()
			value := s.At(rowIdx)
			cell.Value = formatExcelValue(value)
		}
	}

	// Save file
	err = xlFile.Save(filename)
	if err != nil {
		return fmt.Errorf("failed to save Excel file: %v", err)
	}

	return nil
}

// WriteExcelMultiSheet writes multiple DataFrames to different sheets in an Excel file
func WriteExcelMultiSheet(filename string, sheets map[string]*dataframe.DataFrame) error {
	// Create new Excel file
	xlFile := xlsx.NewFile()

	for sheetName, df := range sheets {
		sheet, err := xlFile.AddSheet(sheetName)
		if err != nil {
			return fmt.Errorf("failed to create sheet '%s': %v", sheetName, err)
		}

		// Write header
		headerRow := sheet.AddRow()
		columns := df.Columns()
		for _, colName := range columns {
			cell := headerRow.AddCell()
			cell.Value = colName
		}

		// Write data rows
		seriesList := df.GetSeries()
		numRows := df.Len()

		for rowIdx := 0; rowIdx < numRows; rowIdx++ {
			row := sheet.AddRow()
			for _, s := range seriesList {
				cell := row.AddCell()
				value := s.At(rowIdx)
				cell.Value = formatExcelValue(value)
			}
		}
	}

	// Save file
	err := xlFile.Save(filename)
	if err != nil {
		return fmt.Errorf("failed to save Excel file: %v", err)
	}

	return nil
}

// GetExcelSheetNames returns the names of all sheets in an Excel file
func GetExcelSheetNames(filename string) ([]string, error) {
	xlFile, err := xlsx.OpenFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open Excel file: %v", err)
	}

	sheetNames := make([]string, len(xlFile.Sheets))
	for i, sheet := range xlFile.Sheets {
		sheetNames[i] = sheet.Name
	}

	return sheetNames, nil
}

// formatExcelValue converts a value to its string representation for Excel
func formatExcelValue(value interface{}) string {
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

// Note: isAllIntegers and isAllNumbers functions are defined in csv.go
