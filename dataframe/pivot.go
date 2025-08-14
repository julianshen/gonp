package dataframe

import (
	"fmt"
	"sort"

	"github.com/julianshen/gonp/series"
)

// PivotTable creates a pivot table from the DataFrame
// rowColumn: column to use for row index
// colColumn: column to use for column headers
// valueColumn: column to aggregate for values
// aggFunc: aggregation function ("sum", "mean", "count", "min", "max")
func (df *DataFrame) PivotTable(rowColumn, colColumn, valueColumn, aggFunc string) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("DataFrame cannot be nil")
	}

	if df.Len() == 0 {
		return nil, fmt.Errorf("cannot pivot empty DataFrame")
	}

	// Validate columns exist
	if _, err := df.GetColumn(rowColumn); err != nil {
		return nil, fmt.Errorf("row column '%s' not found: %v", rowColumn, err)
	}
	if _, err := df.GetColumn(colColumn); err != nil {
		return nil, fmt.Errorf("column column '%s' not found: %v", colColumn, err)
	}
	if _, err := df.GetColumn(valueColumn); err != nil {
		return nil, fmt.Errorf("value column '%s' not found: %v", valueColumn, err)
	}

	// Validate aggregation function
	validAggFuncs := map[string]bool{"sum": true, "mean": true, "count": true, "min": true, "max": true}
	if !validAggFuncs[aggFunc] {
		return nil, fmt.Errorf("invalid aggregation function '%s'. Valid options: sum, mean, count, min, max", aggFunc)
	}

	// Get unique values for rows and columns
	rowValues := getUniqueValues(df, rowColumn)
	colValues := getUniqueValues(df, colColumn)

	// Create pivot table structure
	resultColumns := make([]string, 1+len(colValues))
	resultColumns[0] = rowColumn
	copy(resultColumns[1:], colValues)

	// Initialize result data
	resultData := make([][]interface{}, len(resultColumns))
	for i := range resultData {
		resultData[i] = make([]interface{}, len(rowValues))
	}

	// Fill row values
	for i, rowVal := range rowValues {
		resultData[0][i] = rowVal
	}

	// Build pivot table
	for i, rowVal := range rowValues {
		for j, colVal := range colValues {
			// Find all rows matching this combination
			matchingRows := findMatchingRows(df, rowColumn, colColumn, rowVal, colVal)

			if len(matchingRows) == 0 {
				resultData[j+1][i] = nil
			} else {
				// Extract values and aggregate
				values := extractValues(df, valueColumn, matchingRows)
				aggValue, err := aggregateValues(values, aggFunc)
				if err != nil {
					return nil, fmt.Errorf("aggregation failed for %v,%v: %v", rowVal, colVal, err)
				}
				resultData[j+1][i] = aggValue
			}
		}
	}

	// Create result series
	resultSeries := make([]*series.Series, len(resultColumns))
	for i, colName := range resultColumns {
		series, err := series.FromSlice(resultData[i], nil, colName)
		if err != nil {
			return nil, fmt.Errorf("failed to create series for column '%s': %v", colName, err)
		}
		resultSeries[i] = series
	}

	return FromSeries(resultSeries)
}

// Stack converts wide format to long format
// columns: columns to stack
// varName: name for the variable column (column names)
// valueName: name for the value column (column values)
func (df *DataFrame) Stack(columns []string, varName, valueName string) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("DataFrame cannot be nil")
	}

	if len(columns) == 0 {
		return nil, fmt.Errorf("at least one column must be specified for stacking")
	}

	// Validate columns exist
	for _, col := range columns {
		if _, err := df.GetColumn(col); err != nil {
			return nil, fmt.Errorf("column '%s' not found: %v", col, err)
		}
	}

	// Get non-stacked columns (identity columns)
	allColumns := df.Columns()
	stackSet := make(map[string]bool)
	for _, col := range columns {
		stackSet[col] = true
	}

	var idColumns []string
	for _, col := range allColumns {
		if !stackSet[col] {
			idColumns = append(idColumns, col)
		}
	}

	// Calculate result size
	numRows := df.Len() * len(columns)

	// Create result data
	resultColumns := append(idColumns, varName, valueName)
	resultData := make([][]interface{}, len(resultColumns))
	for i := range resultData {
		resultData[i] = make([]interface{}, numRows)
	}

	// Fill result data
	resultIdx := 0
	for rowIdx := 0; rowIdx < df.Len(); rowIdx++ {
		for _, stackCol := range columns {
			// Copy identity columns
			for i, idCol := range idColumns {
				idSeries, _ := df.GetColumn(idCol)
				resultData[i][resultIdx] = idSeries.At(rowIdx)
			}

			// Set variable name
			resultData[len(idColumns)][resultIdx] = stackCol

			// Set value
			stackSeries, _ := df.GetColumn(stackCol)
			resultData[len(idColumns)+1][resultIdx] = stackSeries.At(rowIdx)

			resultIdx++
		}
	}

	// Create result series
	resultSeries := make([]*series.Series, len(resultColumns))
	for i, colName := range resultColumns {
		series, err := series.FromSlice(resultData[i], nil, colName)
		if err != nil {
			return nil, fmt.Errorf("failed to create series for column '%s': %v", colName, err)
		}
		resultSeries[i] = series
	}

	return FromSeries(resultSeries)
}

// Unstack converts long format to wide format
// indexColumn: column to use as row identifier
// columnColumn: column whose values become new column names
// valueColumn: column whose values fill the new columns
func (df *DataFrame) Unstack(indexColumn, columnColumn, valueColumn string) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("DataFrame cannot be nil")
	}

	if df.Len() == 0 {
		return nil, fmt.Errorf("cannot unstack empty DataFrame")
	}

	// Validate columns exist
	if _, err := df.GetColumn(indexColumn); err != nil {
		return nil, fmt.Errorf("index column '%s' not found: %v", indexColumn, err)
	}
	if _, err := df.GetColumn(columnColumn); err != nil {
		return nil, fmt.Errorf("column column '%s' not found: %v", columnColumn, err)
	}
	if _, err := df.GetColumn(valueColumn); err != nil {
		return nil, fmt.Errorf("value column '%s' not found: %v", valueColumn, err)
	}

	// Get unique values for index and columns
	indexValues := getUniqueValues(df, indexColumn)
	colValues := getUniqueValues(df, columnColumn)

	// Create result structure
	resultColumns := make([]string, 1+len(colValues))
	resultColumns[0] = indexColumn
	copy(resultColumns[1:], colValues)

	// Initialize result data
	resultData := make([][]interface{}, len(resultColumns))
	for i := range resultData {
		resultData[i] = make([]interface{}, len(indexValues))
	}

	// Fill index values
	for i, indexVal := range indexValues {
		resultData[0][i] = indexVal
	}

	// Fill values
	for i, indexVal := range indexValues {
		for j, colVal := range colValues {
			// Find value for this combination
			value := findUnstackValue(df, indexColumn, columnColumn, valueColumn, indexVal, colVal)
			resultData[j+1][i] = value
		}
	}

	// Create result series
	resultSeries := make([]*series.Series, len(resultColumns))
	for i, colName := range resultColumns {
		series, err := series.FromSlice(resultData[i], nil, colName)
		if err != nil {
			return nil, fmt.Errorf("failed to create series for column '%s': %v", colName, err)
		}
		resultSeries[i] = series
	}

	return FromSeries(resultSeries)
}

// Transpose creates a transposed version of the DataFrame
// Transposes the DataFrame: rows become columns, columns become rows
func (df *DataFrame) Transpose() (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("DataFrame cannot be nil")
	}

	if df.Len() == 0 {
		return nil, fmt.Errorf("cannot transpose empty DataFrame")
	}

	// Get current dimensions
	numRows := df.Len()
	numCols := len(df.Columns())

	// The transposed DataFrame will have:
	// - numCols rows (one for each original column)
	// - numRows columns (one for each original row)

	// Create new column names based on original row indices
	newColumns := make([]string, numRows)
	for i := 0; i < numRows; i++ {
		newColumns[i] = fmt.Sprintf("row_%d", i)
	}

	// Create result data: transpose[j][i] = original[i][j]
	resultData := make([][]interface{}, numRows) // numRows series for new columns
	for i := range resultData {
		resultData[i] = make([]interface{}, numCols) // each contains numCols values
	}

	// Fill transposed data
	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
			originalValue, _ := df.IAt(i, j)
			resultData[i][j] = originalValue
		}
	}

	// Create result series - we need numRows series (new columns)
	resultSeries := make([]*series.Series, numRows)
	for i := 0; i < numRows; i++ {
		series, err := series.FromSlice(resultData[i], nil, newColumns[i])
		if err != nil {
			return nil, fmt.Errorf("failed to create series for column '%s': %v", newColumns[i], err)
		}
		resultSeries[i] = series
	}

	return FromSeries(resultSeries)
}

// Helper functions

// getUniqueValues returns unique values from a column, sorted
func getUniqueValues(df *DataFrame, columnName string) []string {
	series, _ := df.GetColumn(columnName)
	uniqueMap := make(map[string]bool)

	for i := 0; i < series.Len(); i++ {
		val := series.At(i)
		if val != nil {
			strVal := fmt.Sprintf("%v", val)
			uniqueMap[strVal] = true
		}
	}

	var unique []string
	for val := range uniqueMap {
		unique = append(unique, val)
	}

	sort.Strings(unique)
	return unique
}

// findMatchingRows returns row indices where rowColumn=rowVal and colColumn=colVal
func findMatchingRows(df *DataFrame, rowColumn, colColumn string, rowVal, colVal string) []int {
	rowSeries, _ := df.GetColumn(rowColumn)
	colSeries, _ := df.GetColumn(colColumn)

	var matches []int
	for i := 0; i < df.Len(); i++ {
		rowMatch := fmt.Sprintf("%v", rowSeries.At(i)) == rowVal
		colMatch := fmt.Sprintf("%v", colSeries.At(i)) == colVal

		if rowMatch && colMatch {
			matches = append(matches, i)
		}
	}

	return matches
}

// extractValues extracts values from valueColumn for given row indices
func extractValues(df *DataFrame, valueColumn string, rowIndices []int) []interface{} {
	valueSeries, _ := df.GetColumn(valueColumn)
	values := make([]interface{}, len(rowIndices))

	for i, idx := range rowIndices {
		values[i] = valueSeries.At(idx)
	}

	return values
}

// aggregateValues performs aggregation on a slice of values
func aggregateValues(values []interface{}, aggFunc string) (interface{}, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("no values to aggregate")
	}

	switch aggFunc {
	case "count":
		var count int64 = 0
		for _, val := range values {
			if val != nil {
				count++
			}
		}
		return count, nil

	case "sum":
		return sumValues(values)

	case "mean":
		sum, err := sumValues(values)
		if err != nil {
			return nil, err
		}
		var count int64 = 0
		for _, val := range values {
			if val != nil {
				count++
			}
		}
		if count == 0 {
			return 0.0, nil
		}

		switch s := sum.(type) {
		case int64:
			return float64(s) / float64(count), nil
		case float64:
			return s / float64(count), nil
		default:
			return nil, fmt.Errorf("cannot compute mean of non-numeric values")
		}

	case "min":
		return minValues(values)

	case "max":
		return maxValues(values)

	default:
		return nil, fmt.Errorf("unsupported aggregation function: %s", aggFunc)
	}
}

// sumValues sums numeric values
func sumValues(values []interface{}) (interface{}, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("no values to sum")
	}

	// Find first non-nil value to determine type
	var firstVal interface{}
	for _, val := range values {
		if val != nil {
			firstVal = val
			break
		}
	}

	if firstVal == nil {
		return 0.0, nil
	}

	switch firstVal.(type) {
	case int64:
		var sum int64 = 0
		for _, val := range values {
			if val != nil {
				sum += val.(int64)
			}
		}
		return sum, nil

	case float64:
		var sum float64 = 0.0
		for _, val := range values {
			if val != nil {
				sum += val.(float64)
			}
		}
		return sum, nil

	default:
		return nil, fmt.Errorf("cannot sum non-numeric values")
	}
}

// minValues finds minimum of numeric values
func minValues(values []interface{}) (interface{}, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("no values to find min")
	}

	// Find first non-nil value
	var minVal interface{}
	for _, val := range values {
		if val != nil {
			minVal = val
			break
		}
	}

	if minVal == nil {
		return nil, nil
	}

	switch minVal.(type) {
	case int64:
		min := minVal.(int64)
		for _, val := range values {
			if val != nil && val.(int64) < min {
				min = val.(int64)
			}
		}
		return min, nil

	case float64:
		min := minVal.(float64)
		for _, val := range values {
			if val != nil && val.(float64) < min {
				min = val.(float64)
			}
		}
		return min, nil

	default:
		return nil, fmt.Errorf("cannot find min of non-numeric values")
	}
}

// maxValues finds maximum of numeric values
func maxValues(values []interface{}) (interface{}, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("no values to find max")
	}

	// Find first non-nil value
	var maxVal interface{}
	for _, val := range values {
		if val != nil {
			maxVal = val
			break
		}
	}

	if maxVal == nil {
		return nil, nil
	}

	switch maxVal.(type) {
	case int64:
		max := maxVal.(int64)
		for _, val := range values {
			if val != nil && val.(int64) > max {
				max = val.(int64)
			}
		}
		return max, nil

	case float64:
		max := maxVal.(float64)
		for _, val := range values {
			if val != nil && val.(float64) > max {
				max = val.(float64)
			}
		}
		return max, nil

	default:
		return nil, fmt.Errorf("cannot find max of non-numeric values")
	}
}

// findUnstackValue finds the value for a specific index/column combination
// If multiple matches exist, returns the last one encountered
func findUnstackValue(df *DataFrame, indexColumn, columnColumn, valueColumn string, indexVal, colVal string) interface{} {
	indexSeries, _ := df.GetColumn(indexColumn)
	columnSeries, _ := df.GetColumn(columnColumn)
	valueSeries, _ := df.GetColumn(valueColumn)

	var lastValue interface{} = nil
	found := false

	for i := 0; i < df.Len(); i++ {
		indexMatch := fmt.Sprintf("%v", indexSeries.At(i)) == indexVal
		colMatch := fmt.Sprintf("%v", columnSeries.At(i)) == colVal

		if indexMatch && colMatch {
			lastValue = valueSeries.At(i)
			found = true
		}
	}

	if found {
		return lastValue
	}
	return nil // No match found
}
