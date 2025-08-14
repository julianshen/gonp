package dataframe

import (
	"fmt"

	"github.com/julianshen/gonp/series"
)

// GroupBy represents a grouped DataFrame
type GroupBy struct {
	dataframe *DataFrame
	groupKeys []string
	groups    map[string]*DataFrame
	keys      []interface{}
	keyValues map[string][]interface{} // Store actual key values for each group
}

// GroupBy groups the DataFrame by the specified column
func (df *DataFrame) GroupBy(column string) (*GroupBy, error) {
	return df.GroupByMultiple([]string{column})
}

// GroupByMultiple groups the DataFrame by multiple columns
func (df *DataFrame) GroupByMultiple(columns []string) (*GroupBy, error) {
	if df == nil {
		return nil, fmt.Errorf("DataFrame cannot be nil")
	}

	if len(columns) == 0 {
		return nil, fmt.Errorf("at least one grouping column must be specified")
	}

	if df.Len() == 0 {
		return nil, fmt.Errorf("cannot group empty DataFrame")
	}

	// Validate columns exist
	for _, col := range columns {
		if _, err := df.GetColumn(col); err != nil {
			return nil, fmt.Errorf("grouping column '%s' not found: %v", col, err)
		}
	}

	// Build groups
	groups := make(map[string]*DataFrame)
	keyValues := make(map[string][]interface{})
	var keys []interface{}
	keySet := make(map[string]bool)

	// Group rows by key
	rowsByKey := make(map[string][]int)

	for i := 0; i < df.Len(); i++ {
		key := buildGroupKey(df, columns, i)
		keyStr := groupKeyToString(key)

		if !keySet[keyStr] {
			keySet[keyStr] = true
			keyValues[keyStr] = key.values
			if len(columns) == 1 {
				keys = append(keys, key.values[0])
			} else {
				keys = append(keys, key.values)
			}
		}

		rowsByKey[keyStr] = append(rowsByKey[keyStr], i)
	}

	// Create DataFrame for each group
	for keyStr, rowIndices := range rowsByKey {
		groupDF, err := df.selectRows(rowIndices)
		if err != nil {
			return nil, fmt.Errorf("failed to create group DataFrame: %v", err)
		}
		groups[keyStr] = groupDF
	}

	return &GroupBy{
		dataframe: df,
		groupKeys: columns,
		groups:    groups,
		keys:      keys,
		keyValues: keyValues,
	}, nil
}

// Keys returns the group keys
func (gb *GroupBy) Keys() []interface{} {
	return append([]interface{}{}, gb.keys...)
}

// GetGroup returns the DataFrame for a specific group key
func (gb *GroupBy) GetGroup(key interface{}) (*DataFrame, error) {
	var keyStr string
	if len(gb.groupKeys) == 1 {
		keyStr = fmt.Sprintf("[%v]", key)
	} else {
		keyStr = fmt.Sprintf("%v", key)
	}

	group, exists := gb.groups[keyStr]
	if !exists {
		return nil, fmt.Errorf("group key '%v' not found", key)
	}

	return group, nil
}

// Sum performs sum aggregation on grouped data
func (gb *GroupBy) Sum() (*DataFrame, error) {
	return gb.aggregate("sum")
}

// Mean performs mean aggregation on grouped data
func (gb *GroupBy) Mean() (*DataFrame, error) {
	return gb.aggregate("mean")
}

// Count performs count aggregation on grouped data
func (gb *GroupBy) Count() (*DataFrame, error) {
	return gb.aggregate("count")
}

// Min performs min aggregation on grouped data
func (gb *GroupBy) Min() (*DataFrame, error) {
	return gb.aggregate("min")
}

// Max performs max aggregation on grouped data
func (gb *GroupBy) Max() (*DataFrame, error) {
	return gb.aggregate("max")
}

// aggregate performs the specified aggregation function
func (gb *GroupBy) aggregate(operation string) (*DataFrame, error) {
	if len(gb.groups) == 0 {
		return nil, fmt.Errorf("no groups to aggregate")
	}

	// Determine columns to aggregate (exclude grouping columns)
	allColumns := gb.dataframe.Columns()
	var aggColumns []string
	groupKeySet := make(map[string]bool)
	for _, key := range gb.groupKeys {
		groupKeySet[key] = true
	}

	for _, col := range allColumns {
		if !groupKeySet[col] {
			aggColumns = append(aggColumns, col)
		}
	}

	// Create result columns: grouping columns + aggregated columns
	resultColumns := append(gb.groupKeys, aggColumns...)
	resultSeries := make([]*series.Series, len(resultColumns))

	// Initialize result data arrays
	numGroups := len(gb.keys)
	resultData := make([][]interface{}, len(resultColumns))
	for i := range resultData {
		resultData[i] = make([]interface{}, numGroups)
	}

	// Process each group
	groupIdx := 0
	for keyStr, groupDF := range gb.groups {
		// Set grouping column values
		keyVals := gb.keyValues[keyStr]
		for i := range gb.groupKeys {
			resultData[i][groupIdx] = keyVals[i]
		}

		// Aggregate non-grouping columns
		for i, aggCol := range aggColumns {
			colIdx := len(gb.groupKeys) + i
			colSeries, _ := groupDF.GetColumn(aggCol)

			var aggValue interface{}
			var err error

			switch operation {
			case "sum":
				aggValue, err = gb.sumSeries(colSeries)
			case "mean":
				aggValue, err = gb.meanSeries(colSeries)
			case "count":
				aggValue, err = gb.countSeries(colSeries)
			case "min":
				aggValue, err = gb.minSeries(colSeries)
			case "max":
				aggValue, err = gb.maxSeries(colSeries)
			default:
				return nil, fmt.Errorf("unsupported aggregation operation: %s", operation)
			}

			if err != nil {
				return nil, fmt.Errorf("failed to aggregate column '%s': %v", aggCol, err)
			}

			resultData[colIdx][groupIdx] = aggValue
		}

		groupIdx++
	}

	// Create result series
	for i, colName := range resultColumns {
		series, err := series.FromSlice(resultData[i], nil, colName)
		if err != nil {
			return nil, fmt.Errorf("failed to create series for column '%s': %v", colName, err)
		}
		resultSeries[i] = series
	}

	return FromSeries(resultSeries)
}

// Helper functions for aggregations
func (gb *GroupBy) sumSeries(s *series.Series) (interface{}, error) {
	if s.Len() == 0 {
		return nil, fmt.Errorf("cannot sum empty series")
	}

	// Check if numeric
	firstVal := s.At(0)
	switch firstVal.(type) {
	case int64:
		var sum int64 = 0
		for i := 0; i < s.Len(); i++ {
			if val := s.At(i); val != nil {
				sum += val.(int64)
			}
		}
		return sum, nil
	case float64:
		var sum float64 = 0.0
		for i := 0; i < s.Len(); i++ {
			if val := s.At(i); val != nil {
				sum += val.(float64)
			}
		}
		return sum, nil
	default:
		return nil, fmt.Errorf("cannot sum non-numeric series")
	}
}

func (gb *GroupBy) meanSeries(s *series.Series) (interface{}, error) {
	if s.Len() == 0 {
		return nil, fmt.Errorf("cannot compute mean of empty series")
	}

	// Check if numeric
	firstVal := s.At(0)
	switch firstVal.(type) {
	case int64:
		var sum int64 = 0
		var count int64 = 0
		for i := 0; i < s.Len(); i++ {
			if val := s.At(i); val != nil {
				sum += val.(int64)
				count++
			}
		}
		if count == 0 {
			return 0.0, nil
		}
		return float64(sum) / float64(count), nil
	case float64:
		var sum float64 = 0.0
		var count int64 = 0
		for i := 0; i < s.Len(); i++ {
			if val := s.At(i); val != nil {
				sum += val.(float64)
				count++
			}
		}
		if count == 0 {
			return 0.0, nil
		}
		return sum / float64(count), nil
	default:
		return nil, fmt.Errorf("cannot compute mean of non-numeric series")
	}
}

func (gb *GroupBy) countSeries(s *series.Series) (interface{}, error) {
	var count int64 = 0
	for i := 0; i < s.Len(); i++ {
		if s.At(i) != nil {
			count++
		}
	}
	return count, nil
}

func (gb *GroupBy) minSeries(s *series.Series) (interface{}, error) {
	if s.Len() == 0 {
		return nil, fmt.Errorf("cannot find min of empty series")
	}

	// Find first non-nil value
	var minVal interface{}
	for i := 0; i < s.Len(); i++ {
		if val := s.At(i); val != nil {
			minVal = val
			break
		}
	}

	if minVal == nil {
		return nil, nil
	}

	// Compare remaining values
	switch minVal.(type) {
	case int64:
		min := minVal.(int64)
		for i := 0; i < s.Len(); i++ {
			if val := s.At(i); val != nil {
				if val.(int64) < min {
					min = val.(int64)
				}
			}
		}
		return min, nil
	case float64:
		min := minVal.(float64)
		for i := 0; i < s.Len(); i++ {
			if val := s.At(i); val != nil {
				if val.(float64) < min {
					min = val.(float64)
				}
			}
		}
		return min, nil
	default:
		return nil, fmt.Errorf("cannot find min of non-numeric series")
	}
}

func (gb *GroupBy) maxSeries(s *series.Series) (interface{}, error) {
	if s.Len() == 0 {
		return nil, fmt.Errorf("cannot find max of empty series")
	}

	// Find first non-nil value
	var maxVal interface{}
	for i := 0; i < s.Len(); i++ {
		if val := s.At(i); val != nil {
			maxVal = val
			break
		}
	}

	if maxVal == nil {
		return nil, nil
	}

	// Compare remaining values
	switch maxVal.(type) {
	case int64:
		max := maxVal.(int64)
		for i := 0; i < s.Len(); i++ {
			if val := s.At(i); val != nil {
				if val.(int64) > max {
					max = val.(int64)
				}
			}
		}
		return max, nil
	case float64:
		max := maxVal.(float64)
		for i := 0; i < s.Len(); i++ {
			if val := s.At(i); val != nil {
				if val.(float64) > max {
					max = val.(float64)
				}
			}
		}
		return max, nil
	default:
		return nil, fmt.Errorf("cannot find max of non-numeric series")
	}
}

// selectRows creates a new DataFrame with only the specified row indices
func (df *DataFrame) selectRows(indices []int) (*DataFrame, error) {
	if len(indices) == 0 {
		// Return empty DataFrame with same structure
		emptySeries := make([]*series.Series, len(df.index))
		for i, colName := range df.index {
			originalSeries := df.columns[colName]
			emptySeries[i] = series.Empty(originalSeries.DType(), colName)
		}
		return FromSeries(emptySeries)
	}

	newSeries := make([]*series.Series, len(df.index))

	for i, colName := range df.index {
		originalSeries := df.columns[colName]
		data := make([]interface{}, len(indices))

		for j, rowIdx := range indices {
			if rowIdx < 0 || rowIdx >= originalSeries.Len() {
				return nil, fmt.Errorf("row index %d out of bounds", rowIdx)
			}
			data[j] = originalSeries.At(rowIdx)
		}

		series, err := series.FromSlice(data, nil, colName)
		if err != nil {
			return nil, fmt.Errorf("failed to create series for column '%s': %v", colName, err)
		}
		newSeries[i] = series
	}

	return FromSeries(newSeries)
}

// groupKey represents a composite key for grouping
type groupKey struct {
	values []interface{}
}

// buildGroupKey creates a group key for a specific row
func buildGroupKey(df *DataFrame, columns []string, row int) groupKey {
	values := make([]interface{}, len(columns))
	for i, col := range columns {
		series, _ := df.GetColumn(col)
		values[i] = series.At(row)
	}
	return groupKey{values: values}
}

// groupKeyToString converts a group key to a string for use as map key
func groupKeyToString(key groupKey) string {
	return fmt.Sprintf("%v", key.values)
}
