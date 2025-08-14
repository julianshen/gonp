package dataframe

import (
	"fmt"

	"github.com/julianshen/gonp/series"
)

// DataFrame represents a 2D labeled data structure
type DataFrame struct {
	columns map[string]*series.Series
	index   []string // column names in order
}

// FromSeries creates a DataFrame from a slice of Series
func FromSeries(seriesList []*series.Series) (*DataFrame, error) {
	if len(seriesList) == 0 {
		return nil, fmt.Errorf("cannot create DataFrame from empty series list")
	}

	// Check that all series have the same length
	expectedLen := seriesList[0].Len()
	for i, s := range seriesList {
		if s.Len() != expectedLen {
			return nil, fmt.Errorf("series %d has length %d, expected %d", i, s.Len(), expectedLen)
		}
	}

	columns := make(map[string]*series.Series)
	index := make([]string, len(seriesList))

	for i, s := range seriesList {
		name := s.Name()
		if name == "" {
			name = fmt.Sprintf("col_%d", i)
		}
		columns[name] = s
		index[i] = name
	}

	return &DataFrame{
		columns: columns,
		index:   index,
	}, nil
}

// Len returns the number of rows in the DataFrame
func (df *DataFrame) Len() int {
	if len(df.index) == 0 {
		return 0
	}
	return df.columns[df.index[0]].Len()
}

// Columns returns the column names
func (df *DataFrame) Columns() []string {
	return append([]string{}, df.index...) // return a copy
}

// GetColumn returns the Series for the given column name
func (df *DataFrame) GetColumn(name string) (*series.Series, error) {
	series, exists := df.columns[name]
	if !exists {
		return nil, fmt.Errorf("column '%s' not found", name)
	}
	return series, nil
}

// GetSeries returns all series in column order
func (df *DataFrame) GetSeries() []*series.Series {
	result := make([]*series.Series, len(df.index))
	for i, name := range df.index {
		result[i] = df.columns[name]
	}
	return result
}

// IAt returns the value at the given row and column indices
func (df *DataFrame) IAt(row, col int) (interface{}, error) {
	if col < 0 || col >= len(df.index) {
		return nil, fmt.Errorf("column index %d out of bounds (0-%d)", col, len(df.index)-1)
	}

	colName := df.index[col]
	series := df.columns[colName]

	if row < 0 || row >= series.Len() {
		return nil, fmt.Errorf("row index %d out of bounds (0-%d)", row, series.Len()-1)
	}

	return series.At(row), nil
}

// Slice returns a new DataFrame with rows from start to end (exclusive)
func (df *DataFrame) Slice(start, end int) (*DataFrame, error) {
	if len(df.index) == 0 {
		return &DataFrame{columns: make(map[string]*series.Series), index: []string{}}, nil
	}

	// Normalize indices
	if start < 0 {
		start = 0
	}
	if end > df.Len() {
		end = df.Len()
	}
	if start >= end {
		// Return empty DataFrame with same column structure
		emptyColumns := make(map[string]*series.Series)
		for _, colName := range df.index {
			originalSeries := df.columns[colName]
			emptySeries := series.Empty(originalSeries.DType(), originalSeries.Name())
			emptyColumns[colName] = emptySeries
		}
		return &DataFrame{
			columns: emptyColumns,
			index:   append([]string{}, df.index...),
		}, nil
	}

	// Slice each series
	newColumns := make(map[string]*series.Series)
	for _, colName := range df.index {
		originalSeries := df.columns[colName]
		slicedSeries, err := originalSeries.Slice(start, end)
		if err != nil {
			return nil, fmt.Errorf("failed to slice column '%s': %v", colName, err)
		}
		newColumns[colName] = slicedSeries
	}

	return &DataFrame{
		columns: newColumns,
		index:   append([]string{}, df.index...), // copy column names
	}, nil
}
