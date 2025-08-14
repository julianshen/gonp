package dataframe

import (
	"fmt"
	"reflect"

	"github.com/julianshen/gonp/series"
)

// JoinType represents the type of join operation
type JoinType int

const (
	Inner JoinType = iota
	Left
	Right
	Outer
)

// InnerJoin performs an inner join between two DataFrames on the specified columns
func InnerJoin(left, right *DataFrame, leftOn, rightOn string) (*DataFrame, error) {
	return joinDataFrames(left, right, []string{leftOn}, []string{rightOn}, Inner)
}

// LeftJoin performs a left join between two DataFrames on the specified columns
func LeftJoin(left, right *DataFrame, leftOn, rightOn string) (*DataFrame, error) {
	return joinDataFrames(left, right, []string{leftOn}, []string{rightOn}, Left)
}

// RightJoin performs a right join between two DataFrames on the specified columns
func RightJoin(left, right *DataFrame, leftOn, rightOn string) (*DataFrame, error) {
	return joinDataFrames(left, right, []string{leftOn}, []string{rightOn}, Right)
}

// OuterJoin performs a full outer join between two DataFrames on the specified columns
func OuterJoin(left, right *DataFrame, leftOn, rightOn string) (*DataFrame, error) {
	return joinDataFrames(left, right, []string{leftOn}, []string{rightOn}, Outer)
}

// InnerJoinMultiple performs an inner join on multiple columns
func InnerJoinMultiple(left, right *DataFrame, leftOn, rightOn []string) (*DataFrame, error) {
	return joinDataFrames(left, right, leftOn, rightOn, Inner)
}

// joinDataFrames is the core implementation for all join operations
func joinDataFrames(left, right *DataFrame, leftOn, rightOn []string, joinType JoinType) (*DataFrame, error) {
	// Validate inputs
	if left == nil {
		return nil, fmt.Errorf("left DataFrame cannot be nil")
	}
	if right == nil {
		return nil, fmt.Errorf("right DataFrame cannot be nil")
	}
	if len(leftOn) != len(rightOn) {
		return nil, fmt.Errorf("number of left keys (%d) must match number of right keys (%d)", len(leftOn), len(rightOn))
	}
	if len(leftOn) == 0 {
		return nil, fmt.Errorf("at least one join key must be specified")
	}

	// Validate that join columns exist
	for _, col := range leftOn {
		if _, err := left.GetColumn(col); err != nil {
			return nil, fmt.Errorf("left join column '%s' not found: %v", col, err)
		}
	}
	for _, col := range rightOn {
		if _, err := right.GetColumn(col); err != nil {
			return nil, fmt.Errorf("right join column '%s' not found: %v", col, err)
		}
	}

	// Build indices for join keys
	leftIndex := buildJoinIndex(left, leftOn)
	rightIndex := buildJoinIndex(right, rightOn)

	// Determine which rows to include based on join type
	var resultRows []joinedRow
	switch joinType {
	case Inner:
		resultRows = innerJoinRows(leftIndex, rightIndex)
	case Left:
		resultRows = leftJoinRows(leftIndex, rightIndex)
	case Right:
		resultRows = rightJoinRows(leftIndex, rightIndex)
	case Outer:
		resultRows = outerJoinRows(leftIndex, rightIndex)
	default:
		return nil, fmt.Errorf("unsupported join type: %v", joinType)
	}

	// Build result DataFrame
	return buildJoinedDataFrame(left, right, leftOn, rightOn, resultRows)
}

// joinedRow represents a row in the joined result
type joinedRow struct {
	leftRow  *int // nil if no matching left row
	rightRow *int // nil if no matching right row
}

// joinKey represents a composite key for joining
type joinKey struct {
	values []interface{}
}

// Equal checks if two join keys are equal
func (k joinKey) Equal(other joinKey) bool {
	if len(k.values) != len(other.values) {
		return false
	}
	for i, v := range k.values {
		if !reflect.DeepEqual(v, other.values[i]) {
			return false
		}
	}
	return true
}

// buildJoinIndex creates a map from join keys to row indices
func buildJoinIndex(df *DataFrame, columns []string) map[string][]int {
	index := make(map[string][]int)

	for row := 0; row < df.Len(); row++ {
		key := buildJoinKey(df, columns, row)
		keyStr := joinKeyToString(key)
		index[keyStr] = append(index[keyStr], row)
	}

	return index
}

// buildJoinKey creates a join key for a specific row
func buildJoinKey(df *DataFrame, columns []string, row int) joinKey {
	values := make([]interface{}, len(columns))
	for i, col := range columns {
		series, _ := df.GetColumn(col)
		values[i] = series.At(row)
	}
	return joinKey{values: values}
}

// joinKeyToString converts a join key to a string for use as map key
func joinKeyToString(key joinKey) string {
	return fmt.Sprintf("%v", key.values)
}

// innerJoinRows finds rows for inner join
func innerJoinRows(leftIndex, rightIndex map[string][]int) []joinedRow {
	var result []joinedRow

	for key, leftRows := range leftIndex {
		if rightRows, exists := rightIndex[key]; exists {
			// Cartesian product of matching rows
			for _, leftRow := range leftRows {
				for _, rightRow := range rightRows {
					result = append(result, joinedRow{
						leftRow:  &leftRow,
						rightRow: &rightRow,
					})
				}
			}
		}
	}

	return result
}

// leftJoinRows finds rows for left join
func leftJoinRows(leftIndex, rightIndex map[string][]int) []joinedRow {
	var result []joinedRow

	for key, leftRows := range leftIndex {
		if rightRows, exists := rightIndex[key]; exists {
			// Cartesian product of matching rows
			for _, leftRow := range leftRows {
				for _, rightRow := range rightRows {
					result = append(result, joinedRow{
						leftRow:  &leftRow,
						rightRow: &rightRow,
					})
				}
			}
		} else {
			// No match in right, include left row with nil right
			for _, leftRow := range leftRows {
				result = append(result, joinedRow{
					leftRow:  &leftRow,
					rightRow: nil,
				})
			}
		}
	}

	return result
}

// rightJoinRows finds rows for right join
func rightJoinRows(leftIndex, rightIndex map[string][]int) []joinedRow {
	var result []joinedRow

	for key, rightRows := range rightIndex {
		if leftRows, exists := leftIndex[key]; exists {
			// Cartesian product of matching rows
			for _, leftRow := range leftRows {
				for _, rightRow := range rightRows {
					result = append(result, joinedRow{
						leftRow:  &leftRow,
						rightRow: &rightRow,
					})
				}
			}
		} else {
			// No match in left, include right row with nil left
			for _, rightRow := range rightRows {
				result = append(result, joinedRow{
					leftRow:  nil,
					rightRow: &rightRow,
				})
			}
		}
	}

	return result
}

// outerJoinRows finds rows for outer join
func outerJoinRows(leftIndex, rightIndex map[string][]int) []joinedRow {
	var result []joinedRow
	processedKeys := make(map[string]bool)

	// Process all keys from left index
	for key, leftRows := range leftIndex {
		processedKeys[key] = true
		if rightRows, exists := rightIndex[key]; exists {
			// Cartesian product of matching rows
			for _, leftRow := range leftRows {
				for _, rightRow := range rightRows {
					result = append(result, joinedRow{
						leftRow:  &leftRow,
						rightRow: &rightRow,
					})
				}
			}
		} else {
			// No match in right, include left row with nil right
			for _, leftRow := range leftRows {
				result = append(result, joinedRow{
					leftRow:  &leftRow,
					rightRow: nil,
				})
			}
		}
	}

	// Process remaining keys from right index
	for key, rightRows := range rightIndex {
		if !processedKeys[key] {
			// No match in left, include right row with nil left
			for _, rightRow := range rightRows {
				result = append(result, joinedRow{
					leftRow:  nil,
					rightRow: &rightRow,
				})
			}
		}
	}

	return result
}

// buildJoinedDataFrame constructs the final DataFrame from joined rows
func buildJoinedDataFrame(left, right *DataFrame, leftOn, rightOn []string, rows []joinedRow) (*DataFrame, error) {
	if len(rows) == 0 {
		// Return empty DataFrame with combined columns
		return buildEmptyJoinedDataFrame(left, right, leftOn, rightOn)
	}

	// Determine column structure
	leftColumns := left.Columns()
	rightColumns := right.Columns()

	// Remove duplicate join columns from right DataFrame
	rightColumnsFiltered := make([]string, 0)
	joinColSet := make(map[string]bool)
	for i, col := range rightOn {
		joinColSet[col] = true
		// Keep the left version of join columns
		if i == 0 {
			// This ensures we don't duplicate the join column
		}
	}

	for _, col := range rightColumns {
		if !joinColSet[col] {
			rightColumnsFiltered = append(rightColumnsFiltered, col)
		}
	}

	allColumns := append(leftColumns, rightColumnsFiltered...)

	// Create series for each column
	resultSeries := make([]*series.Series, len(allColumns))

	// Process left columns
	for i, colName := range leftColumns {
		leftSeries, _ := left.GetColumn(colName)
		data := make([]interface{}, len(rows))

		for j, row := range rows {
			if row.leftRow != nil {
				data[j] = leftSeries.At(*row.leftRow)
			} else {
				data[j] = nil
			}
		}

		series, err := series.FromSlice(data, nil, colName)
		if err != nil {
			return nil, fmt.Errorf("failed to create series for column '%s': %v", colName, err)
		}
		resultSeries[i] = series
	}

	// Process right columns (excluding join columns)
	for i, colName := range rightColumnsFiltered {
		rightSeries, _ := right.GetColumn(colName)
		data := make([]interface{}, len(rows))

		for j, row := range rows {
			if row.rightRow != nil {
				data[j] = rightSeries.At(*row.rightRow)
			} else {
				data[j] = nil
			}
		}

		series, err := series.FromSlice(data, nil, colName)
		if err != nil {
			return nil, fmt.Errorf("failed to create series for column '%s': %v", colName, err)
		}
		resultSeries[len(leftColumns)+i] = series
	}

	return FromSeries(resultSeries)
}

// buildEmptyJoinedDataFrame creates an empty DataFrame with the structure of a join result
func buildEmptyJoinedDataFrame(left, right *DataFrame, leftOn, rightOn []string) (*DataFrame, error) {
	leftColumns := left.Columns()
	rightColumns := right.Columns()

	// Remove duplicate join columns from right DataFrame
	rightColumnsFiltered := make([]string, 0)
	joinColSet := make(map[string]bool)
	for _, col := range rightOn {
		joinColSet[col] = true
	}

	for _, col := range rightColumns {
		if !joinColSet[col] {
			rightColumnsFiltered = append(rightColumnsFiltered, col)
		}
	}

	allColumns := append(leftColumns, rightColumnsFiltered...)
	emptySeries := make([]*series.Series, len(allColumns))

	// Create empty series for each column
	for i, colName := range leftColumns {
		leftSeries, _ := left.GetColumn(colName)
		emptySeries[i] = series.Empty(leftSeries.DType(), colName)
	}

	for i, colName := range rightColumnsFiltered {
		rightSeries, _ := right.GetColumn(colName)
		emptySeries[len(leftColumns)+i] = series.Empty(rightSeries.DType(), colName)
	}

	return FromSeries(emptySeries)
}
