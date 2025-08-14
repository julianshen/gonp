package dataframe

import (
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/series"
)

// Helper function to create test series
func createTestSeries(data []float64, name string) *series.Series {
	arr, _ := array.FromSlice(data)
	s, _ := series.NewSeries(arr, nil, name)
	return s
}

func TestFromSeries(t *testing.T) {
	tests := []struct {
		name     string
		series   []*series.Series
		wantErr  bool
		wantCols []string
		wantRows int
	}{
		{
			name: "valid series with same length",
			series: []*series.Series{
				createTestSeries([]float64{1.0, 2.0, 3.0}, "A"),
				createTestSeries([]float64{4.0, 5.0, 6.0}, "B"),
			},
			wantErr:  false,
			wantCols: []string{"A", "B"},
			wantRows: 3,
		},
		{
			name:    "empty series list",
			series:  []*series.Series{},
			wantErr: true,
		},
		{
			name: "series with different lengths",
			series: []*series.Series{
				createTestSeries([]float64{1.0, 2.0}, "A"),
				createTestSeries([]float64{4.0, 5.0, 6.0}, "B"),
			},
			wantErr: true,
		},
		{
			name: "series without names",
			series: []*series.Series{
				createTestSeries([]float64{1.0, 2.0}, ""),
				createTestSeries([]float64{3.0, 4.0}, ""),
			},
			wantErr:  false,
			wantCols: []string{"col_0", "col_1"},
			wantRows: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			df, err := FromSeries(tt.series)
			if (err != nil) != tt.wantErr {
				t.Errorf("FromSeries() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			if df.Len() != tt.wantRows {
				t.Errorf("DataFrame.Len() = %v, want %v", df.Len(), tt.wantRows)
			}

			cols := df.Columns()
			if len(cols) != len(tt.wantCols) {
				t.Errorf("DataFrame.Columns() length = %v, want %v", len(cols), len(tt.wantCols))
			}
			for i, col := range tt.wantCols {
				if cols[i] != col {
					t.Errorf("DataFrame.Columns()[%d] = %v, want %v", i, cols[i], col)
				}
			}
		})
	}
}

func TestDataFrameGetColumn(t *testing.T) {
	seriesA := createTestSeries([]float64{1.0, 2.0, 3.0}, "A")
	seriesB := createTestSeries([]float64{4.0, 5.0, 6.0}, "B")
	df, _ := FromSeries([]*series.Series{seriesA, seriesB})

	tests := []struct {
		name     string
		colName  string
		wantErr  bool
		wantData []float64
	}{
		{
			name:     "existing column A",
			colName:  "A",
			wantErr:  false,
			wantData: []float64{1.0, 2.0, 3.0},
		},
		{
			name:     "existing column B",
			colName:  "B",
			wantErr:  false,
			wantData: []float64{4.0, 5.0, 6.0},
		},
		{
			name:    "non-existing column",
			colName: "C",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			series, err := df.GetColumn(tt.colName)
			if (err != nil) != tt.wantErr {
				t.Errorf("DataFrame.GetColumn() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			if series.Len() != len(tt.wantData) {
				t.Errorf("Series length = %v, want %v", series.Len(), len(tt.wantData))
			}

			for i, expected := range tt.wantData {
				if val := series.At(i).(float64); val != expected {
					t.Errorf("Series.At(%d) = %v, want %v", i, val, expected)
				}
			}
		})
	}
}

func TestDataFrameGetSeries(t *testing.T) {
	seriesA := createTestSeries([]float64{1.0, 2.0, 3.0}, "A")
	seriesB := createTestSeries([]float64{4.0, 5.0, 6.0}, "B")
	df, _ := FromSeries([]*series.Series{seriesA, seriesB})

	allSeries := df.GetSeries()

	if len(allSeries) != 2 {
		t.Errorf("GetSeries() returned %d series, want 2", len(allSeries))
	}

	if allSeries[0].Name() != "A" {
		t.Errorf("First series name = %v, want A", allSeries[0].Name())
	}

	if allSeries[1].Name() != "B" {
		t.Errorf("Second series name = %v, want B", allSeries[1].Name())
	}
}

// Test for DataFrame indexing functionality
func TestDataFrameIAt(t *testing.T) {
	seriesA := createTestSeries([]float64{1.0, 2.0, 3.0}, "A")
	seriesB := createTestSeries([]float64{4.0, 5.0, 6.0}, "B")
	df, _ := FromSeries([]*series.Series{seriesA, seriesB})

	tests := []struct {
		name     string
		row      int
		col      int
		expected interface{}
		wantErr  bool
	}{
		{
			name:     "valid index [0,0]",
			row:      0,
			col:      0,
			expected: 1.0,
			wantErr:  false,
		},
		{
			name:     "valid index [1,1]",
			row:      1,
			col:      1,
			expected: 5.0,
			wantErr:  false,
		},
		{
			name:    "invalid row index",
			row:     3,
			col:     0,
			wantErr: true,
		},
		{
			name:    "invalid column index",
			row:     0,
			col:     2,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			val, err := df.IAt(tt.row, tt.col)
			if (err != nil) != tt.wantErr {
				t.Errorf("DataFrame.IAt() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && val != tt.expected {
				t.Errorf("DataFrame.IAt(%d, %d) = %v, want %v", tt.row, tt.col, val, tt.expected)
			}
		})
	}
}

// Test for DataFrame slicing functionality
func TestDataFrameSlice(t *testing.T) {
	seriesA := createTestSeries([]float64{1.0, 2.0, 3.0, 4.0}, "A")
	seriesB := createTestSeries([]float64{5.0, 6.0, 7.0, 8.0}, "B")
	df, _ := FromSeries([]*series.Series{seriesA, seriesB})

	tests := []struct {
		name     string
		start    int
		end      int
		wantRows int
		wantCols int
	}{
		{
			name:     "slice first two rows",
			start:    0,
			end:      2,
			wantRows: 2,
			wantCols: 2,
		},
		{
			name:     "slice middle rows",
			start:    1,
			end:      3,
			wantRows: 2,
			wantCols: 2,
		},
		{
			name:     "slice single row",
			start:    2,
			end:      3,
			wantRows: 1,
			wantCols: 2,
		},
		{
			name:     "empty slice",
			start:    2,
			end:      2,
			wantRows: 0,
			wantCols: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sliced, err := df.Slice(tt.start, tt.end)
			if err != nil {
				t.Errorf("DataFrame.Slice() error = %v", err)
				return
			}
			if sliced.Len() != tt.wantRows {
				t.Errorf("Sliced DataFrame rows = %v, want %v", sliced.Len(), tt.wantRows)
			}
			if len(sliced.Columns()) != tt.wantCols {
				t.Errorf("Sliced DataFrame columns = %v, want %v", len(sliced.Columns()), tt.wantCols)
			}
		})
	}
}
