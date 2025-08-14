// Series plotting integration for visualization package
package visualization

import (
	"fmt"

	"github.com/julianshen/gonp/array"
)

// SeriesPlotOptions configures Series plotting
type SeriesPlotOptions struct {
	PlotType    PlotType
	Title       string
	XLabel      string
	YLabel      string
	Color       string
	LineStyle   string
	MarkerStyle string
	Alpha       float64
	Grid        bool
}

// DefaultSeriesPlotOptions returns default plotting options for Series
func DefaultSeriesPlotOptions() *SeriesPlotOptions {
	return &SeriesPlotOptions{
		PlotType:    LinePlot,
		Color:       "blue",
		LineStyle:   "-",
		MarkerStyle: "o",
		Alpha:       1.0,
		Grid:        true,
	}
}

// Plot creates a plot from a Series (this would be added to the Series type)
// For now, we'll create a standalone function that takes a Series-like interface

// SeriesPlotter interface defines what we need from a Series for plotting
type SeriesPlotter interface {
	Data() *array.Array // Use Data() method from actual Series
	Index() interface{} // Could be *array.Array or other index types
	Name() string
	Len() int
}

// Adapter function to get values as array
func getValuesArray(s SeriesPlotter) *array.Array {
	return s.Data()
}

// PlotSeries creates a plot from a Series
func PlotSeries(s SeriesPlotter, options *SeriesPlotOptions) (*PlotChart, error) {
	if options == nil {
		options = DefaultSeriesPlotOptions()
	}

	plot := NewPlot(options.PlotType)

	if options.Title != "" {
		plot.SetTitle(options.Title)
	} else {
		plot.SetTitle(fmt.Sprintf("Plot of %s", s.Name()))
	}

	plot.SetXLabel(options.XLabel)
	plot.SetYLabel(options.YLabel)
	plot.layout.Grid.Show = options.Grid

	// Get data from series
	yData := getValuesArray(s)

	// Create X data from index or use integer sequence
	var xData *array.Array
	var err error

	// Try to convert index to array
	switch idx := s.Index().(type) {
	case *array.Array:
		xData = idx
	case []interface{}:
		// Convert interface{} slice to appropriate numeric array
		if len(idx) > 0 {
			// Try to convert to numeric values
			numericData := make([]float64, len(idx))
			for i, val := range idx {
				switch v := val.(type) {
				case float64:
					numericData[i] = v
				case float32:
					numericData[i] = float64(v)
				case int:
					numericData[i] = float64(v)
				case int64:
					numericData[i] = float64(v)
				case string:
					// For string indices, use position
					numericData[i] = float64(i)
				default:
					numericData[i] = float64(i)
				}
			}
			xData, err = array.FromSlice(numericData)
		}
	default:
		// Fallback: create integer sequence as float64 to match common data types
		sequence := make([]float64, s.Len())
		for i := range sequence {
			sequence[i] = float64(i)
		}
		xData, err = array.FromSlice(sequence)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create x-axis data: %v", err)
	}

	// Add trace to plot
	err = plot.AddTrace(s.Name(), xData, yData)
	if err != nil {
		return nil, fmt.Errorf("failed to add trace: %v", err)
	}

	// Apply styling options
	if len(plot.traces) > 0 {
		trace := &plot.traces[0]
		trace.Options.Color = options.Color
		trace.Options.LineStyle = options.LineStyle
		trace.Options.MarkerStyle = options.MarkerStyle
		trace.Options.Opacity = options.Alpha

		// Configure display based on plot type
		switch options.PlotType {
		case LinePlot:
			trace.Options.ShowLine = true
			trace.Options.ShowMarkers = false
		case ScatterPlot:
			trace.Options.ShowLine = false
			trace.Options.ShowMarkers = true
		case BarPlot:
			trace.Options.ShowLine = false
			trace.Options.ShowMarkers = false
		}
	}

	return plot, nil
}

// Statistical plotting functions for Series

// BoxPlot creates a box plot from Series data
func BoxPlotSeries(s SeriesPlotter, options *SeriesPlotOptions) (*PlotChart, error) {
	// Calculate box plot statistics
	data := getValuesArray(s)
	stats, err := calculateBoxPlotStats(data)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate box plot statistics: %v", err)
	}

	plot := NewPlot(BoxPlot)
	plot.SetTitle(fmt.Sprintf("Box Plot of %s", s.Name()))
	plot.SetYLabel("Value")

	// Create box plot data (simplified representation)
	// In a full implementation, this would create proper box plot traces
	xVals := []float64{0} // Single box
	yVals := []float64{stats.Median}

	xData, _ := array.FromSlice(xVals)
	yData, _ := array.FromSlice(yVals)

	err = plot.AddTrace("Box Plot", xData, yData)
	if err != nil {
		return nil, err
	}

	return plot, nil
}

// HistogramSeries creates a histogram from Series data
func HistogramSeries(s SeriesPlotter, bins int, options *SeriesPlotOptions) (*HistogramPlot, error) {
	if options == nil {
		options = DefaultSeriesPlotOptions()
	}

	hist := NewHistogram(getValuesArray(s), bins)
	hist.title = fmt.Sprintf("Histogram of %s", s.Name())

	if options.XLabel != "" {
		hist.xLabel = options.XLabel
	}
	if options.YLabel != "" {
		hist.yLabel = options.YLabel
	}

	return hist, nil
}

// Box plot statistics
type BoxPlotStats struct {
	Min      float64
	Q1       float64
	Median   float64
	Q3       float64
	Max      float64
	Outliers []float64
}

func calculateBoxPlotStats(data *array.Array) (*BoxPlotStats, error) {
	size := data.Size()
	if size == 0 {
		return nil, fmt.Errorf("cannot calculate box plot statistics for empty data")
	}

	// Convert to float64 slice and sort
	values := make([]float64, size)
	for i := 0; i < size; i++ {
		val := data.At(i)
		switch v := val.(type) {
		case float64:
			values[i] = v
		case float32:
			values[i] = float64(v)
		case int:
			values[i] = float64(v)
		case int64:
			values[i] = float64(v)
		default:
			return nil, fmt.Errorf("unsupported data type for box plot: %T", val)
		}
	}

	// Sort values
	for i := 0; i < len(values)-1; i++ {
		for j := i + 1; j < len(values); j++ {
			if values[i] > values[j] {
				values[i], values[j] = values[j], values[i]
			}
		}
	}

	// Calculate quartiles
	q1Index := size / 4
	medianIndex := size / 2
	q3Index := 3 * size / 4

	stats := &BoxPlotStats{
		Min:    values[0],
		Q1:     values[q1Index],
		Median: values[medianIndex],
		Q3:     values[q3Index],
		Max:    values[size-1],
	}

	// Calculate IQR and outliers
	iqr := stats.Q3 - stats.Q1
	lowerFence := stats.Q1 - 1.5*iqr
	upperFence := stats.Q3 + 1.5*iqr

	for _, val := range values {
		if val < lowerFence || val > upperFence {
			stats.Outliers = append(stats.Outliers, val)
		}
	}

	return stats, nil
}

// TimeSeriesPlot creates a time series plot (for datetime-indexed Series)
func TimeSeriesPlot(s SeriesPlotter, options *SeriesPlotOptions) (*PlotChart, error) {
	if options == nil {
		options = DefaultSeriesPlotOptions()
	}

	// Force line plot for time series
	options.PlotType = LinePlot

	plot, err := PlotSeries(s, options)
	if err != nil {
		return nil, err
	}

	// Additional time series formatting would go here
	plot.SetTitle(fmt.Sprintf("Time Series: %s", s.Name()))
	plot.SetXLabel("Time")

	return plot, nil
}

// Multiple series plotting

// PlotMultipleSeries plots multiple series on the same plot
func PlotMultipleSeries(seriesList []SeriesPlotter, options *SeriesPlotOptions) (*PlotChart, error) {
	if len(seriesList) == 0 {
		return nil, fmt.Errorf("no series provided for plotting")
	}

	if options == nil {
		options = DefaultSeriesPlotOptions()
	}

	plot := NewPlot(options.PlotType)

	if options.Title != "" {
		plot.SetTitle(options.Title)
	} else {
		plot.SetTitle("Multiple Series Plot")
	}

	plot.SetXLabel(options.XLabel)
	plot.SetYLabel(options.YLabel)

	// Add each series as a separate trace
	for i, s := range seriesList {
		yData := getValuesArray(s)

		// Create X data (assuming integer index for simplicity)
		sequence := make([]float64, s.Len())
		for j := range sequence {
			sequence[j] = float64(j)
		}
		xData, err := array.FromSlice(sequence)
		if err != nil {
			return nil, fmt.Errorf("failed to create x-axis data for series %d: %v", i, err)
		}

		err = plot.AddTrace(s.Name(), xData, yData)
		if err != nil {
			return nil, fmt.Errorf("failed to add series %d to plot: %v", i, err)
		}
	}

	return plot, nil
}
