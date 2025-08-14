// Extension methods for Series to add plotting capabilities
package visualization

import (
	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/series"
)

// This file provides plotting extensions for the Series type
// In a real implementation, these would be added to the Series type itself

// SeriesWrapper wraps a Series to implement SeriesPlotter interface
type SeriesWrapper struct {
	*series.Series
}

// WrapSeries wraps a Series to make it compatible with plotting functions
func WrapSeries(s *series.Series) SeriesPlotter {
	return &SeriesWrapper{s}
}

// Data implements SeriesPlotter interface
func (sw *SeriesWrapper) Data() *array.Array {
	return sw.Series.Data()
}

// Index implements SeriesPlotter interface
func (sw *SeriesWrapper) Index() interface{} {
	return sw.Series.Index()
}

// Name implements SeriesPlotter interface
func (sw *SeriesWrapper) Name() string {
	return sw.Series.Name()
}

// Len implements SeriesPlotter interface
func (sw *SeriesWrapper) Len() int {
	return sw.Series.Len()
}

// Plot creates a plot from the wrapped Series
func (sw *SeriesWrapper) Plot(options *SeriesPlotOptions) (*PlotChart, error) {
	return PlotSeries(sw, options)
}

// Hist creates a histogram from the wrapped Series
func (sw *SeriesWrapper) Hist(bins int, options *SeriesPlotOptions) (*HistogramPlot, error) {
	return HistogramSeries(sw, bins, options)
}

// BoxPlot creates a box plot from the wrapped Series
func (sw *SeriesWrapper) BoxPlot(options *SeriesPlotOptions) (*PlotChart, error) {
	return BoxPlotSeries(sw, options)
}
