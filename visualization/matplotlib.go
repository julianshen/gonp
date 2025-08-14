// Matplotlib-style plotting API for GoNP
package visualization

import (
	"fmt"
	"github.com/julianshen/gonp/array"
)

// Figure represents a matplotlib-style figure containing subplots
type Figure struct {
	subplots []Subplot
	figsize  [2]int // width, height
	dpi      int
	title    string
}

// Subplot represents a single plot within a figure
type Subplot struct {
	plot     *PlotChart
	position SubplotPosition
	figure   *Figure
}

// SubplotPosition defines the position of a subplot in a grid
type SubplotPosition struct {
	Rows  int
	Cols  int
	Index int
}

// PlotOptions configures plot appearance
type PlotOptions struct {
	Label      string
	Color      string
	LineStyle  string
	LineWidth  float64
	Marker     string
	MarkerSize float64
	Alpha      float64
}

// NewFigure creates a new matplotlib-style figure
func NewFigure() *Figure {
	return &Figure{
		subplots: make([]Subplot, 0),
		figsize:  [2]int{800, 600},
		dpi:      100,
	}
}

// SetFigsize sets the figure size in pixels
func (f *Figure) SetFigsize(width, height int) {
	f.figsize = [2]int{width, height}
}

// SetTitle sets the figure title
func (f *Figure) SetTitle(title string) {
	f.title = title
}

// AddSubplot adds a subplot to the figure
func (f *Figure) AddSubplot(rows, cols, index int) *Subplot {
	plotChart := NewPlot(LinePlot)

	subplot := Subplot{
		plot: plotChart,
		position: SubplotPosition{
			Rows:  rows,
			Cols:  cols,
			Index: index,
		},
		figure: f,
	}

	f.subplots = append(f.subplots, subplot)
	return &f.subplots[len(f.subplots)-1]
}

// NumSubplots returns the number of subplots in the figure
func (f *Figure) NumSubplots() int {
	return len(f.subplots)
}

// GetSubplot returns a specific subplot by index
func (f *Figure) GetSubplot(index int) *Subplot {
	if index < 0 || index >= len(f.subplots) {
		return nil
	}
	return &f.subplots[index]
}

// SaveFig saves the figure to a file (placeholder implementation)
func (f *Figure) SaveFig(filename string, options *SaveOptions) error {
	// This would integrate with actual image generation libraries
	// For now, we'll return a placeholder
	return fmt.Errorf("SaveFig not yet implemented - would save to %s", filename)
}

// Show displays the figure (placeholder implementation)
func (f *Figure) Show() error {
	// This would integrate with display systems
	return fmt.Errorf("Show not yet implemented - would display figure")
}

// SaveOptions configures figure saving
type SaveOptions struct {
	DPI         int
	Format      string // "png", "pdf", "svg", etc.
	Quality     int    // For JPEG
	Transparent bool
	BBox        string // "tight", "standard"
}

// Subplot methods (matplotlib-style API)

// Plot creates a line plot
func (s *Subplot) Plot(x, y *array.Array, options *PlotOptions) error {
	if options == nil {
		options = &PlotOptions{
			Color:     "blue",
			LineStyle: "-",
			LineWidth: 2.0,
		}
	}

	// Set plot type based on options
	if options.LineStyle == "" || options.LineStyle == "none" {
		s.plot.plotType = ScatterPlot
	} else {
		s.plot.plotType = LinePlot
	}

	err := s.plot.AddTrace(options.Label, x, y)
	if err != nil {
		return err
	}

	// Update trace options
	if len(s.plot.traces) > 0 {
		trace := &s.plot.traces[len(s.plot.traces)-1]
		trace.Options.Color = options.Color
		trace.Options.LineStyle = options.LineStyle
		trace.Options.Width = options.LineWidth
		trace.Options.Opacity = options.Alpha
		if options.Alpha == 0 {
			trace.Options.Opacity = 1.0
		}

		// Configure line/marker display
		trace.Options.ShowLine = options.LineStyle != "" && options.LineStyle != "none"
		trace.Options.ShowMarkers = options.Marker != "" && options.Marker != "none"
	}

	return nil
}

// Scatter creates a scatter plot
func (s *Subplot) Scatter(x, y *array.Array, options *PlotOptions) error {
	if options == nil {
		options = &PlotOptions{
			Color:  "blue",
			Marker: "o",
		}
	}

	s.plot.plotType = ScatterPlot
	err := s.plot.AddTrace(options.Label, x, y)
	if err != nil {
		return err
	}

	// Update trace options for scatter plot
	if len(s.plot.traces) > 0 {
		trace := &s.plot.traces[len(s.plot.traces)-1]
		trace.Options.Color = options.Color
		trace.Options.MarkerStyle = options.Marker
		trace.Options.Opacity = options.Alpha
		if options.Alpha == 0 {
			trace.Options.Opacity = 1.0
		}
		trace.Options.ShowLine = false
		trace.Options.ShowMarkers = true
	}

	return nil
}

// Bar creates a bar plot
func (s *Subplot) Bar(x, height *array.Array, options *PlotOptions) error {
	if options == nil {
		options = &PlotOptions{
			Color: "blue",
		}
	}

	s.plot.plotType = BarPlot
	err := s.plot.AddTrace(options.Label, x, height)
	if err != nil {
		return err
	}

	// Update trace options for bar plot
	if len(s.plot.traces) > 0 {
		trace := &s.plot.traces[len(s.plot.traces)-1]
		trace.Options.Color = options.Color
		trace.Options.Opacity = options.Alpha
		if options.Alpha == 0 {
			trace.Options.Opacity = 1.0
		}
	}

	return nil
}

// Hist creates a histogram
func (s *Subplot) Hist(data *array.Array, bins int, options *PlotOptions) error {
	if options == nil {
		options = &PlotOptions{
			Color: "blue",
		}
	}

	hist := NewHistogram(data, bins)

	// Convert histogram to bar plot data
	binCenters := make([]float64, len(hist.counts))
	counts := make([]float64, len(hist.counts))

	for i, count := range hist.counts {
		binCenters[i] = (hist.bins[i] + hist.bins[i+1]) / 2.0
		counts[i] = float64(count)
	}

	xArray, _ := array.FromSlice(binCenters)
	yArray, _ := array.FromSlice(counts)

	s.plot.plotType = BarPlot
	return s.plot.AddTrace(options.Label, xArray, yArray)
}

// SetTitle sets the subplot title
func (s *Subplot) SetTitle(title string) {
	s.plot.SetTitle(title)
}

// SetXLabel sets the X-axis label
func (s *Subplot) SetXLabel(label string) {
	s.plot.SetXLabel(label)
}

// SetYLabel sets the Y-axis label
func (s *Subplot) SetYLabel(label string) {
	s.plot.SetYLabel(label)
}

// SetXLim sets the X-axis limits (placeholder)
func (s *Subplot) SetXLim(min, max float64) {
	// Would configure X-axis range
}

// SetYLim sets the Y-axis limits (placeholder)
func (s *Subplot) SetYLim(min, max float64) {
	// Would configure Y-axis range
}

// Grid toggles the grid display
func (s *Subplot) Grid(show bool) {
	s.plot.layout.Grid.Show = show
}

// Legend displays the legend
func (s *Subplot) Legend() {
	// Would configure legend display
}

// GetPlot returns the underlying PlotChart object
func (s *Subplot) GetPlot() *PlotChart {
	return s.plot
}

// Convenience functions (module-level matplotlib-style functions)

var defaultFigure *Figure
var defaultSubplot *Subplot

// InitializeDefault initializes the default figure and subplot
func InitializeDefault() {
	defaultFigure = NewFigure()
	defaultSubplot = defaultFigure.AddSubplot(1, 1, 1)
}

// Plot creates a line plot on the default subplot
func Plot(x, y *array.Array, options *PlotOptions) error {
	if defaultSubplot == nil {
		InitializeDefault()
	}
	return defaultSubplot.Plot(x, y, options)
}

// Scatter creates a scatter plot on the default subplot
func Scatter(x, y *array.Array, options *PlotOptions) error {
	if defaultSubplot == nil {
		InitializeDefault()
	}
	return defaultSubplot.Scatter(x, y, options)
}

// Bar creates a bar plot on the default subplot
func Bar(x, height *array.Array, options *PlotOptions) error {
	if defaultSubplot == nil {
		InitializeDefault()
	}
	return defaultSubplot.Bar(x, height, options)
}

// Hist creates a histogram on the default subplot
func Hist(data *array.Array, bins int, options *PlotOptions) error {
	if defaultSubplot == nil {
		InitializeDefault()
	}
	return defaultSubplot.Hist(data, bins, options)
}

// Title sets the title of the default subplot
func Title(title string) {
	if defaultSubplot == nil {
		InitializeDefault()
	}
	defaultSubplot.SetTitle(title)
}

// Xlabel sets the X-axis label of the default subplot
func Xlabel(label string) {
	if defaultSubplot == nil {
		InitializeDefault()
	}
	defaultSubplot.SetXLabel(label)
}

// Ylabel sets the Y-axis label of the default subplot
func Ylabel(label string) {
	if defaultSubplot == nil {
		InitializeDefault()
	}
	defaultSubplot.SetYLabel(label)
}

// Grid toggles the grid on the default subplot
func Grid(show bool) {
	if defaultSubplot == nil {
		InitializeDefault()
	}
	defaultSubplot.Grid(show)
}

// Legend displays the legend on the default subplot
func Legend() {
	if defaultSubplot == nil {
		InitializeDefault()
	}
	defaultSubplot.Legend()
}

// Show displays the default figure
func Show() error {
	if defaultFigure == nil {
		InitializeDefault()
	}
	return defaultFigure.Show()
}

// SaveFig saves the default figure
func SaveFig(filename string, options *SaveOptions) error {
	if defaultFigure == nil {
		InitializeDefault()
	}
	return defaultFigure.SaveFig(filename, options)
}

// GetCurrentFigure returns the current default figure
func GetCurrentFigure() *Figure {
	if defaultFigure == nil {
		InitializeDefault()
	}
	return defaultFigure
}
