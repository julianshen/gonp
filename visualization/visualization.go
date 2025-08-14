// Package visualization provides plotting and charting capabilities for GoNP
package visualization

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/julianshen/gonp/array"
)

// PlotType represents different types of plots
type PlotType int

const (
	LinePlot PlotType = iota
	ScatterPlot
	BarPlot
	Histogram
	Heatmap
	Surface3D
	ContourPlot
	BoxPlot
	ViolinPlot
)

// PlotChart represents a generic plot with data traces
type PlotChart struct {
	plotType PlotType
	title    string
	xLabel   string
	yLabel   string
	traces   []Trace
	layout   PlotLayout
}

// Trace represents a data series in a plot
type Trace struct {
	Name    string
	XData   *array.Array
	YData   *array.Array
	ZData   *array.Array // For 3D plots
	Type    PlotType
	Options TraceOptions
}

// TraceOptions configures the appearance of a trace
type TraceOptions struct {
	Color       string
	LineStyle   string
	MarkerStyle string
	Width       float64
	Opacity     float64
	Fill        string
	ShowLine    bool
	ShowMarkers bool
}

// PlotLayout configures the overall plot layout
type PlotLayout struct {
	Width      int
	Height     int
	Background string
	Grid       GridOptions
	Margin     MarginOptions
	Font       FontOptions
}

// GridOptions configures plot grid
type GridOptions struct {
	Show      bool
	Color     string
	LineWidth float64
}

// MarginOptions configures plot margins
type MarginOptions struct {
	Left   int
	Right  int
	Top    int
	Bottom int
}

// FontOptions configures text appearance
type FontOptions struct {
	Family string
	Size   int
	Color  string
}

// NewPlot creates a new plot of the specified type
func NewPlot(plotType PlotType) *PlotChart {
	return &PlotChart{
		plotType: plotType,
		traces:   make([]Trace, 0),
		layout: PlotLayout{
			Width:  800,
			Height: 600,
			Grid: GridOptions{
				Show:      true,
				Color:     "#e0e0e0",
				LineWidth: 0.5,
			},
			Margin: MarginOptions{
				Left:   60,
				Right:  40,
				Top:    40,
				Bottom: 60,
			},
			Font: FontOptions{
				Family: "Arial, sans-serif",
				Size:   12,
				Color:  "#333333",
			},
		},
	}
}

// Type returns the plot type
func (p *PlotChart) Type() PlotType {
	return p.plotType
}

// SetTitle sets the plot title
func (p *PlotChart) SetTitle(title string) {
	p.title = title
}

// Title returns the plot title
func (p *PlotChart) Title() string {
	return p.title
}

// SetXLabel sets the X-axis label
func (p *PlotChart) SetXLabel(label string) {
	p.xLabel = label
}

// XLabel returns the X-axis label
func (p *PlotChart) XLabel() string {
	return p.xLabel
}

// SetYLabel sets the Y-axis label
func (p *PlotChart) SetYLabel(label string) {
	p.yLabel = label
}

// YLabel returns the Y-axis label
func (p *PlotChart) YLabel() string {
	return p.yLabel
}

// AddTrace adds a new data trace to the plot
func (p *PlotChart) AddTrace(name string, xData, yData *array.Array) error {
	if xData.Size() != yData.Size() {
		return fmt.Errorf("X and Y data must have the same size: %d vs %d",
			xData.Size(), yData.Size())
	}

	trace := Trace{
		Name:  name,
		XData: xData,
		YData: yData,
		Type:  p.plotType,
		Options: TraceOptions{
			Color:       p.getDefaultColor(len(p.traces)),
			LineStyle:   "-",
			Width:       2.0,
			Opacity:     1.0,
			ShowLine:    p.plotType == LinePlot,
			ShowMarkers: p.plotType == ScatterPlot,
		},
	}

	p.traces = append(p.traces, trace)
	return nil
}

// GetTraces returns all traces in the plot
func (p *PlotChart) GetTraces() []Trace {
	return p.traces
}

// SetLayout configures the plot layout
func (p *PlotChart) SetLayout(layout PlotLayout) {
	p.layout = layout
}

// GetLayout returns the current plot layout
func (p *PlotChart) GetLayout() PlotLayout {
	return p.layout
}

// ToPlotlyJSON exports the plot as Plotly JSON format
func (p *PlotChart) ToPlotlyJSON() (string, error) {
	plotlyData := make(map[string]interface{})

	// Convert traces to Plotly format
	dataArray := make([]map[string]interface{}, 0, len(p.traces))

	for _, trace := range p.traces {
		traceData := map[string]interface{}{
			"name": trace.Name,
			"x":    p.arrayToSlice(trace.XData),
			"y":    p.arrayToSlice(trace.YData),
		}

		// Set trace type based on plot type
		switch p.plotType {
		case LinePlot:
			traceData["type"] = "scatter"
			traceData["mode"] = "lines"
		case ScatterPlot:
			traceData["type"] = "scatter"
			traceData["mode"] = "markers"
		case BarPlot:
			traceData["type"] = "bar"
		case Heatmap:
			traceData["type"] = "heatmap"
			if trace.ZData != nil {
				traceData["z"] = p.array2DToSlice(trace.ZData)
			}
		}

		// Add styling options
		if trace.Options.Color != "" {
			traceData["line"] = map[string]interface{}{
				"color": trace.Options.Color,
				"width": trace.Options.Width,
			}
		}

		dataArray = append(dataArray, traceData)
	}

	plotlyData["data"] = dataArray

	// Create layout
	layout := map[string]interface{}{
		"title": map[string]interface{}{
			"text": p.title,
			"font": map[string]interface{}{
				"family": p.layout.Font.Family,
				"size":   p.layout.Font.Size,
				"color":  p.layout.Font.Color,
			},
		},
		"xaxis": map[string]interface{}{
			"title":     map[string]string{"text": p.xLabel},
			"showgrid":  p.layout.Grid.Show,
			"gridcolor": p.layout.Grid.Color,
		},
		"yaxis": map[string]interface{}{
			"title":     map[string]string{"text": p.yLabel},
			"showgrid":  p.layout.Grid.Show,
			"gridcolor": p.layout.Grid.Color,
		},
		"width":  p.layout.Width,
		"height": p.layout.Height,
		"margin": map[string]int{
			"l": p.layout.Margin.Left,
			"r": p.layout.Margin.Right,
			"t": p.layout.Margin.Top,
			"b": p.layout.Margin.Bottom,
		},
	}

	plotlyData["layout"] = layout

	jsonBytes, err := json.Marshal(plotlyData)
	if err != nil {
		return "", fmt.Errorf("failed to marshal Plotly JSON: %v", err)
	}

	return string(jsonBytes), nil
}

// Helper methods

func (p *PlotChart) getDefaultColor(index int) string {
	colors := []string{
		"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
		"#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
	}
	return colors[index%len(colors)]
}

func (p *PlotChart) arrayToSlice(arr *array.Array) []interface{} {
	size := arr.Size()
	slice := make([]interface{}, size)

	for i := 0; i < size; i++ {
		slice[i] = arr.At(i)
	}

	return slice
}

func (p *PlotChart) array2DToSlice(arr *array.Array) [][]interface{} {
	shape := arr.Shape()
	if len(shape) != 2 {
		return nil
	}

	rows, cols := shape[0], shape[1]
	slice := make([][]interface{}, rows)

	for i := 0; i < rows; i++ {
		slice[i] = make([]interface{}, cols)
		for j := 0; j < cols; j++ {
			slice[i][j] = arr.At(i, j)
		}
	}

	return slice
}

// HistogramPlot represents a histogram plot
type HistogramPlot struct {
	data     *array.Array
	numBins  int
	bins     []float64
	counts   []int
	title    string
	xLabel   string
	yLabel   string
	colormap string
}

// NewHistogram creates a new histogram from data
func NewHistogram(data *array.Array, numBins int) *HistogramPlot {
	hist := &HistogramPlot{
		data:    data,
		numBins: numBins,
		title:   "Histogram",
		xLabel:  "Value",
		yLabel:  "Frequency",
	}

	hist.calculateBins()
	return hist
}

// NumBins returns the number of histogram bins
func (h *HistogramPlot) NumBins() int {
	return h.numBins
}

// GetBins returns the bin edges
func (h *HistogramPlot) GetBins() []float64 {
	return h.bins
}

// GetCounts returns the bin counts
func (h *HistogramPlot) GetCounts() []int {
	return h.counts
}

func (h *HistogramPlot) calculateBins() {
	size := h.data.Size()
	if size == 0 {
		return
	}

	// Find min and max values
	minVal, maxVal := math.Inf(1), math.Inf(-1)

	for i := 0; i < size; i++ {
		val := h.convertToFloat64(h.data.At(i))
		if val < minVal {
			minVal = val
		}
		if val > maxVal {
			maxVal = val
		}
	}

	// Create bin edges
	h.bins = make([]float64, h.numBins+1)
	binWidth := (maxVal - minVal) / float64(h.numBins)

	for i := 0; i <= h.numBins; i++ {
		h.bins[i] = minVal + float64(i)*binWidth
	}

	// Count values in each bin
	h.counts = make([]int, h.numBins)

	for i := 0; i < size; i++ {
		val := h.convertToFloat64(h.data.At(i))
		binIndex := int((val - minVal) / binWidth)

		// Handle edge case where value equals maxVal
		if binIndex >= h.numBins {
			binIndex = h.numBins - 1
		}

		if binIndex >= 0 && binIndex < h.numBins {
			h.counts[binIndex]++
		}
	}
}

func (h *HistogramPlot) convertToFloat64(value interface{}) float64 {
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	default:
		return 0.0
	}
}

// Heatmap represents a heatmap plot
type HeatmapPlot struct {
	data     *array.Array
	title    string
	colormap string
	xLabels  []string
	yLabels  []string
}

// NewHeatmap creates a new heatmap plot
func NewHeatmap(data *array.Array) *HeatmapPlot {
	return &HeatmapPlot{
		data:     data,
		title:    "Heatmap",
		colormap: "viridis",
	}
}

// SetTitle sets the heatmap title
func (h *HeatmapPlot) SetTitle(title string) {
	h.title = title
}

// Title returns the heatmap title
func (h *HeatmapPlot) Title() string {
	return h.title
}

// SetColormap sets the color scheme
func (h *HeatmapPlot) SetColormap(colormap string) {
	h.colormap = colormap
}

// Colormap returns the current colormap
func (h *HeatmapPlot) Colormap() string {
	return h.colormap
}

// Surface3DPlot represents a 3D surface plot
type Surface3DPlot struct {
	x        []float64
	y        []float64
	z        [][]float64
	title    string
	colormap string
}

// NewSurface3D creates a new 3D surface plot
func NewSurface3D(x, y []float64, z [][]float64) *Surface3DPlot {
	return &Surface3DPlot{
		x:        x,
		y:        y,
		z:        z,
		title:    "3D Surface",
		colormap: "viridis",
	}
}

// SetTitle sets the surface plot title
func (s *Surface3DPlot) SetTitle(title string) {
	s.title = title
}

// Title returns the surface plot title
func (s *Surface3DPlot) Title() string {
	return s.title
}

// SetColormap sets the color scheme
func (s *Surface3DPlot) SetColormap(colormap string) {
	s.colormap = colormap
}

// Colormap returns the current colormap
func (s *Surface3DPlot) Colormap() string {
	return s.colormap
}

// ToPlotlyJSON exports the 3D surface as Plotly JSON
func (s *Surface3DPlot) ToPlotlyJSON() (string, error) {
	plotlyData := map[string]interface{}{
		"data": []map[string]interface{}{
			{
				"type":       "surface",
				"x":          s.x,
				"y":          s.y,
				"z":          s.z,
				"colorscale": s.colormap,
			},
		},
		"layout": map[string]interface{}{
			"title": map[string]string{"text": s.title},
			"scene": map[string]interface{}{
				"xaxis": map[string]string{"title": "X"},
				"yaxis": map[string]string{"title": "Y"},
				"zaxis": map[string]string{"title": "Z"},
			},
		},
	}

	jsonBytes, err := json.Marshal(plotlyData)
	if err != nil {
		return "", fmt.Errorf("failed to marshal 3D surface JSON: %v", err)
	}

	return string(jsonBytes), nil
}
