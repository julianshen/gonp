package visualization

import (
	"encoding/json"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/series"
)

// Test for basic Plot structure and configuration
func TestPlotCreation(t *testing.T) {
	tests := []struct {
		name     string
		plotType PlotType
		title    string
	}{
		{"Line Plot", LinePlot, "Test Line Plot"},
		{"Scatter Plot", ScatterPlot, "Test Scatter Plot"},
		{"Histogram", Histogram, "Test Histogram"},
		{"Heatmap", Heatmap, "Test Heatmap"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plot := NewPlot(tt.plotType)
			plot.SetTitle(tt.title)

			if plot.Type() != tt.plotType {
				t.Errorf("Expected plot type %d, got %d", tt.plotType, plot.Type())
			}

			if plot.Title() != tt.title {
				t.Errorf("Expected title %s, got %s", tt.title, plot.Title())
			}
		})
	}
}

// Test for adding data traces to plots
func TestPlotTraces(t *testing.T) {
	// Create test data
	xData, err := array.FromSlice([]float64{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatalf("Failed to create x data: %v", err)
	}

	yData, err := array.FromSlice([]float64{2, 4, 6, 8, 10})
	if err != nil {
		t.Fatalf("Failed to create y data: %v", err)
	}

	plot := NewPlot(LinePlot)

	// Test adding a trace
	err = plot.AddTrace("line1", xData, yData)
	if err != nil {
		t.Errorf("Failed to add trace: %v", err)
	}

	traces := plot.GetTraces()
	if len(traces) != 1 {
		t.Errorf("Expected 1 trace, got %d", len(traces))
	}

	if traces[0].Name != "line1" {
		t.Errorf("Expected trace name 'line1', got %s", traces[0].Name)
	}
}

// Test for Plotly JSON export functionality
func TestPlotlyExport(t *testing.T) {
	// Create sample data
	x, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
	y, _ := array.FromSlice([]float64{1, 4, 9, 16, 25})

	plot := NewPlot(ScatterPlot)
	plot.SetTitle("Quadratic Function")
	plot.SetXLabel("X Values")
	plot.SetYLabel("Y Values")
	plot.AddTrace("quadratic", x, y)

	// Export to Plotly JSON
	plotlyJSON, err := plot.ToPlotlyJSON()
	if err != nil {
		t.Fatalf("Failed to export to Plotly JSON: %v", err)
	}

	// Validate JSON structure
	var jsonData map[string]interface{}
	err = json.Unmarshal([]byte(plotlyJSON), &jsonData)
	if err != nil {
		t.Errorf("Invalid JSON output: %v", err)
	}

	// Check for required Plotly fields
	if _, exists := jsonData["data"]; !exists {
		t.Error("Missing 'data' field in Plotly JSON")
	}

	if _, exists := jsonData["layout"]; !exists {
		t.Error("Missing 'layout' field in Plotly JSON")
	}
}

// Test for matplotlib-style API
func TestMatplotlibStyleAPI(t *testing.T) {
	// Create sample data
	x, _ := array.FromSlice([]float64{0, 1, 2, 3, 4})
	y, _ := array.FromSlice([]float64{0, 1, 4, 9, 16})

	// Test matplotlib-style functions
	fig := NewFigure()
	ax := fig.AddSubplot(1, 1, 1)

	err := ax.Plot(x, y, &PlotOptions{
		Label:     "x^2",
		Color:     "blue",
		LineStyle: "-",
	})
	if err != nil {
		t.Errorf("Failed to plot with matplotlib-style API: %v", err)
	}

	ax.SetTitle("Quadratic Function")
	ax.SetXLabel("X")
	ax.SetYLabel("Y")
	ax.Legend()

	// Test figure-level operations
	if fig.NumSubplots() != 1 {
		t.Errorf("Expected 1 subplot, got %d", fig.NumSubplots())
	}
}

// Test for statistical visualization
func TestStatisticalPlots(t *testing.T) {
	// Create sample data
	data, _ := array.FromSlice([]float64{1, 2, 2, 3, 3, 3, 4, 4, 5})

	// Test histogram
	hist := NewHistogram(data, 5) // 5 bins
	if hist.NumBins() != 5 {
		t.Errorf("Expected 5 bins, got %d", hist.NumBins())
	}

	// Test that histogram has proper bin counts
	bins := hist.GetBins()
	counts := hist.GetCounts()

	if len(bins) != len(counts)+1 { // bins should have n+1 edges for n bins
		t.Errorf("Bin edges and counts mismatch: %d bins vs %d counts", len(bins), len(counts))
	}
}

// Test for Series visualization integration
func TestSeriesVisualization(t *testing.T) {
	// Create test data as array
	dataSlice := []float64{10.0, 20.0, 15.0, 25.0, 30.0}
	dataArray, err := array.FromSlice(dataSlice)
	if err != nil {
		t.Fatalf("Failed to create array: %v", err)
	}

	// Create a Series
	s, err := series.NewSeries(dataArray, series.NewIndex([]interface{}{"A", "B", "C", "D", "E"}), "Test Series")
	if err != nil {
		t.Fatalf("Failed to create series: %v", err)
	}

	// Wrap the series for plotting
	wrapper := WrapSeries(s)

	// Test bar plot from Series
	plot, err := PlotSeries(wrapper, &SeriesPlotOptions{
		PlotType: BarPlot,
		Title:    "Series Bar Plot",
		Color:    "green",
	})
	if err != nil {
		t.Errorf("Failed to create plot from Series: %v", err)
		return // Exit early to avoid nil pointer access
	}

	if plot == nil {
		t.Error("Plot is nil")
		return
	}

	if plot.Type() != BarPlot {
		t.Errorf("Expected bar plot, got %d", plot.Type())
	}
}

// Test for heatmap functionality
func TestHeatmap(t *testing.T) {
	// Create 2D data for heatmap
	data := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}

	// Convert to array
	flatData := make([]float64, 0, 9)
	for _, row := range data {
		flatData = append(flatData, row...)
	}

	arr, _ := array.FromSlice(flatData)
	arr = arr.Reshape([]int{3, 3})

	heatmap := NewHeatmap(arr)
	heatmap.SetTitle("Test Heatmap")
	heatmap.SetColormap("viridis")

	if heatmap.Title() != "Test Heatmap" {
		t.Errorf("Expected title 'Test Heatmap', got %s", heatmap.Title())
	}

	if heatmap.Colormap() != "viridis" {
		t.Errorf("Expected colormap 'viridis', got %s", heatmap.Colormap())
	}
}

// Test for 3D surface plots
func TestSurfacePlot(t *testing.T) {
	// Create mesh grid data
	x := []float64{1, 2, 3}
	y := []float64{1, 2, 3}

	// Create Z data (function of X, Y)
	z := make([][]float64, len(y))
	for i := range y {
		z[i] = make([]float64, len(x))
		for j := range x {
			z[i][j] = x[j] * y[i] // Simple multiplication
		}
	}

	surface := NewSurface3D(x, y, z)
	surface.SetTitle("3D Surface")
	surface.SetColormap("plasma")

	// Test Plotly export for 3D
	json, err := surface.ToPlotlyJSON()
	if err != nil {
		t.Errorf("Failed to export 3D surface to Plotly JSON: %v", err)
	}

	if len(json) == 0 {
		t.Error("Empty JSON output for 3D surface")
	}
}

// Benchmark tests
func BenchmarkPlotlyExport(b *testing.B) {
	// Create large dataset
	size := 1000
	xData := make([]float64, size)
	yData := make([]float64, size)

	for i := 0; i < size; i++ {
		xData[i] = float64(i)
		yData[i] = float64(i * i)
	}

	x, _ := array.FromSlice(xData)
	y, _ := array.FromSlice(yData)

	plot := NewPlot(LinePlot)
	plot.AddTrace("benchmark", x, y)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plot.ToPlotlyJSON()
	}
}

func BenchmarkHistogramCalculation(b *testing.B) {
	// Create random-ish data
	size := 10000
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = float64(i%100) + float64(i%7)*0.1 // Pseudo-random pattern
	}

	arr, _ := array.FromSlice(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewHistogram(arr, 50)
	}
}
