// Demonstration of GoNP visualization capabilities
package main

import (
	"fmt"
	"log"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/series"
	"github.com/julianshen/gonp/visualization"
)

func main() {
	fmt.Println("GoNP Visualization Demo")
	fmt.Println("=======================")

	// Create sample data
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{2, 4, 6, 8, 10}

	xArray, err := array.FromSlice(x)
	if err != nil {
		log.Fatal("Failed to create x array:", err)
	}

	yArray, err := array.FromSlice(y)
	if err != nil {
		log.Fatal("Failed to create y array:", err)
	}

	// 1. Basic matplotlib-style plotting
	fmt.Println("\n1. Matplotlib-style API Demo:")
	fig := visualization.NewFigure()
	ax := fig.AddSubplot(1, 1, 1)

	err = ax.Plot(xArray, yArray, &visualization.PlotOptions{
		Label:     "Linear Data",
		Color:     "blue",
		LineStyle: "-",
	})
	if err != nil {
		log.Printf("Plot error: %v", err)
	} else {
		fmt.Println("✓ Created line plot with matplotlib-style API")
	}

	// 2. Plotly JSON export
	fmt.Println("\n2. Plotly JSON Export Demo:")
	plot := visualization.NewPlot(visualization.LinePlot)
	plot.SetTitle("Sample Line Plot")
	plot.SetXLabel("X Values")
	plot.SetYLabel("Y Values")

	err = plot.AddTrace("Sample Data", xArray, yArray)
	if err != nil {
		log.Printf("Add trace error: %v", err)
	} else {
		json, err := plot.ToPlotlyJSON()
		if err != nil {
			log.Printf("JSON export error: %v", err)
		} else {
			fmt.Printf("✓ Generated Plotly JSON (%d characters)\n", len(json))
		}
	}

	// 3. Histogram
	fmt.Println("\n3. Histogram Demo:")
	data := []float64{1, 2, 2, 3, 3, 3, 4, 4, 5}
	dataArray, _ := array.FromSlice(data)

	hist := visualization.NewHistogram(dataArray, 5)
	bins := hist.GetBins()
	counts := hist.GetCounts()

	fmt.Printf("✓ Created histogram with %d bins\n", hist.NumBins())
	fmt.Printf("  Bin edges: %v\n", bins[:3]) // Show first 3
	fmt.Printf("  Counts: %v\n", counts)

	// 4. Series integration
	fmt.Println("\n4. Series Integration Demo:")
	seriesData := []float64{10, 20, 15, 25, 30}
	seriesArray, _ := array.FromSlice(seriesData)
	index := []interface{}{"A", "B", "C", "D", "E"}

	s, err := series.NewSeries(seriesArray, series.NewIndex(index), "Sample Series")
	if err != nil {
		log.Printf("Series creation error: %v", err)
	} else {
		wrapper := visualization.WrapSeries(s)
		plot, err := visualization.PlotSeries(wrapper, &visualization.SeriesPlotOptions{
			PlotType: visualization.BarPlot,
			Title:    "Series Bar Plot",
			Color:    "green",
		})
		if err != nil {
			log.Printf("Series plot error: %v", err)
		} else {
			fmt.Printf("✓ Created bar plot from Series (%s)\n", plot.Title())
		}
	}

	// 5. 3D Surface plot
	fmt.Println("\n5. 3D Surface Plot Demo:")
	xSurf := []float64{1, 2, 3}
	ySurf := []float64{1, 2, 3}
	zSurf := [][]float64{
		{1, 2, 3},
		{2, 4, 6},
		{3, 6, 9},
	}

	surface := visualization.NewSurface3D(xSurf, ySurf, zSurf)
	surface.SetTitle("3D Surface Example")
	surface.SetColormap("viridis")

	surfaceJSON, err := surface.ToPlotlyJSON()
	if err != nil {
		log.Printf("3D surface error: %v", err)
	} else {
		fmt.Printf("✓ Created 3D surface plot JSON (%d characters)\n", len(surfaceJSON))
	}

	fmt.Println("\n✅ All visualization features demonstrated successfully!")
}
