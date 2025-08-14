//go:build ignore
// +build ignore

// Package examples demonstrates data analysis workflows with GoNP
//
// This example shows a complete data analysis pipeline including:
// - Data loading and cleaning
// - Exploratory data analysis
// - Statistical analysis
// - Data visualization (text-based)
//
// Run with: go run examples/data_analysis.go
package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/series"
	"github.com/julianshen/gonp/stats"
)

func main() {
	fmt.Println("=== GoNP Data Analysis Example ===\n")

	// Simulate loading sales data
	salesData := generateSalesData()

	// Perform exploratory data analysis
	exploreData(salesData)

	// Statistical analysis
	statisticalAnalysis(salesData)

	// Trend analysis
	trendAnalysis(salesData)

	// Summary and insights
	generateInsights(salesData)
}

// SalesRecord represents a sales transaction
type SalesRecord struct {
	Month     int
	Sales     float64
	Marketing float64
	Staff     int
}

// generateSalesData simulates loading data from a file or database
func generateSalesData() *dataframe.DataFrame {
	fmt.Println("1. Loading Sales Data")
	fmt.Println("---------------------")

	// Simulate 12 months of sales data
	months := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	sales := []float64{45000, 48000, 52000, 49000, 55000, 58000, 62000, 65000, 59000, 61000, 67000, 72000}
	marketing := []float64{5000, 5200, 5800, 5100, 6200, 6500, 7000, 7200, 6800, 6900, 7500, 8000}
	staff := []float64{5, 5, 6, 6, 6, 7, 7, 8, 7, 7, 8, 8}

	// Create arrays
	monthArr, _ := array.FromSlice(months)
	salesArr, _ := array.FromSlice(sales)
	marketingArr, _ := array.FromSlice(marketing)
	staffArr, _ := array.FromSlice(staff)

	// Create series
	monthSeries, _ := series.NewSeries(monthArr, nil, "month")
	salesSeries, _ := series.NewSeries(salesArr, nil, "sales")
	marketingSeries, _ := series.NewSeries(marketingArr, nil, "marketing")
	staffSeries, _ := series.NewSeries(staffArr, nil, "staff")

	// Create DataFrame
	df, err := dataframe.FromSeries([]*series.Series{monthSeries, salesSeries, marketingSeries, staffSeries})
	if err != nil {
		log.Fatal("Failed to create DataFrame:", err)
	}

	fmt.Printf("Loaded data: %d months × %d metrics\n", df.Len(), len(df.Columns()))
	fmt.Printf("Columns: %v\n", df.Columns())
	fmt.Println()

	return df
}

// exploreData performs exploratory data analysis
func exploreData(df *dataframe.DataFrame) {
	fmt.Println("2. Exploratory Data Analysis")
	fmt.Println("-----------------------------")

	// Show data shape and basic info
	fmt.Printf("Dataset shape: %d rows × %d columns\n", df.Len(), len(df.Columns()))

	// Display first few records
	fmt.Println("\nFirst 5 months:")
	fmt.Printf("%-6s %-8s %-10s %-6s\n", "Month", "Sales", "Marketing", "Staff")
	fmt.Println(strings.Repeat("-", 32))

	for i := 0; i < min(5, df.Len()); i++ {
		month, _ := df.IAt(i, 0)
		sales, _ := df.IAt(i, 1)
		marketing, _ := df.IAt(i, 2)
		staff, _ := df.IAt(i, 3)

		fmt.Printf("%-6.0f $%-7.0f $%-9.0f %-6.0f\n",
			month.(float64), sales.(float64), marketing.(float64), staff.(float64))
	}

	// Calculate basic statistics for each column
	fmt.Println("\nSummary Statistics:")
	columns := []string{"sales", "marketing", "staff"}

	for _, colName := range columns {
		column, _ := df.GetColumn(colName)
		data := column.Data()

		mean, _ := stats.Mean(data)
		median, _ := stats.Median(data)
		std, _ := stats.Std(data)
		min, _ := stats.Min(data)
		max, _ := stats.Max(data)

		fmt.Printf("\n%s:\n", colName)
		fmt.Printf("  Mean:   $%8.0f\n", mean)
		fmt.Printf("  Median: $%8.0f\n", median)
		fmt.Printf("  Std:    $%8.0f\n", std)
		fmt.Printf("  Min:    $%8.0f\n", min)
		fmt.Printf("  Max:    $%8.0f\n", max)
	}

	fmt.Println()
}

// statisticalAnalysis performs correlation and regression analysis
func statisticalAnalysis(df *dataframe.DataFrame) {
	fmt.Println("3. Statistical Analysis")
	fmt.Println("-----------------------")

	// Get data columns
	salesColumn, _ := df.GetColumn("sales")
	marketingColumn, _ := df.GetColumn("marketing")
	staffColumn, _ := df.GetColumn("staff")

	marketingData := marketingColumn.Data()
	staffData := staffColumn.Data()

	// Correlation analysis
	fmt.Println("Correlation Analysis:")

	// Sales vs Marketing
	corrSalesMarketing, _ := stats.Correlation(salesColumn.Data(), marketingData)
	fmt.Printf("Sales ↔ Marketing:  %6.3f", corrSalesMarketing)
	interpretCorrelation(corrSalesMarketing)

	// Sales vs Staff
	corrSalesStaff, _ := stats.Correlation(salesColumn.Data(), staffData)
	fmt.Printf("Sales ↔ Staff:      %6.3f", corrSalesStaff)
	interpretCorrelation(corrSalesStaff)

	// Marketing vs Staff
	corrMarketingStaff, _ := stats.Correlation(marketingData, staffData)
	fmt.Printf("Marketing ↔ Staff:  %6.3f", corrMarketingStaff)
	interpretCorrelation(corrMarketingStaff)

	// Calculate marketing ROI
	fmt.Println("\nMarketing ROI Analysis:")
	avgSales, _ := stats.Mean(salesColumn.Data())
	avgMarketing, _ := stats.Mean(marketingData)
	roi := (avgSales - avgMarketing) / avgMarketing * 100

	fmt.Printf("Average Monthly Sales:     $%8.0f\n", avgSales)
	fmt.Printf("Average Monthly Marketing: $%8.0f\n", avgMarketing)
	fmt.Printf("Marketing ROI:             %8.1f%%\n", roi)

	fmt.Println()
}

// trendAnalysis analyzes trends over time
func trendAnalysis(df *dataframe.DataFrame) {
	fmt.Println("4. Trend Analysis")
	fmt.Println("-----------------")

	// Calculate month-over-month growth
	fmt.Println("Month-over-Month Sales Growth:")
	fmt.Printf("%-6s %-10s %-8s\n", "Month", "Sales", "Growth%")
	fmt.Println(strings.Repeat("-", 26))

	var totalGrowth float64
	growthCount := 0

	for i := 0; i < df.Len(); i++ {
		month, _ := df.IAt(i, 0)
		sales, _ := df.IAt(i, 1)

		if i == 0 {
			fmt.Printf("%-6.0f $%-9.0f %8s\n", month.(float64), sales.(float64), "-")
		} else {
			prevSales, _ := df.IAt(i-1, 1)
			growth := (sales.(float64) - prevSales.(float64)) / prevSales.(float64) * 100
			totalGrowth += growth
			growthCount++

			fmt.Printf("%-6.0f $%-9.0f %7.1f%%\n", month.(float64), sales.(float64), growth)
		}
	}

	avgGrowth := totalGrowth / float64(growthCount)
	fmt.Printf("\nAverage Monthly Growth: %.1f%%\n", avgGrowth)

	// Seasonal analysis (quarters)
	fmt.Println("\nQuarterly Analysis:")
	quarters := []string{"Q1", "Q2", "Q3", "Q4"}

	for q := 0; q < 4; q++ {
		start := q * 3
		end := start + 3
		if end > df.Len() {
			end = df.Len()
		}

		quarterData := make([]float64, end-start)
		for i := start; i < end; i++ {
			sales, _ := df.IAt(i, 1)
			quarterData[i-start] = sales.(float64)
		}

		quarterArr, _ := array.FromSlice(quarterData)
		quarterMean, _ := stats.Mean(quarterArr)
		quarterStd, _ := stats.Std(quarterArr)

		fmt.Printf("%s: Mean $%6.0f, Std $%6.0f\n", quarters[q], quarterMean, quarterStd)
	}

	fmt.Println()
}

// generateInsights provides business insights based on analysis
func generateInsights(df *dataframe.DataFrame) {
	fmt.Println("5. Business Insights")
	fmt.Println("--------------------")

	// Get final quarter data for year-end analysis
	lastQuarter, _ := df.Slice(9, 12) // Last 3 months

	salesColumn, _ := df.GetColumn("sales")
	lastQSalesColumn, _ := lastQuarter.GetColumn("sales")

	yearAvg, _ := stats.Mean(salesColumn.Data())
	lastQAvg, _ := stats.Mean(lastQSalesColumn.Data())

	fmt.Printf("📊 Key Findings:\n\n")

	// Sales performance
	if lastQAvg > yearAvg {
		fmt.Printf("✅ Strong finish: Q4 average ($%.0f) exceeded yearly average ($%.0f)\n",
			lastQAvg, yearAvg)
	} else {
		fmt.Printf("⚠️  Slower finish: Q4 average ($%.0f) below yearly average ($%.0f)\n",
			lastQAvg, yearAvg)
	}

	// Growth trajectory
	firstMonth, _ := df.IAt(0, 1)
	lastMonth, _ := df.IAt(df.Len()-1, 1)
	yearGrowth := (lastMonth.(float64) - firstMonth.(float64)) / firstMonth.(float64) * 100

	fmt.Printf("📈 Annual growth: %.1f%% (from $%.0f to $%.0f)\n",
		yearGrowth, firstMonth.(float64), lastMonth.(float64))

	// Marketing efficiency
	marketingColumn, _ := df.GetColumn("marketing")
	totalSales, _ := stats.Sum(salesColumn.Data())
	totalMarketing, _ := stats.Sum(marketingColumn.Data())

	efficiency := totalSales / totalMarketing
	fmt.Printf("💰 Marketing efficiency: $%.2f revenue per $1 marketing spend\n", efficiency)

	// Staff productivity
	staffColumn, _ := df.GetColumn("staff")
	avgStaff, _ := stats.Mean(staffColumn.Data())
	avgSales, _ := stats.Mean(salesColumn.Data())
	productivity := avgSales / avgStaff

	fmt.Printf("👥 Staff productivity: $%.0f average sales per employee\n", productivity)

	// Recommendations
	fmt.Printf("\n💡 Recommendations:\n")

	if yearGrowth > 15 {
		fmt.Println("• Excellent growth trend - consider expanding capacity")
	} else if yearGrowth > 5 {
		fmt.Println("• Solid growth - maintain current strategies")
	} else {
		fmt.Println("• Consider reviewing and optimizing sales strategies")
	}

	if efficiency > 8 {
		fmt.Println("• Marketing spend is highly effective")
	} else if efficiency > 5 {
		fmt.Println("• Marketing ROI is acceptable - look for optimization opportunities")
	} else {
		fmt.Println("• Consider reviewing marketing effectiveness and targeting")
	}

	if productivity > 8000 {
		fmt.Println("• High staff productivity - good team performance")
	} else {
		fmt.Println("• Consider staff development or process optimization")
	}

	fmt.Println()
}

// Helper functions

func interpretCorrelation(corr float64) {
	if corr > 0.7 {
		fmt.Println(" (Strong positive)")
	} else if corr > 0.3 {
		fmt.Println(" (Moderate positive)")
	} else if corr > -0.3 {
		fmt.Println(" (Weak)")
	} else if corr > -0.7 {
		fmt.Println(" (Moderate negative)")
	} else {
		fmt.Println(" (Strong negative)")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
