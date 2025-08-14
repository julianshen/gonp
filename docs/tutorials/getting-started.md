# Getting Started with GoNP

GoNP (Go NumPy + Pandas) is a high-performance numerical computing library for Go that provides familiar NumPy and Pandas-like functionality. This tutorial will guide you through the core concepts and operations.

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Working with Arrays](#working-with-arrays)
4. [Working with Series](#working-with-series)
5. [Working with DataFrames](#working-with-dataframes)
6. [Mathematical Operations](#mathematical-operations)
7. [Statistical Analysis](#statistical-analysis)
8. [Data I/O](#data-io)
9. [Performance Tips](#performance-tips)
10. [Common Patterns](#common-patterns)

## Installation

```bash
go mod init your-project
go get github.com/julianshen/gonp
```

## Core Concepts

GoNP provides three main data structures:

- **Array**: N-dimensional homogeneous data container (like NumPy arrays)
- **Series**: 1-dimensional labeled array (like Pandas Series)
- **DataFrame**: 2-dimensional labeled data structure (like Pandas DataFrame)

## Working with Arrays

Arrays are the foundation of GoNP, providing efficient storage and operations on homogeneous data.

### Creating Arrays

```go
package main

import (
    "fmt"
    "log"
    "github.com/julianshen/gonp/array"
)

func main() {
    // Create from Go slice
    data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    arr, err := array.FromSlice(data)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Array: %v\n", arr)
    fmt.Printf("Size: %d\n", arr.Size())
    fmt.Printf("Shape: %v\n", arr.Shape())
}
```

### Array Operations

```go
// Element-wise operations
arr1, _ := array.FromSlice([]float64{1.0, 2.0, 3.0})
arr2, _ := array.FromSlice([]float64{4.0, 5.0, 6.0})

// Addition
sum, _ := arr1.Add(arr2)
fmt.Printf("Sum: %v\n", sum) // [5.0, 7.0, 9.0]

// Multiplication
product, _ := arr1.Multiply(arr2)
fmt.Printf("Product: %v\n", product) // [4.0, 10.0, 18.0]

// Accessing elements
value := arr1.At(1) // Get element at index 1
fmt.Printf("arr1[1] = %v\n", value) // 2.0

// Setting elements
err := arr1.Set(10.0, 1) // Set element at index 1 to 10.0
if err != nil {
    log.Fatal(err)
}
```

### Array Creation Functions

```go
import "github.com/julianshen/gonp/internal"

// Create arrays filled with specific values
zeros := array.Zeros(internal.Shape{3, 3}, internal.Float64)
ones := array.Ones(internal.Shape{2, 4}, internal.Float64)
filled := array.Full(internal.Shape{2, 2}, 7.5, internal.Float64)

fmt.Printf("Zeros:\n%v\n", zeros)
fmt.Printf("Ones:\n%v\n", ones)
fmt.Printf("Filled:\n%v\n", filled)
```

## Working with Series

Series provide labeled, one-dimensional arrays with powerful indexing capabilities.

### Creating Series

```go
import (
    "github.com/julianshen/gonp/series"
    "github.com/julianshen/gonp/array"
)

func main() {
    // Create from array
    data, _ := array.FromSlice([]float64{10.5, 20.3, 15.7, 25.1})
    s, err := series.NewSeries(data, nil, "measurements")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Series length: %d\n", s.Len())
    fmt.Printf("Series name: %s\n", s.Name())
    
    // Access elements
    value := s.At(0) // First element
    fmt.Printf("First measurement: %v\n", value)
}
```

### Series Operations

```go
// Create test series
data, _ := array.FromSlice([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
s, _ := series.NewSeries(data, nil, "numbers")

// Slicing
subset, _ := s.Slice(1, 4) // Elements 1 to 3 (exclusive end)
fmt.Printf("Subset: %v\n", subset)

// Head and tail
head := s.Head(3) // First 3 elements
tail := s.Tail(2) // Last 2 elements
fmt.Printf("Head: %v\n", head)
fmt.Printf("Tail: %v\n", tail)

// Statistical operations (using underlying array)
mean, _ := stats.Mean(s.Data())
fmt.Printf("Mean: %v\n", mean)
```

## Working with DataFrames

DataFrames provide powerful 2D labeled data structures for complex data analysis.

### Creating DataFrames

```go
import (
    "github.com/julianshen/gonp/dataframe"
    "github.com/julianshen/gonp/series"
    "github.com/julianshen/gonp/array"
)

func main() {
    // Create sample data
    names := []string{"Alice", "Bob", "Charlie", "Diana"}
    ages := []float64{25, 30, 35, 28}
    salaries := []float64{50000, 60000, 70000, 55000}
    
    // Create arrays
    ageArr, _ := array.FromSlice(ages)
    salaryArr, _ := array.FromSlice(salaries)
    
    // Create series
    ageSeries, _ := series.NewSeries(ageArr, nil, "age")
    salarySeries, _ := series.NewSeries(salaryArr, nil, "salary")
    
    // Create DataFrame
    df, err := dataframe.FromSeries([]*series.Series{ageSeries, salarySeries})
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("DataFrame shape: %d rows, %d columns\n", df.Len(), len(df.Columns()))
    fmt.Printf("Columns: %v\n", df.Columns())
}
```

### DataFrame Operations

```go
// Access specific values
age0, _ := df.IAt(0, 0) // First person's age
salary1, _ := df.IAt(1, 1) // Second person's salary
fmt.Printf("First person's age: %v\n", age0)
fmt.Printf("Second person's salary: %v\n", salary1)

// Get entire columns
ageColumn, _ := df.GetColumn("age")
salaryColumn, _ := df.GetColumn("salary")

// Slice DataFrame (first 2 rows)
subset, _ := df.Slice(0, 2)
fmt.Printf("Subset has %d rows\n", subset.Len())

// Column statistics
avgAge, _ := stats.Mean(ageColumn.Data())
avgSalary, _ := stats.Mean(salaryColumn.Data())
fmt.Printf("Average age: %.1f\n", avgAge)
fmt.Printf("Average salary: %.0f\n", avgSalary)
```

## Mathematical Operations

GoNP provides comprehensive mathematical functions through the `math` package.

### Universal Functions

```go
import "github.com/julianshen/gonp/math"

func main() {
    // Create test data
    data, _ := array.FromSlice([]float64{0.0, 1.0, 2.0, 3.0})
    
    // Trigonometric functions
    sinResult, _ := math.Sin(data)
    cosResult, _ := math.Cos(data)
    
    fmt.Printf("Sin: %v\n", sinResult)
    fmt.Printf("Cos: %v\n", cosResult)
    
    // Exponential functions
    expResult, _ := math.Exp(data)
    logResult, _ := math.Log(data[1:]) // Avoid log(0)
    
    fmt.Printf("Exp: %v\n", expResult)
    fmt.Printf("Log: %v\n", logResult)
    
    // Power functions
    sqrtResult, _ := math.Sqrt(data)
    squareResult, _ := math.Square(data)
    
    fmt.Printf("Sqrt: %v\n", sqrtResult)
    fmt.Printf("Square: %v\n", squareResult)
}
```

### Element-wise Operations

```go
// Create two arrays
arr1, _ := array.FromSlice([]float64{1.0, 4.0, 9.0, 16.0})
arr2, _ := array.FromSlice([]float64{1.0, 2.0, 3.0, 4.0})

// Binary operations
sum, _ := arr1.Add(arr2)
diff, _ := arr1.Subtract(arr2)
product, _ := arr1.Multiply(arr2)
quotient, _ := arr1.Divide(arr2)

fmt.Printf("Addition: %v\n", sum)
fmt.Printf("Subtraction: %v\n", diff)
fmt.Printf("Multiplication: %v\n", product)
fmt.Printf("Division: %v\n", quotient)

// Comparison operations
gt, _ := arr1.Greater(arr2)
eq, _ := arr1.Equal(arr2)

fmt.Printf("Greater than: %v\n", gt)
fmt.Printf("Equal: %v\n", eq)
```

## Statistical Analysis

The `stats` package provides comprehensive statistical functions.

### Descriptive Statistics

```go
import "github.com/julianshen/gonp/stats"

func main() {
    // Sample data
    data, _ := array.FromSlice([]float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0})
    
    // Basic statistics
    mean, _ := stats.Mean(data)
    median, _ := stats.Median(data)
    std, _ := stats.Std(data)
    variance, _ := stats.Var(data)
    
    fmt.Printf("Mean: %.2f\n", mean)
    fmt.Printf("Median: %.2f\n", median)
    fmt.Printf("Standard Deviation: %.2f\n", std)
    fmt.Printf("Variance: %.2f\n", variance)
    
    // Min/Max
    min, _ := stats.Min(data)
    max, _ := stats.Max(data)
    
    fmt.Printf("Min: %.2f\n", min)
    fmt.Printf("Max: %.2f\n", max)
}
```

### Correlation Analysis

```go
// Two related datasets
x, _ := array.FromSlice([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
y, _ := array.FromSlice([]float64{2.0, 4.0, 6.0, 8.0, 10.0})

// Calculate correlation
correlation, err := stats.Correlation(x, y)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Correlation: %.3f\n", correlation) // Should be 1.0 (perfect positive correlation)

// Covariance
covariance, err := stats.Covariance(x, y)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Covariance: %.3f\n", covariance)
```

## Data I/O

GoNP supports reading and writing data in various formats.

### CSV Operations

```go
import "github.com/julianshen/gonp/io"

func main() {
    // Read CSV file
    df, err := io.ReadCSV("data.csv", true) // true = has header
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Loaded DataFrame: %d rows, %d columns\n", df.Len(), len(df.Columns()))
    
    // Write CSV file
    err = io.WriteCSV(df, "output.csv", true) // true = write header
    if err != nil {
        log.Fatal(err)
    }
}
```

### JSON Operations

```go
// Write DataFrame to JSON
err := io.WriteJSON(df, "output.json")
if err != nil {
    log.Fatal(err)
}

// Read DataFrame from JSON
dfFromJSON, err := io.ReadJSON("output.json")
if err != nil {
    log.Fatal(err)
}
```

## Performance Tips

### 1. Reuse Arrays When Possible

```go
// Good: Reuse existing array
arr, _ := array.FromSlice([]float64{1.0, 2.0, 3.0})
err := arr.Set(5.0, 0) // Modify in place

// Less efficient: Create new array each time
newArr, _ := array.FromSlice([]float64{5.0, 2.0, 3.0})
```

### 2. Use Appropriate Data Types

```go
// Use the most appropriate data type for your use case
intData := []int64{1, 2, 3, 4, 5}
intArr, _ := array.FromSlice(intData)

floatData := []float64{1.1, 2.2, 3.3, 4.4, 5.5}
floatArr, _ := array.FromSlice(floatData)
```

### 3. Batch Operations

```go
// Good: Batch operations
data := make([]float64, 1000)
for i := range data {
    data[i] = float64(i)
}
arr, _ := array.FromSlice(data)
result, _ := math.Sin(arr) // Single operation on entire array

// Less efficient: Element-by-element operations in Go
for i := 0; i < arr.Size(); i++ {
    // Individual operations
}
```

## Common Patterns

### 1. Data Analysis Pipeline

```go
func analyzeData(filename string) error {
    // Load data
    df, err := io.ReadCSV(filename, true)
    if err != nil {
        return err
    }
    
    // Get numeric columns
    column, err := df.GetColumn("values")
    if err != nil {
        return err
    }
    
    // Calculate statistics
    mean, _ := stats.Mean(column.Data())
    std, _ := stats.Std(column.Data())
    
    fmt.Printf("Mean: %.2f, Std: %.2f\n", mean, std)
    
    // Filter data (first half)
    filtered, _ := df.Slice(0, df.Len()/2)
    
    // Save results
    return io.WriteCSV(filtered, "filtered_data.csv", true)
}
```

### 2. Time Series Analysis

```go
func analyzeTimeSeries(data []float64) {
    // Convert to series
    arr, _ := array.FromSlice(data)
    ts, _ := series.NewSeries(arr, nil, "timeseries")
    
    // Calculate rolling statistics (manual implementation)
    windowSize := 3
    for i := windowSize - 1; i < ts.Len(); i++ {
        window, _ := ts.Slice(i-windowSize+1, i+1)
        mean, _ := stats.Mean(window.Data())
        fmt.Printf("Window %d mean: %.2f\n", i, mean)
    }
}
```

### 3. Data Transformation

```go
func transformData(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
    // Get column
    values, err := df.GetColumn("raw_values")
    if err != nil {
        return nil, err
    }
    
    // Apply mathematical transformation
    logValues, err := math.Log(values.Data())
    if err != nil {
        return nil, err
    }
    
    // Create new series
    logSeries, err := series.NewSeries(logValues, nil, "log_values")
    if err != nil {
        return nil, err
    }
    
    // Create new DataFrame with transformed data
    return dataframe.FromSeries([]*series.Series{logSeries})
}
```

## Next Steps

Now that you understand the basics, explore:

1. **Advanced Examples**: Check the [examples](../examples/) directory
2. **API Reference**: See detailed API documentation in [docs/api](../api/)
3. **Performance Optimization**: Learn about optimizing your GoNP code
4. **Contributing**: See [CLAUDE.md](../../CLAUDE.md) for development guidelines

## Getting Help

- **Issues**: Report bugs and request features on GitHub
- **Documentation**: Check the API documentation for detailed function signatures
- **Examples**: Look at the integration tests for comprehensive usage patterns

Happy computing with GoNP! 🚀