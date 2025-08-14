// Package series provides a Pandas-like Series data structure for GoNP.
//
// # Overview
//
// The series package implements a one-dimensional labeled array data structure
// similar to Pandas Series. It combines the power of GoNP arrays with flexible
// indexing capabilities, making it ideal for data analysis and time series operations.
//
// Key features:
//   - Labeled data with customizable indices
//   - DateTime index support for time series analysis
//   - Advanced indexing (.loc, .iloc, boolean indexing)
//   - String manipulation methods for text data
//   - Missing data handling with NaN support
//   - Integration with GoNP arrays and mathematical operations
//
// # Quick Start
//
// Basic Series creation and operations:
//
//	import "github.com/julianshen/gonp/array"
//	import "github.com/julianshen/gonp/series"
//
//	// Create a Series from slice
//	data, _ := array.FromSlice([]float64{1.5, 2.3, 3.1, 4.8, 5.2})
//	s, _ := series.NewSeries(data, nil, "values")
//
//	// Series with custom index
//	labels := []interface{}{"A", "B", "C", "D", "E"}
//	index := series.NewIndex(labels)
//	s_labeled, _ := series.NewSeries(data, index, "labeled_values")
//
//	// Access elements
//	first := s.ILoc(0)         // Access by position: 1.5
//	value := s_labeled.Loc("B") // Access by label: 2.3
//
//	// Basic statistics
//	mean := s.Mean()           // Arithmetic mean: 3.38
//	std := s.Std()             // Standard deviation
//	sum := s.Sum()             // Sum of all values
//
// # Series Creation
//
// ## From Arrays and Slices
//
//	// From Go slices
//	numeric, _ := series.FromSlice([]float64{1, 2, 3, 4, 5}, nil, "numbers")
//	integers, _ := series.FromSlice([]int{10, 20, 30}, nil, "counts")
//	strings, _ := series.FromSlice([]string{"a", "b", "c"}, nil, "letters")
//
//	// From GoNP arrays
//	arr, _ := array.FromSlice([]float64{1.1, 2.2, 3.3})
//	s, _ := series.NewSeries(arr, nil, "from_array")
//
//	// With custom index
//	dates := []interface{}{"2023-01-01", "2023-01-02", "2023-01-03"}
//	date_index := series.NewIndex(dates)
//	time_series, _ := series.NewSeries(arr, date_index, "daily_values")
//
// ## From Maps (Key-Value Pairs)
//
//	// Create Series from map
//	data_map := map[string]float64{
//		"apple":  1.25,
//		"banana": 0.75,
//		"orange": 1.50,
//	}
//	fruit_prices, _ := series.FromMap(data_map, "prices")
//
// # Index Types and Operations
//
// ## Default Index (Integer-based)
//
//	// Default integer index (0, 1, 2, ...)
//	s, _ := series.FromSlice([]float64{10, 20, 30}, nil, "values")
//	fmt.Printf("Index: %v\n", s.Index().Labels()) // [0, 1, 2]
//
//	// Access by position
//	value := s.ILoc(1) // Returns 20
//
// ## Custom Index (Labels)
//
//	// String labels
//	labels := []interface{}{"first", "second", "third"}
//	index := series.NewIndex(labels)
//	s, _ := series.NewSeries(data, index, "labeled")
//
//	// Access by label
//	value := s.Loc("second")  // Access by label
//	exists := s.HasLabel("first")  // Check if label exists
//
// ## DateTime Index (Time Series)
//
//	// Create datetime index
//	dates := []string{"2023-01-01", "2023-01-02", "2023-01-03"}
//	dt_index, _ := series.NewDateTimeIndex(dates)
//	ts, _ := series.NewSeries(values, dt_index, "time_series")
//
//	// Time-based operations
//	subset := ts.DateRange("2023-01-01", "2023-01-02")  // Date range selection
//	resampled := ts.Resample("D", "mean")               // Daily resampling
//	rolled := ts.RollingWindow(3, "mean")               // 3-day rolling average
//
// # Data Access and Indexing
//
// ## Position-based Indexing (.iloc)
//
//	// Single element access
//	first := s.ILoc(0)         // First element
//	last := s.ILoc(-1)         // Last element (if supported)
//
//	// Slice access
//	subset := s.ILocSlice(1, 4) // Elements from index 1 to 3
//	every_other := s.ILocStep(0, -1, 2) // Every other element
//
// ## Label-based Indexing (.loc)
//
//	// Single label access
//	value := s.Loc("label_name")
//
//	// Multiple label access
//	labels := []interface{}{"A", "C", "E"}
//	subset := s.LocMultiple(labels)
//
//	// Label range (if index is ordered)
//	range_data := s.LocRange("B", "D") // From "B" to "D" inclusive
//
// ## Boolean Indexing (Conditional Selection)
//
//	// Filter with conditions
//	high_values := s.Where(func(x interface{}) bool {
//		return x.(float64) > 5.0
//	})
//
//	// Multiple conditions
//	filtered := s.Where(func(x interface{}) bool {
//		val := x.(float64)
//		return val > 2.0 && val < 8.0
//	})
//
//	// Using comparison methods
//	greater := s.Gt(5.0)      // Boolean Series for values > 5
//	mask := s.IsNull()        // Boolean Series for null values
//	not_null := s.Where(func(x interface{}) bool { return !s.IsNull().At(0).(bool) })
//
// # Data Operations and Methods
//
// ## Arithmetic Operations
//
//	// Element-wise operations
//	doubled := s.Multiply(2)        // Multiply by scalar
//	sum_series := s1.Add(s2)        // Add two Series (aligned by index)
//	diff := s1.Subtract(s2)         // Subtract Series
//
//	// Mathematical functions
//	sqrt_vals := s.Sqrt()           // Square root of all values
//	log_vals := s.Log()             // Natural logarithm
//	abs_vals := s.Abs()             // Absolute values
//
// ## Statistical Methods
//
//	// Descriptive statistics
//	mean := s.Mean()                // Arithmetic mean
//	median := s.Median()            // 50th percentile
//	mode := s.Mode()                // Most frequent value(s)
//	std := s.Std()                  // Standard deviation
//	var_val := s.Var()              // Variance
//
//	// Distribution properties
//	min_val := s.Min()              // Minimum value
//	max_val := s.Max()              // Maximum value
//	q25 := s.Quantile(0.25)         // 25th percentile
//	q75 := s.Quantile(0.75)         // 75th percentile
//
//	// Summary statistics
//	summary := s.Describe()         // Complete statistical summary
//	fmt.Printf("Count: %d, Mean: %.2f, Std: %.2f\n",
//		summary.Count, summary.Mean, summary.Std)
//
// ## Data Cleaning and Transformation
//
//	// Missing data handling
//	has_null := s.IsNull()          // Boolean Series indicating null values
//	not_null := s.NotNull()         // Boolean Series indicating non-null values
//	cleaned := s.DropNa()           // Remove null values
//	filled := s.FillNa(0.0)         // Fill null values with 0
//	forward_filled := s.FillForward() // Forward fill missing values
//
//	// Data type conversion
//	as_int := s.AsType("int64")     // Convert to integer type
//	as_str := s.AsType("string")    // Convert to string type
//
//	// Value replacement
//	replaced := s.Replace(old_val, new_val) // Replace specific values
//	mapped := s.Map(func(x interface{}) interface{} {
//		return x.(float64) * 2 + 1  // Transform each value
//	})
//
// # String Methods (for Text Data)
//
// When a Series contains string data, additional string manipulation methods are available:
//
//	// Create string Series
//	text_data := []string{"Hello World", "  Python  ", "DATA science"}
//	text_series, _ := series.FromSlice(text_data, nil, "text")
//
//	// String methods via .Str() accessor
//	str_methods := text_series.Str()
//
//	// Case conversion
//	upper := str_methods.Upper()    // "HELLO WORLD", "  PYTHON  ", "DATA SCIENCE"
//	lower := str_methods.Lower()    // "hello world", "  python  ", "data science"
//	title := str_methods.Title()    // "Hello World", "  Python  ", "Data Science"
//
//	// Whitespace handling
//	stripped := str_methods.Strip() // "Hello World", "Python", "DATA science"
//	left_strip := str_methods.LStrip()   // Remove left whitespace
//	right_strip := str_methods.RStrip()  // Remove right whitespace
//
//	// String operations
//	lengths := str_methods.Len()    // Length of each string
//	contains := str_methods.Contains("data")  // Boolean Series for pattern match
//	starts := str_methods.StartsWith("Hello") // Boolean Series for prefix
//	ends := str_methods.EndsWith("science")   // Boolean Series for suffix
//
//	// String splitting and joining
//	split := str_methods.Split(" ")  // Split by delimiter
//	joined := str_methods.Join("-")  // Join with delimiter
//
//	// Pattern matching and replacement
//	replaced := str_methods.Replace("data", "DATA") // String replacement
//	extracted := str_methods.Extract(`(\w+)\s+(\w+)`) // Regex extraction
//
// # DateTime Operations (Time Series)
//
// For Series with DateTime index, additional time-based operations are available:
//
//	// Create time series
//	dates := []string{"2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"}
//	values := []float64{100, 105, 98, 110}
//	dt_index, _ := series.NewDateTimeIndex(dates)
//	ts, _ := series.NewSeries(array.FromSlice(values), dt_index, "prices")
//
//	// Time-based selection
//	jan_data := ts.DateRange("2023-01-01", "2023-01-31")
//	recent := ts.Head(10)           // Last 10 observations
//	oldest := ts.Tail(5)            // First 5 observations
//
//	// Resampling and aggregation
//	daily_mean := ts.Resample("D", "mean")      // Daily averages
//	weekly_sum := ts.Resample("W", "sum")       // Weekly sums
//	monthly_max := ts.Resample("M", "max")      // Monthly maximums
//
//	// Rolling window operations
//	rolling_avg := ts.RollingWindow(5, "mean")  // 5-day moving average
//	rolling_std := ts.RollingWindow(10, "std")  // 10-day rolling standard deviation
//	rolling_max := ts.RollingWindow(7, "max")   // 7-day rolling maximum
//
//	// Time shifts and lags
//	lagged := ts.Shift(1)           // Shift forward by 1 period
//	lead := ts.Shift(-1)            // Shift backward by 1 period
//	pct_change := ts.PctChange()    // Percentage change from previous
//
// # Advanced Operations
//
// ## Sorting and Ranking
//
//	// Sort by values
//	sorted_asc := s.SortValues(true)   // Ascending order
//	sorted_desc := s.SortValues(false) // Descending order
//
//	// Sort by index
//	sorted_by_index := s.SortIndex()
//
//	// Ranking
//	ranks := s.Rank("average")      // Rank values (ties = average)
//	ranks_min := s.Rank("min")      // Rank values (ties = minimum)
//
// ## Unique Values and Counts
//
//	// Unique operations
//	unique_vals := s.Unique()       // Array of unique values
//	n_unique := s.NUnique()         // Count of unique values
//
//	// Value counts
//	counts := s.ValueCounts()       // Series of value frequencies
//	counts_norm := s.ValueCounts(true) // Normalized frequencies (proportions)
//
//	// Duplicate handling
//	has_dups := s.HasDuplicates()   // Check for duplicates
//	no_dups := s.DropDuplicates()   // Remove duplicate values
//	first_dup := s.DropDuplicates("first")  // Keep first occurrence
//
// ## Grouping and Aggregation
//
//	// Group by values (for categorical data)
//	grouped := s.GroupBy()          // Group identical values
//	group_means := grouped.Mean()   // Mean for each group
//	group_sums := grouped.Sum()     // Sum for each group
//	group_counts := grouped.Count() // Count for each group
//
//	// Multiple aggregations
//	agg_results := grouped.Agg([]string{"mean", "std", "min", "max"})
//
// # Combining Series
//
//	// Concatenation
//	combined := series.Concat([]*series.Series{s1, s2, s3})
//
//	// Alignment and merging
//	aligned_s1, aligned_s2 := series.Align(s1, s2)  // Align indices
//	merged := s1.Merge(s2, "outer")  // Outer join on indices
//	inner_merged := s1.Merge(s2, "inner")  // Inner join on indices
//
//	// Set operations
//	union := s1.Union(s2)           // Union of indices
//	intersection := s1.Intersect(s2) // Intersection of indices
//	difference := s1.Difference(s2) // Difference of indices
//
// # Integration with GoNP Ecosystem
//
// ## Array Integration
//
//	// Convert to/from arrays
//	arr := s.Values()               // Get underlying array
//	new_s, _ := series.FromArray(arr, index, "name")
//
//	// Mathematical operations using math package
//	import "github.com/julianshen/gonp/math"
//
//	sin_vals := math.Sin(s.Values()) // Apply sine to values
//	s_sin, _ := series.FromArray(sin_vals, s.Index(), "sin_values")
//
// ## Statistical Analysis
//
//	// Integration with stats package
//	import "github.com/julianshen/gonp/stats"
//
//	correlation := stats.Correlation(s1.Values(), s2.Values())
//	ttest_result := stats.TwoSampleTTest(s1.Values(), s2.Values())
//	regression := stats.LinearRegression(x_series.Values(), y_series.Values())
//
// ## Visualization Integration
//
//	// Integration with visualization package
//	import "github.com/julianshen/gonp/visualization"
//
//	// Create plots directly from Series
//	line_plot := s.Plot("line")     // Line plot
//	hist := s.Plot("hist")          // Histogram
//	box_plot := s.Plot("box")       // Box plot
//
//	// Time series plotting
//	ts_plot := time_series.Plot("line")
//	seasonal := time_series.SeasonalDecompose().Plot("subplots")
//
// # Performance Considerations
//
// ## Memory Efficiency
//
//	// Series operations are memory-efficient
//	// - Views share memory with underlying arrays when possible
//	// - Copy operations are explicit
//	// - String data is efficiently stored
//
//	// Create views (no copying)
//	view := s.ILocSlice(10, 100)    // View of slice [10:100]
//
//	// Explicit copying
//	copy_s := s.Copy()              // Deep copy of Series
//
// ## Performance Tips
//
//	// - Use .iloc for position-based access (fastest)
//	// - Use .loc for label-based access when needed
//	// - Boolean indexing is optimized for large Series
//	// - String operations are vectorized when possible
//	// - DateTime operations leverage optimized time libraries
//
// # Error Handling
//
// Series operations handle common errors gracefully:
//
//	// Index errors
//	value, err := s.TryLoc("nonexistent")  // Returns error if label doesn't exist
//	if err != nil {
//		log.Printf("Label not found: %v", err)
//	}
//
//	// Type conversion errors
//	converted, err := s.TryAsType("int64")  // May fail for non-numeric data
//	if err != nil {
//		log.Printf("Conversion failed: %v", err)
//	}
//
//	// Operation errors
//	result, err := s1.TryAdd(s2)  // May fail if indices don't align
//	if err != nil {
//		log.Printf("Addition failed: %v", err)
//	}
//
// # Migration from Pandas
//
// Common Pandas Series operations and their GoNP equivalents:
//
//	# Pandas                        # GoNP
//	import pandas as pd             import "github.com/julianshen/gonp/series"
//
//	pd.Series([1, 2, 3])            series.FromSlice([]float64{1, 2, 3}, nil, "")
//	s.iloc[0]                       s.ILoc(0)
//	s.loc['key']                    s.Loc("key")
//	s[s > 5]                        s.Where(func(x interface{}) bool { return x.(float64) > 5 })
//	s.mean()                        s.Mean()
//	s.dropna()                      s.DropNa()
//	s.fillna(0)                     s.FillNa(0)
//	s.str.upper()                   s.Str().Upper()
//	s.value_counts()                s.ValueCounts()
//	s.rolling(5).mean()             s.RollingWindow(5, "mean")
package series
