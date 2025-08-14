// Package dataframe provides a Pandas-like DataFrame data structure for GoNP.
//
// # Overview
//
// The dataframe package implements a two-dimensional labeled data structure
// similar to Pandas DataFrame. It provides powerful data manipulation capabilities
// including joins, group operations, pivot tables, and comprehensive data analysis tools.
//
// Key features:
//   - Two-dimensional labeled data (rows and columns)
//   - Flexible indexing with row and column labels
//   - Join and merge operations (inner, outer, left, right)
//   - GroupBy operations with aggregation functions
//   - Pivot tables and data reshaping
//   - Advanced indexing (.loc, .iloc, boolean indexing)
//   - Missing data handling and cleaning
//   - Integration with GoNP arrays and mathematical operations
//
// # Quick Start
//
// Basic DataFrame creation and operations:
//
//	import "github.com/julianshen/gonp/array"
//	import "github.com/julianshen/gonp/dataframe"
//
//	// Create DataFrame from map of arrays
//	data := map[string]*array.Array{
//		"name":   array.FromSlice([]string{"Alice", "Bob", "Charlie"}),
//		"age":    array.FromSlice([]int{25, 30, 35}),
//		"salary": array.FromSlice([]float64{50000, 60000, 70000}),
//	}
//	df, _ := dataframe.FromMap(data)
//
//	// Basic operations
//	fmt.Printf("Shape: %v\n", df.Shape())        // [3, 3]
//	fmt.Printf("Columns: %v\n", df.Columns())    // ["name", "age", "salary"]
//
//	// Access data
//	ages := df.GetColumn("age")          // Get age column as Series
//	first_row := df.ILoc(0)              // Get first row
//	alice_data := df.Loc("Alice")        // Get row by label (if indexed)
//
//	// Statistical summary
//	summary := df.Describe()             // Descriptive statistics
//	mean_salary := df.GetColumn("salary").Mean() // Average salary
//
// # DataFrame Creation
//
// ## From Maps and Slices
//
//	// From map of slices
//	data_map := map[string]interface{}{
//		"product": []string{"A", "B", "C", "A", "B"},
//		"price":   []float64{10.5, 15.0, 20.0, 12.0, 18.0},
//		"qty":     []int{100, 200, 150, 80, 120},
//	}
//	df, _ := dataframe.FromMapInterface(data_map)
//
//	// From 2D slice (matrix)
//	matrix := [][]float64{
//		{1.1, 2.2, 3.3},
//		{4.4, 5.5, 6.6},
//		{7.7, 8.8, 9.9},
//	}
//	columns := []string{"col1", "col2", "col3"}
//	df, _ := dataframe.FromMatrix(matrix, columns)
//
// ## From Arrays with Custom Index
//
//	// Create arrays
//	names, _ := array.FromSlice([]string{"Alice", "Bob", "Charlie"})
//	ages, _ := array.FromSlice([]int{25, 30, 35})
//	salaries, _ := array.FromSlice([]float64{50000, 60000, 70000})
//
//	// Create DataFrame with arrays
//	arrays := []*array.Array{names, ages, salaries}
//	columns := []string{"name", "age", "salary"}
//	df, _ := dataframe.FromArrays(arrays, columns)
//
//	// With custom row index
//	row_labels := []interface{}{"emp1", "emp2", "emp3"}
//	index := series.NewIndex(row_labels)
//	df_indexed, _ := dataframe.FromArraysWithIndex(arrays, columns, index)
//
// ## From Series Collection
//
//	// Create individual Series
//	names_series, _ := series.FromSlice([]string{"Alice", "Bob"}, nil, "name")
//	ages_series, _ := series.FromSlice([]int{25, 30}, nil, "age")
//
//	// Combine Series into DataFrame
//	series_map := map[string]*series.Series{
//		"name": names_series,
//		"age":  ages_series,
//	}
//	df, _ := dataframe.FromSeries(series_map)
//
// # Data Access and Indexing
//
// ## Column Access
//
//	// Single column access
//	age_series := df.GetColumn("age")     // Returns Series
//	age_values := df.GetColumnValues("age") // Returns underlying Array
//
//	// Multiple column access
//	subset := df.GetColumns([]string{"name", "salary"}) // Returns DataFrame
//
//	// Column existence check
//	exists := df.HasColumn("age")         // Returns bool
//
// ## Row Access
//
// ### Position-based Indexing (.iloc)
//
//	// Single row access
//	first_row := df.ILoc(0)               // Returns Series (first row)
//	last_row := df.ILoc(-1)               // Last row (if supported)
//
//	// Multiple row access
//	subset := df.ILocRows([]int{0, 2, 4}) // Rows at positions 0, 2, 4
//	slice_data := df.ILocSlice(1, 5)      // Rows 1 to 4
//
// ### Label-based Indexing (.loc)
//
//	// Single row by label (requires row index)
//	alice_row := df.Loc("Alice")          // Returns Series
//
//	// Multiple rows by labels
//	subset := df.LocRows([]interface{}{"Alice", "Charlie"})
//
//	// Cell access
//	value := df.At("Alice", "salary")     // Single cell value
//	df.Set(75000, "Alice", "salary")      // Set single cell
//
// ## Boolean Indexing (Conditional Selection)
//
//	// Filter rows with conditions
//	high_earners := df.Where("salary", func(x interface{}) bool {
//		return x.(float64) > 55000
//	})
//
//	// Complex conditions
//	young_high_earners := df.WhereMultiple(map[string]func(interface{}) bool{
//		"age": func(x interface{}) bool { return x.(int) < 32 },
//		"salary": func(x interface{}) bool { return x.(float64) > 55000 },
//	})
//
//	// Using comparison methods
//	high_age := df.Gt("age", 28)          // Boolean DataFrame for age > 28
//	filtered := df.BooleanIndex(high_age) // Apply boolean mask
//
// # Data Operations and Methods
//
// ## Basic Statistics and Aggregation
//
//	// Summary statistics
//	summary := df.Describe()              // Complete statistical summary
//	means := df.Mean()                    // Mean for all numeric columns
//	stds := df.Std()                      // Standard deviations
//
//	// Column-specific statistics
//	age_mean := df.GetColumn("age").Mean()
//	salary_median := df.GetColumn("salary").Median()
//
//	// Aggregation functions
//	totals := df.Sum()                    // Sum for all numeric columns
//	maximums := df.Max()                  // Maximum values
//	minimums := df.Min()                  // Minimum values
//	counts := df.Count()                  // Non-null counts
//
// ## Data Cleaning and Transformation
//
//	// Missing data handling
//	has_null := df.IsNull()               // Boolean DataFrame for null values
//	not_null := df.NotNull()              // Boolean DataFrame for non-null values
//	cleaned := df.DropNa()                // Remove rows with any null values
//	filled := df.FillNa(0)                // Fill all null values with 0
//
//	// Column-specific null handling
//	age_filled := df.FillNaColumn("age", df.GetColumn("age").Mean())
//
//	// Data type conversion
//	df_converted := df.AsType(map[string]string{
//		"age": "float64",
//		"qty": "float64",
//	})
//
//	// Value replacement
//	replaced := df.Replace("old_value", "new_value")
//	df_mapped := df.MapColumn("salary", func(x interface{}) interface{} {
//		return x.(float64) * 1.1  // 10% salary increase
//	})
//
// # DataFrame Manipulation
//
// ## Adding and Removing Columns
//
//	// Add new columns
//	bonus := df.GetColumn("salary").Map(func(x interface{}) interface{} {
//		return x.(float64) * 0.1  // 10% bonus
//	})
//	df_with_bonus := df.AddColumn("bonus", bonus)
//
//	// Add calculated columns
//	df_calc := df.AddCalculatedColumn("total_comp", []string{"salary", "bonus"},
//		func(values []interface{}) interface{} {
//			return values[0].(float64) + values[1].(float64)
//		})
//
//	// Remove columns
//	df_trimmed := df.DropColumns([]string{"bonus"})
//	df_selected := df.SelectColumns([]string{"name", "salary"})
//
// ## Adding and Removing Rows
//
//	// Add single row
//	new_row := map[string]interface{}{
//		"name": "David",
//		"age": 28,
//		"salary": 55000,
//	}
//	df_expanded := df.AddRow(new_row)
//
//	// Add multiple rows
//	new_rows := []map[string]interface{}{
//		{"name": "Eve", "age": 26, "salary": 52000},
//		{"name": "Frank", "age": 33, "salary": 68000},
//	}
//	df_more := df.AddRows(new_rows)
//
//	// Remove rows by position
//	df_trimmed := df.DropRows([]int{0, 2})  // Remove rows 0 and 2
//
//	// Remove rows by condition
//	df_filtered := df.DropWhere("age", func(x interface{}) bool {
//		return x.(int) < 25  // Remove employees younger than 25
//	})
//
// # GroupBy Operations
//
// GroupBy operations allow data aggregation and analysis by groups:
//
// ## Basic Grouping
//
//	// Group by single column
//	grouped := df.GroupBy("department")
//
//	// Group statistics
//	group_means := grouped.Mean()         // Mean for each group
//	group_sums := grouped.Sum()           // Sum for each group
//	group_counts := grouped.Count()       // Count for each group
//	group_sizes := grouped.Size()         // Group sizes
//
// ## Multiple Column Grouping
//
//	// Group by multiple columns
//	multi_grouped := df.GroupBy("department", "level")
//	avg_by_dept_level := multi_grouped.Mean()
//
// ## Advanced Aggregation
//
//	// Multiple aggregation functions
//	agg_results := grouped.Agg(map[string][]string{
//		"salary": {"mean", "std", "min", "max"},
//		"age":    {"mean", "median"},
//	})
//
//	// Custom aggregation functions
//	custom_agg := grouped.AggFunc("salary", func(values *array.Array) interface{} {
//		// Calculate coefficient of variation
//		mean := stats.Mean(values)
//		std := stats.StdDev(values)
//		return std / mean
//	})
//
// ## Filtering Groups
//
//	// Filter groups by size
//	large_groups := grouped.FilterSize(5)  // Keep groups with >= 5 members
//
//	// Filter groups by condition
//	high_avg_groups := grouped.FilterMean("salary", 60000) // Groups with mean salary > 60k
//
// # Join and Merge Operations
//
// DataFrame supports various join operations similar to SQL:
//
// ## Basic Joins
//
//	// Inner join (intersection of keys)
//	inner_result := df1.Merge(df2, "employee_id", dataframe.InnerJoin)
//
//	// Left join (all rows from left DataFrame)
//	left_result := df1.Merge(df2, "employee_id", dataframe.LeftJoin)
//
//	// Right join (all rows from right DataFrame)
//	right_result := df1.Merge(df2, "employee_id", dataframe.RightJoin)
//
//	// Outer join (all rows from both DataFrames)
//	outer_result := df1.Merge(df2, "employee_id", dataframe.OuterJoin)
//
// ## Multi-column Joins
//
//	// Join on multiple columns
//	multi_join := df1.MergeOn(df2, []string{"dept", "level"}, dataframe.InnerJoin)
//
// ## Join with Different Column Names
//
//	// Join when key columns have different names
//	diff_names := df1.MergeOnColumns(df2, "employee_id", "emp_id", dataframe.LeftJoin)
//
// ## Advanced Join Options
//
//	// Join with custom suffixes for duplicate columns
//	with_suffixes := df1.MergeWithSuffixes(df2, "id", dataframe.OuterJoin, "_left", "_right")
//
//	// Validate join (check for duplicate keys)
//	validated_join := df1.MergeValidate(df2, "id", dataframe.InnerJoin, "one_to_one")
//
// # Pivot Tables and Reshaping
//
// ## Pivot Tables
//
//	// Basic pivot table
//	pivot := df.PivotTable("department", "level", "salary", "mean")
//
//	// Pivot with multiple values
//	multi_pivot := df.PivotTableMultiple(
//		"department",           // index
//		"level",               // columns
//		[]string{"salary", "bonus"}, // values
//		"mean",                // aggregation function
//	)
//
//	// Pivot with margins (totals)
//	with_margins := df.PivotTableWithMargins("dept", "level", "salary", "sum", true)
//
// ## Data Reshaping
//
//	// Melt (wide to long format)
//	melted := df.Melt(
//		[]string{"name", "department"}, // id_vars (identifiers)
//		[]string{"q1", "q2", "q3", "q4"}, // value_vars (to be melted)
//		"quarter", // var_name
//		"sales",   // value_name
//	)
//
//	// Stack (pivot columns to rows)
//	stacked := df.Stack()
//
//	// Unstack (pivot rows to columns)
//	unstacked := df.Unstack("level")
//
// ## Transposition
//
//	// Transpose DataFrame
//	transposed := df.Transpose()  // Swap rows and columns
//
// # Sorting and Ranking
//
//	// Sort by single column
//	sorted_by_age := df.SortValues("age", true)  // Ascending
//	sorted_by_salary := df.SortValues("salary", false) // Descending
//
//	// Sort by multiple columns
//	multi_sorted := df.SortByColumns(
//		[]string{"department", "salary"},
//		[]bool{true, false}, // department asc, salary desc
//	)
//
//	// Sort by index
//	index_sorted := df.SortIndex()
//
//	// Ranking
//	salary_ranks := df.Rank("salary", "average") // Rank salaries
//
// # Advanced Data Analysis
//
// ## Window Functions
//
//	// Rolling window operations
//	rolling_mean := df.RollingWindow("salary", 3, "mean")  // 3-row rolling average
//	rolling_std := df.RollingWindow("price", 5, "std")     // 5-row rolling std dev
//
//	// Expanding window operations
//	expanding_sum := df.ExpandingWindow("sales", "sum")    // Cumulative sum
//	expanding_max := df.ExpandingWindow("price", "max")    // Running maximum
//
// ## Percent Changes and Differences
//
//	// Percentage change
//	pct_change := df.PctChange("price")  // Period-to-period % change
//
//	// Differences
//	diff := df.Diff("value", 1)          // First difference
//	diff2 := df.Diff("value", 2)         // Second difference (lag-2)
//
// ## Correlation and Covariance
//
//	// Correlation matrix
//	corr_matrix := df.Corr()             // Pearson correlation
//	spearman_corr := df.Corr("spearman") // Spearman rank correlation
//
//	// Covariance matrix
//	cov_matrix := df.Cov()
//
//	// Pairwise correlation
//	age_salary_corr := df.CorrPair("age", "salary")
//
// # Time Series Operations
//
// For DataFrames with DateTime index:
//
//	// Create time series DataFrame
//	dates := []string{"2023-01-01", "2023-01-02", "2023-01-03"}
//	dt_index, _ := series.NewDateTimeIndex(dates)
//	ts_df, _ := dataframe.FromArraysWithIndex(arrays, columns, dt_index)
//
//	// Time-based selection
//	jan_data := ts_df.DateRange("2023-01-01", "2023-01-31")
//	recent := ts_df.Last("7D")          // Last 7 days
//
//	// Resampling
//	daily_avg := ts_df.Resample("D", "mean")    // Daily averages
//	monthly_sum := ts_df.Resample("M", "sum")   // Monthly sums
//
//	// Time zone handling
//	localized := ts_df.TzLocalize("UTC")
//	converted := ts_df.TzConvert("US/Eastern")
//
// # I/O Operations Integration
//
//	// Save and load DataFrames
//	import "github.com/julianshen/gonp/io"
//
//	// CSV operations
//	err := io.WriteCSV(df, "output.csv", nil)
//	loaded_df, err := io.ReadCSV("input.csv")
//
//	// Parquet operations
//	err = io.WriteParquet(df, "data.parquet", nil)
//	parquet_df, err := io.ReadParquet("data.parquet")
//
//	// Database operations
//	err = io.WriteSQL(df, "employee_table", conn, nil)
//	sql_df, err := io.ReadSQL("SELECT * FROM employee_table", conn)
//
// # Visualization Integration
//
//	// Integration with visualization package
//	import "github.com/julianshen/gonp/visualization"
//
//	// Create plots from DataFrame
//	scatter := df.Plot("scatter", "age", "salary") // Scatter plot
//	hist := df.Plot("hist", "salary")              // Histogram
//	box := df.Plot("box", "salary")                // Box plot
//
//	// Group plots
//	group_plot := df.GroupBy("department").Plot("bar", "salary")
//
// # Performance Considerations
//
// ## Memory Efficiency
//
//	// DataFrames use memory-efficient storage
//	// - Columns are stored as separate arrays
//	// - Views share memory when possible
//	// - Copy operations are explicit
//
//	// Create views (no copying)
//	view := df.ILocSlice(100, 1000)     // View of rows 100-999
//	col_view := df.GetColumns(["col1", "col2"]) // View of specific columns
//
//	// Explicit copying
//	copy_df := df.Copy()                // Deep copy of entire DataFrame
//
// ## Performance Tips
//
//	// - Use column-wise operations when possible (vectorized)
//	// - Boolean indexing is optimized for large DataFrames
//	// - GroupBy operations use efficient algorithms
//	// - Join operations are optimized with hash tables
//	// - Use appropriate data types to minimize memory usage
//
// # Error Handling
//
// DataFrame operations provide comprehensive error handling:
//
//	// Column access errors
//	col, err := df.TryGetColumn("nonexistent")
//	if err != nil {
//		log.Printf("Column not found: %v", err)
//	}
//
//	// Join errors
//	result, err := df1.TryMerge(df2, "key", dataframe.InnerJoin)
//	if err != nil {
//		log.Printf("Join failed: %v", err)
//	}
//
//	// Type conversion errors
//	converted, err := df.TryAsType(map[string]string{"text_col": "int64"})
//	if err != nil {
//		log.Printf("Type conversion failed: %v", err)
//	}
//
// # Migration from Pandas
//
// Common Pandas DataFrame operations and their GoNP equivalents:
//
//	# Pandas                          # GoNP
//	import pandas as pd               import "github.com/julianshen/gonp/dataframe"
//
//	pd.DataFrame(data)                dataframe.FromMapInterface(data)
//	df.iloc[0]                        df.ILoc(0)
//	df.loc['key']                     df.Loc("key")
//	df['column']                      df.GetColumn("column")
//	df[df['col'] > 5]                 df.Where("col", func(x) bool { return x.(float64) > 5 })
//	df.mean()                         df.Mean()
//	df.groupby('col').mean()          df.GroupBy("col").Mean()
//	df1.merge(df2, on='key')          df1.Merge(df2, "key", dataframe.InnerJoin)
//	df.pivot_table(...)               df.PivotTable(...)
//	df.dropna()                       df.DropNa()
//	df.fillna(0)                      df.FillNa(0)
//	df.sort_values('col')             df.SortValues("col", true)
package dataframe
