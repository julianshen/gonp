# NumPy/Pandas to GoNP Migration Guide

This guide helps you transition from Python's NumPy/Pandas ecosystem to GoNP, providing direct comparisons and equivalent operations.

## Table of Contents

1. [Basic Setup and Imports](#basic-setup-and-imports)
2. [Array Creation](#array-creation) 
3. [Array Properties and Information](#array-properties-and-information)
4. [Element Access and Indexing](#element-access-and-indexing)
5. [Mathematical Operations](#mathematical-operations)
6. [Statistical Functions](#statistical-functions)
7. [Linear Algebra](#linear-algebra)
8. [Array Manipulation](#array-manipulation)
9. [Series Operations (Pandas)](#series-operations-pandas)
10. [DataFrame Operations (Pandas)](#dataframe-operations-pandas)
11. [I/O Operations](#io-operations)
12. [Performance Considerations](#performance-considerations)

---

## Basic Setup and Imports

### NumPy/Pandas
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### GoNP
```go
import (
    "github.com/julianshen/gonp/array"
    "github.com/julianshen/gonp/series"  
    "github.com/julianshen/gonp/dataframe"
    "github.com/julianshen/gonp/math"
    "github.com/julianshen/gonp/stats"
    "github.com/julianshen/gonp/visualization"
)
```

---

## Array Creation

### From Lists/Slices

| NumPy | GoNP |
|-------|------|
| `np.array([1, 2, 3, 4, 5])` | `array.FromSlice([]float64{1, 2, 3, 4, 5})` |
| `np.array([[1, 2], [3, 4]])` | `array.FromSlice([][]float64{{1, 2}, {3, 4}})` |

### Special Arrays

| NumPy | GoNP |
|-------|------|
| `np.zeros((3, 4))` | construct via `array.FromSlice(make([]float64, 12))` then reshape |
| `np.ones((2, 3))` | construct via `array.FromSlice([]float64{1,1,1,1,1,1})` then reshape |
| `np.empty((3, 3))` | construct via `array.FromSlice(make([]float64, 9))` |
| `np.eye(4)` | build from slices; helper may be added in future |
| `np.linspace(0, 10, 50)` | `array.Linspace(0, 10, 50)` |
| `np.arange(0, 10, 0.5)` | `array.Arange(0, 10, 0.5)` |

### Data Types

| NumPy | GoNP |
|-------|------|
| `dtype=np.float64` | use `[]float64` with `array.FromSlice` |
| `dtype=np.float32` | use `[]float32` with `array.FromSlice` |
| `dtype=np.int64` | use `[]int64` with `array.FromSlice` |
| `dtype=np.int32` | use `[]int32` with `array.FromSlice` |
| `dtype=np.bool` | use `[]bool` with `array.FromSlice` |
| `dtype=np.complex128` | use `[]complex128` with `array.FromSlice` |

**Example:**
```python
# NumPy
arr = np.array([1, 2, 3], dtype=np.float32)
```

```go
// GoNP
arr, _ := array.FromSlice([]float32{1, 2, 3})
```

---

## Array Properties and Information

| NumPy | GoNP |
|-------|------|
| `arr.shape` | `arr.Shape()` |
| `arr.ndim` | `arr.Ndim()` |
| `arr.size` | `arr.Size()` |
| `arr.dtype` | `arr.DType()` |
| `len(arr)` | `arr.Shape()[0]` (for first dimension) |

**Example:**
```python
# NumPy
print(f"Shape: {arr.shape}, Size: {arr.size}, Dims: {arr.ndim}")
```

```go
// GoNP  
fmt.Printf("Shape: %v, Size: %d, Dims: %d\n", 
    arr.Shape(), arr.Size(), arr.Ndim())
```

---

## Element Access and Indexing

### Basic Indexing

| NumPy | GoNP |
|-------|------|
| `arr[2]` | `arr.At(2)` |
| `arr[1, 3]` | `arr.At(1, 3)` |
| `arr[2] = 5` | `arr.Set(5, 2)` |
| `arr[1, 3] = 10` | `arr.Set(10, 1, 3)` |

### Slicing

| NumPy | GoNP |
|-------|------|
| `arr[1:4]` | slice via helper methods; see array indexing API |
| `arr[::2]` | slice via helper methods; see array indexing API |
| `arr[:, 1:3]` | slice via helper methods; see array indexing API |

### Boolean Indexing

| NumPy | GoNP |
|-------|------|
| `arr[arr > 5]` | `mask := arr.Greater(5); arr.BooleanIndex(mask)` |
| `arr[arr < 0] = 0` | `mask := arr.Less(0); arr.SetMask(mask, 0)` |

### Fancy Indexing

| NumPy | GoNP |
|-------|------|
| `arr[[0, 2, 4]]` | `indices, _ := array.FromSlice([]int{0, 2, 4}); arr.FancyIndex(indices)` |

**Example:**
```python
# NumPy
result = arr[arr > np.mean(arr)]
```

```go
// GoNP
mean_val := stats.Mean(arr)
mask := arr.Greater(mean_val)
result, _ := arr.BooleanIndex(mask)
```

---

## Mathematical Operations

### Element-wise Operations

| NumPy | GoNP |
|-------|------|
| `arr1 + arr2` | `arr1.Add(arr2)` |
| `arr1 - arr2` | `arr1.Sub(arr2)` |
| `arr1 * arr2` | `arr1.Mul(arr2)` |
| `arr1 / arr2` | `arr1.Div(arr2)` |
| `arr1 ** arr2` | `arr1.Pow(arr2)` |
| `arr + 5` | `arr.AddScalar(5)` |
| `arr * 2.5` | `arr.MulScalar(2.5)` |

### Universal Functions

| NumPy | GoNP |
|-------|------|
| `np.sin(arr)` | `math.Sin(arr)` |
| `np.cos(arr)` | `math.Cos(arr)` |
| `np.exp(arr)` | `math.Exp(arr)` |
| `np.log(arr)` | `math.Log(arr)` |
| `np.sqrt(arr)` | `math.Sqrt(arr)` |
| `np.power(arr, 2)` | `math.Pow(arr, 2)` |
| `np.abs(arr)` | `math.Abs(arr)` |

### Comparison Operations

| NumPy | GoNP |
|-------|------|
| `arr1 > arr2` | `arr1.Greater(arr2)` |
| `arr1 == arr2` | `arr1.Equal(arr2)` |
| `arr1 != arr2` | `arr1.NotEqual(arr2)` |
| `np.maximum(arr1, arr2)` | `math.Maximum(arr1, arr2)` |
| `np.minimum(arr1, arr2)` | `math.Minimum(arr1, arr2)` |

**Example:**
```python
# NumPy
result = np.sin(arr) ** 2 + np.cos(arr) ** 2
```

```go  
// GoNP
sin_arr, _ := math.Sin(arr)
cos_arr, _ := math.Cos(arr)
sin_squared, _ := sin_arr.Pow(arr.FromScalar(2))
cos_squared, _ := cos_arr.Pow(arr.FromScalar(2))
result, _ := sin_squared.Add(cos_squared)
```

---

## Statistical Functions

| NumPy | GoNP |
|-------|------|
| `np.sum(arr)` | `arr.Sum()` or `stats.Sum(arr)` |
| `np.mean(arr)` | `arr.Mean()` or `stats.Mean(arr)` |
| `np.std(arr)` | `arr.Std()` or `stats.StdDev(arr)` |
| `np.var(arr)` | `arr.Var()` or `stats.Variance(arr)` |
| `np.min(arr)` | `arr.Min()` or `stats.Min(arr)` |
| `np.max(arr)` | `arr.Max()` or `stats.Max(arr)` |
| `np.median(arr)` | `stats.Median(arr)` |
| `np.percentile(arr, 75)` | `stats.Percentile(arr, 0.75)` |

### Along Axes

| NumPy | GoNP |
|-------|------|
| `np.sum(arr, axis=0)` | `arr.Sum(0)` |
| `np.mean(arr, axis=1)` | `arr.Mean(1)` |
| `np.std(arr, axis=0)` | `arr.Std(0)` |

### Correlation

| NumPy | GoNP |
|-------|------|
| `np.corrcoef(x, y)` | `stats.Correlation(x, y)` |
| `scipy.stats.spearmanr(x, y)` | `stats.SpearmanCorrelation(x, y)` |
| `scipy.stats.kendalltau(x, y)` | `stats.KendallTau(x, y)` |

**Example:**
```python
# NumPy
stats_summary = {
    'mean': np.mean(data),
    'std': np.std(data),
    'min': np.min(data),
    'max': np.max(data)
}
```

```go
// GoNP
mean, _ := stats.Mean(data)
std, _ := stats.Std(data)
min, _ := stats.Min(data)
max, _ := stats.Max(data)
stats_summary := map[string]float64{"mean": mean, "std": std, "min": min, "max": max}
```

---

## Linear Algebra

| NumPy | GoNP |
|-------|------|
| `np.dot(a, b)` | `math.Dot(a, b)` |
| `a @ b` | `math.MatMul(a, b)` |
| `np.linalg.inv(matrix)` | `math.Inv(matrix)` |
| `np.linalg.pinv(matrix)` | `math.Pinv(matrix)` |
| `np.linalg.det(matrix)` | `math.Det(matrix)` |
| `np.trace(matrix)` | `math.Trace(matrix)` |
| `np.linalg.norm(matrix)` | `math.Norm(matrix)` |

### Decompositions

| NumPy | GoNP |
|-------|------|
| `u, s, vt = np.linalg.svd(matrix)` | `u, s, vt := math.SVD(matrix)` |
| `q, r = np.linalg.qr(matrix)` | `q, r := math.QR(matrix)` |
| `p, l, u = scipy.linalg.lu(matrix)` | `p, l, u := math.LU(matrix)` |
| `l = np.linalg.cholesky(matrix)` | `l := math.Cholesky(matrix)` |

### Solving Systems

| NumPy | GoNP |
|-------|------|
| `np.linalg.solve(A, b)` | `math.Solve(A, b)` |
| `np.linalg.lstsq(A, b)` | `math.LstSq(A, b)` |

**Example:**
```python
# NumPy
# Solve Ax = b
x = np.linalg.solve(A, b)
residual = np.linalg.norm(A @ x - b)
```

```go
// GoNP  
// Solve Ax = b
x, err := math.Solve(A, b)
if err != nil {
    log.Fatal(err)
}
Ax, _ := math.MatMul(A, x)
residualVec, _ := Ax.Sub(b)
residual, _ := math.Norm(residualVec, nil)
```

---

## Array Manipulation

### Shape Operations

| NumPy | GoNP |
|-------|------|
| `arr.reshape(2, 4)` | `arr.Reshape(/* shape helper */)` |
| `arr.T` | `arr.Transpose()` |
| `arr.flatten()` | `arr.Flatten()` |
| `np.squeeze(arr)` | `arr.Squeeze()` |
| `np.expand_dims(arr, 1)` | `arr.ExpandDims(1)` |

### Concatenation and Splitting

| NumPy | GoNP |
|-------|------|
| `np.concatenate([arr1, arr2])` | `array.Concatenate([]*array.Array{arr1, arr2}, 0)` |
| `np.vstack([arr1, arr2])` | `array.VStack([]*array.Array{arr1, arr2})` |
| `np.hstack([arr1, arr2])` | `array.HStack([]*array.Array{arr1, arr2})` |
| `np.split(arr, 3)` | `array.Split(arr, 3, 0)` |

### Copying

| NumPy | GoNP |
|-------|------|
| `arr.copy()` | `arr.Copy()` |
| `np.asarray(data)` | `array.FromSlice(data)` |

**Example:**
```python
# NumPy
reshaped = arr.reshape(-1, 1).T.flatten()
```

```go
// GoNP
reshaped := arr.Reshape(internal.Shape{-1, 1}).Transpose().Flatten()
```

---

## Series Operations (Pandas)

### Creation

| Pandas | GoNP |
|--------|------|
| `pd.Series([1, 2, 3, 4])` | `series.NewSeries(array.FromSlice([]float64{1, 2, 3, 4}), nil, "")` |
| `pd.Series(data, index=['a', 'b', 'c'])` | `series.NewSeries(arr, series.NewIndex([]interface{}{"a", "b", "c"}), "")` |
| `pd.Series(data, name='values')` | `series.NewSeries(arr, nil, "values")` |

### Indexing and Selection

| Pandas | GoNP |
|--------|------|
| `s.iloc[2]` | `s.ILoc(2)` |
| `s.loc['key']` | `s.Loc("key")` |
| `s[s > 5]` | `s.Where(func(x interface{}) bool { return x.(float64) > 5 })` |

### Operations

| Pandas | GoNP |
|--------|------|
| `s.mean()` | `s.Mean()` |
| `s.std()` | `s.Std()` |
| `s.sum()` | `s.Sum()` |
| `s.value_counts()` | `s.ValueCounts()` |
| `s.unique()` | `s.Unique()` |

### String Methods

| Pandas | GoNP |
|--------|------|
| `s.str.upper()` | `s.Str().Upper()` |
| `s.str.contains('pattern')` | `s.Str().Contains("pattern")` |
| `s.str.len()` | `s.Str().Len()` |
| `s.str.split(',')` | `s.Str().Split(",")` |

**Example:**
```python
# Pandas
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'], name='values')
filtered = s[s > s.mean()]
```

```go
// GoNP
arr, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
index := series.NewIndex([]interface{}{"a", "b", "c", "d", "e"})
s, _ := series.NewSeries(arr, index, "values")

mean_val := s.Mean()
filtered := s.Where(func(x interface{}) bool {
    return x.(float64) > mean_val
})
```

---

## DataFrame Operations (Pandas)

### Creation

| Pandas | GoNP |
|--------|------|
| `pd.DataFrame({'A': [1, 2], 'B': [3, 4]})` | `dataframe.FromMap(map[string]*array.Array{"A": arr1, "B": arr2})` |
| `pd.DataFrame(data, columns=cols)` | `dataframe.FromArrays(arrays, columns)` |

### Selection and Indexing

| Pandas | GoNP |
|--------|------|
| `df['column']` | `df.GetColumn("column")` |
| `df.iloc[2, 1]` | `df.ILoc(2, 1)` |
| `df.loc[idx, 'col']` | `df.Loc(idx, "col")` |
| `df[df['A'] > 5]` | `df.Where("A", func(x interface{}) bool { return x.(float64) > 5 })` |

### Operations

| Pandas | GoNP |
|--------|------|
| `df.mean()` | use groupby aggregations or column-level stats |
| `df.describe()` | derive via column stats (per-series) |
| `df.groupby('column').sum()` | `df.GroupBy("column").Sum()` |
| `df.merge(other, on='key')` | `dataframe.InnerJoin(df, other, "key", "key")` |

### Manipulation

| Pandas | GoNP |
|--------|------|
| `df.drop(columns=['A'])` | not yet available |
| `df.rename(columns={'A': 'X'})` | not yet available |
| `df.pivot_table(values='V', index='I', columns='C')` | `df.PivotTable("I", "C", "V", "mean")` |

**Example:**
```python  
# Pandas
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B'],
    'value': [1, 2, 3, 4]
})
result = df.groupby('group').mean()
```

```go
// GoNP
group_arr, _ := array.FromSlice([]string{"A", "A", "B", "B"})
value_arr, _ := array.FromSlice([]float64{1, 2, 3, 4})

df, _ := dataframe.FromMap(map[string]*array.Array{
    "group": group_arr,
    "value": value_arr,
})

result := df.GroupBy("group").Mean()
```

---

## I/O Operations  

### CSV Files

| Pandas | GoNP |
|--------|------|
| `pd.read_csv('file.csv')` | `io.ReadCSV("file.csv")` |
| `df.to_csv('file.csv')` | `io.WriteCSV(df, "file.csv")` |

### Other Formats

| Pandas | GoNP |
|--------|------|
| `pd.read_parquet('file.parquet')` | `io.ReadParquet("file.parquet")` |
| `df.to_parquet('file.parquet')` | `io.WriteParquet("file.parquet", df)` |
| `pd.read_sql(query, conn)` | `io.ReadSQL(query, conn)` |
| `df.to_sql('table', conn)` | `io.WriteSQL(df, "table", conn, nil)` |

**Example:**
```python
# Pandas
df = pd.read_csv('data.csv')
df.to_parquet('output.parquet', compression='snappy')
```

```go
// GoNP
df, err := io.ReadCSV("data.csv")
if err != nil {
    log.Fatal(err)
}

options := &io.ParquetWriteOptions{
    Compression: "snappy",
}
err = io.WriteParquet(df, "output.parquet", options)
```

---

## Performance Considerations

### Memory Management

| NumPy/Pandas | GoNP |
|--------------|------|
| Views share memory | Use `arr.Slice()` for views |
| `arr.copy()` for deep copy | `arr.Copy()` for deep copy |
| `del arr` for cleanup | Go's garbage collector handles cleanup |

### Performance Tips

| NumPy/Pandas | GoNP |
|--------------|------|
| Use vectorized operations | Use array methods, avoid loops |
| `@` operator for matrix multiply | `math.MatMul()` or `math.Dot()` |
| Use `numba` for acceleration | Built-in SIMD optimization |

### Memory Profiling

```python
# NumPy/Pandas
import memory_profiler
%memit operation()
```

```go
// GoNP
// Use Go's pprof or your APM of choice for memory profiling
// Example: go test -bench . -benchmem ./internal
```

### Benchmarking

```python
# NumPy/Pandas  
%timeit operation()
```

```go
// GoNP
func BenchmarkOperation(b *testing.B) {
    for i := 0; i < b.N; i++ {
        operation()
    }
}
```

---

## Key Differences and Go-Specific Considerations

### Error Handling

NumPy/Pandas often raise exceptions, while GoNP returns errors:

```python
# Python - can raise exceptions
result = np.linalg.inv(singular_matrix)  # Raises LinAlgError
```

```go
// Go - explicit error handling
result, err := math.Inv(singular_matrix)
if err != nil {
    log.Printf("Matrix inversion failed: %v", err)
    return
}
```

### Type Safety

GoNP provides compile-time type safety:

```python
# Python - runtime type checking
arr = np.array([1, 2, 3])
arr[0] = "string"  # Might cause issues later
```

```go
// Go - compile-time type safety  
arr, _ := array.FromSlice([]float64{1, 2, 3})
// arr.Set("string", 0)  // Compile error! Type mismatch
```

### Memory Management

Go's garbage collector handles memory automatically:

```python
# Python - manual memory management sometimes needed
del large_array
gc.collect()
```

```go
// Go - automatic garbage collection
// No manual memory management needed
// large_array goes out of scope and is automatically cleaned up
```

### Concurrency

GoNP provides built-in thread safety for reads:

```python
# NumPy - manual thread safety needed
import threading
lock = threading.Lock()
```

```go
// Go - built-in goroutine support
go func() {
    // Concurrent read operations are safe
    result := arr.Sum()
    // Send result through channel, etc.
}()
```

---

## Migration Checklist

When migrating from NumPy/Pandas to GoNP:

### ✅ Preparation
- [ ] Identify all NumPy/Pandas operations in your codebase
- [ ] Review data types and ensure Go compatibility
- [ ] Plan error handling strategy
- [ ] Consider performance requirements

### ✅ Code Migration  
- [ ] Replace imports and setup code
- [ ] Convert array creation methods
- [ ] Update mathematical operations
- [ ] Migrate statistical functions
- [ ] Convert DataFrame operations
- [ ] Update I/O operations
- [ ] Add proper error handling

### ✅ Testing and Validation
- [ ] Create comprehensive tests
- [ ] Validate numerical accuracy
- [ ] Benchmark performance improvements
- [ ] Test with real datasets
- [ ] Verify memory usage

### ✅ Optimization
- [ ] Profile performance hotspots  
- [ ] Utilize SIMD optimizations
- [ ] Implement concurrent processing where applicable
- [ ] Optimize memory allocation patterns

---

## Getting Help

- **Documentation**: Check package-level documentation for detailed API reference
- **Examples**: See `examples/` directory for comprehensive usage patterns  
- **Performance**: Use built-in profiling tools for optimization
- **Community**: GitHub issues and discussions for questions and contributions

The GoNP library aims to provide familiar functionality with Go's performance and type safety benefits. Most NumPy/Pandas patterns have direct GoNP equivalents with similar or better performance characteristics.
