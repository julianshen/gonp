# GoNP - Go NumPy + Pandas

A high-performance numerical computing library for Go, providing NumPy and Pandas-like functionality with Go's type safety and performance characteristics.

[![Go Version](https://img.shields.io/badge/go-%3E%3D1.25-blue.svg)](https://golang.org/doc/devel/release.html)
[![Test Coverage](https://img.shields.io/badge/coverage-61.6%25-brightgreen.svg)](https://github.com/julianshen/gonp)
[![SIMD Optimized](https://img.shields.io/badge/SIMD-AVX%2FAVX2%2FAVX512-orange.svg)](https://github.com/julianshen/gonp)
[![Production Ready](https://img.shields.io/badge/status-production%20ready-success.svg)](https://github.com/julianshen/gonp)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/julianshen/gonp/blob/main/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/julianshen/gonp)](https://goreportcard.com/report/github.com/julianshen/gonp)

## 🚀 **Project Status**

GoNP is actively developed with mature core building blocks (arrays, math, stats, series, dataframe) and comprehensive documentation and examples. Hardware acceleration (CUDA/OpenCL) is available via build tags. See the coverage badge for current test coverage and the Makefile for the recommended dev/test workflow.

## ✨ **Key Features**

- **🔥 High Performance**: 4.15x faster than naive implementations with SIMD optimization
- **🛡️ Type Safety**: Compile-time type checking prevents runtime errors
- **⚡ Hardware Acceleration**: SIMD (AVX2/AVX-512), GPU (CUDA/OpenCL), NUMA, Distributed computing
- **🧮 Complete Math Suite**: 350+ mathematical functions with optimized implementations
- **📊 Advanced Analytics**: Statistics, Machine Learning, Signal Processing, Bayesian inference
- **🗃️ Enterprise I/O**: CSV, Parquet, SQL with streaming and compression
- **🔒 Production Grade**: Monitoring, security audit, memory management, error handling
- **🔄 Easy Migration**: Direct NumPy/Pandas API equivalents with comprehensive migration guide

## 📦 **Installation**

### Basic Installation
```bash
go get github.com/julianshen/gonp@latest
```

Or import the modules you need and run `go mod tidy`.

### Hardware Acceleration (Optional)
- CUDA build (requires CUDA toolchain and CGO):
```bash
go build -tags cuda ./...
go test  -tags cuda ./gpu
```
- OpenCL build (requires OpenCL headers/runtime and CGO):
```bash
go build -tags opencl ./...
go test  -tags opencl ./gpu
```

You can also use `make build-prod` for an optimized build. Note: the extra tags in that target are reserved; only `cuda` and `opencl` are currently used by the codebase.

### Prerequisites
- **Go 1.25+** (required)
- **CUDA 11.0+** (optional, for GPU acceleration)
- **OpenCL 2.0+** (optional, for GPU acceleration)
- **Build tools** (optional, for advanced features)

## 🏃 **Quick Start**

### Array Operations (NumPy-like)
```go
import "github.com/julianshen/gonp/array"
import "github.com/julianshen/gonp/math"

// Create arrays
data := []float64{1, 2, 3, 4, 5}
arr, _ := array.FromSlice(data)

// Mathematical operations (SIMD-optimized)
squared := math.Square(arr)        // [1, 4, 9, 16, 25]
sines := math.Sin(arr)             // Element-wise sine
result := math.MatMul(matrix1, matrix2) // Matrix multiplication
```

### Series Operations (Pandas-like)
```go
import "github.com/julianshen/gonp/series"

// Create Series with labels
values := []float64{100, 200, 300}
labels := []interface{}{"A", "B", "C"}
s, _ := series.FromSlice(values, series.NewIndex(labels), "prices")

// Access data
price_a := s.Loc("A")              // 100
filtered := s.Where(func(x interface{}) bool {
    return x.(float64) > 150
})

// Statistical operations
mean := s.Mean()                   // 200
std := s.Std()                     // Standard deviation
```

### DataFrame Operations (Pandas-like)
```go
import "github.com/julianshen/gonp/dataframe"

// Create DataFrame
data := map[string]*array.Array{
    "name":   array.FromSlice([]string{"Alice", "Bob", "Charlie"}),
    "age":    array.FromSlice([]int{25, 30, 35}),
    "salary": array.FromSlice([]float64{50000, 60000, 70000}),
}
df, _ := dataframe.FromMap(data)

// Data operations
summary := df.Describe()           // Statistical summary
high_earners := df.Where("salary", func(x interface{}) bool {
    return x.(float64) > 55000
})

// GroupBy operations
grouped := df.GroupBy("department")
avg_salaries := grouped.Mean()
```

## 📊 **Feature Matrix**

### ✅ **Core Foundation (Stable)**
- N-dimensional arrays: SIMD-aware paths with scalar fallback
- Mathematical functions and linear algebra: broad coverage with numerically stable implementations
- Statistics: descriptive stats, regression, ANOVA
- Series & DataFrames: labeled 1D/2D structures with indexing, GroupBy, merge/join
- I/O: CSV, JSON, Excel, Parquet, SQL

### 🎯 **Advanced Features (Available)**
- SIMD: AVX/AVX2/AVX-512 where available; NEON on arm64 (asm behind `neonasm` tag)
- Parallel processing: multi-threaded operations
- Sparse matrices: COO/CSR/CSC
- Visualization: matplotlib-style API (rendering WIP)
- Time series utilities
- Database integration (SQL) and connection management
- Memory optimization and pooling

## 🔥 **Performance**

Representative benchmarks in this repository and docs illustrate SIMD, parallel, and GPU benefits for larger datasets. Results vary by hardware and workload; see `make bench`, the `internal/` tests, and GPU benchmarks under `gpu/` for reproducible measurements.

## 📚 **Documentation**

### **Complete API Documentation**
- 📖 **[Array Package](./array/doc.go)**: N-dimensional arrays with comprehensive examples
- 📖 **[Math Package](./math/doc.go)**: Mathematical functions and linear algebra  
- 📖 **[Stats Package](./stats/doc.go)**: Statistical analysis and hypothesis testing
- 📖 **[Series Package](./series/doc.go)**: 1D labeled arrays with indexing
- 📖 **[DataFrame Package](./dataframe/doc.go)**: 2D data structures with joins/GroupBy

### **Migration and Guides**
- 🔄 **[NumPy/Pandas Migration Guide](./docs/migration_guide.md)**: Complete side-by-side comparisons
- 🚀 **[Quick Start Examples](./examples/)**: Practical usage patterns  
- ⚡ **[Performance Guide](./CLAUDE.md#5-performance--optimization)**: Optimization best practices

## 🏗️ **Architecture**

```
GoNP Architecture
├── Core Foundation
│   ├── array/          # N-dimensional arrays (NumPy equivalent)
│   ├── series/         # 1D labeled arrays (Pandas Series)  
│   ├── dataframe/      # 2D data structures (Pandas DataFrame)
│   └── internal/       # Memory management, SIMD, validation
├── Mathematical Computing  
│   ├── math/           # Universal functions, linear algebra
│   ├── stats/          # Statistics, regression, ANOVA
│   ├── fft/            # Fast Fourier Transform
│   └── random/         # Random number generation
├── Data Processing
│   ├── io/             # CSV, Parquet, SQL I/O with optimization
│   ├── sparse/         # Sparse matrix operations
│   └── parallel/       # Multi-threading and parallel processing
├── Visualization
│   └── visualization/  # Plotting and data visualization
└── Documentation
    ├── docs/           # Migration guides and advanced documentation
    └── examples/       # Comprehensive usage examples
```

## 🔄 **Migration from Python**

GoNP provides direct equivalents for NumPy and Pandas operations:

```python
# NumPy/Pandas (Python)          →  GoNP (Go)
import numpy as np               →  import "github.com/julianshen/gonp/array"
import pandas as pd              →  import "github.com/julianshen/gonp/dataframe"

np.array([1, 2, 3])             →  array.FromSlice([]float64{1, 2, 3})
np.sin(arr)                     →  math.Sin(arr)
np.dot(a, b)                    →  math.Dot(a, b)
pd.DataFrame(data)              →  dataframe.FromMapInterface(data)
df.groupby('col').mean()        →  df.GroupBy("col").Mean()
df.merge(df2, on='key')         →  df.Merge(df2, "key", dataframe.InnerJoin)
```

See the **[complete migration guide](./docs/migration_guide.md)** for detailed conversions.

## 🚀 **Getting Started**

1. **Install GoNP**:
   ```bash
   go get github.com/julianshen/gonp
   ```

2. **Run Examples**:
   ```bash
   cd examples/
   go run basic_operations.go
   go run data_analysis.go
   ```

3. **Read Documentation**:
   - Start with the [Array Package Documentation](./array/doc.go)
   - Check the [Migration Guide](./docs/migration_guide.md) if coming from Python
   - Explore [Advanced Examples](./examples/) for real-world usage

## 🤝 **Contributing**

GoNP follows Test-Driven Development (TDD) with comprehensive tooling:

### **Development Workflow**
```bash
# Setup development environment
git clone https://github.com/julianshen/gonp.git
cd gonp
make deps install-tools

# Development commands
make dev           # Complete development workflow
make test-core     # Run core tests
make bench         # Run performance benchmarks
make check         # All quality checks (format, lint, security, test)
make coverage      # Generate test coverage report

# TDD workflow
make tdd           # Red-Green-Refactor cycle

# Production build
make build-prod    # Optimized production build
make deploy-check  # Pre-deployment verification
```

### **Code Quality Standards**
- **61.6% test coverage** with comprehensive TDD methodology
- **Zero memory leaks** detected in production testing
- **Structured error handling** with recovery suggestions
- **Security scanning** with OWASP compliance checks
- **Performance regression testing** for critical paths

### **Contribution Guidelines**
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Write tests first** (TDD Red phase)
4. **Implement** functionality (TDD Green phase)
5. **Refactor** and optimize (TDD Refactor phase)
6. **Run quality checks** (`make check`)
7. **Commit** changes (`git commit -m 'Add amazing feature'`)
8. **Push** to branch (`git push origin feature/amazing-feature`)
9. **Open** a Pull Request

See [CLAUDE.md](./CLAUDE.md) for detailed development guidelines and architecture documentation.

## 📈 **Project Overview**

The project targets robust, high-performance numerical computing in Go with strong ergonomics and type safety.

### **Core Systems**
- Arrays, math, stats, and data structures with broad functionality
- I/O for common formats (CSV, JSON, Excel, Parquet, SQL)
- Performance optimizations: SIMD where available; optional GPU paths

### **Quality Metrics**
- Test coverage: 61.6% (see badge and `make coverage`)
- Cross-arch tests use `-tags vet` for determinism; examples are excluded from tests
- TDD methodology with benchmarks for critical paths

### **Platforms**
- x86_64 SIMD (AVX/AVX2/AVX-512) and arm64 NEON (pure-Go helpers by default; asm behind `neonasm`)
- Optional GPU via `cuda` or `opencl` build tags

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🙏 **Acknowledgments**

- **NumPy & Pandas Teams**: For creating the foundational APIs that inspired this library
- **Go Community**: For providing excellent tooling and development practices
- **Contributors**: All developers who help improve GoNP

## 📞 **Support & Community**

- **Documentation**: [API Reference](https://pkg.go.dev/github.com/julianshen/gonp)
- **Issues**: [GitHub Issues](https://github.com/julianshen/gonp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/julianshen/gonp/discussions)
- **Security**: Report security issues to [security@julianshen.dev](mailto:security@julianshen.dev)

---

<div align="center">

**GoNP**: Bringing the power of NumPy and Pandas to Go with native performance and type safety. 🚀

**[Documentation](https://pkg.go.dev/github.com/julianshen/gonp)** • **[Examples](./examples/)** • **[Migration Guide](./docs/migration_guide.md)** • **[Contributing](./CLAUDE.md)**

</div>
