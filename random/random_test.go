package random

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
)

// TestNewGenerator tests random number generator creation
func TestNewGenerator(t *testing.T) {
	tests := []struct {
		name string
		seed int64
	}{
		{
			name: "default seed",
			seed: 42,
		},
		{
			name: "zero seed",
			seed: 0,
		},
		{
			name: "negative seed",
			seed: -123,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// This should fail initially - we haven't implemented NewGenerator yet
			gen := NewGenerator(tt.seed)
			if gen == nil {
				t.Fatal("NewGenerator returned nil")
			}
		})
	}
}

// TestUniform tests uniform distribution generation
func TestUniform(t *testing.T) {
	tests := []struct {
		name string
		low  float64
		high float64
		size int
		seed int64
	}{
		{
			name: "0 to 1 range",
			low:  0.0,
			high: 1.0,
			size: 1000,
			seed: 42,
		},
		{
			name: "negative to positive",
			low:  -5.0,
			high: 5.0,
			size: 500,
			seed: 123,
		},
		{
			name: "large numbers",
			low:  1000.0,
			high: 2000.0,
			size: 100,
			seed: 456,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gen := NewGenerator(tt.seed)

			// This should fail initially - we haven't implemented Uniform yet
			result, err := gen.Uniform(tt.low, tt.high, tt.size)
			if err != nil {
				t.Fatalf("Uniform failed: %v", err)
			}

			// Check result properties
			if result.Size() != tt.size {
				t.Errorf("Size mismatch: got %d, want %d", result.Size(), tt.size)
			}

			// Check all values are in range
			for i := 0; i < result.Size(); i++ {
				val := result.At(i)
				floatVal, ok := val.(float64)
				if !ok {
					t.Fatalf("Expected float64, got %T", val)
				}

				if floatVal < tt.low || floatVal >= tt.high {
					t.Errorf("Value %v out of range [%v, %v)", floatVal, tt.low, tt.high)
				}
			}
		})
	}
}

// TestNormal tests normal distribution generation
func TestNormal(t *testing.T) {
	tests := []struct {
		name   string
		mean   float64
		stddev float64
		size   int
		seed   int64
	}{
		{
			name:   "standard normal",
			mean:   0.0,
			stddev: 1.0,
			size:   1000,
			seed:   42,
		},
		{
			name:   "shifted normal",
			mean:   10.0,
			stddev: 2.0,
			size:   500,
			seed:   123,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gen := NewGenerator(tt.seed)

			// This should fail initially - we haven't implemented Normal yet
			result, err := gen.Normal(tt.mean, tt.stddev, tt.size)
			if err != nil {
				t.Fatalf("Normal failed: %v", err)
			}

			// Check result properties
			if result.Size() != tt.size {
				t.Errorf("Size mismatch: got %d, want %d", result.Size(), tt.size)
			}

			// Calculate sample mean and std (rough check)
			if tt.size >= 100 {
				sum := 0.0
				for i := 0; i < result.Size(); i++ {
					val := result.At(i)
					floatVal, ok := val.(float64)
					if !ok {
						t.Fatalf("Expected float64, got %T", val)
					}
					sum += floatVal
				}

				sampleMean := sum / float64(tt.size)
				tolerance := 3.0 * tt.stddev / math.Sqrt(float64(tt.size)) // 3-sigma rule

				if math.Abs(sampleMean-tt.mean) > tolerance {
					t.Errorf("Sample mean %v differs too much from expected %v (tolerance: %v)",
						sampleMean, tt.mean, tolerance)
				}
			}
		})
	}
}

// TestExponential tests exponential distribution generation
func TestExponential(t *testing.T) {
	tests := []struct {
		name string
		rate float64
		size int
		seed int64
	}{
		{
			name: "rate 1.0",
			rate: 1.0,
			size: 1000,
			seed: 42,
		},
		{
			name: "rate 0.5",
			rate: 0.5,
			size: 500,
			seed: 123,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gen := NewGenerator(tt.seed)

			// This should fail initially - we haven't implemented Exponential yet
			result, err := gen.Exponential(tt.rate, tt.size)
			if err != nil {
				t.Fatalf("Exponential failed: %v", err)
			}

			// Check result properties
			if result.Size() != tt.size {
				t.Errorf("Size mismatch: got %d, want %d", result.Size(), tt.size)
			}

			// Check all values are non-negative
			for i := 0; i < result.Size(); i++ {
				val := result.At(i)
				floatVal, ok := val.(float64)
				if !ok {
					t.Fatalf("Expected float64, got %T", val)
				}

				if floatVal < 0 {
					t.Errorf("Negative value %v in exponential distribution", floatVal)
				}
			}
		})
	}
}

// TestBinomial tests binomial distribution generation
func TestBinomial(t *testing.T) {
	tests := []struct {
		name string
		n    int
		p    float64
		size int
		seed int64
	}{
		{
			name: "fair coin 10 flips",
			n:    10,
			p:    0.5,
			size: 1000,
			seed: 42,
		},
		{
			name: "biased coin 20 flips",
			n:    20,
			p:    0.3,
			size: 500,
			seed: 123,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gen := NewGenerator(tt.seed)

			// This should fail initially - we haven't implemented Binomial yet
			result, err := gen.Binomial(tt.n, tt.p, tt.size)
			if err != nil {
				t.Fatalf("Binomial failed: %v", err)
			}

			// Check result properties
			if result.Size() != tt.size {
				t.Errorf("Size mismatch: got %d, want %d", result.Size(), tt.size)
			}

			// Check all values are in valid range [0, n]
			for i := 0; i < result.Size(); i++ {
				val := result.At(i)
				intVal, ok := val.(int64)
				if !ok {
					t.Fatalf("Expected int64, got %T", val)
				}

				if intVal < 0 || intVal > int64(tt.n) {
					t.Errorf("Value %d out of range [0, %d]", intVal, tt.n)
				}
			}
		})
	}
}

// TestChoice tests random sampling without replacement
func TestChoice(t *testing.T) {
	tests := []struct {
		name    string
		data    []float64
		size    int
		replace bool
		seed    int64
	}{
		{
			name:    "sample without replacement",
			data:    []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			size:    5,
			replace: false,
			seed:    42,
		},
		{
			name:    "sample with replacement",
			data:    []float64{1, 2, 3},
			size:    10,
			replace: true,
			seed:    123,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gen := NewGenerator(tt.seed)
			dataArr, _ := array.FromSlice(tt.data)

			// This should fail initially - we haven't implemented Choice yet
			result, err := gen.Choice(dataArr, tt.size, tt.replace)
			if err != nil {
				t.Fatalf("Choice failed: %v", err)
			}

			// Check result properties
			if result.Size() != tt.size {
				t.Errorf("Size mismatch: got %d, want %d", result.Size(), tt.size)
			}

			// If without replacement, check no duplicates (when size <= len(data))
			if !tt.replace && tt.size <= len(tt.data) {
				seen := make(map[float64]bool)
				for i := 0; i < result.Size(); i++ {
					val := result.At(i)
					floatVal, ok := val.(float64)
					if !ok {
						t.Fatalf("Expected float64, got %T", val)
					}

					if seen[floatVal] {
						t.Errorf("Duplicate value %v found in sampling without replacement", floatVal)
					}
					seen[floatVal] = true
				}
			}
		})
	}
}

// TestShuffle tests array shuffling
func TestShuffle(t *testing.T) {
	tests := []struct {
		name string
		data []float64
		seed int64
	}{
		{
			name: "shuffle numbers",
			data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			seed: 42,
		},
		{
			name: "shuffle small array",
			data: []float64{1, 2, 3},
			seed: 123,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gen := NewGenerator(tt.seed)
			dataArr, _ := array.FromSlice(tt.data)

			// This should fail initially - we haven't implemented Shuffle yet
			result, err := gen.Shuffle(dataArr)
			if err != nil {
				t.Fatalf("Shuffle failed: %v", err)
			}

			// Check result properties
			if result.Size() != len(tt.data) {
				t.Errorf("Size mismatch: got %d, want %d", result.Size(), len(tt.data))
			}

			// Check that all original elements are still present
			originalCount := make(map[float64]int)
			for _, val := range tt.data {
				originalCount[val]++
			}

			resultCount := make(map[float64]int)
			for i := 0; i < result.Size(); i++ {
				val := result.At(i)
				floatVal, ok := val.(float64)
				if !ok {
					t.Fatalf("Expected float64, got %T", val)
				}
				resultCount[floatVal]++
			}

			for val, count := range originalCount {
				if resultCount[val] != count {
					t.Errorf("Element %v count mismatch: got %d, want %d", val, resultCount[val], count)
				}
			}
		})
	}
}

// TestPoisson tests Poisson distribution generation
func TestPoisson(t *testing.T) {
	tests := []struct {
		name   string
		lambda float64
		size   int
		seed   int64
	}{
		{
			name:   "lambda 1.0",
			lambda: 1.0,
			size:   1000,
			seed:   42,
		},
		{
			name:   "lambda 5.0",
			lambda: 5.0,
			size:   500,
			seed:   123,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gen := NewGenerator(tt.seed)

			// This should fail initially - we haven't implemented Poisson yet
			result, err := gen.Poisson(tt.lambda, tt.size)
			if err != nil {
				t.Fatalf("Poisson failed: %v", err)
			}

			// Check result properties
			if result.Size() != tt.size {
				t.Errorf("Size mismatch: got %d, want %d", result.Size(), tt.size)
			}

			// Check all values are non-negative integers
			for i := 0; i < result.Size(); i++ {
				val := result.At(i)
				intVal, ok := val.(int64)
				if !ok {
					t.Fatalf("Expected int64, got %T", val)
				}

				if intVal < 0 {
					t.Errorf("Negative value %d in Poisson distribution", intVal)
				}
			}
		})
	}
}

// TestGamma tests Gamma distribution generation
func TestGamma(t *testing.T) {
	tests := []struct {
		name  string
		shape float64
		scale float64
		size  int
		seed  int64
	}{
		{
			name:  "shape 2.0, scale 1.0",
			shape: 2.0,
			scale: 1.0,
			size:  1000,
			seed:  42,
		},
		{
			name:  "shape 1.0, scale 2.0",
			shape: 1.0,
			scale: 2.0,
			size:  500,
			seed:  123,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gen := NewGenerator(tt.seed)

			// This should fail initially - we haven't implemented Gamma yet
			result, err := gen.Gamma(tt.shape, tt.scale, tt.size)
			if err != nil {
				t.Fatalf("Gamma failed: %v", err)
			}

			// Check result properties
			if result.Size() != tt.size {
				t.Errorf("Size mismatch: got %d, want %d", result.Size(), tt.size)
			}

			// Check all values are positive
			for i := 0; i < result.Size(); i++ {
				val := result.At(i)
				floatVal, ok := val.(float64)
				if !ok {
					t.Fatalf("Expected float64, got %T", val)
				}

				if floatVal <= 0 {
					t.Errorf("Non-positive value %v in Gamma distribution", floatVal)
				}
			}
		})
	}
}

// TestBeta tests Beta distribution generation
func TestBeta(t *testing.T) {
	tests := []struct {
		name  string
		alpha float64
		beta  float64
		size  int
		seed  int64
	}{
		{
			name:  "alpha 2.0, beta 3.0",
			alpha: 2.0,
			beta:  3.0,
			size:  1000,
			seed:  42,
		},
		{
			name:  "symmetric beta 1.0, 1.0",
			alpha: 1.0,
			beta:  1.0,
			size:  500,
			seed:  123,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gen := NewGenerator(tt.seed)

			// This should fail initially - we haven't implemented Beta yet
			result, err := gen.Beta(tt.alpha, tt.beta, tt.size)
			if err != nil {
				t.Fatalf("Beta failed: %v", err)
			}

			// Check result properties
			if result.Size() != tt.size {
				t.Errorf("Size mismatch: got %d, want %d", result.Size(), tt.size)
			}

			// Check all values are in [0, 1]
			for i := 0; i < result.Size(); i++ {
				val := result.At(i)
				floatVal, ok := val.(float64)
				if !ok {
					t.Fatalf("Expected float64, got %T", val)
				}

				if floatVal < 0 || floatVal > 1 {
					t.Errorf("Value %v out of range [0, 1] in Beta distribution", floatVal)
				}
			}
		})
	}
}

// TestRandomSeed tests reproducibility with same seed
func TestRandomSeed(t *testing.T) {
	seed := int64(42)
	size := 100

	// Generate two sequences with same seed
	gen1 := NewGenerator(seed)
	result1, err := gen1.Uniform(0.0, 1.0, size)
	if err != nil {
		t.Fatalf("First Uniform failed: %v", err)
	}

	gen2 := NewGenerator(seed)
	result2, err := gen2.Uniform(0.0, 1.0, size)
	if err != nil {
		t.Fatalf("Second Uniform failed: %v", err)
	}

	// Check they're identical
	for i := 0; i < size; i++ {
		val1 := result1.At(i).(float64)
		val2 := result2.At(i).(float64)

		if val1 != val2 {
			t.Errorf("Values differ at index %d: %v != %v", i, val1, val2)
		}
	}
}
