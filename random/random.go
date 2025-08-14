package random

import (
	"fmt"
	"math"

	"github.com/julianshen/gonp/array"
)

// Generator represents a pseudo-random number generator
type Generator struct {
	state uint64 // PCG state
	inc   uint64 // PCG increment (must be odd)
}

// NewGenerator creates a new random number generator with the given seed
func NewGenerator(seed int64) *Generator {
	gen := &Generator{
		state: 0,
		inc:   1, // Default increment (odd number)
	}
	gen.Seed(seed)
	return gen
}

// Seed initializes the generator with a seed value
func (g *Generator) Seed(seed int64) {
	g.state = uint64(seed)
	g.inc = 1 // Ensure odd increment
	// Advance the state once to mix the seed
	g.next()
}

// next generates the next random uint32 using PCG algorithm
func (g *Generator) next() uint32 {
	// PCG algorithm: https://www.pcg-random.org/
	oldstate := g.state
	g.state = oldstate*6364136223846793005 + g.inc
	xorshifted := uint32(((oldstate >> 18) ^ oldstate) >> 27)
	rot := uint32(oldstate >> 59)
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
}

// Float64 generates a random float64 in [0, 1)
func (g *Generator) Float64() float64 {
	// Use upper 53 bits for double precision
	return float64(g.next()>>5) * (1.0 / (1 << 27))
}

// Uniform generates an array of uniformly distributed random numbers in [low, high)
func (g *Generator) Uniform(low, high float64, size int) (*array.Array, error) {
	if size <= 0 {
		return nil, fmt.Errorf("size must be positive, got %d", size)
	}
	if low >= high {
		return nil, fmt.Errorf("low must be less than high: %v >= %v", low, high)
	}

	result := make([]float64, size)
	scale := high - low

	for i := 0; i < size; i++ {
		result[i] = low + g.Float64()*scale
	}

	return array.FromSlice(result)
}

// Normal generates an array of normally distributed random numbers using Box-Muller transform
func (g *Generator) Normal(mean, stddev float64, size int) (*array.Array, error) {
	if size <= 0 {
		return nil, fmt.Errorf("size must be positive, got %d", size)
	}
	if stddev <= 0 {
		return nil, fmt.Errorf("standard deviation must be positive, got %v", stddev)
	}

	result := make([]float64, size)

	// Box-Muller transform generates pairs, so we need to handle odd sizes
	for i := 0; i < size; i += 2 {
		// Generate two uniform random numbers
		u1 := g.Float64()
		u2 := g.Float64()

		// Ensure u1 is not zero to avoid log(0)
		for u1 == 0 {
			u1 = g.Float64()
		}

		// Box-Muller transformation
		mag := stddev * math.Sqrt(-2.0*math.Log(u1))
		z0 := mag * math.Cos(2.0*math.Pi*u2)
		z1 := mag * math.Sin(2.0*math.Pi*u2)

		result[i] = mean + z0
		if i+1 < size {
			result[i+1] = mean + z1
		}
	}

	return array.FromSlice(result)
}

// Exponential generates an array of exponentially distributed random numbers
func (g *Generator) Exponential(rate float64, size int) (*array.Array, error) {
	if size <= 0 {
		return nil, fmt.Errorf("size must be positive, got %d", size)
	}
	if rate <= 0 {
		return nil, fmt.Errorf("rate must be positive, got %v", rate)
	}

	result := make([]float64, size)

	for i := 0; i < size; i++ {
		u := g.Float64()
		// Ensure u is not zero to avoid log(0)
		for u == 0 {
			u = g.Float64()
		}
		// Inverse transform sampling: -ln(1-u)/rate
		// Since u is uniform [0,1), (1-u) is also uniform [0,1)
		result[i] = -math.Log(u) / rate
	}

	return array.FromSlice(result)
}

// Binomial generates an array of binomial distributed random numbers
func (g *Generator) Binomial(n int, p float64, size int) (*array.Array, error) {
	if size <= 0 {
		return nil, fmt.Errorf("size must be positive, got %d", size)
	}
	if n < 0 {
		return nil, fmt.Errorf("n must be non-negative, got %d", n)
	}
	if p < 0 || p > 1 {
		return nil, fmt.Errorf("p must be in [0, 1], got %v", p)
	}

	result := make([]int64, size)

	for i := 0; i < size; i++ {
		// Simple method: sum of Bernoulli trials
		count := int64(0)
		for j := 0; j < n; j++ {
			if g.Float64() < p {
				count++
			}
		}
		result[i] = count
	}

	return array.FromSlice(result)
}

// Poisson generates an array of Poisson distributed random numbers
func (g *Generator) Poisson(lambda float64, size int) (*array.Array, error) {
	if size <= 0 {
		return nil, fmt.Errorf("size must be positive, got %d", size)
	}
	if lambda <= 0 {
		return nil, fmt.Errorf("lambda must be positive, got %v", lambda)
	}

	result := make([]int64, size)

	// For small lambda, use Knuth's algorithm
	// For large lambda, use acceptance-rejection method
	if lambda < 30 {
		// Knuth's algorithm
		L := math.Exp(-lambda)
		for i := 0; i < size; i++ {
			k := int64(0)
			p := 1.0
			for p > L {
				k++
				u := g.Float64()
				p *= u
			}
			result[i] = k - 1
		}
	} else {
		// Approximate with normal distribution for large lambda
		// Poisson(λ) ≈ Normal(λ, λ) for large λ
		for i := 0; i < size; i++ {
			// Box-Muller for normal
			u1 := g.Float64()
			u2 := g.Float64()
			for u1 == 0 {
				u1 = g.Float64()
			}

			z := math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
			val := lambda + math.Sqrt(lambda)*z

			// Ensure non-negative integer
			if val < 0 {
				val = 0
			}
			result[i] = int64(math.Round(val))
		}
	}

	return array.FromSlice(result)
}

// Gamma generates an array of Gamma distributed random numbers using Marsaglia-Tsang method
func (g *Generator) Gamma(shape, scale float64, size int) (*array.Array, error) {
	if size <= 0 {
		return nil, fmt.Errorf("size must be positive, got %d", size)
	}
	if shape <= 0 {
		return nil, fmt.Errorf("shape must be positive, got %v", shape)
	}
	if scale <= 0 {
		return nil, fmt.Errorf("scale must be positive, got %v", scale)
	}

	result := make([]float64, size)

	for i := 0; i < size; i++ {
		result[i] = g.gammaRand(shape) * scale
	}

	return array.FromSlice(result)
}

// gammaRand generates a single Gamma(shape, 1) random number
func (g *Generator) gammaRand(shape float64) float64 {
	if shape < 1 {
		// Use rejection method for shape < 1
		// Generate Gamma(shape + 1) and multiply by uniform^(1/shape)
		gamma1 := g.gammaRand(shape + 1)
		u := g.Float64()
		for u == 0 {
			u = g.Float64()
		}
		return gamma1 * math.Pow(u, 1.0/shape)
	}

	// Marsaglia-Tsang method for shape >= 1
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)

	for {
		// Generate normal random variable
		u1 := g.Float64()
		u2 := g.Float64()
		for u1 == 0 {
			u1 = g.Float64()
		}

		z := math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
		v := 1.0 + c*z

		if v <= 0 {
			continue
		}

		v = v * v * v
		u := g.Float64()

		if u < 1.0-0.0331*(z*z)*(z*z) {
			return d * v
		}

		if math.Log(u) < 0.5*z*z+d*(1.0-v+math.Log(v)) {
			return d * v
		}
	}
}

// Beta generates an array of Beta distributed random numbers
func (g *Generator) Beta(alpha, beta float64, size int) (*array.Array, error) {
	if size <= 0 {
		return nil, fmt.Errorf("size must be positive, got %d", size)
	}
	if alpha <= 0 {
		return nil, fmt.Errorf("alpha must be positive, got %v", alpha)
	}
	if beta <= 0 {
		return nil, fmt.Errorf("beta must be positive, got %v", beta)
	}

	result := make([]float64, size)

	// Beta distribution using ratio of Gamma variables
	// If X ~ Gamma(α, 1) and Y ~ Gamma(β, 1), then X/(X+Y) ~ Beta(α, β)
	for i := 0; i < size; i++ {
		x := g.gammaRand(alpha)
		y := g.gammaRand(beta)
		result[i] = x / (x + y)
	}

	return array.FromSlice(result)
}

// Choice samples from an array with or without replacement
func (g *Generator) Choice(arr *array.Array, size int, replace bool) (*array.Array, error) {
	if arr == nil {
		return nil, fmt.Errorf("input array cannot be nil")
	}
	if size <= 0 {
		return nil, fmt.Errorf("size must be positive, got %d", size)
	}
	if arr.Size() == 0 {
		return nil, fmt.Errorf("cannot sample from empty array")
	}
	if !replace && size > arr.Size() {
		return nil, fmt.Errorf("cannot sample %d items without replacement from array of size %d", size, arr.Size())
	}

	result := make([]interface{}, size)

	if replace {
		// Sampling with replacement
		for i := 0; i < size; i++ {
			idx := int(g.Float64() * float64(arr.Size()))
			if idx >= arr.Size() {
				idx = arr.Size() - 1 // Handle edge case
			}
			result[i] = arr.At(idx)
		}
	} else {
		// Sampling without replacement using Fisher-Yates shuffle
		indices := make([]int, arr.Size())
		for i := range indices {
			indices[i] = i
		}

		// Partial Fisher-Yates shuffle
		for i := 0; i < size; i++ {
			j := i + int(g.Float64()*float64(arr.Size()-i))
			if j >= arr.Size() {
				j = arr.Size() - 1
			}
			indices[i], indices[j] = indices[j], indices[i]
			result[i] = arr.At(indices[i])
		}
	}

	return array.FromSlice(result)
}

// Shuffle returns a shuffled copy of the input array
func (g *Generator) Shuffle(arr *array.Array) (*array.Array, error) {
	if arr == nil {
		return nil, fmt.Errorf("input array cannot be nil")
	}
	if arr.Size() == 0 {
		return nil, fmt.Errorf("cannot shuffle empty array")
	}

	// Copy all elements
	result := make([]interface{}, arr.Size())
	for i := 0; i < arr.Size(); i++ {
		result[i] = arr.At(i)
	}

	// Fisher-Yates shuffle
	for i := len(result) - 1; i > 0; i-- {
		j := int(g.Float64() * float64(i+1))
		if j > i {
			j = i
		}
		result[i], result[j] = result[j], result[i]
	}

	return array.FromSlice(result)
}
