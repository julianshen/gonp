//go:build !amd64
// +build !amd64

package internal

// Fallback implementations for non-x86 architectures
// These just use the scalar implementations

func (p *AVXProvider) AddFloat64(a, b, result []float64, n int) {
	p.ScalarProvider.AddFloat64(a, b, result, n)
}

func (p *AVXProvider) MulFloat64(a, b, result []float64, n int) {
	p.ScalarProvider.MulFloat64(a, b, result, n)
}

func (p *AVXProvider) SumFloat64(a []float64, n int) float64 {
	return p.ScalarProvider.SumFloat64(a, n)
}

func (p *AVXProvider) AddScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	p.ScalarProvider.AddScalarFloat64(a, scalar, result, n)
}

func (p *AVXProvider) MulScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	p.ScalarProvider.MulScalarFloat64(a, scalar, result, n)
}

func (p *AVX2Provider) AddFloat64(a, b, result []float64, n int) {
	p.AVXProvider.AddFloat64(a, b, result, n)
}

func (p *AVX2Provider) MulFloat64(a, b, result []float64, n int) {
	p.AVXProvider.MulFloat64(a, b, result, n)
}

func (p *AVX2Provider) SumFloat64(a []float64, n int) float64 {
	return p.AVXProvider.SumFloat64(a, n)
}

func (p *AVX2Provider) AddScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	p.AVXProvider.AddScalarFloat64(a, scalar, result, n)
}

func (p *AVX512Provider) AddFloat64(a, b, result []float64, n int) {
	p.AVX2Provider.AddFloat64(a, b, result, n)
}

func (p *AVX512Provider) MulFloat64(a, b, result []float64, n int) {
	p.AVX2Provider.MulFloat64(a, b, result, n)
}

func (p *AVX512Provider) SumFloat64(a []float64, n int) float64 {
	return p.AVX2Provider.SumFloat64(a, n)
}

func (p *AVX512Provider) AddScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	p.AVX2Provider.AddScalarFloat64(a, scalar, result, n)
}
