package stats

import (
	"errors"
	"math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TTestResult represents the result of a t-test
type TTestResult struct {
	Statistic        float64 // t-statistic
	PValue           float64 // p-value (two-tailed)
	DegreesOfFreedom float64 // degrees of freedom
}

// OneSampleTTest performs a one-sample t-test
// H0: population mean = hypothesized mean
func OneSampleTTest(arr *array.Array, populationMean float64) (*TTestResult, error) {
	ctx := internal.StartProfiler("Stats.OneSampleTTest")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("OneSampleTTest", "array cannot be nil")
	}

	// Calculate sample statistics
	n := float64(arr.Size())
	if n < 2 {
		return nil, errors.New("need at least 2 observations for t-test")
	}

	sampleMean, err := Mean(arr)
	if err != nil {
		return nil, err
	}

	sampleStd, err := Std(arr)
	if err != nil {
		return nil, err
	}

	if sampleStd == 0 {
		return nil, errors.New("cannot perform t-test with zero standard deviation")
	}

	// Calculate t-statistic
	tStat := (sampleMean - populationMean) / (sampleStd / math.Sqrt(n))
	df := n - 1

	// Calculate p-value (two-tailed)
	pValue := 2.0 * (1.0 - studentTCDF(math.Abs(tStat), df))

	return &TTestResult{
		Statistic:        tStat,
		PValue:           pValue,
		DegreesOfFreedom: df,
	}, nil
}

// TwoSampleTTest performs a two-sample t-test (Welch's t-test for unequal variances)
// H0: mean1 = mean2
func TwoSampleTTest(x, y *array.Array) (*TTestResult, error) {
	ctx := internal.StartProfiler("Stats.TwoSampleTTest")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if x == nil || y == nil {
		return nil, internal.NewValidationErrorWithMsg("TwoSampleTTest", "arrays cannot be nil")
	}

	// Calculate sample statistics for both groups
	n1 := float64(x.Size())
	n2 := float64(y.Size())

	if n1 < 2 || n2 < 2 {
		return nil, errors.New("need at least 2 observations in each group for t-test")
	}

	mean1, err := Mean(x)
	if err != nil {
		return nil, err
	}

	mean2, err := Mean(y)
	if err != nil {
		return nil, err
	}

	var1, err := Var(x)
	if err != nil {
		return nil, err
	}

	var2, err := Var(y)
	if err != nil {
		return nil, err
	}

	// Welch's t-test (unequal variances)
	se1 := var1 / n1
	se2 := var2 / n2
	pooledSE := math.Sqrt(se1 + se2)

	if pooledSE == 0 {
		return nil, errors.New("cannot perform t-test with zero pooled standard error")
	}

	// Calculate t-statistic
	tStat := (mean1 - mean2) / pooledSE

	// Welch-Satterthwaite degrees of freedom
	df := math.Pow(se1+se2, 2) / (math.Pow(se1, 2)/(n1-1) + math.Pow(se2, 2)/(n2-1))

	// Calculate p-value (two-tailed)
	pValue := 2.0 * (1.0 - studentTCDF(math.Abs(tStat), df))

	return &TTestResult{
		Statistic:        tStat,
		PValue:           pValue,
		DegreesOfFreedom: df,
	}, nil
}

// PairedTTest performs a paired t-test
// H0: mean difference = 0
func PairedTTest(x, y *array.Array) (*TTestResult, error) {
	ctx := internal.StartProfiler("Stats.PairedTTest")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if x == nil || y == nil {
		return nil, internal.NewValidationErrorWithMsg("PairedTTest", "arrays cannot be nil")
	}

	if x.Size() != y.Size() {
		return nil, errors.New("arrays must have the same size for paired t-test")
	}

	// Calculate differences
	differences := make([]float64, 0, x.Size())
	xFlat := x.Flatten()
	yFlat := y.Flatten()

	for i := 0; i < xFlat.Size(); i++ {
		xVal := convertToFloat64(xFlat.At(i))
		yVal := convertToFloat64(yFlat.At(i))

		if !math.IsNaN(xVal) && !math.IsNaN(yVal) {
			differences = append(differences, xVal-yVal)
		}
	}

	if len(differences) < 2 {
		return nil, errors.New("need at least 2 valid pairs for paired t-test")
	}

	// Create array from differences and perform one-sample t-test against 0
	diffArray, err := array.FromSlice(differences)
	if err != nil {
		return nil, err
	}

	return OneSampleTTest(diffArray, 0.0)
}

// ChiSquareGoodnessOfFit performs a chi-square goodness of fit test
type ChiSquareResult struct {
	Statistic        float64
	PValue           float64
	DegreesOfFreedom int
}

// ChiSquareGoodnessOfFit tests if observed frequencies match expected frequencies
func ChiSquareGoodnessOfFit(observed, expected *array.Array) (*ChiSquareResult, error) {
	ctx := internal.StartProfiler("Stats.ChiSquareGoodnessOfFit")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if observed == nil || expected == nil {
		return nil, internal.NewValidationErrorWithMsg("ChiSquareGoodnessOfFit", "arrays cannot be nil")
	}

	if observed.Size() != expected.Size() {
		return nil, errors.New("observed and expected arrays must have the same size")
	}

	if observed.Size() == 0 {
		return nil, errors.New("cannot perform chi-square test on empty arrays")
	}

	var chiSquare float64
	var validCategories int

	obsFlat := observed.Flatten()
	expFlat := expected.Flatten()

	for i := 0; i < obsFlat.Size(); i++ {
		obsVal := convertToFloat64(obsFlat.At(i))
		expVal := convertToFloat64(expFlat.At(i))

		if math.IsNaN(obsVal) || math.IsNaN(expVal) {
			continue
		}

		if expVal <= 0 {
			return nil, errors.New("expected frequencies must be positive")
		}

		if obsVal < 0 {
			return nil, errors.New("observed frequencies cannot be negative")
		}

		diff := obsVal - expVal
		chiSquare += (diff * diff) / expVal
		validCategories++
	}

	if validCategories == 0 {
		return nil, errors.New("no valid categories for chi-square test")
	}

	df := validCategories - 1
	if df <= 0 {
		return nil, errors.New("insufficient degrees of freedom for chi-square test")
	}

	// Calculate p-value using chi-square distribution
	pValue := 1.0 - chiSquareCDF(chiSquare, float64(df))

	return &ChiSquareResult{
		Statistic:        chiSquare,
		PValue:           pValue,
		DegreesOfFreedom: df,
	}, nil
}

// NormalityTestResult represents the result of a normality test
type NormalityTestResult struct {
	Statistic float64
	PValue    float64
	TestName  string
}

// ShapiroWilkTest performs the Shapiro-Wilk test for normality (simplified implementation)
func ShapiroWilkTest(arr *array.Array) (*NormalityTestResult, error) {
	ctx := internal.StartProfiler("Stats.ShapiroWilkTest")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if arr == nil {
		return nil, internal.NewValidationErrorWithMsg("ShapiroWilkTest", "array cannot be nil")
	}

	// Extract non-NaN values and sort
	var values []float64
	flatArr := arr.Flatten()
	for i := 0; i < flatArr.Size(); i++ {
		val := convertToFloat64(flatArr.At(i))
		if !math.IsNaN(val) {
			values = append(values, val)
		}
	}

	n := len(values)
	if n < 3 {
		return nil, errors.New("need at least 3 observations for Shapiro-Wilk test")
	}

	if n > 5000 {
		return nil, errors.New("Shapiro-Wilk test not recommended for n > 5000")
	}

	// This is a simplified implementation
	// A full implementation would use the exact Shapiro-Wilk coefficients
	// For now, we'll use a correlation-based approximation

	// Sort the data
	sortedValues := make([]float64, len(values))
	copy(sortedValues, values)
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if sortedValues[i] > sortedValues[j] {
				sortedValues[i], sortedValues[j] = sortedValues[j], sortedValues[i]
			}
		}
	}

	// Calculate sample mean and variance
	mean := 0.0
	for _, v := range sortedValues {
		mean += v
	}
	mean /= float64(n)

	var variance float64
	for _, v := range sortedValues {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(n - 1)

	// Generate expected normal order statistics (approximation)
	expectedNormal := make([]float64, n)
	for i := 0; i < n; i++ {
		// Approximate expected value of i-th order statistic from standard normal
		p := (float64(i) + 0.375) / (float64(n) + 0.25)
		expectedNormal[i] = normalInverse(p)
	}

	// Calculate correlation coefficient between sorted data and expected normal values
	var sumXY, sumX, sumY, sumX2, sumY2 float64
	for i := 0; i < n; i++ {
		x := sortedValues[i]
		y := expectedNormal[i]
		sumXY += x * y
		sumX += x
		sumY += y
		sumX2 += x * x
		sumY2 += y * y
	}

	numerator := float64(n)*sumXY - sumX*sumY
	denominator := math.Sqrt((float64(n)*sumX2 - sumX*sumX) * (float64(n)*sumY2 - sumY*sumY))

	if denominator == 0 {
		return nil, errors.New("cannot calculate Shapiro-Wilk statistic")
	}

	correlation := numerator / denominator
	wStatistic := correlation * correlation

	// Approximate p-value (this is a very rough approximation)
	// In practice, you'd use lookup tables or more sophisticated methods
	pValue := math.Exp(-2.0 * math.Abs(math.Log(1.0-wStatistic)))

	return &NormalityTestResult{
		Statistic: wStatistic,
		PValue:    pValue,
		TestName:  "Shapiro-Wilk",
	}, nil
}

// Helper functions for statistical distributions

// studentTCDF computes the cumulative distribution function of Student's t-distribution
// This is a simplified approximation
func studentTCDF(t, df float64) float64 {
	if df <= 0 {
		return 0.5
	}

	// For large df, approximate with standard normal
	if df > 100 {
		return normalCDF(t)
	}

	// Simple approximation for Student's t
	// In practice, you'd use more accurate methods
	x := t / math.Sqrt(df)
	return 0.5 + (x/(1.0+math.Abs(x)))*0.5
}

// normalCDF computes the cumulative distribution function of standard normal distribution
func normalCDF(x float64) float64 {
	// Approximation using error function
	return 0.5 * (1.0 + math.Erf(x/math.Sqrt(2.0)))
}

// normalInverse computes the inverse of the standard normal CDF (approximation)
func normalInverse(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}
	if p == 0.5 {
		return 0
	}

	// Rational approximation (Beasley-Springer-Moro algorithm approximation)
	sign := 1.0
	if p < 0.5 {
		p = 1 - p
		sign = -1.0
	}

	t := math.Sqrt(-2.0 * math.Log(1.0-p))

	// Approximation coefficients
	c0 := 2.515517
	c1 := 0.802853
	c2 := 0.010328
	d1 := 1.432788
	d2 := 0.189269
	d3 := 0.001308

	x := t - (c0+c1*t+c2*t*t)/(1.0+d1*t+d2*t*t+d3*t*t*t)

	return sign * x
}

// chiSquareCDF computes the cumulative distribution function of chi-square distribution
// This is a simplified approximation
func chiSquareCDF(x, df float64) float64 {
	if x <= 0 {
		return 0
	}
	if df <= 0 {
		return 0
	}

	// For large df, use normal approximation
	if df > 100 {
		mean := df
		variance := 2.0 * df
		standardized := (x - mean) / math.Sqrt(variance)
		return normalCDF(standardized)
	}

	// Improved approximation using Wilson-Hilferty transformation for chi-square
	// Convert to approximately normal
	h := 2.0 / (9.0 * df)
	z := (math.Pow(x/df, 1.0/3.0) - (1.0 - h)) / math.Sqrt(h)

	// For small df, use a different approach
	if df <= 2 {
		if df == 1 {
			// For df=1, chi-square is square of standard normal
			return 2.0*normalCDF(math.Sqrt(x)) - 1.0
		} else if df == 2 {
			// For df=2, chi-square has simple exponential form
			return 1.0 - math.Exp(-x/2.0)
		}
	}

	return normalCDF(z)
}
