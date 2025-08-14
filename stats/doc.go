// Package stats provides statistical analysis functions for GoNP arrays and data structures.
//
// # Overview
//
// The stats package offers a comprehensive suite of statistical functions including
// descriptive statistics, correlation analysis, hypothesis testing, regression analysis,
// and advanced statistical methods like ANOVA. All functions are optimized for
// performance and numerical stability.
//
// Key features:
//   - SIMD-optimized calculations for large datasets
//   - Numerically stable algorithms for critical computations
//   - Comprehensive hypothesis testing suite
//   - Advanced regression and ANOVA analysis
//   - Integration with GoNP arrays and data structures
//
// # Quick Start
//
// Basic statistical analysis:
//
//	import "github.com/julianshen/gonp/array"
//	import "github.com/julianshen/gonp/stats"
//
//	// Create sample data
//	data, _ := array.FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
//
//	// Descriptive statistics
//	mean := stats.Mean(data)           // Arithmetic mean: 5.5
//	median := stats.Median(data)       // Middle value: 5.5
//	stdDev := stats.StdDev(data)       // Standard deviation
//	variance := stats.Variance(data)   // Population variance
//
//	// Distribution properties
//	min := stats.Min(data)             // Minimum value: 1
//	max := stats.Max(data)             // Maximum value: 10
//	q25 := stats.Quantile(data, 0.25)  // First quartile
//	q75 := stats.Quantile(data, 0.75)  // Third quartile
//	iqr := stats.IQR(data)             // Interquartile range
//
// # Descriptive Statistics
//
// ## Central Tendency
//
//	// Measures of central tendency
//	mean := stats.Mean(data)           // Arithmetic mean
//	geomean := stats.GeometricMean(data) // Geometric mean
//	harmean := stats.HarmonicMean(data)  // Harmonic mean
//	median := stats.Median(data)       // 50th percentile
//	mode := stats.Mode(data)           // Most frequent value(s)
//
// ## Dispersion Measures
//
//	// Measures of spread
//	variance := stats.Variance(data)   // Population variance
//	stddev := stats.StdDev(data)       // Standard deviation
//	mad := stats.MAD(data)             // Mean absolute deviation
//	range_val := stats.Range(data)     // Max - Min
//	iqr := stats.IQR(data)             // Interquartile range
//
//	// Coefficient of variation
//	cv := stats.CV(data)               // StdDev / Mean
//
// ## Distribution Shape
//
//	// Skewness and kurtosis
//	skewness := stats.Skewness(data)   // Measure of asymmetry
//	kurtosis := stats.Kurtosis(data)   // Measure of tail heaviness
//
//	// Quantile functions
//	percentiles := stats.Percentile(data, []float64{25, 50, 75, 95})
//	quartiles := stats.Quartiles(data) // Q1, Q2 (median), Q3
//
// # Correlation Analysis
//
// ## Pairwise Correlations
//
//	// Pearson correlation (linear relationships)
//	corr := stats.Correlation(x, y)    // Range: [-1, 1]
//
//	// Rank-based correlations (non-linear relationships)
//	spearman := stats.SpearmanCorr(x, y)  // Spearman's rank correlation
//	kendall := stats.KendallTau(x, y)     // Kendall's tau
//
//	// Partial correlation (controlling for other variables)
//	partial := stats.PartialCorr(x, y, z) // Correlation of x,y given z
//
// ## Distance Metrics
//
//	// Distance measures between arrays
//	euclidean := stats.EuclideanDistance(a, b)   // L2 distance
//	manhattan := stats.ManhattanDistance(a, b)   // L1 distance
//	chebyshev := stats.ChebyshevDistance(a, b)   // L∞ distance
//	cosine := stats.CosineDistance(a, b)         // Angular distance
//
//	// Similarity measures
//	cosine_sim := stats.CosineSimilarity(a, b)   // Cosine similarity
//	jaccard := stats.JaccardSimilarity(a, b)     // Jaccard index
//
// # Hypothesis Testing
//
// ## One-Sample Tests
//
//	// Test if sample mean differs from population value
//	ttest := stats.OneSampleTTest(data, 0.0)     // Test against 0
//	fmt.Printf("t-statistic: %.4f, p-value: %.4f\n", ttest.TStatistic, ttest.PValue)
//
//	// Normality tests
//	shapiro := stats.ShapiroWilk(data)           // Test for normality
//	ks := stats.KSTest(data, "normal")           // Kolmogorov-Smirnov test
//
// ## Two-Sample Tests
//
//	// Compare means of two independent groups
//	ttest := stats.TwoSampleTTest(group1, group2)
//	fmt.Printf("Two-sample t-test: t=%.4f, p=%.4f\n", ttest.TStatistic, ttest.PValue)
//
//	// Non-parametric alternatives
//	mann_whitney := stats.MannWhitneyU(group1, group2)  // Rank-sum test
//	ks_test := stats.KSTest2Sample(group1, group2)      // Two-sample KS test
//
//	// Paired samples
//	paired_t := stats.PairedTTest(before, after)        // Paired t-test
//	wilcoxon := stats.WilcoxonSigned(before, after)     // Wilcoxon signed-rank
//
// ## Chi-Square Tests
//
//	// Goodness of fit
//	observed, _ := array.FromSlice([]float64{10, 15, 20, 25})
//	expected, _ := array.FromSlice([]float64{12, 18, 18, 22})
//	chi2 := stats.ChiSquareGoodnessOfFit(observed, expected)
//
//	// Test of independence (requires contingency table)
//	contingency := array.FromSlice([][]float64{{10, 15}, {20, 25}})
//	independence := stats.ChiSquareIndependence(contingency)
//
// # Linear Regression Analysis
//
// ## Simple Linear Regression
//
//	// Fit y = β₀ + β₁x + ε
//	reg := stats.LinearRegression(x, y)
//
//	// Extract results
//	fmt.Printf("Intercept: %.4f ± %.4f\n", reg.Intercept, reg.InterceptStdErr)
//	fmt.Printf("Slope: %.4f ± %.4f\n", reg.Slope, reg.SlopeStdErr)
//	fmt.Printf("R-squared: %.4f\n", reg.RSquared)
//	fmt.Printf("F-statistic: %.4f (p=%.4f)\n", reg.FStatistic, reg.FPValue)
//
//	// Predictions and intervals
//	predictions := reg.Predict(new_x)
//	conf_intervals := reg.ConfidenceInterval(new_x, 0.95)
//	pred_intervals := reg.PredictionInterval(new_x, 0.95)
//
// ## Multiple Linear Regression
//
//	// Fit y = β₀ + β₁x₁ + β₂x₂ + ... + ε
//	// X should be a matrix where each column is a predictor
//	reg := stats.MultipleRegression(X, y)
//
//	// Model diagnostics
//	residuals := reg.Residuals
//	fitted := reg.FittedValues
//	leverage := reg.Leverage()           // Hat diagonal
//	cook_d := reg.CooksDistance()        // Cook's distance
//
//	// Model comparison
//	aic := reg.AIC()                     // Akaike Information Criterion
//	bic := reg.BIC()                     // Bayesian Information Criterion
//	adj_r2 := reg.AdjustedRSquared()     // Adjusted R-squared
//
// # Analysis of Variance (ANOVA)
//
// ## One-Way ANOVA
//
//	// Test if means of multiple groups differ
//	groups := []*array.Array{group1, group2, group3, group4}
//	anova := stats.OneWayANOVA(groups)
//
//	fmt.Printf("F-statistic: %.4f\n", anova.FStatistic)
//	fmt.Printf("p-value: %.4f\n", anova.PValue)
//	fmt.Printf("Effect size (η²): %.4f\n", anova.EtaSquared)
//
//	// Post-hoc comparisons (if significant)
//	if anova.PValue < 0.05 {
//		tukey := stats.TukeyHSD(groups)   // All pairwise comparisons
//		bonferroni := stats.Bonferroni(groups) // Bonferroni correction
//	}
//
// ## Two-Way ANOVA
//
//	// Test main effects and interaction
//	anova2 := stats.TwoWayANOVA(values, factorA, factorB)
//
//	fmt.Printf("Factor A: F=%.4f, p=%.4f\n",
//		anova2.FactorA.FStatistic, anova2.FactorA.PValue)
//	fmt.Printf("Factor B: F=%.4f, p=%.4f\n",
//		anova2.FactorB.FStatistic, anova2.FactorB.PValue)
//	fmt.Printf("Interaction: F=%.4f, p=%.4f\n",
//		anova2.Interaction.FStatistic, anova2.Interaction.PValue)
//
// ## Advanced ANOVA Designs
//
//	// Repeated measures ANOVA (within-subjects)
//	repeated := stats.RepeatedMeasuresANOVA(data) // Subject x Time matrix
//
//	// Multivariate ANOVA (MANOVA)
//	manova := stats.MANOVA(groups)               // Multiple dependent variables
//
//	// Mixed-effects models (random and fixed effects)
//	mixed := stats.MixedEffectsANOVA(data, fixed, random)
//
// # Non-Parametric Tests
//
// Alternative methods when assumptions are violated:
//
//	// Alternative to one-way ANOVA
//	kruskal := stats.KruskalWallisTest(groups)   // Rank-based ANOVA
//
//	// Alternative to repeated measures ANOVA
//	friedman := stats.FriedmanTest(data)         // Non-parametric RM-ANOVA
//
//	// Rank correlations
//	spearman := stats.SpearmanCorrelation(x, y)  // Monotonic relationships
//	kendall := stats.KendallTau(x, y)            // Robust rank correlation
//
// # Advanced Statistical Methods
//
// ## Multivariate Statistics
//
//	// Principal Component Analysis
//	pca := stats.PCA(data_matrix)
//	components := pca.Components    // Principal components
//	explained_var := pca.ExplainedVariance
//
//	// Factor Analysis
//	fa := stats.FactorAnalysis(data_matrix, n_factors)
//	loadings := fa.Loadings        // Factor loadings
//
//	// Cluster Analysis
//	kmeans := stats.KMeans(data_matrix, k_clusters)
//	hierarchical := stats.HierarchicalClustering(data_matrix)
//
// ## Time Series Analysis
//
//	// Autocorrelation and partial autocorrelation
//	acf := stats.AutoCorrelation(timeseries, max_lag)
//	pacf := stats.PartialAutoCorrelation(timeseries, max_lag)
//
//	// Trend detection
//	trend := stats.MannKendallTrend(timeseries)  // Non-parametric trend test
//	seasonal := stats.SeasonalDecomposition(timeseries, period)
//
// # Performance and Accuracy
//
// ## Numerical Stability
//
// The package uses numerically stable algorithms:
//
//   - Welford's algorithm for variance calculation
//   - Kahan summation for reducing floating-point errors
//   - QR decomposition for regression to avoid matrix inversion
//   - Pivoting strategies in matrix operations
//
// ## SIMD Optimization
//
//   - Automatic SIMD acceleration for large datasets
//   - Vectorized operations where mathematically sound
//   - Efficient memory access patterns
//   - Multi-threading for compute-intensive operations
//
// ## Missing Data Handling
//
//	// Functions automatically handle NaN values
//	data_with_missing := []float64{1, 2, NaN, 4, 5, NaN, 7}
//	arr, _ := array.FromSlice(data_with_missing)
//
//	mean := stats.MeanIgnoreNaN(arr)         // Ignores NaN values
//	complete_cases := stats.RemoveMissing(arr) // Remove NaN entries
//
// # Integration Examples
//
// ## Data Pipeline Example
//
//	// Complete statistical analysis pipeline
//	func analyzeData(rawData *array.Array) {
//		// 1. Descriptive statistics
//		desc := stats.Describe(rawData)
//		fmt.Printf("Mean: %.2f, Std: %.2f\n", desc.Mean, desc.StdDev)
//
//		// 2. Check normality
//		normality := stats.ShapiroWilk(rawData)
//		if normality.PValue < 0.05 {
//			fmt.Println("Data is not normally distributed")
//			// Use non-parametric methods
//		}
//
//		// 3. Outlier detection
//		outliers := stats.DetectOutliers(rawData, "iqr") // IQR method
//		cleaned := stats.RemoveOutliers(rawData, outliers)
//
//		// 4. Correlation analysis
//		if len(otherVars) > 0 {
//			corrMatrix := stats.CorrelationMatrix([][]float64{rawData, otherVars...})
//			fmt.Printf("Correlation matrix:\n%v\n", corrMatrix)
//		}
//	}
//
// ## A/B Testing Example
//
//	func abTest(control, treatment *array.Array) {
//		// Descriptive statistics for both groups
//		controlStats := stats.Describe(control)
//		treatmentStats := stats.Describe(treatment)
//
//		fmt.Printf("Control: μ=%.3f, σ=%.3f, n=%d\n",
//			controlStats.Mean, controlStats.StdDev, controlStats.Count)
//		fmt.Printf("Treatment: μ=%.3f, σ=%.3f, n=%d\n",
//			treatmentStats.Mean, treatmentStats.StdDev, treatmentStats.Count)
//
//		// Test for equal variances
//		levene := stats.LeveneTest(control, treatment)
//		equal_var := levene.PValue > 0.05
//
//		// Two-sample t-test
//		ttest := stats.TwoSampleTTest(control, treatment, equal_var)
//
//		// Effect size
//		cohens_d := stats.CohensD(control, treatment)
//
//		// Results
//		fmt.Printf("t-test: t=%.3f, p=%.4f\n", ttest.TStatistic, ttest.PValue)
//		fmt.Printf("Effect size (Cohen's d): %.3f\n", cohens_d)
//
//		if ttest.PValue < 0.05 {
//			fmt.Printf("Significant difference detected!\n")
//		}
//	}
package stats
