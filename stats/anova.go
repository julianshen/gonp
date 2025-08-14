package stats

import (
	"errors"
	"fmt"
	"math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	mathlib "github.com/julianshen/gonp/math"
)

// ANOVAResult represents the result of a one-way ANOVA test
type ANOVAResult struct {
	FStatistic float64 // F-statistic
	PValue     float64 // p-value
	DFBetween  int     // degrees of freedom between groups
	DFWithin   int     // degrees of freedom within groups
	SSBetween  float64 // sum of squares between groups
	SSWithin   float64 // sum of squares within groups
	MSBetween  float64 // mean square between groups
	MSWithin   float64 // mean square within groups
}

// TwoWayANOVAResult represents the result of a two-way ANOVA test
type TwoWayANOVAResult struct {
	FStatisticA           float64 // F-statistic for factor A
	FStatisticB           float64 // F-statistic for factor B
	FStatisticInteraction float64 // F-statistic for interaction
	PValueA               float64 // p-value for factor A
	PValueB               float64 // p-value for factor B
	PValueInteraction     float64 // p-value for interaction
	DFFactorA             int     // degrees of freedom for factor A
	DFFactorB             int     // degrees of freedom for factor B
	DFInteraction         int     // degrees of freedom for interaction
	DFError               int     // degrees of freedom for error
	SSFactorA             float64 // sum of squares for factor A
	SSFactorB             float64 // sum of squares for factor B
	SSInteraction         float64 // sum of squares for interaction
	SSError               float64 // sum of squares for error
}

// RepeatedMeasuresANOVAResult represents the result of repeated measures ANOVA
type RepeatedMeasuresANOVAResult struct {
	FStatisticTime float64 // F-statistic for time effect
	PValueTime     float64 // p-value for time effect
	DFTime         int     // degrees of freedom for time
	DFError        int     // degrees of freedom for error
	SSTime         float64 // sum of squares for time
	SSError        float64 // sum of squares for error
	SSSubjects     float64 // sum of squares for subjects
}

// MANOVAResult represents the result of multivariate ANOVA
type MANOVAResult struct {
	WilksLambda  float64 // Wilks' Lambda test statistic
	PillaiTrace  float64 // Pillai's trace test statistic
	FStatistic   float64 // F-statistic
	PValue       float64 // p-value
	DFHypothesis int     // hypothesis degrees of freedom
	DFError      int     // error degrees of freedom
}

// KruskalWallisResult represents the result of Kruskal-Wallis test
type KruskalWallisResult struct {
	HStatistic       float64 // H-statistic (chi-square distributed)
	PValue           float64 // p-value
	DegreesOfFreedom int     // degrees of freedom
}

// FriedmanResult represents the result of Friedman test
type FriedmanResult struct {
	ChiSquareStatistic float64 // chi-square statistic
	PValue             float64 // p-value
	DegreesOfFreedom   int     // degrees of freedom
}

// OneWayANOVA performs one-way analysis of variance
// H0: all group means are equal
func OneWayANOVA(groups []*array.Array) (*ANOVAResult, error) {
	ctx := internal.StartProfiler("Stats.OneWayANOVA")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if len(groups) < 2 {
		return nil, errors.New("need at least 2 groups for ANOVA")
	}

	// Validate groups and collect data
	var allData []float64
	var groupSizes []int
	var groupMeans []float64
	var groupSumSquares []float64

	totalN := 0
	for i, group := range groups {
		if group == nil {
			return nil, fmt.Errorf("group %d cannot be nil", i)
		}

		flatGroup := group.Flatten()
		if flatGroup.Size() < 2 {
			return nil, fmt.Errorf("group %d must have at least 2 observations", i)
		}

		// Extract values
		groupData := make([]float64, 0, flatGroup.Size())
		for j := 0; j < flatGroup.Size(); j++ {
			val := convertToFloat64(flatGroup.At(j))
			if !math.IsNaN(val) {
				groupData = append(groupData, val)
				allData = append(allData, val)
			}
		}

		if len(groupData) < 2 {
			return nil, fmt.Errorf("group %d has insufficient valid observations", i)
		}

		groupSizes = append(groupSizes, len(groupData))
		totalN += len(groupData)

		// Calculate group mean
		sum := 0.0
		for _, val := range groupData {
			sum += val
		}
		groupMean := sum / float64(len(groupData))
		groupMeans = append(groupMeans, groupMean)

		// Calculate sum of squares within group
		ss := 0.0
		for _, val := range groupData {
			diff := val - groupMean
			ss += diff * diff
		}
		groupSumSquares = append(groupSumSquares, ss)
	}

	// Calculate grand mean
	grandSum := 0.0
	for _, val := range allData {
		grandSum += val
	}
	grandMean := grandSum / float64(totalN)

	// Calculate sum of squares between groups (treatment effect)
	ssBetween := 0.0
	for i, groupMean := range groupMeans {
		diff := groupMean - grandMean
		ssBetween += float64(groupSizes[i]) * diff * diff
	}

	// Calculate sum of squares within groups (error)
	ssWithin := 0.0
	for _, ss := range groupSumSquares {
		ssWithin += ss
	}

	// Degrees of freedom
	dfBetween := len(groups) - 1
	dfWithin := totalN - len(groups)

	if dfWithin <= 0 {
		return nil, errors.New("insufficient degrees of freedom for error term")
	}

	// Mean squares
	msBetween := ssBetween / float64(dfBetween)
	msWithin := ssWithin / float64(dfWithin)

	// F-statistic
	var fStatistic float64
	if msWithin == 0 {
		if msBetween == 0 {
			fStatistic = 0 // No variance within or between
		} else {
			return nil, errors.New("zero within-group variance with non-zero between-group variance")
		}
	} else {
		fStatistic = msBetween / msWithin
	}

	// P-value using F-distribution
	pValue := fDistributionPValue(fStatistic, dfBetween, dfWithin)
	if fStatistic == 0 {
		pValue = 1.0 // No effect when F = 0
	}

	return &ANOVAResult{
		FStatistic: fStatistic,
		PValue:     pValue,
		DFBetween:  dfBetween,
		DFWithin:   dfWithin,
		SSBetween:  ssBetween,
		SSWithin:   ssWithin,
		MSBetween:  msBetween,
		MSWithin:   msWithin,
	}, nil
}

// TwoWayANOVA performs two-way analysis of variance
// Tests main effects of two factors and their interaction
func TwoWayANOVA(values, factorA, factorB *array.Array) (*TwoWayANOVAResult, error) {
	ctx := internal.StartProfiler("Stats.TwoWayANOVA")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if values == nil || factorA == nil || factorB == nil {
		return nil, errors.New("arrays cannot be nil")
	}

	if values.Size() != factorA.Size() || values.Size() != factorB.Size() {
		return nil, errors.New("all arrays must have the same size")
	}

	n := values.Size()
	if n < 4 {
		return nil, errors.New("need at least 4 observations for two-way ANOVA")
	}

	// Extract data
	valData := make([]float64, n)
	factorAData := make([]int, n)
	factorBData := make([]int, n)

	valFlat := values.Flatten()
	factorAFlat := factorA.Flatten()
	factorBFlat := factorB.Flatten()

	for i := 0; i < n; i++ {
		valData[i] = convertToFloat64(valFlat.At(i))
		factorAData[i] = int(convertToFloat64(factorAFlat.At(i)))
		factorBData[i] = int(convertToFloat64(factorBFlat.At(i)))
	}

	// Find unique levels
	levelsA := make(map[int]bool)
	levelsB := make(map[int]bool)
	for i := 0; i < n; i++ {
		levelsA[factorAData[i]] = true
		levelsB[factorBData[i]] = true
	}

	aLevels := len(levelsA)
	bLevels := len(levelsB)

	if aLevels < 2 || bLevels < 2 {
		return nil, errors.New("each factor must have at least 2 levels")
	}

	// Calculate grand mean
	grandSum := 0.0
	for _, val := range valData {
		grandSum += val
	}
	grandMean := grandSum / float64(n)

	// Calculate cell means and marginal means
	cellSums := make(map[string]float64)
	cellCounts := make(map[string]int)
	marginASum := make(map[int]float64)
	marginACount := make(map[int]int)
	marginBSum := make(map[int]float64)
	marginBCount := make(map[int]int)

	for i := 0; i < n; i++ {
		a := factorAData[i]
		b := factorBData[i]
		val := valData[i]

		// Cell means
		cellKey := fmt.Sprintf("%d_%d", a, b)
		cellSums[cellKey] += val
		cellCounts[cellKey]++

		// Marginal means
		marginASum[a] += val
		marginACount[a]++
		marginBSum[b] += val
		marginBCount[b]++
	}

	// Calculate sum of squares
	ssTotal := 0.0
	ssA := 0.0
	ssB := 0.0
	ssAB := 0.0
	ssError := 0.0

	for i := 0; i < n; i++ {
		val := valData[i]
		a := factorAData[i]
		b := factorBData[i]
		cellKey := fmt.Sprintf("%d_%d", a, b)

		// Total SS
		ssTotal += math.Pow(val-grandMean, 2)

		// Cell mean
		cellMean := cellSums[cellKey] / float64(cellCounts[cellKey])
		ssError += math.Pow(val-cellMean, 2)
	}

	// Main effect A
	for a, sum := range marginASum {
		count := marginACount[a]
		marginMean := sum / float64(count)
		ssA += float64(count) * math.Pow(marginMean-grandMean, 2)
	}

	// Main effect B
	for b, sum := range marginBSum {
		count := marginBCount[b]
		marginMean := sum / float64(count)
		ssB += float64(count) * math.Pow(marginMean-grandMean, 2)
	}

	// Interaction AB
	for cellKey, sum := range cellSums {
		count := cellCounts[cellKey]
		cellMean := sum / float64(count)

		// Parse cell key to get factor levels
		var a, b int
		fmt.Sscanf(cellKey, "%d_%d", &a, &b)

		marginMeanA := marginASum[a] / float64(marginACount[a])
		marginMeanB := marginBSum[b] / float64(marginBCount[b])

		expected := marginMeanA + marginMeanB - grandMean
		ssAB += float64(count) * math.Pow(cellMean-expected, 2)
	}

	// Degrees of freedom
	dfA := aLevels - 1
	dfB := bLevels - 1
	dfAB := dfA * dfB
	dfError := n - aLevels*bLevels

	if dfError <= 0 {
		return nil, errors.New("insufficient error degrees of freedom")
	}

	// Mean squares
	msA := ssA / float64(dfA)
	msB := ssB / float64(dfB)
	msAB := ssAB / float64(dfAB)
	msError := ssError / float64(dfError)

	// F-statistics
	var fA, fB, fAB float64
	if msError > 0 {
		fA = msA / msError
		fB = msB / msError
		fAB = msAB / msError
	}

	// P-values
	pA := fDistributionPValue(fA, dfA, dfError)
	pB := fDistributionPValue(fB, dfB, dfError)
	pAB := fDistributionPValue(fAB, dfAB, dfError)

	return &TwoWayANOVAResult{
		FStatisticA:           fA,
		FStatisticB:           fB,
		FStatisticInteraction: fAB,
		PValueA:               pA,
		PValueB:               pB,
		PValueInteraction:     pAB,
		DFFactorA:             dfA,
		DFFactorB:             dfB,
		DFInteraction:         dfAB,
		DFError:               dfError,
		SSFactorA:             ssA,
		SSFactorB:             ssB,
		SSInteraction:         ssAB,
		SSError:               ssError,
	}, nil
}

// RepeatedMeasuresANOVA performs repeated measures ANOVA
// Data should be organized as subjects x time points (2D array)
func RepeatedMeasuresANOVA(data *array.Array) (*RepeatedMeasuresANOVAResult, error) {
	ctx := internal.StartProfiler("Stats.RepeatedMeasuresANOVA")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if data == nil {
		return nil, errors.New("data cannot be nil")
	}

	shape := data.Shape()
	if shape.Ndim() != 2 {
		return nil, errors.New("data must be 2-dimensional (subjects x time)")
	}

	subjects := shape[0]
	timePoints := shape[1]

	if subjects < 2 {
		return nil, errors.New("need at least 2 subjects")
	}
	if timePoints < 2 {
		return nil, errors.New("need at least 2 time points")
	}

	// Extract data
	dataValues := make([][]float64, subjects)
	for i := 0; i < subjects; i++ {
		dataValues[i] = make([]float64, timePoints)
		for j := 0; j < timePoints; j++ {
			dataValues[i][j] = convertToFloat64(data.At(i, j))
		}
	}

	// Calculate grand mean
	grandSum := 0.0
	n := subjects * timePoints
	for i := 0; i < subjects; i++ {
		for j := 0; j < timePoints; j++ {
			grandSum += dataValues[i][j]
		}
	}
	grandMean := grandSum / float64(n)

	// Calculate subject means
	subjectMeans := make([]float64, subjects)
	for i := 0; i < subjects; i++ {
		sum := 0.0
		for j := 0; j < timePoints; j++ {
			sum += dataValues[i][j]
		}
		subjectMeans[i] = sum / float64(timePoints)
	}

	// Calculate time means
	timeMeans := make([]float64, timePoints)
	for j := 0; j < timePoints; j++ {
		sum := 0.0
		for i := 0; i < subjects; i++ {
			sum += dataValues[i][j]
		}
		timeMeans[j] = sum / float64(subjects)
	}

	// Calculate sum of squares
	ssTotal := 0.0
	ssSubjects := 0.0
	ssTime := 0.0
	ssError := 0.0

	// Total SS
	for i := 0; i < subjects; i++ {
		for j := 0; j < timePoints; j++ {
			diff := dataValues[i][j] - grandMean
			ssTotal += diff * diff
		}
	}

	// Subject SS
	for i := 0; i < subjects; i++ {
		diff := subjectMeans[i] - grandMean
		ssSubjects += float64(timePoints) * diff * diff
	}

	// Time SS
	for j := 0; j < timePoints; j++ {
		diff := timeMeans[j] - grandMean
		ssTime += float64(subjects) * diff * diff
	}

	// Error SS (residual)
	ssError = ssTotal - ssSubjects - ssTime

	// Degrees of freedom
	dfTime := timePoints - 1
	dfError := (subjects - 1) * (timePoints - 1)

	if dfError <= 0 {
		return nil, errors.New("insufficient error degrees of freedom")
	}

	// Mean squares
	msTime := ssTime / float64(dfTime)
	msError := ssError / float64(dfError)

	// F-statistic
	var fTime float64
	if msError > 0 {
		fTime = msTime / msError
	}

	// P-value
	pTime := fDistributionPValue(fTime, dfTime, dfError)
	if fTime == 0 {
		pTime = 1.0 // No effect when F = 0
	}

	return &RepeatedMeasuresANOVAResult{
		FStatisticTime: fTime,
		PValueTime:     pTime,
		DFTime:         dfTime,
		DFError:        dfError,
		SSTime:         ssTime,
		SSError:        ssError,
		SSSubjects:     ssSubjects,
	}, nil
}

// MANOVA performs multivariate analysis of variance
// groups should contain 2D arrays (observations x variables)
func MANOVA(groups []*array.Array) (*MANOVAResult, error) {
	ctx := internal.StartProfiler("Stats.MANOVA")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if len(groups) < 2 {
		return nil, errors.New("need at least 2 groups for MANOVA")
	}

	// Validate groups
	var nVars int
	var totalObs int
	groupSizes := make([]int, len(groups))

	for i, group := range groups {
		if group == nil {
			return nil, fmt.Errorf("group %d cannot be nil", i)
		}

		shape := group.Shape()
		if shape.Ndim() != 2 {
			return nil, fmt.Errorf("group %d must be 2-dimensional", i)
		}

		if i == 0 {
			nVars = shape[1]
		} else if shape[1] != nVars {
			return nil, fmt.Errorf("all groups must have the same number of variables")
		}

		groupSizes[i] = shape[0]
		totalObs += shape[0]
	}

	if nVars < 1 {
		return nil, errors.New("need at least 1 variable")
	}

	// Extract all data and calculate grand means
	allData := make([][]float64, totalObs)
	grandMeans := make([]float64, nVars)

	idx := 0
	for _, group := range groups {
		shape := group.Shape()
		for i := 0; i < shape[0]; i++ {
			allData[idx] = make([]float64, nVars)
			for j := 0; j < nVars; j++ {
				val := convertToFloat64(group.At(i, j))
				allData[idx][j] = val
				grandMeans[j] += val
			}
			idx++
		}
	}

	for j := 0; j < nVars; j++ {
		grandMeans[j] /= float64(totalObs)
	}

	// Calculate group means
	groupMeans := make([][]float64, len(groups))
	idx = 0
	for g, group := range groups {
		shape := group.Shape()
		groupMeans[g] = make([]float64, nVars)

		for j := 0; j < nVars; j++ {
			sum := 0.0
			for i := 0; i < shape[0]; i++ {
				sum += convertToFloat64(group.At(i, j))
			}
			groupMeans[g][j] = sum / float64(shape[0])
		}
	}

	// Calculate SSCP matrices (sum of squares and cross-products)
	// Total SSCP
	totalSSCP := make([][]float64, nVars)
	for i := range totalSSCP {
		totalSSCP[i] = make([]float64, nVars)
	}

	for _, obs := range allData {
		for i := 0; i < nVars; i++ {
			for j := 0; j < nVars; j++ {
				diff1 := obs[i] - grandMeans[i]
				diff2 := obs[j] - grandMeans[j]
				totalSSCP[i][j] += diff1 * diff2
			}
		}
	}

	// Between-groups SSCP (hypothesis)
	betweenSSCP := make([][]float64, nVars)
	for i := range betweenSSCP {
		betweenSSCP[i] = make([]float64, nVars)
	}

	for g, groupMean := range groupMeans {
		n := float64(groupSizes[g])
		for i := 0; i < nVars; i++ {
			for j := 0; j < nVars; j++ {
				diff1 := groupMean[i] - grandMeans[i]
				diff2 := groupMean[j] - grandMeans[j]
				betweenSSCP[i][j] += n * diff1 * diff2
			}
		}
	}

	// Within-groups SSCP (error)
	withinSSCP := make([][]float64, nVars)
	for i := range withinSSCP {
		withinSSCP[i] = make([]float64, nVars)
		for j := range withinSSCP[i] {
			withinSSCP[i][j] = totalSSCP[i][j] - betweenSSCP[i][j]
		}
	}

	// Calculate test statistics
	// For simplicity, we'll calculate Wilks' Lambda and Pillai's trace
	// This requires matrix operations

	// Convert to arrays for matrix operations
	withinMatrix := matrixToArray(withinSSCP)

	// Calculate Wilks' Lambda = |E| / |E + H|
	// where E is error (within) matrix and H is hypothesis (between) matrix
	detWithin, err := mathlib.Det(withinMatrix)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate determinant of within matrix: %v", err)
	}

	// E + H
	totalMatrix := make([][]float64, nVars)
	for i := range totalMatrix {
		totalMatrix[i] = make([]float64, nVars)
		for j := range totalMatrix[i] {
			totalMatrix[i][j] = withinSSCP[i][j] + betweenSSCP[i][j]
		}
	}

	totalMatrixArray := matrixToArray(totalMatrix)
	detTotal, err := mathlib.Det(totalMatrixArray)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate determinant of total matrix: %v", err)
	}

	wilksLambda := detWithin / detTotal
	if math.IsNaN(wilksLambda) || math.IsInf(wilksLambda, 0) {
		wilksLambda = 1.0 // Default when determinants are problematic
	}

	// Pillai's trace (simplified calculation)
	pillaiTrace := 0.0
	for i := 0; i < nVars; i++ {
		if totalSSCP[i][i] > 0 {
			pillaiTrace += betweenSSCP[i][i] / totalSSCP[i][i]
		}
	}

	// Approximate F-statistic and p-value
	dfHypothesis := (len(groups) - 1) * nVars
	dfError := totalObs - len(groups) - nVars + 1

	if dfError <= 0 {
		return nil, errors.New("insufficient error degrees of freedom")
	}

	// Approximate F from Wilks' Lambda
	fStatistic := ((1 - wilksLambda) / wilksLambda) * (float64(dfError) / float64(dfHypothesis))
	pValue := fDistributionPValue(fStatistic, dfHypothesis, dfError)

	return &MANOVAResult{
		WilksLambda:  wilksLambda,
		PillaiTrace:  pillaiTrace,
		FStatistic:   fStatistic,
		PValue:       pValue,
		DFHypothesis: dfHypothesis,
		DFError:      dfError,
	}, nil
}

// KruskalWallisTest performs the Kruskal-Wallis H test (non-parametric one-way ANOVA)
func KruskalWallisTest(groups []*array.Array) (*KruskalWallisResult, error) {
	ctx := internal.StartProfiler("Stats.KruskalWallisTest")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if len(groups) < 2 {
		return nil, errors.New("need at least 2 groups for Kruskal-Wallis test")
	}

	// Collect all data with group labels
	type dataPoint struct {
		value float64
		group int
	}

	var allData []dataPoint
	groupSizes := make([]int, len(groups))

	for g, group := range groups {
		if group == nil {
			return nil, fmt.Errorf("group %d cannot be nil", g)
		}

		flatGroup := group.Flatten()
		for i := 0; i < flatGroup.Size(); i++ {
			val := convertToFloat64(flatGroup.At(i))
			if !math.IsNaN(val) {
				allData = append(allData, dataPoint{value: val, group: g})
				groupSizes[g]++
			}
		}
	}

	n := len(allData)
	if n < 3 {
		return nil, errors.New("need at least 3 total observations")
	}

	// Sort all data
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if allData[i].value > allData[j].value {
				allData[i], allData[j] = allData[j], allData[i]
			}
		}
	}

	// Assign ranks (handle ties by averaging)
	ranks := make([]float64, n)
	i := 0
	for i < n {
		j := i
		// Find end of tied values
		for j < n && allData[j].value == allData[i].value {
			j++
		}

		// Calculate average rank for tied values
		avgRank := float64(i+j+1) / 2.0 // +1 because ranks start at 1
		for k := i; k < j; k++ {
			ranks[k] = avgRank
		}
		i = j
	}

	// Calculate rank sums for each group
	rankSums := make([]float64, len(groups))
	for i, point := range allData {
		rankSums[point.group] += ranks[i]
	}

	// Calculate H statistic
	h := 0.0
	for g, rankSum := range rankSums {
		if groupSizes[g] > 0 {
			h += (rankSum * rankSum) / float64(groupSizes[g])
		}
	}

	h = (12.0/(float64(n)*float64(n+1)))*h - 3.0*(float64(n)+1.0)

	// Degrees of freedom
	df := len(groups) - 1

	// P-value using chi-square distribution
	pValue := 1.0 - chiSquareCDF(h, float64(df))

	return &KruskalWallisResult{
		HStatistic:       h,
		PValue:           pValue,
		DegreesOfFreedom: df,
	}, nil
}

// FriedmanTest performs the Friedman test (non-parametric repeated measures ANOVA)
// Data should be organized as subjects x treatments (2D array)
func FriedmanTest(data *array.Array) (*FriedmanResult, error) {
	ctx := internal.StartProfiler("Stats.FriedmanTest")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if data == nil {
		return nil, errors.New("data cannot be nil")
	}

	shape := data.Shape()
	if shape.Ndim() != 2 {
		return nil, errors.New("data must be 2-dimensional (subjects x treatments)")
	}

	subjects := shape[0]
	treatments := shape[1]

	if subjects < 2 {
		return nil, errors.New("need at least 2 subjects")
	}
	if treatments < 2 {
		return nil, errors.New("need at least 2 treatments")
	}

	// Rank each subject's data
	rankSums := make([]float64, treatments)

	for i := 0; i < subjects; i++ {
		// Get subject's data
		subjectData := make([]float64, treatments)
		for j := 0; j < treatments; j++ {
			subjectData[j] = convertToFloat64(data.At(i, j))
		}

		// Rank within subject
		subjectRanks := rankArray(subjectData)

		// Add to rank sums
		for j, rank := range subjectRanks {
			rankSums[j] += rank
		}
	}

	// Calculate chi-square statistic
	chiSquare := 0.0
	expectedRankSum := float64(subjects*(treatments+1)) / 2.0

	for _, rankSum := range rankSums {
		diff := rankSum - expectedRankSum
		chiSquare += diff * diff
	}

	chiSquare = (12.0 / (float64(subjects) * float64(treatments) * float64(treatments+1))) * chiSquare

	// Degrees of freedom
	df := treatments - 1

	// P-value
	pValue := 1.0 - chiSquareCDF(chiSquare, float64(df))

	return &FriedmanResult{
		ChiSquareStatistic: chiSquare,
		PValue:             pValue,
		DegreesOfFreedom:   df,
	}, nil
}

// Helper functions

// rankArray assigns ranks to values (handles ties by averaging)
func rankArray(values []float64) []float64 {
	n := len(values)
	if n == 0 {
		return nil
	}

	// Create index array to track original positions
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	// Sort indices by values
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if values[indices[i]] > values[indices[j]] {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}

	// Assign ranks
	ranks := make([]float64, n)
	i := 0
	for i < n {
		j := i
		// Find end of tied values
		currentValue := values[indices[i]]
		for j < n && values[indices[j]] == currentValue {
			j++
		}

		// Calculate average rank for tied values
		avgRank := float64(i+j+1) / 2.0 // +1 because ranks start at 1
		for k := i; k < j; k++ {
			ranks[indices[k]] = avgRank
		}
		i = j
	}

	return ranks
}

// matrixToArray converts 2D slice to Array
func matrixToArray(matrix [][]float64) *array.Array {
	if len(matrix) == 0 {
		return nil
	}

	rows := len(matrix)
	cols := len(matrix[0])
	data := make([]float64, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = matrix[i][j]
		}
	}

	arr, _ := array.NewArrayWithShape(data, []int{rows, cols})
	return arr
}

// F-distribution p-value calculation is already defined in regression.go
