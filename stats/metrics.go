package stats

import (
	"errors"
	"math"
	"sort"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// ClassMetrics represents metrics for a single class
type ClassMetrics struct {
	Precision float64
	Recall    float64
	F1Score   float64
	Support   int // Number of samples in this class
}

// ClassificationReport represents a comprehensive classification report
type ClassificationReport struct {
	Accuracy    float64               // Overall accuracy
	PerClass    map[int]*ClassMetrics // Per-class metrics
	MacroAvg    *ClassMetrics         // Macro-averaged metrics
	WeightedAvg *ClassMetrics         // Weighted-averaged metrics
}

// Accuracy computes classification accuracy
func Accuracy(yTrue, yPred *array.Array) (float64, error) {
	if yTrue == nil || yPred == nil {
		return 0, errors.New("yTrue and yPred cannot be nil")
	}

	if yTrue.Size() != yPred.Size() {
		return 0, errors.New("yTrue and yPred must have the same size")
	}

	correct := 0
	total := yTrue.Size()

	for i := 0; i < total; i++ {
		if convertToFloat64(yTrue.At(i)) == convertToFloat64(yPred.At(i)) {
			correct++
		}
	}

	return float64(correct) / float64(total), nil
}

// Precision computes precision for a specific class
func Precision(yTrue, yPred *array.Array, targetClass float64) (float64, error) {
	if yTrue == nil || yPred == nil {
		return 0, errors.New("yTrue and yPred cannot be nil")
	}

	if yTrue.Size() != yPred.Size() {
		return 0, errors.New("yTrue and yPred must have the same size")
	}

	truePositives := 0
	falsePositives := 0

	for i := 0; i < yTrue.Size(); i++ {
		predicted := convertToFloat64(yPred.At(i))
		actual := convertToFloat64(yTrue.At(i))

		if predicted == targetClass {
			if actual == targetClass {
				truePositives++
			} else {
				falsePositives++
			}
		}
	}

	if truePositives+falsePositives == 0 {
		return 0.0, nil // No predictions for this class
	}

	return float64(truePositives) / float64(truePositives+falsePositives), nil
}

// Recall computes recall for a specific class
func Recall(yTrue, yPred *array.Array, targetClass float64) (float64, error) {
	if yTrue == nil || yPred == nil {
		return 0, errors.New("yTrue and yPred cannot be nil")
	}

	if yTrue.Size() != yPred.Size() {
		return 0, errors.New("yTrue and yPred must have the same size")
	}

	truePositives := 0
	falseNegatives := 0

	for i := 0; i < yTrue.Size(); i++ {
		predicted := convertToFloat64(yPred.At(i))
		actual := convertToFloat64(yTrue.At(i))

		if actual == targetClass {
			if predicted == targetClass {
				truePositives++
			} else {
				falseNegatives++
			}
		}
	}

	if truePositives+falseNegatives == 0 {
		return 0.0, nil // No actual instances of this class
	}

	return float64(truePositives) / float64(truePositives+falseNegatives), nil
}

// F1Score computes F1 score for a specific class
func F1Score(yTrue, yPred *array.Array, targetClass float64) (float64, error) {
	precision, err := Precision(yTrue, yPred, targetClass)
	if err != nil {
		return 0, err
	}

	recall, err := Recall(yTrue, yPred, targetClass)
	if err != nil {
		return 0, err
	}

	if precision+recall == 0 {
		return 0.0, nil
	}

	return 2 * (precision * recall) / (precision + recall), nil
}

// ConfusionMatrix computes the confusion matrix
func ConfusionMatrix(yTrue, yPred *array.Array) (*array.Array, []int, error) {
	if yTrue == nil || yPred == nil {
		return nil, nil, errors.New("yTrue and yPred cannot be nil")
	}

	if yTrue.Size() != yPred.Size() {
		return nil, nil, errors.New("yTrue and yPred must have the same size")
	}

	// Get unique classes
	classSet := make(map[int]bool)
	for i := 0; i < yTrue.Size(); i++ {
		classSet[int(convertToFloat64(yTrue.At(i)))] = true
		classSet[int(convertToFloat64(yPred.At(i)))] = true
	}

	// Convert to sorted slice
	var classes []int
	for class := range classSet {
		classes = append(classes, class)
	}
	sort.Ints(classes)

	nClasses := len(classes)
	classToIndex := make(map[int]int)
	for i, class := range classes {
		classToIndex[class] = i
	}

	// Create confusion matrix
	cm := array.Zeros(internal.Shape{nClasses, nClasses}, internal.Int64)

	for i := 0; i < yTrue.Size(); i++ {
		trueClass := int(convertToFloat64(yTrue.At(i)))
		predClass := int(convertToFloat64(yPred.At(i)))

		trueIdx := classToIndex[trueClass]
		predIdx := classToIndex[predClass]

		currentVal := cm.At(trueIdx, predIdx).(int64)
		cm.Set(currentVal+1, trueIdx, predIdx)
	}

	return cm, classes, nil
}

// GenerateClassificationReport generates a comprehensive classification report
func GenerateClassificationReport(yTrue, yPred *array.Array) (*ClassificationReport, error) {
	if yTrue == nil || yPred == nil {
		return nil, errors.New("yTrue and yPred cannot be nil")
	}

	// Calculate overall accuracy
	accuracy, err := Accuracy(yTrue, yPred)
	if err != nil {
		return nil, err
	}

	// Get unique classes
	classSet := make(map[int]bool)
	for i := 0; i < yTrue.Size(); i++ {
		classSet[int(convertToFloat64(yTrue.At(i)))] = true
	}

	var classes []int
	for class := range classSet {
		classes = append(classes, class)
	}
	sort.Ints(classes)

	// Calculate per-class metrics
	perClass := make(map[int]*ClassMetrics)
	for _, class := range classes {
		classFloat := float64(class)

		precision, err := Precision(yTrue, yPred, classFloat)
		if err != nil {
			return nil, err
		}

		recall, err := Recall(yTrue, yPred, classFloat)
		if err != nil {
			return nil, err
		}

		f1, err := F1Score(yTrue, yPred, classFloat)
		if err != nil {
			return nil, err
		}

		// Count support (number of true instances)
		support := 0
		for i := 0; i < yTrue.Size(); i++ {
			if int(convertToFloat64(yTrue.At(i))) == class {
				support++
			}
		}

		perClass[class] = &ClassMetrics{
			Precision: precision,
			Recall:    recall,
			F1Score:   f1,
			Support:   support,
		}
	}

	// Calculate macro averages
	var macroPrec, macroRec, macroF1 float64
	for _, metrics := range perClass {
		macroPrec += metrics.Precision
		macroRec += metrics.Recall
		macroF1 += metrics.F1Score
	}
	nClasses := float64(len(classes))
	macroAvg := &ClassMetrics{
		Precision: macroPrec / nClasses,
		Recall:    macroRec / nClasses,
		F1Score:   macroF1 / nClasses,
		Support:   yTrue.Size(),
	}

	// Calculate weighted averages
	var weightedPrec, weightedRec, weightedF1 float64
	totalSupport := 0
	for _, metrics := range perClass {
		weight := float64(metrics.Support)
		weightedPrec += metrics.Precision * weight
		weightedRec += metrics.Recall * weight
		weightedF1 += metrics.F1Score * weight
		totalSupport += metrics.Support
	}
	weightedAvg := &ClassMetrics{
		Precision: weightedPrec / float64(totalSupport),
		Recall:    weightedRec / float64(totalSupport),
		F1Score:   weightedF1 / float64(totalSupport),
		Support:   totalSupport,
	}

	return &ClassificationReport{
		Accuracy:    accuracy,
		PerClass:    perClass,
		MacroAvg:    macroAvg,
		WeightedAvg: weightedAvg,
	}, nil
}

// ROCAUC computes the Area Under the ROC Curve for binary classification
func ROCAUC(yTrue, yScores *array.Array) (float64, error) {
	if yTrue == nil || yScores == nil {
		return 0, errors.New("yTrue and yScores cannot be nil")
	}

	if yTrue.Size() != yScores.Size() {
		return 0, errors.New("yTrue and yScores must have the same size")
	}

	// Check if binary classification
	classSet := make(map[int]bool)
	for i := 0; i < yTrue.Size(); i++ {
		classSet[int(convertToFloat64(yTrue.At(i)))] = true
	}

	if len(classSet) != 2 {
		return 0, errors.New("ROC AUC requires binary classification (exactly 2 classes)")
	}

	// Create sorted pairs of (score, true_label)
	type scorePair struct {
		score float64
		label int
	}

	pairs := make([]scorePair, yTrue.Size())
	for i := 0; i < yTrue.Size(); i++ {
		pairs[i] = scorePair{
			score: convertToFloat64(yScores.At(i)),
			label: int(convertToFloat64(yTrue.At(i))),
		}
	}

	// Sort by score in descending order
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score > pairs[j].score
	})

	// Count positive and negative examples
	nPos, nNeg := 0, 0
	for _, pair := range pairs {
		if pair.label == 1 {
			nPos++
		} else {
			nNeg++
		}
	}

	if nPos == 0 || nNeg == 0 {
		return 0, errors.New("ROC AUC requires both positive and negative examples")
	}

	// Calculate AUC using the trapezoidal rule
	tpr := 0.0 // True Positive Rate
	fpr := 0.0 // False Positive Rate
	auc := 0.0

	for _, pair := range pairs {
		if pair.label == 1 {
			tpr += 1.0 / float64(nPos)
		} else {
			// When we encounter a negative, we add the area under the curve
			// The area is the TPR * width (1/nNeg)
			auc += tpr * (1.0 / float64(nNeg))
			fpr += 1.0 / float64(nNeg)
		}
	}

	return auc, nil
}

// ROCCurve computes the ROC curve points (FPR, TPR, thresholds)
func ROCCurve(yTrue, yScores *array.Array) (*array.Array, *array.Array, *array.Array, error) {
	if yTrue == nil || yScores == nil {
		return nil, nil, nil, errors.New("yTrue and yScores cannot be nil")
	}

	if yTrue.Size() != yScores.Size() {
		return nil, nil, nil, errors.New("yTrue and yScores must have the same size")
	}

	// Check if binary classification
	classSet := make(map[int]bool)
	for i := 0; i < yTrue.Size(); i++ {
		classSet[int(convertToFloat64(yTrue.At(i)))] = true
	}

	if len(classSet) != 2 {
		return nil, nil, nil, errors.New("ROC curve requires binary classification")
	}

	// Create pairs and sort by score
	type scorePair struct {
		score float64
		label int
	}

	pairs := make([]scorePair, yTrue.Size())
	for i := 0; i < yTrue.Size(); i++ {
		pairs[i] = scorePair{
			score: convertToFloat64(yScores.At(i)),
			label: int(convertToFloat64(yTrue.At(i))),
		}
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score > pairs[j].score
	})

	// Count totals
	nPos, nNeg := 0, 0
	for _, pair := range pairs {
		if pair.label == 1 {
			nPos++
		} else {
			nNeg++
		}
	}

	// Calculate ROC points
	var fprPoints, tprPoints, thresholds []float64

	// Start at (0, 0)
	fprPoints = append(fprPoints, 0.0)
	tprPoints = append(tprPoints, 0.0)
	thresholds = append(thresholds, pairs[0].score+1.0) // Higher than any score

	fp, tp := 0, 0
	prevScore := math.Inf(1)

	for _, pair := range pairs {
		if pair.score != prevScore {
			// Add point for previous threshold
			fpr := float64(fp) / float64(nNeg)
			tpr := float64(tp) / float64(nPos)
			fprPoints = append(fprPoints, fpr)
			tprPoints = append(tprPoints, tpr)
			thresholds = append(thresholds, pair.score)
		}

		if pair.label == 1 {
			tp++
		} else {
			fp++
		}
		prevScore = pair.score
	}

	// Add final point (1, 1)
	fprPoints = append(fprPoints, 1.0)
	tprPoints = append(tprPoints, 1.0)
	thresholds = append(thresholds, pairs[len(pairs)-1].score-1.0) // Lower than any score

	// Convert to arrays
	fprArray, _ := array.FromSlice(fprPoints)
	tprArray, _ := array.FromSlice(tprPoints)
	thresholdArray, _ := array.FromSlice(thresholds)

	return fprArray, tprArray, thresholdArray, nil
}
