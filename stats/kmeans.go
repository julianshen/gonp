package stats

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// KMeansResult contains the results of K-means clustering
type KMeansResult struct {
	Labels      []int        // Cluster labels for each data point (0 to k-1)
	Centers     *array.Array // Cluster centers (k x n_features)
	Inertia     float64      // Sum of squared distances to cluster centers
	NClusters   int          // Number of clusters
	NIterations int          // Number of iterations until convergence
	Converged   bool         // Whether the algorithm converged
}

// KMeansOptions contains options for K-means clustering
type KMeansOptions struct {
	InitMethod     string       // Initialization method: "random", "kmeans++", "manual"
	MaxIterations  int          // Maximum number of iterations (default: 300)
	Tolerance      float64      // Convergence tolerance (default: 1e-4)
	RandomSeed     int64        // Random seed for reproducibility (default: current time)
	InitialCenters *array.Array // Manual initial centers (only for "manual" init)
}

// KMeans performs K-means clustering on the input data
//
// Parameters:
//
//	data: Input data matrix (n_samples x n_features)
//	k: Number of clusters
//	options: Clustering options (can be nil for defaults)
//
// Returns: KMeansResult with cluster assignments and centers
func KMeans(data *array.Array, k int, options *KMeansOptions) (*KMeansResult, error) {
	if data == nil {
		return nil, fmt.Errorf("data array cannot be nil")
	}

	if data.Ndim() != 2 {
		return nil, fmt.Errorf("data must be 2-dimensional, got %dD", data.Ndim())
	}

	shape := data.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	if nSamples == 0 || nFeatures == 0 {
		return nil, fmt.Errorf("data cannot be empty")
	}

	if k <= 0 {
		return nil, fmt.Errorf("number of clusters must be positive, got %d", k)
	}

	if k > nSamples {
		return nil, fmt.Errorf("number of clusters (%d) cannot exceed number of samples (%d)",
			k, nSamples)
	}

	// Set default options
	if options == nil {
		options = &KMeansOptions{}
	}
	if options.InitMethod == "" {
		options.InitMethod = "kmeans++"
	}
	if options.MaxIterations <= 0 {
		options.MaxIterations = 300
	}
	if options.Tolerance <= 0 {
		options.Tolerance = 1e-4
	}
	if options.RandomSeed == 0 {
		options.RandomSeed = time.Now().UnixNano()
	}

	// Initialize random number generator
	rng := rand.New(rand.NewSource(options.RandomSeed))

	// Initialize cluster centers
	centers, err := initializeCenters(data, k, options, rng)
	if err != nil {
		return nil, fmt.Errorf("center initialization failed: %v", err)
	}

	// Initialize labels
	labels := make([]int, nSamples)
	previousInertia := math.Inf(1)
	converged := false

	var iterations int
	for iterations = 0; iterations < options.MaxIterations; iterations++ {
		// Assignment step: assign each point to nearest cluster center
		changed := false
		for i := 0; i < nSamples; i++ {
			nearestCluster := findNearestCluster(data, i, centers)
			if labels[i] != nearestCluster {
				labels[i] = nearestCluster
				changed = true
			}
		}

		// If no assignments changed, we've converged
		if !changed && iterations > 0 {
			converged = true
			break
		}

		// Update step: compute new cluster centers
		err = updateCenters(data, labels, centers, k)
		if err != nil {
			return nil, fmt.Errorf("center update failed: %v", err)
		}

		// Compute inertia (within-cluster sum of squares)
		inertia := computeInertia(data, labels, centers)

		// Check for convergence based on inertia change
		if math.Abs(previousInertia-inertia) < options.Tolerance {
			converged = true
			break
		}

		previousInertia = inertia
	}

	// Final inertia calculation
	finalInertia := computeInertia(data, labels, centers)

	result := &KMeansResult{
		Labels:      labels,
		Centers:     centers,
		Inertia:     finalInertia,
		NClusters:   k,
		NIterations: iterations + 1,
		Converged:   converged,
	}

	return result, nil
}

// initializeCenters initializes cluster centers using the specified method
func initializeCenters(data *array.Array, k int, options *KMeansOptions, rng *rand.Rand) (*array.Array, error) {
	shape := data.Shape()
	nFeatures := shape[1]

	centers := array.Empty(internal.Shape{k, nFeatures}, internal.Float64)

	switch options.InitMethod {
	case "random":
		return initializeRandomCenters(data, k, rng)

	case "kmeans++":
		return initializeKMeansPlusPlusCenters(data, k, rng)

	case "manual":
		if options.InitialCenters == nil {
			return nil, fmt.Errorf("manual initialization requires InitialCenters")
		}

		centerShape := options.InitialCenters.Shape()
		if centerShape[0] != k || centerShape[1] != nFeatures {
			return nil, fmt.Errorf("InitialCenters shape (%v) must match (k=%d, n_features=%d)",
				centerShape, k, nFeatures)
		}

		// Copy the provided centers
		for i := 0; i < k; i++ {
			for j := 0; j < nFeatures; j++ {
				val := options.InitialCenters.At(i, j)
				centers.Set(val, i, j)
			}
		}
		return centers, nil

	default:
		return nil, fmt.Errorf("unknown initialization method: %s", options.InitMethod)
	}
}

// initializeRandomCenters randomly selects k data points as initial centers
func initializeRandomCenters(data *array.Array, k int, rng *rand.Rand) (*array.Array, error) {
	shape := data.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	centers := array.Empty(internal.Shape{k, nFeatures}, internal.Float64)

	// Select k random data points as initial centers
	selectedIndices := make(map[int]bool)
	for i := 0; i < k; i++ {
		// Find a unique random index
		var idx int
		for {
			idx = rng.Intn(nSamples)
			if !selectedIndices[idx] {
				selectedIndices[idx] = true
				break
			}
		}

		// Copy the selected point as a center
		for j := 0; j < nFeatures; j++ {
			val := data.At(idx, j)
			centers.Set(val, i, j)
		}
	}

	return centers, nil
}

// initializeKMeansPlusPlusCenters uses K-means++ initialization for better initial centers
func initializeKMeansPlusPlusCenters(data *array.Array, k int, rng *rand.Rand) (*array.Array, error) {
	shape := data.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	centers := array.Empty(internal.Shape{k, nFeatures}, internal.Float64)

	// Step 1: Choose first center randomly
	firstIdx := rng.Intn(nSamples)
	for j := 0; j < nFeatures; j++ {
		val := data.At(firstIdx, j)
		centers.Set(val, 0, j)
	}

	// Step 2: Choose remaining centers using weighted probability
	for centerIdx := 1; centerIdx < k; centerIdx++ {
		// Compute distances from each point to nearest existing center
		distances := make([]float64, nSamples)
		totalDistance := 0.0

		for i := 0; i < nSamples; i++ {
			minDist := math.Inf(1)

			// Find distance to nearest existing center
			for c := 0; c < centerIdx; c++ {
				dist := euclideanDistance(data, i, centers, c)
				if dist < minDist {
					minDist = dist
				}
			}

			distances[i] = minDist * minDist // Use squared distance for weighting
			totalDistance += distances[i]
		}

		// Choose next center with probability proportional to squared distance
		if totalDistance == 0 {
			// All points are identical to existing centers, choose randomly
			randomIdx := rng.Intn(nSamples)
			for j := 0; j < nFeatures; j++ {
				val := data.At(randomIdx, j)
				centers.Set(val, centerIdx, j)
			}
		} else {
			threshold := rng.Float64() * totalDistance
			cumulative := 0.0

			for i := 0; i < nSamples; i++ {
				cumulative += distances[i]
				if cumulative >= threshold {
					for j := 0; j < nFeatures; j++ {
						val := data.At(i, j)
						centers.Set(val, centerIdx, j)
					}
					break
				}
			}
		}
	}

	return centers, nil
}

// findNearestCluster finds the cluster center nearest to a data point
func findNearestCluster(data *array.Array, pointIdx int, centers *array.Array) int {
	k := centers.Shape()[0]

	minDistance := math.Inf(1)
	nearestCluster := 0

	for clusterIdx := 0; clusterIdx < k; clusterIdx++ {
		distance := euclideanDistance(data, pointIdx, centers, clusterIdx)
		if distance < minDistance {
			minDistance = distance
			nearestCluster = clusterIdx
		}
	}

	return nearestCluster
}

// euclideanDistance computes the Euclidean distance between a data point and a center
func euclideanDistance(data *array.Array, pointIdx int, centers *array.Array, centerIdx int) float64 {
	nFeatures := data.Shape()[1]

	sum := 0.0
	for j := 0; j < nFeatures; j++ {
		dataVal := convertToFloat64(data.At(pointIdx, j))
		centerVal := convertToFloat64(centers.At(centerIdx, j))
		diff := dataVal - centerVal
		sum += diff * diff
	}

	return math.Sqrt(sum)
}

// updateCenters computes new cluster centers as the mean of assigned points
func updateCenters(data *array.Array, labels []int, centers *array.Array, k int) error {
	shape := data.Shape()
	nSamples := shape[0]
	nFeatures := shape[1]

	// Count points in each cluster
	clusterCounts := make([]int, k)
	for _, label := range labels {
		clusterCounts[label]++
	}

	// Reset centers to zero
	for i := 0; i < k; i++ {
		for j := 0; j < nFeatures; j++ {
			centers.Set(0.0, i, j)
		}
	}

	// Sum up all points assigned to each cluster
	for i := 0; i < nSamples; i++ {
		cluster := labels[i]
		for j := 0; j < nFeatures; j++ {
			currentCenter := convertToFloat64(centers.At(cluster, j))
			dataVal := convertToFloat64(data.At(i, j))
			newCenter := currentCenter + dataVal
			centers.Set(newCenter, cluster, j)
		}
	}

	// Divide by cluster size to get means
	for i := 0; i < k; i++ {
		if clusterCounts[i] > 0 {
			for j := 0; j < nFeatures; j++ {
				currentSum := convertToFloat64(centers.At(i, j))
				mean := currentSum / float64(clusterCounts[i])
				centers.Set(mean, i, j)
			}
		} else {
			// For empty clusters, set center to first data point
			// This handles cases where we have more clusters than unique points
			for j := 0; j < nFeatures; j++ {
				val := convertToFloat64(data.At(0, j))
				centers.Set(val, i, j)
			}
		}
	}

	return nil
}

// computeInertia calculates the within-cluster sum of squared distances
func computeInertia(data *array.Array, labels []int, centers *array.Array) float64 {
	nSamples := data.Shape()[0]

	totalInertia := 0.0
	for i := 0; i < nSamples; i++ {
		cluster := labels[i]
		distance := euclideanDistance(data, i, centers, cluster)
		totalInertia += distance * distance
	}

	return totalInertia
}
