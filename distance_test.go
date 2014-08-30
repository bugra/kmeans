package kmeans

/*
TODO: Figure out the limit in the column number and
how to format the unfinished lines due to the limitation

Test for Weighted Minkowski Distance should be improved
*/

import (
	"math"
	"testing"
)

func TestLPNorm(t *testing.T) {
	vector := []float64{3., 4.}
	const l1Out, l2Out = 7., 5.

	l1Norm, _ := LPNorm(vector, 1.)
	l2Norm, _ := LPNorm(vector, 2.)
	if l1Norm != l1Out {
		t.Errorf("Computed l1 Norm: %f\nActual l1 Norm: %f", l1Norm, l1Out)
	}
	if l2Norm != l2Out {
		t.Errorf("Computed l2 Norm: %f\nActual l2 Norm: %f", l2Norm, l2Out)
	}
}

func TestManhattanDistance(t *testing.T) {
	firstVector := []float64{1., 2., 3., 2}
	secondVector := []float64{3., 4., 5., -1}
	const out = 9.
	mDistance, _ := ManhattanDistance(firstVector, secondVector)
	if mDistance != out {
		t.Errorf("\nComputed Manhattan Distance: %f\nActual Manhattan Distance: %f", mDistance, out)
	}
}

func TestEuclideanDistance(t *testing.T) {
	firstVector := []float64{5., 12.}
	secondVector := []float64{0., 0.}
	thirdVector := []float64{8., 15.}
	fourthVector := []float64{20., 20.}
	out2 := math.Sqrt(18)
	const out1, out3 = 13., 17.
	firstEuclideanDistance, _ := EuclideanDistance(firstVector, secondVector)
	secondEuclideanDistance, _ := EuclideanDistance(firstVector, thirdVector)
	thirdEuclideanDistance, _ := EuclideanDistance(thirdVector, secondVector)
	anotherFirst, _ := EuclideanDistance(thirdVector, fourthVector)

	if out1 != firstEuclideanDistance {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", firstEuclideanDistance, out1)
	}
	if out1 != anotherFirst {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", firstEuclideanDistance, out1)
	}
	if out2 != secondEuclideanDistance {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", secondEuclideanDistance, out2)
	}
	if out3 != thirdEuclideanDistance {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", thirdEuclideanDistance, out3)
	}
}

func TestSquareEuclideanDistance(t *testing.T) {
	firstVector := []float64{5., 12.}
	secondVector := []float64{0., 0.}
	thirdVector := []float64{8., 15.}
	fourthVector := []float64{20., 20.}
	const out1, out3 = 169., 289.
	firstSquaredEuclideanDistance, _ := SquaredEuclideanDistance(firstVector, secondVector)
	thirdSquaredEuclideanDistance, _ := SquaredEuclideanDistance(thirdVector, secondVector)
	anotherFirst, _ := SquaredEuclideanDistance(thirdVector, fourthVector)

	if out1 != firstSquaredEuclideanDistance {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", firstSquaredEuclideanDistance, out1)
	}
	if out1 != anotherFirst {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", firstSquaredEuclideanDistance, out1)
	}
	if out3 != thirdSquaredEuclideanDistance {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", thirdSquaredEuclideanDistance, out3)
	}
}

func TestMinkowskiDistance(t *testing.T) {
	// p = 1 Test
	// Should be equal to Manhattan Distance
	firstVector := []float64{1., 2., 3., 2}
	secondVector := []float64{3., 4., 5., -1}
	mDistance, _ := ManhattanDistance(firstVector, secondVector)
	l1MinkowskiDistance, _ := MinkowskiDistance(firstVector, secondVector, 1.)
	if mDistance != l1MinkowskiDistance {
		t.Errorf("\nComputed l1 Minkowski Distance: %f\nComputed Manhattan Distance: %f", l1MinkowskiDistance, mDistance)
	}

	// p = 2 Test
	// Should be equal to Euclidean Distance
	firstVector = []float64{5., 12.}
	secondVector = []float64{0., 0.}
	thirdVector := []float64{8., 15.}
	fourthVector := []float64{20., 20.}

	firstEuclideanDistance, _ := EuclideanDistance(firstVector, secondVector)
	anotherFirstEuclideanDistance, _ := EuclideanDistance(thirdVector, fourthVector)
	secondEuclideanDistance, _ := EuclideanDistance(firstVector, thirdVector)
	thirdEuclideanDistance, _ := EuclideanDistance(thirdVector, secondVector)

	firstl2MinkowskiDistance, _ := MinkowskiDistance(firstVector, secondVector, 2.)
	anotherFirstl2MinkowskiDistance, _ := MinkowskiDistance(thirdVector, fourthVector, 2.)
	secondl2MinkowskiDistance, _ := MinkowskiDistance(firstVector, thirdVector, 2.)
	thirdl2MinkowskiDistance, _ := MinkowskiDistance(thirdVector, secondVector, 2.)

	if firstEuclideanDistance != firstl2MinkowskiDistance {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nComputed Euclidean Distance: %f", firstl2MinkowskiDistance, firstEuclideanDistance)
	}
	if secondEuclideanDistance != secondl2MinkowskiDistance {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nComputed Euclidean Distance: %f", secondl2MinkowskiDistance, secondEuclideanDistance)
	}
	if thirdEuclideanDistance != thirdl2MinkowskiDistance {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nComputed Euclidean Distance: %f", thirdl2MinkowskiDistance, thirdEuclideanDistance)
	}
	if anotherFirstEuclideanDistance != firstl2MinkowskiDistance {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nComputed Euclidean Distance: %f", anotherFirstl2MinkowskiDistance, anotherFirstEuclideanDistance)
	}

	// p = 3 and p = 4 Test
	const l3Minkowski, l4Minkowski, precision = 12.282642, 12.089418, 1000000.

	computedl3Minkowski, _ := MinkowskiDistance(firstVector, secondVector, 3.)
	computedl4Minkowski, _ := MinkowskiDistance(firstVector, secondVector, 4.)
	computedl3Minkowski = float64(int(computedl3Minkowski*precision)) / precision
	computedl4Minkowski = float64(int(computedl4Minkowski*precision)) / precision

	if l3Minkowski != computedl3Minkowski {
		t.Errorf("\nComputed l3 Minkowski Distance: %f\nActual l3 Minkowski Distance: %f", computedl3Minkowski, l3Minkowski)
	}

	if l4Minkowski != computedl4Minkowski {
		t.Errorf("\nComputed l4 Minkowski Distance: %f\nActual l4 Minkowski Distance: %f", computedl4Minkowski, l4Minkowski)
	}
}

func TestWeightedMinkowskiDistance(t *testing.T) {
	// Weight Vector is all 1.
	// Results should be same when we do not apply any weighting vector
	firstVector := []float64{1., 2., 3., 2}
	secondVector := []float64{3., 4., 5., -1}
	weightVector := []float64{1., 1., 1., 1.}
	l1MinkowskiDistance, _ := MinkowskiDistance(firstVector, secondVector, 1.)
	l1WeightedMinkowskiDistance, _ := WeightedMinkowskiDistance(firstVector, secondVector, weightVector, 1.)

	if l1MinkowskiDistance != l1WeightedMinkowskiDistance {
		t.Errorf("\nComputed l1 Minkowski Distance: %f\nActual Weighted(1., 1., ...) l1 Minkowski Distance: %f", l1MinkowskiDistance, l1WeightedMinkowskiDistance)
	}

	firstVector = []float64{5., 12.}
	secondVector = []float64{0., 0.}
	thirdVector := []float64{8., 15.}
	fourthVector := []float64{20., 20.}

	firstl2MinkowskiDistance, _ := MinkowskiDistance(firstVector, secondVector, 2.)
	anotherFirstl2MinkowskiDistance, _ := MinkowskiDistance(thirdVector, fourthVector, 2.)
	secondl2MinkowskiDistance, _ := MinkowskiDistance(firstVector, thirdVector, 2.)
	thirdl2MinkowskiDistance, _ := MinkowskiDistance(thirdVector, secondVector, 2.)

	firstl2WeightedMinkowskiDistance, _ := WeightedMinkowskiDistance(firstVector, secondVector, weightVector, 2.)
	anotherFirstl2WeightedMinkowskiDistance, _ := WeightedMinkowskiDistance(thirdVector, fourthVector, weightVector, 2.)
	secondl2WeightedMinkowskiDistance, _ := WeightedMinkowskiDistance(firstVector, thirdVector, weightVector, 2.)
	thirdl2WeightedMinkowskiDistance, _ := WeightedMinkowskiDistance(thirdVector, secondVector, weightVector, 2.)

	if firstl2MinkowskiDistance != firstl2WeightedMinkowskiDistance {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nActual Weighted(1., 1., ...) l2 Minkowski Distance: %f", firstl2MinkowskiDistance, firstl2WeightedMinkowskiDistance)
	}
	if anotherFirstl2MinkowskiDistance != anotherFirstl2WeightedMinkowskiDistance {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nActual Weighted(1., 1., ...) l2 Minkowski Distance: %f", anotherFirstl2MinkowskiDistance, anotherFirstl2WeightedMinkowskiDistance)
	}
	if secondl2MinkowskiDistance != secondl2WeightedMinkowskiDistance {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nActual Weighted(1., 1., ...) l2 Minkowski Distance: %f", secondl2MinkowskiDistance, secondl2WeightedMinkowskiDistance)
	}
	if thirdl2MinkowskiDistance != thirdl2WeightedMinkowskiDistance {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nActual Weighted(1., 1., ...) l2 Minkowski Distance: %f", thirdl2MinkowskiDistance, thirdl2WeightedMinkowskiDistance)
	}

}

func TestChebyshevDistance(t *testing.T) {
	firstVector := []float64{1., 2., 3., 4.}
	secondVector := []float64{3., -4., 6., 1.5}
	thirdVector := []float64{4., 3., -2.5, -5.}
	const firstActual, secondActual = 6., 8.5
	firstComputed, _ := ChebyshevDistance(firstVector, secondVector)
	secondComputed, _ := ChebyshevDistance(secondVector, thirdVector)
	if firstComputed != firstActual {
		t.Errorf("\nComputed Chebyshev Distance: %f\nActual Chebyshev Distance: %f", firstComputed, firstActual)
	}
	if secondComputed != secondActual {
		t.Errorf("\nComputed Chebyshev Distance: %f\nActual Chebyshev Distance: %f", secondComputed, secondActual)
	}
}

func TestHammingDistance(t *testing.T) {
	firstVector := []float64{1., 2., 2.5, 3., 4.}
	secondVector := []float64{1., 2.5, 3., 3., 4.}
	thirdVector := []float64{1., 2., 3., 4., 5., 6.}
	fourthVector := []float64{1., 1., 1., 1., 1., 1.}
	const firstActual, secondActual = 2, 5
	firstComputed, _ := HammingDistance(firstVector, secondVector)
	secondComputed, _ := HammingDistance(thirdVector, fourthVector)

	if firstComputed != firstActual {
		t.Errorf("\nComputed Hamming Distance: %f\nActual Hamming Distance: %f", firstComputed, firstActual)
	}
	if secondComputed != secondActual {
		t.Errorf("\nComputed Hamming Distance: %f\nActual Hmming Distance: %f", secondComputed, secondActual)
	}
}

func TestBrayCurtisDistance(t *testing.T) {
	firstVector := []float64{1., 2., 3., 4., 5.}
	secondVector := []float64{1.5, 2.5, 5., 5., 6.}

	thirdVector := []float64{3., 2., 4., 6.5, 7}
	fourthVector := []float64{1., 6., 3., 5.5, 4.5}
	const firstActual, secondActual, precision = 0.14285, 0.24705, 100000

	firstComputed, _ := BrayCurtisDistance(firstVector, secondVector)
	secondComputed, _ := BrayCurtisDistance(thirdVector, fourthVector)

	firstComputed = float64(int(firstComputed*precision)) / precision
	secondComputed = float64(int(secondComputed*precision)) / precision

	if firstComputed != firstActual {
		t.Errorf("\nComputed Bray Curtis Distance: %f\nActual Bray Curtis Distance: %f", firstComputed, firstActual)
	}
	if secondComputed != secondActual {
		t.Errorf("\nComputed Bray Curtis Distance: %f\nActual Bray Curtis Distance: %f", secondComputed, secondActual)
	}
}

func TestCanberraDistance(t *testing.T) {
	firstVector := []float64{3., 4., 5., -2., 4.}
	secondVector := []float64{2., 6., 5., 3., -1.}
	const firstActual = 2.4
	firstComputed, _ := CanberraDistance(firstVector, secondVector)
	if firstActual != firstComputed {
		t.Errorf("Computed Canberra Distance: %f\n Actual Canberra Distance: %f", firstComputed, firstActual)
	}
}
