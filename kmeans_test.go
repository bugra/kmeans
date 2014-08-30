package kmeans

import (
	"io/ioutil"
	"log"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// Test K-Means Algorithm in Iris Dataset
func TestKmeans(t *testing.T) {
	filePath, err := filepath.Abs("data/iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		log.Fatal(err)
	}

	lines := strings.Split(string(content), "\n")
	irisData := make([][]float64, len(lines))
	irisLabels := make([]string, len(lines))
	for ii, line := range lines {
		vector := strings.Split(line, ",")
		label := vector[len(vector)-1]
		vector = vector[:len(vector)-1]
		floatVector := make([]float64, len(vector))
		for jj := range vector {
			floatVector[jj], err = strconv.ParseFloat(vector[jj], 64)
		}
		irisData[ii] = floatVector
		irisLabels[ii] = label
	}
	threshold := 10
	// Best Distance for Iris is Canberra Distance
	labels, err := Kmeans(irisData, 3, CanberraDistance, threshold)
	if err != nil {
		log.Fatal(err)
	}

	misclassifiedOnes := 0
	for ii, jj := range labels {
		if ii < 50 {
			if jj != 2 {
				misclassifiedOnes++
			}
		} else if ii < 100 {
			if jj != 1 {
				misclassifiedOnes++
			}
		} else {
			if jj != 0 {
				misclassifiedOnes++
			}
		}
	}
}
