package glasso

import (
	"testing"

	"github.com/bmizerany/assert"
)

func TestLeastSquares(t *testing.T) {
	// make linearly independent columns
	data := []float64{
		-1.1931518, -0.6704931, -0.474932601, -0.3698354709,
		0.4190582, -0.1185333, -0.125655095, -0.1175419928,
		0.2793721, 0.6082578, -0.006221774, -0.0095660929,
		0.2095291, 0.5616490, 0.910557371, 0.0001879049,
	}
	labels := []string{"temp", "2", "3", "$"}

	// make the data frame
	df, _ := DF(data, labels)

	// response variable for regression
	y := []float64{1.0, 2.0, 3.0, 4.0}

	// instantiate OLS struct
	lm := NewOLS(df)

	err := lm.Train(y)

	assert.Equal(t, err, nil)
}
