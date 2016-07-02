package glasso

import (
	"testing"

	"github.com/bmizerany/assert"
)

// Stackloss Data set from R
var data = [][]float64{
	{80.0, 27.0, 89.0},
	{80.0, 27.0, 88.0},
	{75.0, 25.0, 90.0},
	{62.0, 24.0, 87.0},
	{62.0, 22.0, 87.0},
	{62.0, 23.0, 87.0},
	{62.0, 24.0, 93.0},
	{62.0, 24.0, 93.0},
	{58.0, 23.0, 87.0},
	{58.0, 18.0, 80.0},
	{58.0, 18.0, 89.0},
	{58.0, 17.0, 88.0},
	{58.0, 18.0, 82.0},
	{58.0, 19.0, 93.0},
	{50.0, 18.0, 89.0},
	{50.0, 18.0, 86.0},
	{50.0, 19.0, 72.0},
	{50.0, 19.0, 79.0},
	{50.0, 20.0, 80.0},
	{56.0, 20.0, 82.0},
	{70.0, 20.0, 91.0},
}

// response
var y = []float64{42.0, 37.0, 37.0, 28.0, 18.0, 18.0, 19.0, 20.0, 15.0, 14.0, 14.0, 13.0, 11.0, 12.0, 8.0, 7.0, 8.0, 8.0, 9.0, 15.0, 15.0}

func TestLeastSquares(t *testing.T) {
	// make the data frame
	df := NewDF(data)

	// instantiate OLS struct
	lm := NewOLS(df)

	// train model
	err := lm.Train(y)
	assert.Equal(t, err, nil)

	// compare values to output in summary() function in R
	t.Logf("Betas: %v", roundAll(lm.Coefficients()))
	t.Logf("Residuals: %v...", roundAll(lm.Residuals()[0:4]))
	t.Logf("Yhat: %v...", roundAll(lm.Yhat()[0:4]))
	t.Log(lm)
}
