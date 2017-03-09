package glasso

import (
	"testing"

	"github.com/bmizerany/assert"
)

func TestLeastSquares(t *testing.T) {
	// Stackloss Data set from R
	data := [][]float64{
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

	// make the data frame
	df := NewDataFrame(data)

	// response
	y := []float64{42.0, 37.0, 37.0, 28.0, 18.0, 18.0, 19.0, 20.0, 15.0, 14.0, 14.0, 13.0, 11.0, 12.0, 8.0, 7.0, 8.0, 8.0, 9.0, 15.0, 15.0}

	trainer := NewOlsTrainer()
	model, summary, err := trainer.Train(df, y)
	assert.Equal(t, nil, err)

	// compare values to output in summary() function in R
	t.Logf("Betas: %v", roundAll(summary.Coefficients()))
	t.Logf("Residuals: %v...", roundAll(summary.Residuals()[0:4]))
	t.Logf("Yhat: %v...", roundAll(summary.Yhat()[0:4]))
	t.Logf("predict=%v", model.Predict([]float64{70.0, 20.0, 91.0}))
}
