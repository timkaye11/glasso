package glasso

import (
	"testing"

	"github.com/bmizerany/assert"
	"github.com/gonum/matrix/mat64"
)

var (
	model   Model
	summary Summary
)

func init() {
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

	// response variable for regression
	y := []float64{42.0, 37.0, 37.0, 28.0, 18.0, 18.0, 19.0, 20.0, 15.0, 14.0, 14.0, 13.0, 11.0, 12.0, 8.0, 7.0, 8.0, 8.0, 9.0, 15.0, 15.0}

	// instantiate OLS struct
	trainer := NewOlsTrainer()
	var err error
	model, summary, err = trainer.Train(df, y)
	if err != nil {
		panic(err)
	}
}

func TestLeverage(t *testing.T) {
	// compare leverage values with output from R to make sure it's correct
	leverage := roundAll(LeveragePoints(summary))
	assert.Equal(t, leverage[0], .302)
	assert.Equal(t, leverage[1], .318)
	assert.Equal(t, leverage[20], 0.285)
}

func TestCooksDistance(t *testing.T) {
	// compare cooks distances with output from R to make sure it's correct
	cooks := roundAll(CooksDistance(summary))
	assert.Equal(t, cooks[0], 0.154)
	assert.Equal(t, cooks[1], 0.06)
	assert.Equal(t, cooks[20], 0.692)
}

func TestStudentized(t *testing.T) {
	// compare studentized residuals with output from R
	students := roundAll(StudentizedResiduals(summary))
	assert.Equal(t, len(students), 21)
	assert.Equal(t, students[0], 1.193)
	assert.Equal(t, students[1], -0.716)
}

func TestVarianceCovariance(t *testing.T) {
	var_cov, err := VarCov(summary)
	assert.Equal(t, nil, err)
	assert.Equal(t, roundAll(mat64.Col(nil, 0, var_cov.Data())), []float64{141.515, 0.288, -0.652, -1.677})
	assert.Equal(t, roundAll(mat64.Col(nil, 1, var_cov.Data())), []float64{0.288, 0.018, -0.037, -0.008})
	assert.Equal(t, roundAll(mat64.Col(nil, 2, var_cov.Data())), []float64{-0.652, -0.037, 0.135, 0})
	assert.Equal(t, roundAll(mat64.Col(nil, 3, var_cov.Data())), []float64{-1.677, -0.008, 0, 0.024})
}
