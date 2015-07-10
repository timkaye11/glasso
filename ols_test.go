package glasso

import (
	"testing"

	"github.com/bmizerany/assert"
)

// make sure everything is constructed OK
func TestMakeDF(t *testing.T) {
	// make dataframe
	data := []float64{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9}
	labels := []string{"a", "", "c"}

	df, err := DF(data, labels)
	assert.Equal(t, err, nil)
	assert.Equal(t, df.rows, 3)
	assert.Equal(t, df.cols, 3)
	assert.Equal(t, df.colToIdx["a"], 0)
}

// test transform dataframe function
func TestTransformDF(t *testing.T) {
	// a, b, c |
	//---------|
	// 1, 2, 3 |
	// 4, 5, 6 |
	// 7, 8, 9 |
	//---------/
	// sum : 12, 15, 18

	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	labels := []string{"a", "b", "c"}
	df, _ := DF(data, labels)

	// silyl transformation function
	add1 := func(x float64) float64 {
		return x + 1
	}

	// add 1 to every number in cols "a" & "c"
	df.Transform(add1, "a", "c")
	newA, _ := df.GetCol("a")
	newB, _ := df.GetCol("b")
	newC, _ := df.GetCol("c")

	assert.Equal(t, sum(newA), 15.0)
	assert.Equal(t, sum(newB), 15.0) // shouldn't change
	assert.Equal(t, sum(newC), 21.0)
}

// test apply function
func TestApplyDF(t *testing.T) {
	// make data
	data := []float64{
		1, 2, 3,
		2, 3, 1,
		3, 1, 2,
	}
	labels := []string{"a", "b", "c"}
	df, _ := DF(data, labels)

	colProds := df.Apply(mult, true, 0, 1, 2)
	rowProds := df.Apply(mult, false, 0, 1, 2)

	// all the products should equal 6
	for i := 0; i < 3; i++ {
		assert.T(t, colProds[i] == rowProds[i])
	}

}

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
