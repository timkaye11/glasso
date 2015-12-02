package glasso

import (
	"testing"

	"github.com/bmizerany/assert"
)

func makeDF() *DataFrame {
	data := [][]float64{
		{1.1, 4.4, 7.7},
		{2.2, 5.5, 8.8},
		{3.3, 6.6, 9.9},
	}
	labels := []string{"a", "b", "c"}

	return NewDataFrame(data, labels)
}

func TestMakeDF(t *testing.T) {
	df := makeDF()
	assert.Equal(t, df.n, 3)
	assert.Equal(t, df.c, 3)
}

func TestAppend(t *testing.T) {
	df := makeDF()

	col := []float64{4.7, 8.8, 9.2}
	err := df.AppendCol(col)
	assert.Equal(t, nil, err)
	assert.Equal(t, col, df.GetCol(df.c-1))

	row := []float64{3.1, 4.5, 6.5, 3.3}
	err = df.AppendRow(row)
	assert.Equal(t, nil, err)
	assert.Equal(t, row, df.GetRow(df.n-1))
}

func TestPush(t *testing.T) {
	df := makeDF()

	col := []float64{4.7, 8.8, 9.2}
	err := df.PushCol(col)
	assert.Equal(t, nil, err)
	assert.Equal(t, col, df.GetCol(0))

	row := []float64{3.1, 4.5, 6.5, 3.3}
	err = df.PushRow(row)
	assert.Equal(t, nil, err)
	assert.Equal(t, row, df.GetRow(0))
}

func TestTransformDF(t *testing.T) {
	df := makeDF()

	// use a trivial evaluator.
	var simple Evaluator
	simple = func(x float64) float64 {
		return x + 1
	}

	// apply transformation to 0th, and 2nd columns.
	df.Transform(simple, 0, 2)
	vals := []float64{df.GetCol(0), df.GetCol(1), df.GetCol(2)}

	// (1.1 + 2.2 + 3.3) + 1 = 7.6
	// (4.4 + 5.5 + 6.6) 	 = 16.5
	// (7.7 + 8.8 + 9.9) + 1 = 27.4
	expected := []float64{7.6, 16.5, 27.4}
	assertEqual(t, vals, expected)
}

func assertEqual(t *testing.T, x, y []float64) {
	assert.Equal(t, len(x), len(y))
	for i := range x {
		assert.Equal(t, x[i], y[i])
	}
}

func TestApplyDF(t *testing.T) {
	df := makeDF()

	{
		colsums := df.Apply(sum, true, nil)
		expected := []float64{6.6, 16.5, 26.4}
		assertEqual(t, colsums, expected)

	}
	{
		rowsums := df.Apply(sum, false, nil)
		expected := []float64{13.2, 16.5, 19.8}
		assertEqual(t, rowsums, expected)
	}
}
