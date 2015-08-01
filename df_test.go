package glasso

import (
	"testing"

	"github.com/bmizerany/assert"
)

// make sure everything is constructed OK
func TestMakeDF(t *testing.T) {
	// make dataframe
	// fill by column
	//
	// 1.1 	4.4   7.7
	// 2.2 ...
	data := []float64{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9}
	labels := []string{"a", "", "c"}

	df, err := DF(data, labels)
	assert.Equal(t, err, nil)
	assert.Equal(t, df.rows, 3)
	assert.Equal(t, df.cols, 3)

	t.Logf("First column: %v", df.data.Col(nil, 0))
	t.Logf("Second column: %v", df.data.Col(nil, 1))
}

func TestNewDF(t *testing.T) {
	data := [][]float64{
		{1.0, 2.2, 4.4},
		{4.4, 2.2, 1.0},
		{3.3, 2.1, 5.6},
	}

	df := NewDF(data)
	t.Logf("First column: %v", df.data.Col(nil, 0))
	t.Logf("Second column: %v", df.data.Col(nil, 1))
}

func TestAppend(t *testing.T) {
	data := [][]float64{
		{1.0, 2.2, 4.4},
		{4.4, 2.2, 1.0},
		{3.3, 2.1, 5.6},
	}

	df := NewDF(data)

	df.AppendCol([]float64{4.7, 8.8, 9.2})
	assert.Equal(t, df.data.Col(nil, 3), []float64{4.7, 8.8, 9.2})

	df.AppendRow([]float64{3.1, 4.5, 6.5, 3.3})
	assert.Equal(t, df.data.Row(nil, 3), []float64{3.1, 4.5, 6.5, 3.3})
}

func TestPush(t *testing.T) {
	data := [][]float64{
		{1.0, 2.2, 4.4},
		{4.4, 2.2, 1.0},
		{3.3, 2.1, 5.6},
	}

	df := NewDF(data)

	df.PushCol([]float64{4.7, 8.8, 9.2})
	assert.Equal(t, df.data.Col(nil, 0), []float64{4.7, 8.8, 9.2})

	df.PushRow([]float64{3.1, 4.5, 6.5, 3.3})
	assert.Equal(t, df.data.Row(nil, 0), []float64{3.1, 4.5, 6.5, 3.3})
}

/*
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
	df.Transform(add1, 0, 2)
	newA := df.data.Col(nil, 0)
	newB := df.data.Col(nil, 1)
	newC := df.data.Col(nil, 2)

	assert.Equal(t, sum(newA), 15.0)
	assert.Equal(t, sum(newB), 15.0) // shouldn't change
	assert.Equal(t, sum(newC), 21.0)
}
*/

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
