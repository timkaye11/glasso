package glasso

import (
	"testing"

	"github.com/bmizerany/assert"
	"github.com/gonum/matrix/mat64"
)

func TestRemoveRow(t *testing.T) {
	data := mat64.NewDense(2, 4, append(Rep(1.0, 4), Rep(2.0, 4)...))
	// 1,1,1,1
	// 2,2,2,2
	for _, x := range data.Row(nil, 0) {
		assert.Equal(t, x, 1.0)
	}
	r, c := data.Dims()
	assert.Equal(t, r, 2)
	assert.Equal(t, c, 4)

	data = RemoveRow(data, 0)
	for _, x := range data.Row(nil, 0) {
		assert.Equal(t, x, 2.0)
	}

	r, c = data.Dims()
	assert.Equal(t, r, 1)
	assert.Equal(t, c, 4)
}

func TestRemoveCol(t *testing.T) {
	data := mat64.NewDense(4, 2, []float64{1, 2, 1, 2, 1, 2, 1, 2})
	// 1, 2
	// 1, 2
	// 1, 2
	// 1, 2
	r, c := data.Dims()
	assert.Equal(t, r, 4)
	assert.Equal(t, c, 2)

	data = RemoveCol(data, 0)
	for _, x := range data.Col(nil, 0) {
		assert.Equal(t, x, 2.0)
	}

	r, c = data.Dims()
	assert.Equal(t, r, 4)
	assert.Equal(t, c, 1)
}
