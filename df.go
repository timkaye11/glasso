package glasso

import (
	"errors"
	"fmt"
	"github.com/gonum/matrix/mat64"
)

var (
	DimensionError = errors.New("Error caused by wrong dimensionality")
	LabelError     = errors.New("Missing labels for columns")
)

type DataFrame struct {
	data       *mat64.Dense
	cols, rows int
	labels     []string
}

func DfFromMat(mat *mat64.Dense) *DataFrame {
	rows, cols := mat.Dims()
	return &DataFrame{
		data: mat,
		rows: rows,
		cols: cols,
	}
}

func DF(data []float64, labels []string) (*DataFrame, error) {
	cols := len(labels)
	ents := len(data)
	// dimensions gotta be right
	if ents%cols != 0 {
		fmt.Println(cols, len(data))
		return nil, DimensionError
	}

	x := mat64.NewDense(ents/cols, cols, rep(0.0, ents))
	rows := ents / cols
	for i := 0; i < rows; i++ {
		idx := i * cols
		x.SetCol(i, data[idx:idx+cols])
	}

	return &DataFrame{
		data:   x,
		labels: labels,
		cols:   cols,
		rows:   ents / cols,
	}, nil
}

func NewDF(data [][]float64) *DataFrame {
	rows := len(data)
	cols := len(data[0])
	x := make([]float64, 0, cols*rows)

	for _, d := range data {
		x = append(x, d...)
	}

	df := mat64.NewDense(rows, cols, x)

	return &DataFrame{
		data: df,
		cols: cols,
		rows: rows,
	}
}

func (df *DataFrame) Values() []float64 {
	vals := make([]float64, 0, df.rows*df.cols)

	for r := 0; r <= df.rows; r++ {
		vals = append(vals, df.data.Col(nil, r)...)
	}
	return vals
}

func (df *DataFrame) Dim() (int, int) {
	return df.data.Dims()
}

func (df *DataFrame) Transform(f func(x float64) float64, cols ...int) {
	fc := func(f func(x float64) float64, buf []float64) []float64 {
		for i := 0; i < len(buf); i++ {
			buf[i] = f(buf[i])
		}
		return buf
	}

	for _, col := range cols {
		buf := make([]float64, df.rows)
		df.data.Col(buf, col)
		df.data.SetCol(col, fc(f, buf))
	}
	return
}

// Similar to R, if margin is set to true, the function f is
// applied on the columns. Else, apply the function to the rows
func (df *DataFrame) Apply(f func(x []float64) float64, margin bool, idxs ...int) []float64 {
	if margin {
		return df.applyCols(f, idxs)
	}
	return df.applyRows(f, idxs)
}

func (df *DataFrame) applyCols(f func(x []float64) float64, cols []int) []float64 {
	if len(cols) > df.cols {
		panic("wtf")
	}

	output := make([]float64, len(cols))

	for i, col := range cols {
		if col > df.cols {
			panic("wtf")
		}
		x := df.data.Col(nil, col)
		output[i] = f(x)
	}

	return output
}

func (df *DataFrame) applyRows(f func(x []float64) float64, rows []int) []float64 {
	if len(rows) > df.rows {
		panic("wtf")
	}

	output := make([]float64, len(rows))

	for i, row := range rows {
		if row > df.rows {
			panic("wtf")
		}
		x := df.data.Row(nil, row)
		output[i] = f(x)
	}

	return output
}
