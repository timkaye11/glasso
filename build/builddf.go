package build

import (
	"errors"
	"log"
	"sync"

	"github.com/gonum/matrix/mat64"
)

var (
	DimensionError = errors.New("wrong dimension")
	LabelError     = errors.New("missing labels for columns")
)

type DataFrame struct {
	X      *mat64.Dense
	n, c   int      // memoize # rows, columns
	labels []string // optional column names
}

func Mat64ToDF(mat *mat64.Dense) *DataFrame {
	rows, cols := mat.Dims()
	return &DataFrame{
		X: mat,
		n: rows,
		c: cols,
	}
}

func NewDataFrame(data [][]float64, labels ...[]string) *DataFrame {
	rows := len(data)
	cols := len(data[0])
	x := make([]float64, 0, cols*rows)

	for _, d := range data {
		x = append(x, d...)
	}

	df := &DataFrame{
		X: mat64.NewDense(rows, cols, x),
		c: cols,
		n: rows,
	}

	if len(labels) > 0 {
		df.labels = labels[0]
	}

	return df
}

func (d *DataFrame) GetRow(i int) []float64 {
	if i > d.n {
		return nil
	}
	return mat64.Row(nil, i, d.X)
}

func (d *DataFrame) GetCol(j int) []float64 {
	if j > d.c {
		return nil
	}
	return mat64.Col(nil, j, d.X)
}

func (d *DataFrame) Rows() int { return d.n }
func (d *DataFrame) Cols() int { return d.c }
func (d *DataFrame) Data() *mat64.Dense {
	return mat64.DenseCopyOf(d.X)
}

func (d *DataFrame) Copy() *DataFrame {
	return &DataFrame{
		X: d.Data(),
		n: d.Rows(),
		c: d.Cols(),
	}
}

// Transform applies a function to the columns of the DataFrame.
// Cols indicates which columns to apply the function for.
// If nil, every column is evaluated.
func (d *DataFrame) Transform(f Evaluator, cols ...int) {
	d.X.Apply(func(_, c int, v float64) float64 {
		if containsInt(c, cols) && cols != nil {
			return f(v)
		}
		return v
	}, d.X)
}

// AppendCol appends a column to the end of the DataFrame.
func (d *DataFrame) AppendCol(col []float64) error {
	if len(col) != d.n {
		return DimensionError
	}

	d.X = mat64.DenseCopyOf(d.X.Grow(0, 1))
	d.n, d.c = d.X.Dims()
	d.X.SetCol(d.c-1, col)

	return nil
}

// AppendCol appends a row to the end of the DataFrame.
func (d *DataFrame) AppendRow(row []float64) error {
	if len(row) != d.c {
		return DimensionError
	}

	d.X = mat64.DenseCopyOf(d.X.Grow(1, 0))
	d.n, d.c = d.X.Dims()
	d.X.SetRow(d.n-1, row)

	return nil
}

// PushCol appends a column to the front of the DataFrame.
func (d *DataFrame) PushCol(col []float64) error {
	if len(col) != d.n {
		return DimensionError
	}

	d.c++
	x := mat64.NewDense(d.n, d.c, nil)
	x.SetCol(0, col)

	for c := 1; c < d.c; c++ {
		x.SetCol(c, d.GetCol(c-1))
	}
	d.X = x

	return nil
}

// PushRow appends a row to the front of the DataFrame.
func (d *DataFrame) PushRow(row []float64) error {
	if len(row) != d.c {
		return DimensionError
	}

	d.n++
	x := mat64.NewDense(d.n, d.c, nil)
	x.SetRow(0, row)

	for r := 1; r < d.n; r++ {
		x.SetRow(r, d.GetRow(r-1))
	}
	d.X = x

	return nil
}

// RemoveCol removes a specified column from the Dataframe.
func (d *DataFrame) RemoveCol(col int) error {
	if col > d.c {
		return DimensionError
	}

	tmp := mat64.NewDense(d.n, d.c-1, nil)
	j := 0
	for i := 0; i < d.c; i++ {
		if i != col {
			tmp.SetCol(j, d.GetCol(i))
			j++
		}
	}
	d.c--
	d.X = tmp
	return nil
}

// RemoveRow removes a specified row from the Dataframe
func (d *DataFrame) RemoveRow(row int) error {
	if row > d.n {
		return DimensionError
	}

	tmp := mat64.NewDense(d.n-1, d.c, nil)
	j := 0
	for i := 0; i < d.n; i++ {
		if i != row {
			tmp.SetRow(j, d.GetRow(i))
			j++
		}
	}
	d.n--
	d.X = tmp
	return nil
}

// Similar to the R equivalent: apply a function across all rows / columns of a DataFrame.
// If margin == true, evaluate column wise, else evaluator rowwise.
// Each aggregation is run in a goroutine.
func (d *DataFrame) Apply(f Aggregator, margin bool, idxs ...int) []float64 {
	if margin {
		return d.ApplyCols(f, idxs)
	}
	return d.ApplyRows(f, idxs)
}

// ApplyCols aggregates values over the columns of the Dataframe.
func (d *DataFrame) ApplyCols(agg Aggregator, cols []int) []float64 {
	if len(cols) > d.c {
		log.Println("cannot apply function to more columns than present")
		return nil
	}

	if len(cols) == 0 {
		cols = seq(0, d.c-1, 1)
	}

	output := make([]float64, len(cols))
	var wg sync.WaitGroup
	for i, col := range cols {
		if col > d.c {
			log.Printf("Column Out of Range: %v > # columns(%v)", col, d.c)
			return nil
		}

		wg.Add(1)
		go func(i, c int) {
			output[i] = agg(d.GetCol(c))
			wg.Done()
		}(i, col)
	}
	wg.Wait()

	return output
}

// ApplyRows aggregates values over the rows of the Dataframe.
func (d *DataFrame) ApplyRows(agg Aggregator, rows []int) []float64 {
	if len(rows) > d.n {
		log.Println("cannot apply function to more rows than present")
		return nil
	}

	if len(rows) == 0 {
		rows = seq(0, d.n-1, 1)
	}

	output := make([]float64, len(rows))
	var wg sync.WaitGroup
	for i, row := range rows {
		if row > d.n {
			log.Printf("Row Out of Range: %v > # rows(%v)", row, d.n)
			return nil
		}
		wg.Add(1)
		go func(i, n int) {
			output[i] = agg(d.GetRow(n))
			wg.Done()
		}(i, row)
	}
	wg.Wait()

	return output
}
