package util

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

/*
func qt(alpha float64, df int) {
	norm := dist.UnitNormal
	X := stng.gaussian.StdGaussian() / math.Sqrt(stng.chisquared.ChiSquared(freedom)/float64(freedom))
	return X
}

func confidenceInterval(theta, sigma2, alpha float64, df int) []float64 {
	t := qt(alpha, df)

	low := theta - sigma2*t
	hi := theta + sigma2*t

	return []float64{low, hi}
}

*/

func qt(alpha float64, df int) float64 {
	return 1.0
}

func Sum(x []float64) float64 {
	y := 0.0
	for _, z := range x {
		y += z
	}
	return y
}

func Mult(x []float64) float64 {
	y := 1.0
	for _, z := range x {
		y *= z
	}
	return y
}

func Mean(x []float64) float64 {
	return Sum(x) / float64(len(x))
}

func Variance(x []float64) float64 {
	n := float64(len(x))
	if n == 1 {
		return 0
	} else if n < 2 {
		n = 2
	}

	m := Mean(x)

	ss := 0.0
	for _, v := range x {
		ss += math.Pow(v-m, 2.0)
	}

	return ss / (n - 1)
}

func Sd(x []float64) float64 {
	return math.Sqrt(Variance(x))
}

func Rep(val float64, times int) []float64 {
	out := make([]float64, times)
	for i := 0; i < times; i++ {
		out[i] = val
	}
	return out
}

func RemoveCol(df *mat64.Dense, col int) *mat64.Dense {
	r, c := df.Dims()
	if col > c || col < 0 {
		panic("Column Index not supported")
	}

	cop := mat64.NewDense(r, c-1, nil)

	m := 0

	for i := 0; i < c; i++ {
		if i != col {
			cop.SetCol(m, df.Col(nil, i))
			m++
		}
	}

	return cop
}

func RemoveRow(df *mat64.Dense, row int) *mat64.Dense {
	r, c := df.Dims()
	if row > r || row < 0 {
		panic("Row Index not supported")
	}

	cop := mat64.NewDense(r-1, c, nil)

	m := 0

	for i := 0; i < r; i++ {
		if i != row {
			cop.SetRow(m, df.Row(nil, i))
			m++
		}
	}

	return cop
}
