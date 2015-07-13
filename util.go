package glasso

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

func (o *OLS) sdResiduals() float64 {
	ybar := mean(o.response)

	ss := 0.0
	for i := 0; i < o.n; i++ {
		ss += math.Pow(ybar-o.fitted[i], 2.0)
	}

	return math.Sqrt(ss / float64(o.n-2))
}

/*
func (o *OLS) ci_ybar(alpha, val float64) {
	ybar := mean(o.response)

	s := o.sdResiduals()

	t := qt(alpha, o.df)

}
*/

func sum(x []float64) float64 {
	y := 0.0
	for _, z := range x {
		y += z
	}
	return y
}

func mult(x []float64) float64 {
	y := 1.0
	for _, z := range x {
		y *= z
	}
	return y
}

func mean(x []float64) float64 {
	return sum(x) / float64(len(x))
}

func variance(x []float64) float64 {
	n := float64(len(x))
	if n == 1 {
		return 0
	} else if n < 2 {
		n = 2
	}

	m := mean(x)

	ss := 0.0
	for _, v := range x {
		ss += math.Pow(v-m, 2.0)
	}

	return ss / (n - 1)
}

func sd(x []float64) float64 {
	return math.Sqrt(variance(x))
}

func rep(val float64, times int) []float64 {
	out := make([]float64, times)
	for i := 0; i < times; i++ {
		out[i] = val
	}
	return out
}

func removeCol(df *mat64.Dense, col int) *mat64.Dense {
	r, c := df.Dims()
	if col > c || col < 0 {
		panic("Column Index not supported")
	}

	cop := mat64.NewDense(r, c-1, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if j == col {
				continue
			}
			cop.Set(i, j, df.At(i, j))
		}
	}
	return cop
}

func removeRow(df *mat64.Dense, row int) *mat64.Dense {
	r, c := df.Dims()
	if row > r || row < 0 {
		panic("Row Index not supported")
	}

	cop := mat64.NewDense(r, c-1, nil)
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			if j == row {
				continue
			}
			cop.Set(j, i, df.At(j, i))
		}
	}
	return cop
}
