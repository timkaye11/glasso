package glasso

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

func qt(alpha float64, df int) float64 {
	return 1.0
}

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

func prod(x, y []float64) []float64 {
	p := make([]float64, len(x))

	for i, _ := range x {
		p[i] = x[i] * y[i]
	}
	return p
}

func cor(x, y []float64) float64 {
	n := float64(len(x))

	xy := prod(x, y)

	sx := sd(x)
	sy := sd(y)

	mx := mean(x)
	my := mean(y)

	return (sum(xy) - n*mx*my) / ((n - 1) * sx * sy)
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

	m := 0

	for i := 0; i < c; i++ {
		if i != col {
			cop.SetCol(m, df.Col(nil, i))
			m++
		}
	}

	return cop
}

func removeRow(df *mat64.Dense, row int) *mat64.Dense {
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

func standardize(x []float64) []float64 {
	m := mean(x)
	dev := sd(x)

	cp := make([]float64, len(x))
	copy(x, cp)

	for i := 0; i < len(cp); i++ {
		cp[i] -= m
		cp[i] /= dev
	}
	return cp
}

func normalize(x []float64) []float64 {
	s := sum(x)

	cp := make([]float64, len(x))
	copy(x, cp)

	for i := 0; i < len(cp); i++ {
		cp[i] /= s
	}
	return cp
}

func subtractMean(x []float64) []float64 {
	m := mean(x)

	cp := make([]float64, len(x))
	copy(x, cp)

	for i := 0; i < len(cp); i++ {
		cp[i] -= m
	}
	return cp
}

func max(x []float64) float64 {
	m := x[0]
	for i, _ := range x {
		if x[i] > m {
			m = x[i]
		}
	}
	return m
}

func sign(x float64) float64 {
	if x > 0.0 {
		return 1.0
	}
	return -1.0
}

func addSlice(x []float64, val float64) []float64 {
	cp := make([]float64, len(x))
	copy(x, cp)

	for i, _ := range cp {
		cp[i] += val
	}
	return cp
}

func subSlice(x []float64, val float64) []float64 {
	cp := make([]float64, len(x))
	copy(x, cp)

	for i, _ := range cp {
		cp[i] -= val
	}
	return cp
}

func multSlice(x []float64, val float64) []float64 {
	cp := make([]float64, len(x))
	copy(x, cp)

	for i, _ := range cp {
		cp[i] *= val
	}
	return cp
}

func diff(x, y []float64) []float64 {
	cp := make([]float64, len(x))
	copy(x, cp)

	for i, val := range y {
		cp[i] -= val
	}
	return cp
}
