package glasso

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

func round(val float64, places int) float64 {
	var round float64
	pow := math.Pow(10, float64(places))
	digit := pow * val
	_, div := math.Modf(digit)
	if div >= 0.5 {
		round = math.Ceil(digit)
	} else {
		round = math.Floor(digit)
	}
	return round / pow
}

func roundAll(x []float64) []float64 {
	f := make([]float64, len(x))
	for i, val := range f {
		f[i] = round(val, 3)
	}
	return f
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

func removeCol(df *mat64.Dense, col int) (*mat64.Dense, error) {
	r, c := df.Dims()
	if col > c || col < 0 {
		return nil, DimensionError
	}

	cp := mat64.NewDense(r, c-1, nil)
	m := 0
	for i := 0; i < c; i++ {
		if i != col {
			cp.SetCol(m, df.Col(nil, i))
			m++
		}
	}

	return cp, nil
}

func removeRow(df *mat64.Dense, row int) (*mat64.Dense, error) {
	r, c := df.Dims()
	if row > r || row < 0 {
		return nil, DimensionError
	}

	cp := mat64.NewDense(r-1, c, nil)
	m := 0
	for i := 0; i < r; i++ {
		if i != row {
			cp.SetRow(m, df.Row(nil, i))
			m++
		}
	}

	return cp, nil
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

	for i := 0; i < len(cp); i++ {
		cp[i] = x[i] / s
	}
	return cp
}

func subtractMean(x []float64) []float64 {
	m := mean(x)
	cp := make([]float64, len(x))

	for i := 0; i < len(cp); i++ {
		cp[i] = x[i] - m
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
	for i := range cp {
		cp[i] = x[i] + val
	}
	return cp
}

func subSlice(x []float64, val float64) []float64 {
	cp := make([]float64, len(x))
	for i := range cp {
		cp[i] = x[i] - val
	}
	return cp
}

func multSlice(x []float64, val float64) []float64 {
	cp := make([]float64, len(x))
	for i := range cp {
		cp[i] = x[i] * val
	}
	return cp
}

func diff(x, y []float64) []float64 {
	if len(x) != len(y) {
		return nil
	}

	cp := make([]float64, len(x))
	for i := range y {
		cp[i] = x[i] - y[i]
	}
	return cp
}

func contains(x interface{}, values []interface{}) bool {
	for _, v := range values {
		if v == x {
			return true
		}
	}
	return false
}

func seq(start, end, by int) []int {
	if start > end || by > (end-start) {
		fmt.Println("fuck")
		return nil
	}

	s := make([]int, (end-start)/by+1)
	for i := range s {
		s[i] = start + i*by
	}
	return s
}
