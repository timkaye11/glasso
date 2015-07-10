package glasso

import "math"

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
