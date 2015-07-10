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

// Calculates the variance-covariance matrix of the regression coefficients
// defined as (XtX)-1
// Using QR decomposition: X = QR
// ((QR)tQR)-1 ---> (RtQtQR)-1 ---> (RtR)-1 ---> R-1Rt-1
//
func (o *OLS) varianceCovarianceMatrix() *mat64.Dense {
	x := o.x.data

	// it's easier to do things with X = QR
	qrFactor := mat64.QR(x)
	R := qrFactor.R()

	Raug := mat64.NewDense(o.p, o.p, nil)
	for i := 0; i < o.p; i++ {
		for j := 0; j < o.p; j++ {
			Raug.Set(i, j, R.At(i, j))
		}
	}

	Rinverse, err := mat64.Inverse(Raug)
	if err != nil {
		panic("R matrix is not invertible")
	}

	varCov := mat64.NewDense(o.p, o.p, nil)
	varCov.MulTrans(Rinverse, false, Rinverse, true)

	return varCov
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
