package glasso

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// Cooks Distance: do this concurrently
//
// D_{i} = \frac{r_{i}^2}{p * MSE} * \frac{h_{ii}}{(1 - h_{ii})^2}
//
func CooksDistance(o *OLS, bounds ...int) []float64 {

	h := LeveragePoints(o)
	mse := o.meanSquaredError()

	dists := make(chan tuple, o.n)

	for i := 0; i < o.n; i++ {
		go func(idx int) {
			left := math.Pow(o.residuals[i], 2.0) / (float64(o.p) * mse)
			right := h[i] / math.Pow(1-h[i], 2)
			dists <- tuple{left * right, idx}
		}(i)
	}

	// drain the channel
	output := make([]float64, o.n)
	for {
		select {
		case tup, ok := <-dists:
			if ok {
				output[tup.i] = tup.val
			}
		}
	}

	return output
}

type tuple struct {
	val float64
	i   int
}

// Leverage Points, the diagonal of the hat matrix
// H = X(X'X)^-1X'  , X = QR,  X' = R'Q'
//   = QR(R'Q'QR)-1 R'Q'
//	 = QR(R'R)-1 R'Q'
//	 = QRR'-1 R-1 R'Q'
//	 = QQ' (the first p cols of Q, where X = n x p)
//
func LeveragePoints(o *OLS) []float64 {
	x := o.x.data
	qrf := mat64.QR(x)
	q := qrf.Q()

	// need to get first first p columns only
	n, p := q.Dims()
	trans := mat64.NewDense(n, p, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j && i < p {
				trans.Set(i, j, 1.0)
			}
			trans.Set(i, j, 0.0)
		}
	}

	H := &mat64.Dense{}
	H.Mul(q, trans)
	H.MulTrans(H, false, q, true)

	o.hat = H

	// get diagonal elements
	diag := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if j == i {
				diag[i] = H.At(i, j)
			}
		}
	}
	return diag
}

// Gosset (student)  - studentized resids
// found by dividing residual by estimate of std deviation
//
// t_{i} = \frac{\hat{\epsilon}}{\sigma * \sqrt{1 - h_{ii}}}
// \hat{\epsilon} =
func StudentizedResiduals(o *OLS) []float64 {
	t := make([]float64, o.n)
	sigma := sd(o.residuals)
	h := LeveragePoints(o)

	for i := 0; i < o.n; i++ {
		t[i] = o.residuals[i] / (sigma * math.Sqrt(1-h[i]))
	}

	return t
}
