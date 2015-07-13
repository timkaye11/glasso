package glasso

import (
	"math"

	"github.com/gonum/matrix/mat64"

	. "github.com/timkaye11/glasso/util"
)

// Cooks Distance: do this concurrently
//
// D_{i} = \frac{r_{i}^2}{p * MSE} * \frac{h_{ii}}{(1 - h_{ii})^2}
//
func (o *OLS) CooksDistance() []float64 {

	h := o.LeveragePoints()
	mse := o.MeanSquaredError()

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
func (o *OLS) LeveragePoints() []float64 {
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
func (o *OLS) StudentizedResiduals() []float64 {
	t := make([]float64, o.n)
	sigma := Sd(o.residuals)
	h := o.LeveragePoints()

	for i := 0; i < o.n; i++ {
		t[i] = o.residuals[i] / (sigma * math.Sqrt(1-h[i]))
	}

	return t
}

// PRESS (Predicted Error Sum of Squares)
// This is used as estimate the model's ability to predict new observations
// R^2_prediction = 1 - (PRESS / TSS)
func (o *OLS) PRESS() []float64 {
	press := make([]float64, o.n)
	h_diag := o.LeveragePoints()

	for i := 0; i < o.n; i++ {
		press[i] = o.residuals[i] / (1.0 - h_diag[i])
	}

	return press
}

// Calculates the variance-covariance matrix of the regression coefficients
// defined as (XtX)-1
// Using QR decomposition: X = QR
// ((QR)tQR)-1 ---> (RtQtQR)-1 ---> (RtR)-1 ---> R-1Rt-1
//
func (o *OLS) VarianceCovarianceMatrix() *mat64.Dense {
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

// A simple approach to identify collinearity among explanatory variables is the use of variance inflation factors (VIF).
// VIF calculations are straightforward and easily comprehensible; the higher the value, the higher the collinearity
// A VIF for a single explanatory variable is obtained using the r-squared value of the regression of that
// variable against all other explanatory variables:
//
// VIF_{j} = \frac{1}{1 - R_{j}^2}
//
func (o *OLS) VarianceInflationFactors() []float64 {
	// save a copy of the data
	orig := mat64.DenseCopyOf(o.x.data)

	vifs := make([]float64, o.p)

	for idx := 0; idx < o.p; idx++ {
		x := o.x.data

		col := x.Col(nil, idx)

		x.SetCol(idx, Rep(0.0, o.n))

		err := o.Train(col)
		if err != nil {
			panic("Error Occured calculating VIF")
		}

		vifs[idx] = 1.0 / (1.0 - o.RSquared())
	}

	// reset the data
	o.x.data = orig

	return vifs
}
