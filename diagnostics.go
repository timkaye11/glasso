package glasso

import (
	"math"
	"runtime"

	"github.com/drewlanenga/govector"
	"github.com/gonum/matrix/mat64"
	"github.com/timkaye11/gostat/stat"
)

const (
	NCPU = 4
)

// Cooks Distance: do this concurrently
//
// D_{i} = \frac{r_{i}^2}{p * MSE} * \frac{h_{ii}}{(1 - h_{ii})^2}
//
func (o *OLS) CooksDistance() []float64 {
	runtime.GOMAXPROCS(NCPU)

	h := o.LeveragePoints()
	mse := o.MeanSquaredError()

	c := make(chan int, NCPU)

	dists := make([]float64, o.n)

	for i := 0; i < o.n; i++ {
		go func(idx int) {
			left := math.Pow(o.residuals[i], 2.0) / (float64(o.p) * mse)
			right := h[i] / math.Pow(1-h[i], 2)
			dists[idx] = left * right
			c <- 1
		}(i)
	}

	// drain the channel
	for i := 0; i < NCPU; i++ {
		<-c
	}

	return dists
}

// Leverage Points, the diagonal of the hat matrix
// H = X(X'X)^-1X'  , X = QR,  X' = R'Q'
//   = QR(R'Q'QR)-1 R'Q'
//	 = QR(R'R)-1 R'Q'
//	 = QRR'-1 R-1 R'Q'
//	 = QQ' (the first p cols of Q, where X = n x p)
//
// Leverage points are considered large if they exceed 2p/ n
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
	sigma := sd(o.residuals)
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

		x.SetCol(idx, rep(0.0, o.n))

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

// DFBETAS
//
//
func (o *OLS) DFBETA() []float64 {
	runtime.GOMAXPROCS(NCPU)

	c := make(chan int, NCPU)

	dfs := make([]float64, o.n)

	for i := 0; i < o.n; i++ {
		go func() {
			dfs[i] = 0.0
			c <- 1
		}()
	}

	for i := 0; i < NCPU; i++ {
		<-c
	}

	return dfs
}

// DFFITS - influence of single fitted value
// = \hat{Y_{i}} - \hat{Y_{i(i)}} / \sqrt{MSE_{(i)} h_{ii}}
// influential if larger than 1
//
func (o *OLS) DFFITS() []float64 {
	orig := o.x.data
	fitted := o.fitted
	leverage := o.LeveragePoints()

	runtime.GOMAXPROCS(NCPU)

	c := make(chan int, NCPU)

	dffits := make([]float64, o.n)

	o.n--

	for i := 0; i < len(dffits); i++ {
		go func(i int) {
			o.x.data = removeRow(o.x.data, i)

			err := o.Train(o.residuals)
			if err != nil {
				panic(err)
			}

			loo_fitted := o.fitted

			dffits[i] = fitted[i] - loo_fitted[i]
			dffits[i] /= math.Sqrt(o.MeanSquaredError() * leverage[i])

			c <- 1
		}(i)
	}

	for i := 0; i < NCPU; i++ {
		<-c
	}

	o.x.data = orig
	o.n++

	return dffits
}

// var(\beta) = \sigma * (Xt X_)-1
// 			  = \sigma * ((QR)t QR) -1
// 			  = \sigma * (RtQt QR) -1
//			  = \sigma * (Rt R) -1
//
func (o *OLS) VarBeta() []float64 {
	// use the unbiased estimator for sigma^2
	sig := o.ResidualSumofSquares() / float64(o.n-o.p-1)

	var_cov := o.VarianceCovarianceMatrix()

	var_cov_diag := make([]float64, len(o.betas))

	for i := 0; i < len(o.betas); i++ {
		var_cov_diag[i] = var_cov.At(i, i)
	}

	varbetas := make([]float64, len(o.betas))

	for i, diag := range var_cov_diag {
		varbetas[i] = math.Sqrt(sig * diag)
	}

	return varbetas
}

// To test a hypothesis that a coefficient B_j = 0, we form
// the standardized coefficient or Z-score
// Z_j = \frac{B_j}{\sigma * sqrt{v_{j}}}
// where v_j is the jth diagonal element from the variance covariance matrix: (XtX)-1
func (o *OLS) Z_Scores() []float64 {
	z := make([]float64, len(o.betas))
	v := make([]float64, len(o.betas))

	sigma := math.Sqrt(o.ResidualSumofSquares() / float64(o.n-o.p-1))

	var_cov := o.VarianceCovarianceMatrix()
	for i := 0; i < len(z); i++ {
		v[i] = var_cov.At(i, i)
	}

	for i, beta_j := range o.betas {
		z[i] = beta_j / (sigma * math.Sqrt(v[i]))
	}

	return z
}

// The F statistic measures the change in residual sum-of-squares per
// additional parameter in the bigger model, and it is normalized by an estimate of sigma2
//
//
func (o *OLS) F_Test(toRemove ...int) (fval, pval float64) {
	if len(toRemove) > (o.p - 1) {
		panic("Too many columns to remove")
	}

	data := o.x.data
	for _, col := range toRemove {
		data = removeCol(data, col)
	}

	cols, rows := data.Dims()
	df := &DataFrame{data: data, cols: cols, rows: rows}
	ols := NewOLS(df)

	err := ols.Train(o.response)
	if err != nil {
		panic(err)
	}

	d1 := float64(o.p - ols.p)
	d2 := float64(o.n - o.p)

	f := (ols.ResidualSumofSquares() - o.ResidualSumofSquares()) / d1
	f /= o.ResidualSumofSquares() / d2

	Fdist := stat.F_CDF(d1, d2)
	p := 1 - Fdist(f)

	return f, p
}

// Durbin Watson Test for Autocorrelatoin of the Residuals
// d = \sum_i=2 ^ n (e_i  - e_i-1)^2 / \sum_i=1^n e_i^2
//
// Does not calculate the p-value
func (o *OLS) DW_Test() float64 {
	e, err := govector.AsVector(o.residuals)
	if err != nil {
		panic(err)
	}

	square := func(x float64) float64 { return math.Pow(x, 2) }

	d := e.Diff().Apply(square).Sum()
	d /= e.Apply(square).Sum()

	return d
}
