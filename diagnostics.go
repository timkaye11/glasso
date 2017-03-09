package glasso

import (
	"math"
	"sync"

	"github.com/drewlanenga/govector"
	"github.com/ematvey/gostat"
	"github.com/gonum/matrix/mat64"
)

// CooksDistance concurrently calculates the cooks distances for the model
//
// D_{i} = \frac{r_{i}^2}{p * MSE} * \frac{h_{ii}}{(1 - h_{ii})^2}
func CooksDistance(m Summary) []float64 {
	h := LeveragePoints(m)
	residuals := m.Residuals()
	distances := make([]float64, m.Data().Rows())
	p := float64(m.Data().Cols())
	mse := MseAdjusted(m)
	wg := sync.WaitGroup{}
	mu := sync.Mutex{}
	cooks := func(i int) {
		left := math.Pow(residuals[i], 2.0) / (p * mse)
		right := h[i] / math.Pow(1-h[i], 2.0)

		mu.Lock()
		distances[i] = left * right
		mu.Unlock()
	}

	for i := 0; i < m.Data().Rows(); i++ {
		wg.Add(1)
		go func(j int) {
			cooks(j)
			wg.Done()
		}(i)
	}
	wg.Wait()

	return distances
}

func Mse(m Summary) float64 {
	return m.SumOfSquares() / float64(m.Data().Rows())
}

func MseAdjusted(m Summary) float64 {
	return m.SumOfSquares() / float64(m.Data().Rows()-m.Data().Cols())
}

// LeveragePoints returns the diagonal of the hat matrix
// H = X(X'X)^-1X'  , X = QR,  X' = R'Q'
//   = QR(R'Q'QR)-1 R'Q'
//	 = QR(R'R)-1 R'Q'
//	 = QRR'-1 R-1 R'Q'
//	 = QQ' (the first p cols of Q, where X = n x p)
//
// Leverage points are considered large if they exceed 2p/ n

func LeveragePoints(m Summary) []float64 {
	q := &mat64.Dense{}
	h := &mat64.Dense{}
	qr := &mat64.QR{}
	qr.Factorize(m.Data().X)
	q.QFromQR(qr)

	// get the first p columns of Q
	n, p := m.Data().X.Dims()
	q = q.View(0, 0, n, p).(*mat64.Dense)
	h.Mul(q, q.T())

	n = m.Data().Rows()
	diagonals := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := -0; j < n; j++ {
			if i == j {
				diagonals[i] = h.At(i, j)
			}
		}
	}

	return diagonals
}

// StudentizedResiduals returns the studentized residuals,
// found by dividing residual by estimate of std deviation
//
// t_{i} = \frac{\hat{\epsilon}}{\sigma * \sqrt{1 - h_{ii}}}
// \hat{\epsilon} =
func StudentizedResiduals(m Summary) []float64 {
	n, c := m.Data().Rows(), m.Data().Cols()
	sigma := math.Sqrt(m.SumOfSquares() / float64(n-c))
	h := LeveragePoints(m)
	t := make([]float64, n)
	residuals := m.Residuals()
	for i := 0; i < m.Data().Rows(); i++ {
		t[i] = residuals[i] / (sigma * math.Sqrt(1-h[i]))
	}

	return t
}

// Press returns the Predicted Error Sum of Squares (Press) of the model.
// This is used as estimate the model's ability to predict new observations
// R^2_prediction = 1 - (PRESS / TSS)
func Press(m Summary) []float64 {
	press := make([]float64, m.Data().Rows())
	hdiag := LeveragePoints(m)
	residuals := m.Residuals()
	for i := 0; i < m.Data().Rows(); i++ {
		press[i] = residuals[i] / (1.0 - hdiag[i])
	}
	return press
}

// VarCov calculates the variance-covariance matrix of the regression coefficients
// defined as sigma*(XtX)-1
// Using QR decomposition: X = QR
// ((QR)tQR)-1 ---> (RtQtQR)-1 ---> (RtR)-1 ---> R-1Rt-1 --> sigma*R-1Rt-1
func VarCov(m Summary) (*DataFrame, error) {
	r := &mat64.Dense{}
	qr := &mat64.QR{}
	qr.Factorize(m.Data().X)
	r.RFromQR(qr)

	var rinv mat64.Dense
	var rtinv mat64.Dense
	_, columns := r.Dims()
	rCopy := mat64.NewDense(columns, columns, nil)
	rCopy.Copy(r)

	rt := mat64.NewDense(columns, columns, nil)
	rt.Copy(rCopy.T())
	if err := rtinv.Inverse(rt); err != nil {
		return nil, err
	}

	if err := rinv.Inverse(rCopy); err != nil {
		return nil, err
	}

	cols := m.Data().Cols()
	varCov := mat64.NewDense(cols, cols, nil)
	varCov.Mul(&rinv, &rtinv)
	mse := MseAdjusted(m)
	varCov.Apply(func(_, _ int, v float64) float64 { return v * mse }, varCov)
	return Mat64ToDF(varCov), nil
}

// VarianceInflationFactors calculates the VIFs for the model.
// VIF calculations are straightforward and easily comprehensible; the higher the value, the higher the collinearity
// A VIF for a single explanatory variable is obtained using the r-squared value of the regression of that
// variable against all other explanatory variables:
//
// VIF_{j} = \frac{1}{1 - R_{j}^2}
//
func VarianceInflationFactors(m Summary, trainer Trainer) ([]float64, error) {
	vifs := make([]float64, m.Data().Cols())
	for i := 0; i < m.Data().Cols(); i++ {
		data := m.Data().Copy()
		if err := data.RemoveCol(i); err != nil {
			return nil, err
		}

		_, summary, err := trainer.Train(data, m.Response())
		if err != nil {
			return nil, err
		}
		vifs[i] = 1.0 / (1.0 - summary.SumOfSquares())
	}
	return vifs, nil
}

// VarBeta returns the variance of the coefficients for the model.
// var(\beta) = \sigma * (Xt X_)-1
// 			  = \sigma * ((QR)t QR) -1
// 			  = \sigma * (RtQt QR) -1
//			  = \sigma * (Rt R) -1
//
func VarBeta(m Summary) []float64 {
	// use the unbiased estimator for sigma^2
	sig := m.SumOfSquares() / float64(m.Data().Rows()-m.Data().Cols()-1)
	vc, err := VarCov(m)
	if err != nil {
		return nil
	}

	vcdiag := make([]float64, vc.Rows())
	for i := 0; i < vc.Rows(); i++ {
		vcdiag[i] = vc.X.At(i, i)
	}

	varbetas := make([]float64, len(vcdiag))
	for i := range vcdiag {
		varbetas[i] = math.Sqrt(sig * vcdiag[i])
	}

	return varbetas
}

// Z Scores returns the Z score for each coefficient in the model.
// To test a hypothesis that a coefficient B_j = 0, we form
// the standardized coefficient or Z-score
// Z_j = \frac{B_j}{\sigma * sqrt{v_{j}}}
// where v_j is the jth diagonal element from the variance covariance matrix: (XtX)-1
func ZScores(m Summary) []float64 {
	z := make([]float64, m.Data().Cols())
	v := make([]float64, m.Data().Cols())
	sigma := math.Sqrt(m.SumOfSquares() / float64(m.Data().Rows()-m.Data().Cols()-1))

	vc, err := VarCov(m)
	if err != nil {
		return nil
	}
	for i := 0; i < len(v); i++ {
		v[i] = vc.X.At(i, i)
	}

	for i, beta := range m.Coefficients() {
		z[i] = beta / (sigma * math.Sqrt(v[i]))
	}

	return z
}

// The F statistic measures the change in residual sum-of-squares per
// additional parameter in the bigger model, and it is normalized by an estimate of sigma2
func FTest(m Summary, trainer Trainer, toRemove []int) (fval, pval float64, err error) {
	tmp := m.Data()
	n, c := m.Data().Rows(), m.Data().Cols()
	oldSS := m.SumOfSquares()

	for _, col := range toRemove {
		err = tmp.RemoveCol(col)
		if err != nil {
			return
		}
	}
	_, summary, err := trainer.Train(tmp, m.Response())
	if err != nil {
		return
	}

	tmpN, tmpC := tmp.Data().Dims()
	d1 := float64(c - tmpC)
	d2 := float64(n - tmpN)
	fval = (summary.SumOfSquares() / oldSS) / d1
	fval /= oldSS / d2
	Fdist := stat.F_CDF(d1, d2)
	pval = 1 - Fdist(fval)
	return
}

// Durbin Watson Test for Autocorrelatoin of the Residuals
// d = \sum_i=2 ^ n (e_i  - e_i-1)^2 / \sum_i=1^n e_i^2
//
// Does not calculate the p-value
func DW(m Summary) float64 {
	e, err := govector.AsVector(m.Residuals())
	if err != nil {
		return 0.0
	}

	square := func(x float64) float64 { return math.Pow(x, 2) }
	d := e.Diff().Apply(square).Sum()
	d /= e.Apply(square).Sum()
	return d
}

// n log(SSE(M) + 2(p(M)+1)
// AIC = n log(SSE/n) + 2(p + 1).
func AIC(m Summary) float64 {
	n, p := float64(m.Data().Rows()), float64(m.Data().Cols())
	return n*math.Log(m.SumOfSquares()/n) + (2.0*p + 1)
}

// n log(SSE(M) + (p(M)+1)log(n)
// BIC = n log(SSE/n) + log(n)(p + 1).
func BIC(m Summary) float64 {
	n, p := float64(m.Data().Rows()), float64(m.Data().Cols())
	return n*math.Log(m.SumOfSquares()/n) + (math.Log(n)*p + 1)
}
