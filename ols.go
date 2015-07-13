package glasso

import (
	"fmt"
	"math"
	"strings"

	"github.com/drewlanenga/govector"
	"github.com/gonum/matrix/mat64"
	. "github.com/timkaye11/glasso/util"
)

// Regression Output
type Model interface {
	// Build a linear model. Additional arguments specified in the constructor
	Train(response []float64) error

	// Predict values based on linear model
	Predict(x []float64) []float64

	// To Output diagnostics
	String() string

	Residuals() []float64
}

// Ordinary Least Squares regression using QR factorization
// Y = β_0 + Σ x_j β_j
// β = (XtX)^-1 Xt y
// X = Q*R
// XtX = (QR)t(QR) = RtQtQR = RtR
// Rβ = Qt y
type OLS struct {
	hat       *mat64.Dense
	x         *DataFrame
	n, p, df  int
	xbar      float64
	betas     []float64
	residuals []float64
	fitted    []float64
	response  []float64
}

func NewOLS(data *DataFrame) *OLS {
	rows := data.rows
	cols := data.cols
	return &OLS{
		x:         data,
		betas:     make([]float64, cols),
		residuals: make([]float64, rows),
		fitted:    make([]float64, rows),
		df:        rows - 1,
		n:         rows,
		p:         cols,
	}
}

func (o *OLS) Train(yvector []float64) error {
	// sanity check
	if len(yvector) != o.n {
		return DimensionError
	}

	o.response = yvector
	y := mat64.NewDense(len(yvector), 1, yvector)
	x := o.x.data

	// it's easier to do things with X = QR
	qrFactor := mat64.QR(x)
	Q := qrFactor.Q()
	R := qrFactor.R()

	// calculate yhat (fitted values)
	// y_hat = Q*Qt*y
	qqt := &mat64.Dense{}
	qqt.MulTrans(Q, false, Q, true)

	yhat := &mat64.Dense{}
	yhat.Mul(qqt, y)

	o.fitted = yhat.Col(nil, 0)

	// gotta find the betas (coefficients)
	// Rβ = Qt y
	Qty := &mat64.Dense{}
	Qty.MulTrans(Q, true, y, false)

	beta, err := mat64.Solve(R, Qty)
	if err != nil {
		return err
	}

	o.betas = beta.Col(nil, 0)

	// all good
	return nil
}

//func (o *OLS) prediction

func (o *OLS) Predict(x []float64) []float64 {
	return []float64{}
}

func (o *OLS) String() string {
	q, _ := govector.AsVector(o.residuals)
	points := []float64{0.0, 0.25, 0.5, 0.75, 1.0}
	p, _ := govector.AsVector(points)
	quantiles := q.Quantiles(p)
	betas := []float64{1.0, 2.0}

	return fmt.Sprintf(`
		\n Formula: response ~ %v
		\n Residuals: 
		\n\t Min \t 25 \t 50 \t 75 \t Max: 
		\n\t %v  \t %v \t %v \t %v \t %v 
		\n
		\n Coefficients: %v
		\n 
		\n RSS: %v 
		\n Adjusted R-Squared: %v
		\n R-squared: %v`,
		strings.Join(o.x.labels, ","),
		quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4],
		betas, o.ResidualSumofSquares(), o.AdjustedRSquared(), o.RSquared())
}

func (o *OLS) Residuals() []float64 {
	return o.residuals
}

func (o *OLS) TotalSumofSquares() float64 {
	// no chance this could error
	y, _ := govector.AsVector(o.response)
	ybar := Mean(o.response)

	squaredDiff := func(x float64) float64 {
		return math.Pow(x-ybar, 2.0)
	}

	return y.Apply(squaredDiff).Sum()
}

func (o *OLS) ResidualSumofSquares() float64 {
	res, _ := govector.AsVector(o.residuals)

	rss, err := govector.DotProduct(res, res)
	if err != nil {
		panic(err)
	}
	return rss
}

func (o *OLS) RSquared() float64 {
	return float64(1 - o.ResidualSumofSquares()/o.TotalSumofSquares())
}

func (o *OLS) MeanSquaredError() float64 {
	n := float64(o.x.rows)
	return o.ResidualSumofSquares() / (n - 2.0)
}

// the adjusted r-squared adjusts the r-squared value to reflect the importance of predictor variables
// https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
func (o *OLS) AdjustedRSquared() float64 {
	dfe := float64(o.x.rows)
	dft := dfe - float64(o.x.cols)
	return 1 - (o.ResidualSumofSquares()*(dfe-1.0))/(o.TotalSumofSquares()*dft)
}

func (o *OLS) sdResiduals() float64 {
	ybar := Mean(o.response)

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
