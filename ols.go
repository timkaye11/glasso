package glasso

import (
	"fmt"
	"math"

	u "github.com/araddon/gou"
	"github.com/drewlanenga/govector"
	"github.com/gonum/matrix/mat64"
	"github.com/timkaye11/gostat/stat"
)

// Regression Output
type Model interface {
	// Build a linear model. Additional arguments specified in the constructor
	Train(response []float64) error
	//Predict(x []float64) []float64
	Residuals() []float64
	Data() *DataFrame
	Coefficients() []float64
	Yhat() []float64
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

func NewOLS(x *DataFrame) *OLS {
	rows := x.rows
	cols := x.cols
	//	cols := x.cols + 1

	//	d := mat64.DenseCopyOf(x.data.Grow(0, 1))
	//	d.SetCol(0, rep(1.0, rows))

	return &OLS{
		x:         x,
		betas:     make([]float64, cols),
		residuals: make([]float64, rows),
		fitted:    make([]float64, rows),
		df:        rows - 1,
		n:         rows,
		p:         cols,
		response:  make([]float64, rows),
	}
}

func (o *OLS) Train(yvector []float64) error {
	// sanity check
	if len(yvector) != o.n {
		return DimensionError
	}

	copy(o.response, yvector)
	y := mat64.NewDense(len(yvector), 1, yvector)

	o.x.PushCol(rep(1.0, o.x.rows))
	x := o.x.data

	// it's easier to do things with X = QR
	qrFactor := mat64.QR(mat64.DenseCopyOf(x))
	Q := qrFactor.Q()

	betas := qrFactor.Solve(mat64.DenseCopyOf(y))
	o.betas = betas.Col(nil, 0)
	if len(o.betas) != o.p {
		u.Warnf("Unexpected dimension error. Betas: %v", o.betas)
	}

	// calculate residuals and fitted vals
	/*
		fitted := &mat64.Dense{}
		fitted.Mul(x, betas)
		o.fitted = fitted.Col(nil, 0)
		y.Sub(y, fitted)
		o.residuals = y.Col(nil, 0)
	*/

	// y_hat = Q Qt y
	// e = y - y_hat
	qqt := &mat64.Dense{}
	qqt.MulTrans(Q, false, Q, true)
	yhat := &mat64.Dense{}
	yhat.Mul(qqt, y)
	o.fitted = yhat.Col(nil, 0)
	y.Sub(y, yhat)
	o.residuals = y.Col(nil, 0)

	return nil
}

//func (o *OLS) prediction

func (o *OLS) Predict(x []float64) float64 {
	return sum(prod(x, o.betas))
}

func (o *OLS) String() string {
	q, _ := govector.AsVector(o.residuals)
	points := []float64{0.0, 0.25, 0.5, 0.75, 1.0}
	p, _ := govector.AsVector(points)
	qnt := q.Quantiles(p)
	f, fp := o.F_Statistic()

	return fmt.Sprintf(`
		Residuals: 
		Min  25  50t 75  Max: 
		%v
		
		Coefficients: 
		%v

		RSS: %v 
		MSE: %v
		Adjusted R-Squared: %v
		R-squared: %v
		F-statistic: %v with P-value: %v`,
		roundAll(qnt),
		roundAll(o.betas),
		round(o.ResidualSumofSquares(), 3),
		round(o.MeanSquaredError(), 3),
		round(o.AdjustedRSquared(), 3),
		round(o.RSquared(), 3),
		round(f, 4), round(fp, 10),
	)
}

// interface methods
func (o *OLS) Data() *DataFrame        { return o.x }
func (o *OLS) Coefficients() []float64 { return o.betas }
func (o *OLS) Residuals() []float64    { return o.residuals }
func (o *OLS) Yhat() []float64         { return o.fitted }

func (o *OLS) TotalSumofSquares() float64 {
	// no chance this could error
	y, _ := govector.AsVector(o.response)
	ybar := mean(o.response)

	squaredDiff := func(x float64) float64 {
		return math.Pow(x-ybar, 2.0)
	}

	return y.Apply(squaredDiff).Sum()
}

func (o *OLS) ResidualSumofSquares() float64 {
	return sum(prod(o.residuals, o.residuals))
}

func (o *OLS) RSquared() float64 {
	return float64(1 - (o.ResidualSumofSquares() / o.TotalSumofSquares()))
}

func (o *OLS) MeanSquaredError() float64 {
	n, p := o.x.data.Dims()
	return o.ResidualSumofSquares() / (float64(n - p))
}

// the adjusted r-squared adjusts the r-squared value to reflect the importance of predictor variables
// https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
func (o *OLS) AdjustedRSquared() float64 {
	dfe := float64(o.x.rows)
	dft := dfe - float64(o.x.cols)
	return 1 - (o.ResidualSumofSquares()*(dfe-1.0))/(o.TotalSumofSquares()*dft)
}

func (o *OLS) sdResiduals() float64 {
	ybar := mean(o.response)

	ss := 0.0
	for i := 0; i < o.n; i++ {
		ss += math.Pow(ybar-o.fitted[i], 2.0)
	}

	return math.Sqrt(ss / float64(o.n-2))
}

func (o *OLS) F_Statistic() (float64, float64) {
	r1 := o.TotalSumofSquares()
	r2 := o.ResidualSumofSquares()
	p1 := float64(1)
	p2 := float64(o.p)
	denom1 := p2 - p1 + 1
	denom2 := float64(o.n) - p2 - 1

	f := (r1 - r2) / denom1
	f /= r2 / denom2

	Fdist := stat.F_CDF(denom1, denom2)
	return f, 1.0 - Fdist(f)
}

func (o *OLS) Confidence_interval(alpha float64) [][2]float64 {
	tdist := stat.StudentsT_PDF(float64(o.df))

	t := tdist(1 - alpha)

	std_err := o.VarBeta()

	cis := make([][2]float64, len(o.betas))

	for i, b := range o.betas {
		v := math.Sqrt(std_err[i])
		cis[i] = [2]float64{b - t*v, b + t*v}
	}

	return cis
}
