package build

import (
	"fmt"
	"math"

	"github.com/drewlanenga/govector"
	"github.com/gonum/matrix/mat64"
	"github.com/timkaye11/gostat/stat"
)

// Ordinary Least Squares regression using QR factorization
// Y = β_0 + Σ x_j β_j
// β = (XtX)^-1 Xt y
// X = Q*R
// XtX = (QR)t(QR) = RtQtQR = RtR
// Rβ = Qt y
type OLS struct {
	hat       *mat64.Dense
	dataframe *DataFrame
	n, p, df  int
	xbar      float64
	betas     []float64
	residuals []float64
	fitted    []float64
	response  []float64
}

func (*OLS) Generator() Generator {
	return NewOLS
}

func NewOLS(x *DataFrame) Model {
	rows, cols := x.Rows(), x.Cols()
	//	cols := x.cols + 1
	//	d := mat64.DenseCopyOf(x.data.Grow(0, 1))
	//	d.SetCol(0, rep(1.0, rows))
	return &OLS{
		dataframe: x,
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
	x := o.dataframe.Copy()
	x.PushCol(rep(1., x.Rows()))
	// remove?
	o.dataframe = x

	// it's easier to do things with X = QR
	betas := &mat64.Dense{}
	qr := &mat64.QR{}
	data := x.Data()
	qr.Factorize(data)
	if err := betas.SolveQR(qr, false, y); err != nil {
		return err
	}
	// first one is intercept
	o.betas = mat64.Col(nil, 0, betas)

	fitted := &mat64.Dense{}
	fitted.Mul(x.Data(), betas)
	o.fitted = mat64.Col(nil, 0, fitted)

	residuals := &mat64.Dense{}
	residuals.Sub(y, fitted)
	o.residuals = mat64.Col(nil, 0, residuals)

	// fmt.Printf("betas=%v\n", o.betas)
	// Q := &mat64.Dense{}
	// Q.QFromQR(qr)
	// fmt.Printf("q=%+v\nn\n\n\n\n", Q)
	// qqt := &mat64.Dense{}
	// qqt.Mul(Q, Q.T())
	// fmt.Printf("qqt=%+v\n", qqt)
	// yhat := &mat64.Dense{}
	// yhat.Mul(qqt, y)
	// o.fitted = mat64.Col(nil, 0, yhat)
	// fmt.Printf("fitted=%v\n", o.fitted)
	// y.Sub(y, yhat)
	// o.residuals = mat64.Col(nil, 0, y)
	return nil
}

//func (o *OLS) prediction
func (o *OLS) Predict(x []float64) float64 {
	return o.betas[0] + sum(prod(x, o.betas[1:]))
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
func (o *OLS) Data() *DataFrame        { return o.dataframe }
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

func (o *OLS) SumOfSquares() float64 {
	fmt.Printf("\n\nRESIDUALS=%v\n\n", o.residuals)
	return o.ResidualSumofSquares()
}

func (o *OLS) ResidualSumofSquares() float64 {

	return sum(prod(o.residuals, o.residuals))
}

func (o *OLS) RSquared() float64 {
	return float64(1 - (o.ResidualSumofSquares() / o.TotalSumofSquares()))
}

func (o *OLS) MeanSquaredError() float64 {
	return o.ResidualSumofSquares() / (float64(o.dataframe.Rows() - o.dataframe.Cols()))
}

// the adjusted r-squared adjusts the r-squared value to reflect the importance of predictor variables
// https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
func (o *OLS) AdjustedRSquared() float64 {
	dfe := float64(o.dataframe.Rows())
	dft := dfe - float64(o.dataframe.Cols())
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

func (o *OLS) Response() []float64 {
	return o.response
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
	std_err := VarBeta(o)
	cis := make([][2]float64, len(o.betas))
	for i, b := range o.betas {
		v := math.Sqrt(std_err[i])
		cis[i] = [2]float64{b - t*v, b + t*v}
	}

	return cis
}
