package glasso

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
	betas []float64
	n, p  int
}

func NewOlsTrainer() Trainer {
	return &olsTrainer{}
}

type olsTrainer struct{}

func (o *olsTrainer) Train(x *DataFrame, yvector []float64) (Model, Summary, error) {
	rows, cols := x.Rows(), x.Cols()
	//	cols := x.cols + 1
	//	d := mat64.DenseCopyOf(x.data.Grow(0, 1))
	//	d.SetCol(0, rep(1.0, rows))
	dataframe := x
	betas := make([]float64, cols)
	residuals := make([]float64, rows)
	fitted := make([]float64, rows)
	n := rows
	p := cols
	response := make([]float64, rows)

	// sanity check
	if len(yvector) != n {
		return nil, nil, DimensionError
	}

	copy(response, yvector)
	y := mat64.NewDense(len(yvector), 1, yvector)

	// remove?
	x.PushCol(rep(1., x.Rows()))

	// it's easier to do things with X = QR
	betaMat := &mat64.Dense{}
	qr := &mat64.QR{}
	data := x.Data()
	qr.Factorize(data)
	if err := betaMat.SolveQR(qr, false, y); err != nil {
		return nil, nil, err
	}
	// first one is intercept
	betas = mat64.Col(nil, 0, betaMat)
	fittedMat := &mat64.Dense{}
	fittedMat.Mul(x.Data(), betaMat)
	fitted = mat64.Col(nil, 0, fittedMat)

	residualMat := &mat64.Dense{}
	residualMat.Sub(y, fittedMat)
	residuals = mat64.Col(nil, 0, residualMat)

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

	return &OLS{
			betas: betas,
		},
		OlsSummary{
			betas:     betas,
			residuals: residuals,
			fitted:    fitted,
			response:  response,
			n:         n,
			p:         p,
			data:      dataframe,
		}, nil
}

//func (o *OLS) prediction
func (o *OLS) Predict(x []float64) float64 {
	return o.betas[0] + sum(prod(x, o.betas[1:]))
}

func (o OlsSummary) String() string {
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

type OlsSummary struct {
	betas     []float64
	residuals []float64
	fitted    []float64
	response  []float64
	n, p      int
	data      *DataFrame
}

func (o OlsSummary) Data() *DataFrame        { return o.data }
func (o OlsSummary) Coefficients() []float64 { return o.betas }
func (o OlsSummary) Residuals() []float64    { return o.residuals }
func (o OlsSummary) Yhat() []float64         { return o.fitted }

func (o OlsSummary) TotalSumofSquares() float64 {
	y := govector.Vector(o.response)
	ybar := y.Mean()
	squaredDiff := func(x float64) float64 {
		return math.Pow(x-ybar, 2.0)
	}
	return y.Apply(squaredDiff).Sum()
}

func (o OlsSummary) SumOfSquares() float64 {
	return o.ResidualSumofSquares()
}

func (o OlsSummary) ResidualSumofSquares() float64 {
	return sum(prod(o.residuals, o.residuals))
}

func (o OlsSummary) RSquared() float64 {
	return float64(1 - (o.ResidualSumofSquares() / o.TotalSumofSquares()))
}

func (o OlsSummary) MeanSquaredError() float64 {
	return o.ResidualSumofSquares() / float64(o.n) // - o.dataframe.Cols()))
}

// the adjusted r-squared adjusts the r-squared value to reflect the importance of predictor variables
// https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
func (o OlsSummary) AdjustedRSquared() float64 {
	dfe := float64(o.n)
	dft := dfe - float64(o.p)
	return 1 - (o.ResidualSumofSquares()*(dfe-1.0))/(o.TotalSumofSquares()*dft)
}

func (o OlsSummary) sdResiduals() float64 {
	ybar := mean(o.response)
	ss := 0.0
	for i := 0; i < o.n; i++ {
		ss += math.Pow(ybar-o.fitted[i], 2.0)
	}
	return math.Sqrt(ss / float64(o.n-2))
}

func (o OlsSummary) Response() []float64 {
	return o.response
}

func (o OlsSummary) F_Statistic() (float64, float64) {
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

func (o OlsSummary) Confidence_interval(alpha float64) [][2]float64 {
	tdist := stat.StudentsT_PDF(float64(o.n - 1))
	t := tdist(1 - alpha)
	std_err := VarBeta(o)
	cis := make([][2]float64, len(o.betas))
	for i, b := range o.betas {
		v := math.Sqrt(std_err[i])
		cis[i] = [2]float64{b - t*v, b + t*v}
	}

	return cis
}
