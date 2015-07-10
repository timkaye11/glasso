package glasso

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/drewlanenga/govector"
	"github.com/gonum/matrix/mat64"
)

var (
	DimensionError = errors.New("Error caused by wrong dimensionality")
	LabelError     = errors.New("Missing labels for columns")
)

type DataFrame struct {
	data       *mat64.Dense
	cols, rows int
	labels     []string
	colToIdx   map[string]int
}

func DF(data []float64, labels []string) (*DataFrame, error) {
	cols := len(labels)
	ents := len(data)
	// dimensions gotta be right
	if ents%cols != 0 {
		return nil, DimensionError
	}

	lookup := make(map[string]int)
	for col := 0; col < cols; col++ {
		if name := labels[col]; name != "" {
			lookup[name] = col
			continue
		}
		colname := fmt.Sprintf("$%d", col)
		lookup[colname] = col
		labels[col] = colname
	}

	return &DataFrame{
		data:     mat64.NewDense(ents/cols, cols, data),
		labels:   labels,
		colToIdx: lookup,
		cols:     cols,
		rows:     ents / cols,
	}, nil
}

func (df *DataFrame) Dim() (int, int) {
	return df.data.Dims()
}

func (df *DataFrame) GetCol(col string) ([]float64, int) {
	idx, ok := df.colToIdx[col]
	if !ok {
		return nil, 0
	}
	return df.data.Col(nil, idx), idx
}

func (df *DataFrame) Transform(f func(x float64) float64, cols ...interface{}) {
	fc := func(f func(x float64) float64, buf []float64) []float64 {
		for i := 0; i < len(buf); i++ {
			buf[i] = f(buf[i])
		}
		return buf
	}

	for _, col := range cols {
		switch v := col.(type) {
		case string:
			buf, idx := df.GetCol(v)
			df.data.SetCol(idx, fc(f, buf))
		case int, int32, int64:
			idx := v.(int)
			buf := make([]float64, df.rows)
			df.data.Col(buf, idx)
			df.data.SetCol(idx, fc(f, buf))
		default:
		}
	}
	return
}

// Similar to R, if margin is set to true, the function f is
// applied on the columns. Else, apply the function to the rows
func (df *DataFrame) Apply(f func(x []float64) float64, margin bool, idxs ...int) []float64 {
	if margin {
		return df.applyCols(f, idxs)
	}
	return df.applyRows(f, idxs)
}

func (df *DataFrame) applyCols(f func(x []float64) float64, cols []int) []float64 {
	if len(cols) > df.cols {
		panic("wtf")
	}

	output := make([]float64, len(cols))

	for i, col := range cols {
		if col > df.cols {
			panic("wtf")
		}
		x := df.data.Col(nil, col)
		output[i] = f(x)
	}

	return output
}

func (df *DataFrame) applyRows(f func(x []float64) float64, rows []int) []float64 {
	if len(rows) > df.rows {
		panic("wtf")
	}

	output := make([]float64, len(rows))

	for i, row := range rows {
		if row > df.rows {
			panic("wtf")
		}
		x := df.data.Row(nil, row)
		output[i] = f(x)
	}

	return output
}

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
		betas, o.residualSumofSquares(), o.adjustedRSquared(), o.rSquared())
}

func (o *OLS) Residuals() []float64 {
	return o.residuals
}

func (o *OLS) totalSumofSquares() float64 {
	// no chance this could error
	y, _ := govector.AsVector(o.response)
	ybar := mean(o.response)

	squaredDiff := func(x float64) float64 {
		return math.Pow(x-ybar, 2.0)
	}

	return y.Apply(squaredDiff).Sum()
}

func (o *OLS) residualSumofSquares() float64 {
	res, _ := govector.AsVector(o.residuals)

	rss, err := govector.DotProduct(res, res)
	if err != nil {
		panic(err)
	}
	return rss
}

func (o *OLS) rSquared() float64 {
	return float64(1 - o.residualSumofSquares()/o.totalSumofSquares())
}

func (o *OLS) meanSquaredError() float64 {
	n := o.x.rows
	return o.residualSumofSquares() / float64(n-2)
}

// the adjusted r-squared adjusts the r-squared value to reflect the importance of predictor variables
// https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
func (o *OLS) adjustedRSquared() float64 {
	dfe := float64(o.x.rows)
	dft := dfe - float64(o.x.cols)
	return 1 - (o.residualSumofSquares()*(dfe-1))/(o.totalSumofSquares()*dft)
}
