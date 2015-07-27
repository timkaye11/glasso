package glasso

import (
	"sort"

	"github.com/gonum/matrix/mat64"
)

func (df *DataFrame) Standardize() {
	d := df.data

	n, p := d.Dims()

	col := make([]float64, n)
	for i := 0; i < p; i++ {
		d.Col(col, i)

		d.SetCol(i, standardize(col))
	}
}

func (df *DataFrame) Normalize() {
	d := df.data

	n, p := d.Dims()

	col := make([]float64, n)
	for i := 0; i < p; i++ {
		d.Col(col, i)

		d.SetCol(i, normalize(col))
	}
}

// Analogous to least squares boosting (trees = predictors)
type ForwardStage struct {
	x              *DataFrame
	epsilon, delta float64
	y              []float64
	betas          []float64
	p              int
	firstRun       bool
}

// Start with initial residual r = y, and β1 = β2 = · · · = βp = 0.
// Find the predictor Zj (j = 1, . . . , p) most correlated with r
// Update βj ← βj + δj
// Set r ← r − δjZj
// Repeat
//
func (f *ForwardStage) Train(y []float64) error {
	// first we need to standardize the matrix and scale y
	// and set up variables
	f.x.Standardize() // make sure x_j_bar = 0
	n, p := f.x.rows, f.x.cols
	f.betas = rep(0.0, p)
	y = subtractMean(y) // make sure y_bar = 0
	x := mat64.NewDense(n, p, rep(0.0, n*p))
	f.firstRun = true

	// how do we know when to stop?
	for f.isCorrelation(y) {

		// find the most correlated variable
		cors := make([]float64, 0, f.x.cols)
		for i := 0; i < f.x.cols; i++ {
			cors[i] = cor(f.x.data.Col(nil, i), y)
		}
		maxCor := max(cors)
		maxIdx := sort.SearchFloat64s(cors, maxCor)

		// update beta_j
		// beta_j = beta_j + delta_j
		// where delta_j = epsilon * sign(y, x_j)
		x.SetCol(maxIdx, f.x.data.Col(nil, maxIdx))
		ols := NewOLS(&DataFrame{x, n, p, nil})
		ols.Train(y)
		delta := f.epsilon * sign(maxCor)
		ols.betas[maxIdx] += delta

		// set y = y - delta_j * x_j
		y = diff(y, multSlice(x.Col(nil, maxIdx), delta))

	}
	return nil
}

// we continue until the residuals are uncorrelated with the predictors up
// to a certain delta
func (f *ForwardStage) isCorrelation(y []float64) bool {
	if f.firstRun {
		f.firstRun = false
		return true
	}

	// find the most correlated variable
	cors := make([]float64, 0, f.x.cols)
	for i := 0; i < f.x.cols; i++ {
		cors[i] = cor(f.x.data.Col(nil, i), y)
	}
	if max(cors) < f.delta {
		return false
	}
	return true
}
