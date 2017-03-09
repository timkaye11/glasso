package build

import (
	"sort"

	"github.com/gonum/matrix/mat64"
)

type fsTrainer struct {
	delta   float64
	epsilon float64
}

func NewForwardStageWiseTrainer(delta, epsilon float64) Trainer {
	return &fsTrainer{
		delta:   delta,
		epsilon: epsilon,
	}
}

type fsModel struct {
	betas []float64
}

func (r *fsModel) Predict(x []float64) float64 {
	return r.betas[0] + sum(prod(x, r.betas[1:]))
}

func calculateCorrelation(x *mat64.Dense, y []float64) []float64 {
	_, p := x.Dims()
	cors := make([]float64, 0, p)
	for i := 0; i < p; i++ {
		cors[i] = cor(mat64.Col(nil, i, x), y)
	}
	return cors
}

// Start with initial residual r = y, and β1 = β2 = · · · = βp = 0.
// Find the predictor Zj (j = 1, . . . , p) most correlated with r
// Update βj ← βj + δj
// Set r ← r − δjZj
// Repeat
//
// Pretty much the same as least squares boosting
func (f *fsTrainer) Train(df *DataFrame, y []float64) (Model, Summary, error) {
	// first we need to standardize the matrix and scale y
	// and set up variables
	df.Standardize() // make sure x_j_bar = 0
	n, p := df.Rows(), df.Cols()

	// set all betas to 0
	betas := rep(0.0, p)

	// center y
	r := subtractMean(y) // make sure y_bar = 0
	x := mat64.NewDense(n, p, rep(0.0, n*p))
	data := df.Data()
	firstRun := true

	// we continue until the residuals are uncorrelated with the predictors up
	// to a certain delta
	isCorrelation := func(y []float64) bool {
		if firstRun {
			firstRun = false
			return true
		}

		// find the most correlated variable
		cors := calculateCorrelation(data, y)
		if max(cors) < f.delta {
			return false
		}
		return true
	}

	// how do we know when to stop?
	for isCorrelation(r) {

		// find the most correlated variable
		cors := calculateCorrelation(data, y)
		maxCor := max(cors)
		maxIdx := sort.SearchFloat64s(cors, maxCor)

		// update beta_j
		// beta_j = beta_j + delta_j
		// where delta_j = epsilon * sign(y, x_j)
		x.SetCol(maxIdx, mat64.Col(nil, maxIdx, data))
		//ols := NewOLS(&DataFrame{x, n, p, nil})
		//ols.Train(r)

		// update beta
		delta := f.epsilon * sign(sum(prod(mat64.Col(nil, maxIdx, x), r)))
		betas[maxIdx] += delta

		// set r = r - delta_j * x_j
		r = diff(r, multSlice(mat64.Col(nil, maxIdx, x), delta))
	}

	return &fsModel{
			betas: betas,
		}, OlsSummary{
			data: df,
			n:    n,
			p:    p,
			// fitted:    fitted,
			// residuals: residuals,
			// response:  response,
			betas: betas,
		}, nil
}
