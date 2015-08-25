package glasso

/*
import (
	"sort"

	"github.com/gonum/matrix/mat64"
)

// Analogous to least squares boosting (trees = predictors)
type LeastAngle struct {
	x        *DataFrame
	epsilon  float64 // how much to increase each beta by
	delta    float64 // limit to the max correlation amongst variables
	y        []float64
	betas    []float64
	p        int
	firstRun bool
}

// Start with initial residual r = y, and β1 = β2 = · · · = βp = 0.
// Find the predictor Zj (j = 1, . . . , p) most correlated with r
// Update βj ← βj + δj
// Set r ← r − δjZj
// Repeat
//
// Pretty much the same as least squares boosting
func (f *LeastAngle) Train(y []float64) error {
	// first we need to standardize the matrix and scale y
	// and set up variables
	f.x.Standardize() // make sure x_j_bar = 0
	n, p := f.x.rows, f.x.cols

	// set all betas to 0
	f.betas = rep(0.0, p)

	// center y
	r := subtractMean(y) // make sure y_bar = 0
	x := mat64.NewDense(n, p, rep(0.0, n*p))
	f.firstRun = true

	// find the most correlated variable
	cors := make([]float64, 0, f.x.cols)
	for i := 0; i < f.x.cols; i++ {
		cors[i] = cor(f.x.data.Col(nil, i), y)
	}
	maxCor := max(cors)
	maxIdx := sort.SearchFloat64s(cors, maxCor)

	stopCor := 1
	for f.stop() {
		x.SetCol(maxIdx, f.x.data.Col(nil, maxIdx))

		delta := f.epsilon * sign(sum(prod(x.Col(nil, maxIdx), r)))
		f.betas[maxIdx] += delta

		r = diff(y, multSlice(x.Col(nil, maxIdx), delta))

		maxCor = cor(f.x.data.Col(nil, maxIdx), r)
		for i := 0; i < f.x.Cols && i != maxIdx; i++ {
			c := cor(f.x.data.Col(nil, i), r)
			if c >= maxCor {
				maxIdx = i
			}
		}
	}

	// update beta_j
	// beta_j = beta_j + delta_j
	// where delta_j = epsilon * sign(y, x_j)
	x.SetCol(maxIdx, f.x.data.Col(nil, maxIdx))
	//ols := NewOLS(&DataFrame{x, n, p, nil})
	//ols.Train(r)

	// update beta
	delta := f.epsilon * sign(sum(prod(x.Col(nil, maxIdx), r)))
	f.betas[maxIdx] += delta

	// set r = r - delta_j * x_j
	r = diff(r, multSlice(x.Col(nil, maxIdx), delta))

	return nil
}

// we continue until the residuals are uncorrelated with the predictors up
// to a certain delta
func (f *LeastAngle) isCorrelation(y []float64) bool {
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

*/
