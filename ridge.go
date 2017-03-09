package glasso

import (
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

// Beta_ridge = min( \sum(y_i - \beta_0  - \sum x_ij * \beta_j)^2 + \lambda \sum \beta_j^2)
// where \lambda >= 0 controls the amount of shrinkage
// Beta_ridge = (XTX + \lambdaI)−1 XTy
//
// The ridge estimate is the mode of the posterior distribution;
// Using Single value Decomposition, we can easily solve for Beta_ridge
// X = UDVt
//
// We can solve for Beta_ridge using:
// X Beta_ridge = U D(D2 + \lambda I)−1D UT y
//
// type ridgeSummary struct {
// 	x          *DataFrame
// 	lambda     float64
// 	n, c       int
// 	fitted     []float64
// 	residuals  []float64
// 	response   []float64
// 	beta_ridge []float64
// }
type Ridge struct {
	betas []float64
}

func (r *Ridge) Predict(x []float64) float64 {
	return r.betas[0] + sum(prod(x, r.betas[1:]))
}

// x = n x c
// U = n x c
// D = c x c
// V = c x c
type ridgeTrainer struct {
	lambda float64
}

// Ridge regression for model shrinkage
//
// Larger lambda equals more shrinkage of the variables.
// lambda -> 0 equals the least squares solution
// lambda -> oo means all coeffients equal 0
func (r *ridgeTrainer) Train(x *DataFrame, y []float64) (Model, Summary, error) {
	n, c := x.Data().Dims()
	var (
		fitted     = make([]float64, n)
		residuals  = make([]float64, n)
		beta_ridge = make([]float64, c)
	)

	// standarize matrix and have y_bar = 0
	x.Normalize()
	y = subtractMean(y)
	response := y
	svd := &mat64.SVD{}
	svd.Factorize(mat64.DenseCopyOf(x.Data()), matrix.SVDFull)

	U := (&mat64.Dense{})
	U.UFromSVD(svd)
	V := (&mat64.Dense{})
	V.VFromSVD(svd)
	d := svd.Values(nil)

	// convert the c x c diagonal matrix D into a mat64.Dense matrix
	D := mat64.NewDense(c, c, rep(0.0, c*c))
	for i := 0; i < c; i++ {
		val := d[i] / (d[i] + r.lambda)
		D.Set(i, i, val)
	}

	// solve for beta_ridge
	betaMat := &mat64.Dense{}
	betaMat.Mul(V, D)
	betaMat.Mul(betaMat, U.T())
	Y := mat64.NewDense(len(y), 1, y)
	betaMat.Mul(betaMat, Y)

	// save beta values
	beta_ridge = mat64.Col(nil, 0, betaMat)

	// find the fitted values : X * \beta_ridge
	fittedMat := &mat64.Dense{}
	fittedMat.Mul(x.Data(), betaMat)
	fitted = mat64.Col(nil, 0, fittedMat)

	// get residuals
	fittedMat.Sub(fittedMat, Y)
	residuals = mat64.Col(nil, 0, fittedMat)

	return &Ridge{
			betas: beta_ridge,
		}, OlsSummary{
			data: x,
			//lambda:     r.lambda,
			n:         n,
			p:         c,
			fitted:    fitted,
			residuals: residuals,
			response:  response,
			betas:     beta_ridge,
		}, nil
}
