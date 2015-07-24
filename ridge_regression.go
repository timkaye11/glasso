package glasso

import (
	"math"

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
// 				= UUT y

type Ridge struct {
	x          *mat64.Dense
	lambda     float64
	n, c       int
	fitted     []float64
	residuals  []float64
	response   []float64
	beta_ridge []float64
}

// Larger lambda equals more shrinkage
func NewRidge(x *mat64.Dense, lambda float64) *Ridge {
	n, c := x.Dims()
	return &Ridge{
		x:          x,
		n:          n,
		c:          c,
		fitted:     make([]float64, n),
		residuals:  make([]float64, n),
		beta_ridge: make([]float64, c),
	}
}

// x = n x c
// U = n x c
// D = c x c
// V = c x c
func (r *Ridge) Train(y []float64) error {
	r.response = y

	epsilon := math.Pow(2, -52.0)
	small := math.Pow(2, -966.0)

	svd := mat64.SVD(mat64.DenseCopyOf(r.x), epsilon, small, true, true)

	U := svd.U
	// D[0] >= D[1] >= ... >= D[n-1]
	d := svd.Sigma
	V := svd.V

	// convert the c x c diagonal matrix D into a mat64.Dense matrix
	D := mat64.NewDense(r.c, r.c, rep(0.0, r.c*r.c))
	for i := 0; i < r.c; i++ {
		val := d[i] / (d[i] + r.lambda)
		D.Set(i, i, val)
	}

	// solve for beta_ridge
	beta := &mat64.Dense{}
	beta.Mul(V, D)
	beta.MulTrans(beta, false, U, true)
	Y := mat64.NewDense(len(y), 1, y)
	beta.Mul(beta, Y)

	r.beta_ridge = beta.Col(nil, 0)

	// find the fitted values : X * \beta_ridge
	fitted := &mat64.Dense{}
	fitted.Mul(r.x, beta)

	r.fitted = fitted.Col(nil, 0)

	return nil
}
