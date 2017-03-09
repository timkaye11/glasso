package glasso

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

const DefaultTolerance = .000001

// The GamConfig specifies the desired family, and other model configurations
type GLMConfig struct {
	F         Family  // Distribution family for the GLM
	MaxIt     int64   // upper bound on number of model iterations
	Tolerance float64 // tolerance for model training
}

func NewGLMConfig(fam Family, maxit int64, tol float64) *GLMConfig {
	return &GLMConfig{
		F:         fam,
		MaxIt:     maxit,
		Tolerance: tol,
	}
}

type glmTrainer struct {
	config *GLMConfig
}

func NewGlmTrainer(config *GLMConfig) Trainer {
	return &glmTrainer{
		config: config,
	}
}

// Iterative Re-weighting Least Squares Estimation for Generalized Linear Models
func (l *glmTrainer) Train(df *DataFrame, b []float64) (Model, Summary, error) {
	if l.config == nil {
		return nil, nil, fmt.Errorf("config not set")
	}

	A := df.Data()
	nrow, ncol := A.Dims()
	x := mat64.NewDense(ncol, 1, rep(0.0, ncol))

	var i int64
	var err error
	for ; i < l.config.MaxIt; i++ {
		eta := matrixMult(A, x)
		etaCol := mat64.Col(nil, 0, eta)

		var (
			g      = make([]float64, nrow) // g = invLink(eta)
			gprime = make([]float64, nrow) // g = derivativeFn(eta)
			w      = make([]float64, nrow) // w = gprime^2 / variance(g)
		)

		for i, val := range etaCol {
			g[i] = l.config.F.LinkFn(val)
			gprime[i] = l.config.F.DerivativeFn(val)
			w[i] = math.Pow(gprime[i], 2.0) / l.config.F.VarianceFn(g[i])
		}

		// z = eta + (b - g) / gprime
		z := mat64.NewDense(nrow, 1, nil)
		eta.Clone(z)
		z.Apply(func(i, j int, eta float64) float64 {
			return eta + (b[i]-g[i])/gprime[i]
		}, z)

		// convert w = w * I
		wMat := mat64.NewDense(nrow, nrow, rep(0.0, nrow*nrow))
		for i := 0; i < nrow; i++ {
			wMat.Set(i, i, w[i])
		}

		var (
			wa     = matrixMult(wMat, A)
			cprod1 = matrixMult(wa.T(), A)
			wz     = matrixMult(wMat, z)
			cprod2 = matrixMult(wz.T(), A)
		)

		// save xold for evaluating convergence
		xold := mat64.NewDense(ncol, 1, nil)
		x.Clone(xold)

		// xnew = solve(crossprod(A,W*A), crossprod(A,W*z))
		x = &mat64.Dense{}
		err = x.Solve(cprod1, cprod2.T())
		if err != nil {
			return nil, nil, err
		}

		// convergence = sqrt(crossprod(x - xold)) <= tolerance
		diff := &mat64.Dense{}
		diff.Sub(x, xold)
		conv := matrixMult(diff.T(), diff)
		if math.Sqrt(conv.At(0, 0)) <= l.config.Tolerance {
			break
		}
	}

	coef := mat64.Col(nil, 0, x)
	fmt.Printf("coef=%v x=%v", coef, x)
	return nil, nil, nil
}

func matrixMult(a, b mat64.Matrix) *mat64.Dense {
	out := &mat64.Dense{}
	out.Mul(a, b)
	return out
}

type linkFunc func(float64) float64

type fnType uint8

const (
	Link fnType = iota
	Derivative
	Variance
)

type evalFn func(float64) float64

type Family struct {
	LinkFn       evalFn
	VarianceFn   evalFn
	DerivativeFn evalFn
}

func NewFamily(l, d, v evalFn) Family {
	return Family{
		LinkFn:       l,
		VarianceFn:   v,
		DerivativeFn: d,
	}
}

var (
	Binomial  = NewFamily(binomialLink, binomialDerivative, binomialVariance)
	Poisson   = NewFamily(poissonLink, poissonDerivative, poissonVariance)
	Gamma     = NewFamily(gammaLink, gammaDerivative, gammaVariance)
	InvNormal = NewFamily(invnLink, invnDerivative, invnVariance)
)

// -------------------------- //
//          Binomial
// -------------------------- //

// logistic link function
func binomialLink(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// derivative of logistic f(x) = f(x) * (1 - f(x))
func binomialDerivative(x float64) float64 {
	l := 1 / (1 + math.Exp(-x))
	ans := l * (1 - l)
	return ans
}

// mean = x 	variance = np(1 - p) = p - p^2
func binomialVariance(x float64) float64 {
	return x - math.Pow(x, 2.0)
}

// -------------------------- //
//          Poisson
// -------------------------- //

// exponential link function
func poissonLink(x float64) float64 {
	return math.Exp(x)
}

func poissonDerivative(x float64) float64 {
	return poissonLink(x)
}

func poissonVariance(x float64) float64 {
	return x
}

// -------------------------- //
//          Gamma
// -------------------------- //

// inverse link function: 1/x
func gammaLink(x float64) float64 {
	return 1 / x
}

// derivative of link: 1/x^2
func gammaDerivative(x float64) float64 {
	return 1 / math.Pow(x, 2.0)
}

// variance of gamma dist: kx^2, but k=1
func gammaVariance(x float64) float64 {
	return math.Pow(x, 2.0)
}

// -------------------------- //
//       Inverse Normal
// -------------------------- //

// Link function: 1 / sqrt(x)

func invnLink(x float64) float64 { return 1 / math.Sqrt(x) }

// Derivative of Link: - (x ^ 3/2) / 2

func invnDerivative(x float64) float64 {
	return -0.5 * math.Pow(x, -1.5)
}

// Variance : mu^3 / lambda , lambda = 1
func invnVariance(x float64) float64 {
	return math.Pow(x, 3.0)
}
