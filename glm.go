package glasso

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

type GlmTrainer struct {
	fam   Family
	maxIt int
}

func NewGlmTrainer(family string, maxIt int) Trainer {
	fam := families[family]
	return &GlmTrainer{
		fam:   fam,
		maxIt: maxIt,
	}
}

var DefaultTolerance = .000001

func (g *GlmTrainer) Train(df *DataFrame, b []float64) (Model, Summary, error) {
	var (
		A          = df.Data()
		nrow, ncol = A.Dims()
		link       = g.fam.Link()
		mu_eta     = g.fam.Derivative()
		variance   = g.fam.Variance()
		empty      = rep(0.0, ncol)
		x          = mat64.NewDense(ncol, 1, empty)
	)

	for i := 0; i < g.maxIt; i++ {
		eta := &mat64.Dense{}
		eta.Mul(A, x)

		g := make([]float64, 0, nrow)
		gprime := make([]float64, 0, nrow)
		w := make([]float64, 0, nrow)

		for i, val := range mat64.Col(nil, 0, eta) {
			g[i] = link(val)
			// gprime[i] = mu_eta(val)
			gprime[i] = mu_eta(val)

			w[i] = math.Pow(gprime[i], 2.0) / variance(g[i])
		}

		// z = eta + (b - g) / gprime
		z := make([]float64, nrow)
		for i, et := range mat64.Col(nil, 0, eta) {
			z[i] = et + (b[i]-g[i])/gprime[i]
		}

		/*
			AWA := _

			AWZ := _

			x, err := mat64.Solve(AWA, AWZ)
			if err != nil {
				return err
			}

			C := mat64.Sol
		*/
	}
	return nil, nil, nil
}

type linkFunc func(float64) float64

type Family interface {
	Link() linkFunc
	Derivative() linkFunc
	Variance() linkFunc
}

// Binomial
type binomial struct{}

// logistic link function
func (b *binomial) Link() linkFunc {
	return func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	}
}

// derivative of logistic f(x) = f(x) * (1 - f(x))
func (b *binomial) Derivative() linkFunc {
	return func(x float64) float64 {
		l := 1 / (1 + math.Exp(-x))
		return l * (1 - l)
	}
}

// mean = x 	variance = np(1 - p) = p - p^2
func (b *binomial) Variance() linkFunc {
	return func(x float64) float64 {
		return x - math.Pow(x, 2.0)
	}
}

// Poisson
type poisson struct{}

// exponential link function
func (p *poisson) Link() linkFunc {
	return func(x float64) float64 {
		return math.Exp(x)
	}
}

func (p *poisson) Derivative() linkFunc {
	return p.Link()
}

func (p *poisson) Variance() linkFunc {
	return func(x float64) float64 { return x }
}

// Gammma
type gamma struct{}

// inverse link function: 1/x
func (g *gamma) Link() linkFunc {
	return func(x float64) float64 {
		return 1 / x
	}
}

// derivative of link: 1/x^2
func (g *gamma) Derivative() linkFunc {
	return func(x float64) float64 {
		return 1 / math.Pow(x, 2.0)
	}
}

// variance of gamma dist: kx^2, but k=1
func (g *gamma) Variance() linkFunc {
	return func(x float64) float64 {
		return math.Pow(x, 2.0)
	}
}

// Inverse Normal
type invNormal struct{}

// Link function: 1 / sqrt(x)
func (i *invNormal) Link() linkFunc {
	return func(x float64) float64 { return 1 / math.Sqrt(x) }
}

// Derivative of Link: - (x ^ 3/2) / 2
func (i *invNormal) Derivative() linkFunc {
	return func(x float64) float64 {
		return -0.5 * math.Pow(x, -1.5)
	}
}

// Variance : mu^3 / lambda , lambda = 1
func (i *invNormal) Variance() linkFunc {
	return func(x float64) float64 {
		return math.Pow(x, 3.0)
	}
}

/*
var canonicalLinks = map[string]linkFunc{
	//"normal":      Identity,
	"exponential": Inverse,
	"gamma":       Inverse,
	"poisson":     Inverse,
	"bernoulli":   Logit,
	"binomial":    Logit,
}
*/

var families = map[string]Family{
	"binomial":         new(binomial),
	"poisson":          new(poisson),
	"inverse gaussian": new(invNormal),
	"inverse normal":   new(invNormal),
	"gamma":            new(gamma),
	"bernoulli":        new(binomial),
}
