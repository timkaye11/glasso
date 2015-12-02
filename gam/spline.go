package np

import "fmt"

var (
	DimensionError = fmt.Errorf("dimension mismatch")
)

type Coefficients [4]float64

// Application of the Stone-Weierstrauus Theorem
// |f(x) - P(x)| < e
//
// Piecewise Polynomial Approximation
//
// Satisfies the natural boundary conditions (smoothness conditions)
// a = x[0], ..., x[n] = b
// S''(a) = S''(b) = 0
func CubicSpline(x, y []float64) ([]float64, []Coefficients, error) {
	if len(x) != len(y) {
		return nil, nil, DimensionError
	}

	n := len(y) - 1
	h := make([]float64, n)
	for i := range h {
		h[i] = x[i+1] - x[i]
	}

	alpha := make([]float64, n)
	for i := 1; i < n; i++ {
		alpha[i] = (3/h[i])*(y[i+1]-y[i]) - (3/h[i-1])*(y[i]-y[i-1])
	}

	l := make([]float64, n)
	l[0] = 1
	mu := make([]float64, n)
	mu[0] = 0
	z := make([]float64, n)
	z[0] = 0

	for i := 1; i < n; i++ {
		l[i] = 2*(x[i+1]-x[i-1]) - (h[i-1] * mu[i-1])
		mu[i] = h[i] / l[i]
		z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]
	}

	l[n] = 1
	z[n] = 0
	c := make([]float64, n)
	c[n] = 0
	b := make([]float64, n)
	d := make([]float64, n)

	for j := n - 1; j >= 0; j-- {
		c[j] = z[j] - mu[j]*c[j+1]
		b[j] = (y[j+1] - y[j]) / (h[j] - h[j]*(c[j+1]+2*c[j])) / 3
		d[j] = (c[j+1] - c[j]) / (3 * h[j])
	}

	coefficients := make([]Coefficients, n)
	for i := 0; i < n; i++ {
		coefficients[i] = Coefficients{y[i], c[i], b[i], d[i]}
	}

	return x, coefficients, nil
}
