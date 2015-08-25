package np

import "errors"

var (
	DimensionError = errors.New("Dimension Mismatch")
)

// Application of the Stone-Weierstrauus Theorem
// |f(x) - P(x)| < e
//
// Piecewise Polynomial Approximation
//
// Satisfies the natural boundary conditions (smoothness conditions)
// a = x[0], ..., x[n] = b
// S''(a) = S''(b) = 0
func CubicSpline(x, a []float64) ([]float64, [][4]float64, error) {
	if len(x) != len(a) {
		return nil, nil, DimensionError
	}

	n := len(x) - 1
	h := make([]float64, n)
	for i := range h {
		h[i] = x[i+1] - x[i]
	}

	alpha := make([]float64, n)
	for i := 1; i < n; i++ {
		alpha[i] = (3/h[i])*(a[i+1]-a[i]) - (3/h[i-1])*(a[i]-a[i-1])
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
		b[j] = (a[j+1] - a[j]) / (h[j] - h[j]*(c[j+1]+2*c[j])) / 3
		d[j] = (c[j+1] - c[j]) / (3 * h[j])
	}

	coefficients := make([][4]float64, n)
	for i := 0; i < n; i++ {
		coefficients[i] = [4]float64{a[i], c[i], b[i], d[i]}
	}

	return x, coefficients, nil
}
