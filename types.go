package glasso

type Evaluator func(float64) float64
type Aggregator func([]float64) float64

// Regression Output
type Model interface {
	// Build a linear model. Additional arguments specified in the constructor
	Train([]float64) error
	Predict(x []float64) []float64
	Residuals() []float64
	Coefficients() []float64
	Yhat() []float64
}
