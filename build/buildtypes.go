package build

type Evaluator func(float64) float64
type Aggregator func([]float64) float64

type Trainer interface {
	Train(*DataFrame, []float64) (Model, Summary, error)
}

type Model interface {
	Predict(x []float64) float64
}

type Summary interface {
	Data() *DataFrame
	Residuals() []float64
	Coefficients() []float64
	Yhat() []float64
	Response() []float64
	SumOfSquares() float64
}
