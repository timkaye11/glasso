package glasso

// Train stuff
type Trainer interface {
	Train(*DataFrame, []float64) (Model, Summary, error)
}

// Predict stuff
type Model interface {
	Predict(x []float64) float64
}

// summarize the model
type Summary interface {
	Data() *DataFrame
	Residuals() []float64
	Coefficients() []float64
	Yhat() []float64
	Response() []float64
	SumOfSquares() float64
}
