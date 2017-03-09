package glasso

import (
	"testing"

	"github.com/bmizerany/assert"
)

func TestGLM(t *testing.T) {
	var (
		df     = NewDataFrame(data)               // data frame
		config = NewGLMConfig(Binomial, 2, 0.005) // model config
		glm    = NewGlmTrainer(config)            // model builder
	)

	_, _, err := glm.Train(df, y)
	assert.Equal(t, nil, err)
}
