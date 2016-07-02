package glasso

import "testing"

func TestGLM(t *testing.T) {
	var (
		df     = NewDF(data)                      // data frame
		config = NewGLMConfig(Binomial, 2, 0.005) // model config
		glm    = NewGLM(config)                   // model builder
	)

	coefficients, err := glm.Train(df.data, y)
	if err != nil {
		t.Fatalf("could not train GLM: %v", err)
	}
	t.Logf("coefs: %v", coefficients)
}
