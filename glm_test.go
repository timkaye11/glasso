package glasso

import "testing"

func TestGLM(t *testing.T) {
	df := NewDF(data)

	config := NewGLMConfig(Binomial, 2, 0.005)

	glm := NewGLM()
	coefficients, err := glm.Train(df.data, y, config)

	if err != nil {
		t.Fatalf("could not train GLM: %v", err)
	}
	t.Logf("coefs: %v", coefficients)
}
