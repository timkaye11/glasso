package glasso

import (
	"encoding/json"
	"testing"

	"github.com/bmizerany/assert"
)

func TestEncoding(t *testing.T) {
	// fake data
	x := [][]float64{
		{4.7, 7.4, 2.3, 4.4},
		{2.2, 3.3, 1.9, 4.7},
		{1.0, 9.2, 4.2, 2.2},
	}

	// fake OLS
	o := &OLS{
		x:         NewDF(x),
		betas:     []float64{1.0, 2.0, 3.0},
		residuals: []float64{0.8, 0.7, 0.2},
		fitted:    []float64{1.1, 2.2, 3.3},
	}

	// check dimension
	r, c := o.Data().data.Dims()
	assert.Equal(t, r, 3)
	assert.Equal(t, c, 4)

	buf, err := marshalGeneral(o)
	assert.Equal(t, nil, err)

	// do a basic test
	jsonable := &jsonableModel{}
	err = json.Unmarshal(buf, &jsonable)
	assert.Equal(t, nil, err)
	assert.Equal(t, len(jsonable.Betas), 3)

	// type test
	o2 := &OLS{}
	m, err := unmarshalGeneral(o2, buf)
	assert.Equal(t, nil, err)
	o2, ok := m.(*OLS)
	assert.T(t, ok)

	// roundtrip
	buf, err = json.Marshal(o)
	assert.Equal(t, nil, err)
	ols := &OLS{}
	err = json.Unmarshal(buf, ols)
	assert.Equal(t, nil, err)
	t.Logf("Residuals: %v", ols.Residuals())
}
