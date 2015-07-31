package glasso

import (
	"encoding/json"
	"io/ioutil"
)

type jsonableModel struct {
	df        *DataFrame `json:"data"`
	betas     *[]float64 `json:"betas"`
	fitted    *[]float64 `json:"yhat"`
	residuals *[]float64 `json:"residuals"`
}

func (o *OLS) MarshalJSON() ([]byte, error) {
	check := func(x []float64) *[]float64 {
		if len(x) == 0 {
			return nil
		}
		return &x
	}

	return json.Marshal(&jsonableModel{
		o.x,
		check(o.betas),
		check(o.fitted),
		check(o.residuals),
	})
}

func (o *OLS) UnmarshalJSON(buf []byte) error {
	j := jsonableModel{}

	if err := json.Unmarshal(buf, &j); err != nil {
		return err
	}

	// copy fields over
	*o = *NewOLS(j.df)
	o.betas = *j.betas
	o.fitted = *j.fitted
	o.residuals = *j.residuals

	return nil
}

func NewModelFromJSON(buf []byte) (*OLS, error) {
	ols := &OLS{}

	if err := ols.UnmarshalJSON(buf); err != nil {
		return nil, err
	}
	return ols, nil
}

func NewModelFromFile(file string) (*OLS, error) {
	buf, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	return NewModelFromJSON(buf)
}
