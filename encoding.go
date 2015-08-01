package glasso

import (
	"encoding/json"
	"fmt"
	"reflect"
)

type jsonableModel struct {
	Df        *DataFrame `json:"data"`
	Betas     []float64  `json:"betas"`
	Fitted    []float64  `json:"yhat"`
	Residuals []float64  `json:"residuals"`
}

func marshalGeneral(o Model) ([]byte, error) {
	check := func(x []float64) []float64 {
		if len(x) == 0 {
			return nil
		}
		return x
	}

	return json.Marshal(&jsonableModel{
		Df:        o.Data(),
		Betas:     check(o.Coefficients()),
		Fitted:    check(o.Yhat()),
		Residuals: check(o.Residuals()),
	})
}

func unmarshalGeneral(o Model, buf []byte) (interface{}, error) {
	j := jsonableModel{}

	if err := json.Unmarshal(buf, &j); err != nil {
		return nil, err
	}

	rv := reflect.ValueOf(o)
	switch rv.Interface().(type) {
	case *OLS:
		ols := &OLS{}
		ols.betas = j.Betas
		ols.fitted = j.Fitted
		ols.residuals = j.Residuals
		ols.x = j.Df

		return ols, nil
	case Ridge:
		rid := &Ridge{}
		rid.beta_ridge = j.Betas
		rid.fitted = j.Fitted
		rid.residuals = j.Residuals
		rid.x = j.Df

		return rid, nil
	default:
		return nil, fmt.Errorf("Object type not found: %v", rv)
	}
}

func (o *OLS) MarshalJSON() ([]byte, error) {
	return marshalGeneral(o)
}

func (o *OLS) UnmarshalJSON(buf []byte) error {
	mod, err := unmarshalGeneral(o, buf)
	if err != nil {
		return err
	}

	m, ok := mod.(*OLS)

	*o = *m

	if !ok {
		return fmt.Errorf("Could not convert to OLS")
	}

	return nil
}

func (r *Ridge) MarshalJSON() ([]byte, error) {
	return marshalGeneral(r)
}

func (r *Ridge) UnmarshalJSON(buf []byte) error {
	mod, err := unmarshalGeneral(r, buf)
	if err != nil {
		return err
	}

	r, ok := mod.(*Ridge)
	if !ok {
		return fmt.Errorf("Could not convert to Ridge")
	}
	return nil
}

type jsonableDf struct {
	Values []float64 `json:"values"`
	Rows   int       `json:"rows"`
	Cols   int       `json:"cols"`
}

func (df *DataFrame) MarshalJSON() ([]byte, error) {
	return json.Marshal(&jsonableDf{
		Values: df.Values(),
		Rows:   df.rows,
		Cols:   df.cols,
	})
}

func (df *DataFrame) UnmarshalJSON(buf []byte) error {
	j := &jsonableDf{}
	err := json.Unmarshal(buf, j)
	if err != nil {
		return err
	}

	data, err := DF(j.Values, make([]string, j.Cols))
	if err != nil {
		return err
	}

	*df = *data
	return nil
}
