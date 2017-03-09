package glasso

/*
type PCR struct {
	x *DataFrame
	z *DataFrame
	m int
}

func NewPCR(x *DataFrame, m int) *PCR {
	z := mat64.DenseCopyOf(x.data)
	r, c := z.Dims()

	// need to standardize x for best results
	x.Standardize()

	return &PCR{
		x: x,
		z: &DataFrame{z, r, c, nil},
		m: m,
	}
}

func (p *PCR) GetPrinComps() *mat64.Dense {
	epsilon := math.Pow(2, -52.0)
	small := math.Pow(2, -966.0)

	svd := mat64.SVD(mat64.DenseCopyOf(p.x.data), epsilon, small, false, true)

	V := svd.V

	r, c := p.x.rows, p.x.cols

	for i := 0; i < p.x.cols; i++ {
		v_i := V.Col(nil, i)
		xv := mat64.NewDense(r, c, v_i)

		xv.Mul(p.x.data, xv)

		p.z.data.SetCol(i, xv.Col(nil, i))
	}

	return p.z.data
}

// Principal component regression forms the derived input columns zm = Xvm
// and then regresses y on z1, z2, . . . , zM for some M <= p. Since the zm
// are orthogonal, this regression is just a sum of univariate regressions:
//
// y_hat = y_bar + sum \theta_m z_m
// where \theta_m = <z_m, y> / <z_m, z_m>
//
func (p *PCR) Train(y []float64) error {
	//ybar := mean(y)

	theta := make([]float64, p.m)

	for i := 0; i < p.m; i++ {
		z_m := p.z.data.Col(nil, i)
		theta[i] = sum(prod(z_m, y)) / sum(prod(z_m, z_m))
	}

	// ...
	return nil
}

*/
