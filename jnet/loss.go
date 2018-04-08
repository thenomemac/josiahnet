package jnet

import "math"

type Loss interface {
	Loss(predicted Tensor, actual Tensor) float64
	Grad(predicted Tensor, actual Tensor) Tensor
}

type MSE struct{}

func (mse MSE) Loss(predicted Tensor, actual Tensor) float64 {
	sum := 0.
	preds := predicted.Data
	obs := actual.Data

	for i, vals := range preds {
		for j, val := range vals {
			sum += math.Pow(val-obs[i][j], 2)
		}
	}
	return sum
}

func (mse MSE) Grad(predicted Tensor, actual Tensor) Tensor {

	grad := NewTensor(predicted.Shape)
	shape := grad.Shape

	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			grad.Data[i][j] = 2 * (predicted.Data[i][j] - actual.Data[i][j])
		}
	}

	return grad
}
