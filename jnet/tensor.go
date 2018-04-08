package jnet

import (
	"math/rand"
)

type TensorShape [2]int

func NewTensorShape(rows int, cols int) TensorShape {
	return TensorShape{rows, cols}
}

type Tensor struct {
	Data  [][]float64
	Shape TensorShape
}

func NewTensor(shape TensorShape) Tensor {
	data := make([][]float64, shape[0], shape[0])
	for i, _ := range data {
		data[i] = make([]float64, shape[1], shape[1])
	}

	t := Tensor{
		Data:  data,
		Shape: shape,
	}

	return t
}

func NewTensorFromFloat64(shape TensorShape, initValue float64) Tensor {
	t := NewTensor(shape)

	data := t.Data

	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			data[i][j] = initValue
		}
	}

	return t

}

func NewTensorRandomNormal(shape TensorShape) Tensor {
	t := NewTensor(shape)

	data := t.Data

	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			data[i][j] = rand.NormFloat64()
		}
	}

	return t
}

// Only supports dim 2 matrix multiplication
func DotFloat64(x Tensor, y Tensor) Tensor {
	out := NewTensor(NewTensorShape(x.Shape[0], y.Shape[1]))

	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < y.Shape[1]; j++ {
			o := 0.
			for k := 0; k < x.Shape[1]; k++ {
				o += x.Data[i][k] * y.Data[k][j]
			}
			out.Data[i][j] = o
		}
	}

	return out
}

// NOTE: this expects tensor (m, n) @ (1, n)
// TODO: generalize
func AddFloat64(x Tensor, y Tensor) Tensor {
	out := x
	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < x.Shape[1]; j++ {
			out.Data[i][j] += y.Data[0][j]
		}
	}

	return out
}

// NOTE: only supports dim 2 tensor
func TransposeFloat64(x Tensor) Tensor {
	out := NewTensor(NewTensorShape(x.Shape[1], x.Shape[0]))
	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < x.Shape[1]; j++ {
			out.Data[j][i] = x.Data[i][j]
		}
	}

	return out
}

// NOTE: only sums accros axis = 0 currently for dim 2 tensor
func SumAcross0Float64(x Tensor) Tensor {
	out := NewTensor(NewTensorShape(1, x.Shape[1]))
	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < x.Shape[1]; j++ {
			out.Data[0][j] += x.Data[i][j]
		}
	}

	return out
}

// NOTE: only supports dim 2 tensors
func MaxFloat64(x, y Tensor) Tensor {
	out := NewTensor(x.Shape)
	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < x.Shape[1]; j++ {
			out.Data[i][j] = y.Data[i][j]
			if x.Data[i][j] > y.Data[i][j] {
				out.Data[i][j] = x.Data[i][j]
			}
		}
	}

	return out
}

// NOTE: only supports dim 2 tensors
func WhereFloat64(x, y Tensor) Tensor {
	out := NewTensor(x.Shape)

	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < x.Shape[1]; j++ {
			if x.Data[i][j] > 0 {
				out.Data[i][j] = y.Data[i][j]
			}
		}
	}

	return out
}

// NOTE: only supports dim 2 tensors
func MulFloat64(x, y Tensor) Tensor {
	out := NewTensor(x.Shape)
	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < x.Shape[1]; j++ {
			out.Data[i][j] = x.Data[i][j] * y.Data[i][j]
		}
	}

	return out
}

// NOTE: only supports dim 2 tensors
func SubFloat64(x, y Tensor) Tensor {
	out := NewTensor(x.Shape)
	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < x.Shape[1]; j++ {
			out.Data[i][j] = x.Data[i][j] - y.Data[i][j]
		}
	}

	return out
}
