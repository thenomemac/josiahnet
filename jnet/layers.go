package jnet

type Layer interface {
	Forward(inputs Tensor) Tensor
	Backward(grad Tensor) Tensor
	GetWeights() map[string]Tensor
	GetGrads() map[string]Tensor
}

type Linear struct {
	inputs  map[string]Tensor
	grads   map[string]Tensor
	weights map[string]Tensor
}

func NewLinear(inputSize TensorShape, outSize TensorShape) Linear {
	l := Linear{
		inputs:  make(map[string]Tensor),
		grads:   make(map[string]Tensor),
		weights: make(map[string]Tensor),
	}

	l.weights["w"] = NewTensorRandomNormal(NewTensorShape(inputSize[1], outSize[1]))
	l.weights["b"] = NewTensorRandomNormal(NewTensorShape(1, outSize[1]))
	return l
}

func (l Linear) GetWeights() map[string]Tensor {
	return l.weights
}

func (l Linear) GetGrads() map[string]Tensor {
	return l.grads
}

func (l Linear) Forward(inputs Tensor) Tensor {
	l.inputs["x"] = inputs

	out := AddFloat64(DotFloat64(l.inputs["x"], l.weights["w"]), l.weights["b"])

	return out
}

func (l Linear) Backward(grad Tensor) Tensor {
	l.grads["b"] = SumAcross0Float64(grad)
	l.grads["w"] = DotFloat64(TransposeFloat64(l.inputs["x"]), grad)

	gradDelta := DotFloat64(grad, TransposeFloat64(l.weights["w"]))
	return gradDelta
}

type Relu struct {
	inputs  map[string]Tensor
	grads   map[string]Tensor
	weights map[string]Tensor
}

func NewRelu(inputSize TensorShape, outSize TensorShape) Relu {
	l := Relu{
		inputs:  make(map[string]Tensor),
		grads:   make(map[string]Tensor),
		weights: make(map[string]Tensor),
	}
	return l
}

func (l Relu) GetWeights() map[string]Tensor {
	return l.weights
}

func (l Relu) GetGrads() map[string]Tensor {
	return l.grads
}

func (l Relu) Forward(inputs Tensor) Tensor {
	l.inputs["x"] = inputs
	return MaxFloat64(l.inputs["x"], NewTensor(inputs.Shape))
}

func (l Relu) Backward(grad Tensor) Tensor {
	return WhereFloat64(MaxFloat64(l.inputs["x"], NewTensor(l.inputs["x"].Shape)), grad)
}
