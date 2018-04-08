package jnet

type Optimizer interface {
	Step()
}

type SGD struct {
	LR  float64
	Net NeuralNet
}

func (sgd SGD) Step() {
	for _, layer := range sgd.Net.GetLayers() {

		weights := layer.GetWeights()
		grads := layer.GetGrads()
		for name, weight := range weights {
			grad := grads[name]
			gradscaled := MulFloat64(NewTensorFromFloat64(grad.Shape, sgd.LR), grad)
			weights[name] = SubFloat64(weight, gradscaled)
		}
	}
}
