package jnet

type NeuralNet interface {
	Forward(inputs Tensor) Tensor
	Backward(grad Tensor)
	GetLayers() []Layer
}

type Sequential struct {
	layers []Layer
}

func NewSequential(layers []Layer) Sequential {
	seq := Sequential{layers}
	return seq
}

func (seq Sequential) Forward(inputs Tensor) Tensor {
	for _, layer := range seq.layers {
		inputs = layer.Forward(inputs)
	}

	return inputs
}

func (seq Sequential) Backward(grad Tensor) {
	revLayers := make([]Layer, len(seq.layers))

	for i := 0; i < len(revLayers); i++ {
		revLayers[i] = seq.layers[len(seq.layers)-i-1]
	}

	for _, layer := range revLayers {
		grad = layer.Backward(grad)
	}
}

func (seq Sequential) GetLayers() []Layer {
	return seq.layers
}
