package main

import (
	"fmt"

	"github.com/thenomemac/josiahnet/jnet"
)

// cononical XOR example that every neural net must do!
func main() {

	inputs := jnet.Tensor{
		Data: [][]float64{
			[]float64{0, 0},
			[]float64{1, 0},
			[]float64{0, 1},
			[]float64{1, 1},
		},
		Shape: jnet.NewTensorShape(4, 2),
	}

	targets := jnet.Tensor{
		Data: [][]float64{
			[]float64{1, 0},
			[]float64{0, 1},
			[]float64{0, 1},
			[]float64{1, 0},
		},
		Shape: jnet.NewTensorShape(4, 2),
	}

	hiddenSize := 8
	batchSize := 4

	net := jnet.NewSequential([]jnet.Layer{
		jnet.NewLinear(
			jnet.NewTensorShape(batchSize, 2),
			jnet.NewTensorShape(batchSize, hiddenSize),
		),
		jnet.NewRelu(
			jnet.NewTensorShape(batchSize, hiddenSize),
			jnet.NewTensorShape(batchSize, hiddenSize),
		),
		jnet.NewLinear(
			jnet.NewTensorShape(batchSize, hiddenSize),
			jnet.NewTensorShape(batchSize, 2),
		),
	})

	fmt.Print("----- Begin Training -----\n\n")

	jnet.Train(net, inputs, targets, 500, 0.01)

	fmt.Print("\n----- End Training -----\n\n")

	var preds [4]float64
	for i, row := range net.Forward(inputs).Data {
		preds[i] = row[0]
	}

	var trues [4]float64
	for i, row := range targets.Data {
		trues[i] = row[0]
	}

	// log results
	fmt.Println("Predictions:\t", preds)
	fmt.Println("Targets:\t", trues)
	fmt.Println("")
	fmt.Println("And we're done! Deep Learning is fun.")
}
