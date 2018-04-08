package jnet

import "fmt"

func Train(net NeuralNet, inputs Tensor, targets Tensor, numEpochs int, lr float64) {

	loss := MSE{}
	optimizer := SGD{lr, net}

	for epoch := 0; epoch < numEpochs; epoch++ {
		epochLoss := 0.0
		predicted := net.Forward(inputs)
		epochLoss += loss.Loss(predicted, targets)
		grad := loss.Grad(predicted, targets)
		net.Backward(grad)
		optimizer.Step()

		if epoch%10 == 0 {
			fmt.Printf("Epoch/Loss: %v\t| %.03f\n", epoch, epochLoss)
		}
	}
}
