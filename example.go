package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	trainingIn := mat.NewDense(7, 3, []float64{
		0, 0, 0,
		0, 1, 0,
		0, 0, 1,
		0, 1, 1,
		1, 0, 0,
		1, 1, 0,
		1, 1, 1})
	trainingOut := mat.NewDense(7, 1, []float64{0, 0, 0, 0, 1, 1, 1})

	synapticWeights := KaimingInitialization(3, 1)

	fmt.Println("\nInitial weights:")
	Print(synapticWeights)

	shuffled, target := Shuffle(trainingIn, trainingOut)

	fmt.Println("Target:")
	Print(target)

	var activations mat.Matrix
	for i := 0; i < 200000; i++ {

		// Multiply the i and w to create weightedOutputs
		//
		// Input *	Weights ->	Weighted Outputs:
		// ⎡1 1⎤	⎡w1⎤		⎡wo1⎤
		// ⎢0 1⎥	⎣w2⎦		⎢wo2⎥
		// ⎢1 0⎥				⎢wo3⎥
		// ⎣0 0⎦				⎣wo4⎦
		weightedOutputs := Multiply(shuffled, synapticWeights) // z

		// Map activation function (sigmoid) over all weightedOutputs -> activations
		//
		// Sigmoid(wo) = Activations:
		// ⎡fn(wo1)⎤ = ⎡a1⎤
		// ⎢fn(wo2)⎥ = ⎢a2⎥
		// ⎢fn(wo3)⎥ = ⎢a3⎥
		// ⎣fn(wo4)⎦ = ⎣a4⎦
		activations = Map(Sigmoid, weightedOutputs) // y

		// Subtract the truth/target by the activations
		//
		// Truth - Activations = Error
		// ⎡t1 - a1⎤ = ⎡e1⎤
		// ⎢t2 - a2⎥ = ⎢e2⎥
		// ⎢t3 - a3⎥ = ⎢e3⎥
		// ⎣t4 - a4⎦ = ⎣e4⎦
		error := Subtract(target, activations) // der of cost w.r.t y

		// Map the derivative of sigmoid over the activations
		// DerSigmoid(a)
		// ⎡fn(a1)⎤ = ⎡o1⎤
		// ⎢fn(a2)⎥ = ⎢o2⎥
		// ⎢fn(a3)⎥ = ⎢o3⎥
		// ⎣fn(a4)⎦ = ⎣o4⎦
		sigDer := Map(SigmoidDerivative, activations) // der of act w.r.t y

		errorSidDer := MultiplyElems(error, sigDer)

		// Create the new weights
		// Input.Transposed() *	Adjustments = New Weights
		// ⎡1 0 1 0⎤			⎡d1⎤		  ⎡nw1⎤
		// ⎣1 1 0 0⎦			⎢d2⎥		  ⎣nw2⎦
		//						⎢d3⎥
		//						⎣d4⎦
		transposedInputsErrorSidDer := Multiply(shuffled.T(), errorSidDer) // gradient descent

		// Apply the adjustments
		adjustedWeights := Add(synapticWeights, transposedInputsErrorSidDer)

		// Update weights
		synapticWeights = Update(adjustedWeights)
	}

	fmt.Println("Weights after training:")
	Print(synapticWeights)

	fmt.Println("\nResults after training:")
	Print(activations)
}
