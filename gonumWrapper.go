package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Multiply returns a matrix that contains the product of two matrices A, B
func Multiply(A, B mat.Matrix) mat.Matrix {
	rowsA, _ := A.Dims()
	_, colsB := B.Dims()
	matrix := mat.NewDense(rowsA, colsB, nil)
	matrix.Mul(A, B)

	return matrix
}

// Add returns a matrix that contains the sum of two matrices A, B
func Add(A, B mat.Matrix) mat.Matrix {
	rows, cols := A.Dims()
	matrix := mat.NewDense(rows, cols, nil)
	matrix.Add(A, B)

	return matrix
}

// Subtract returns a matrix that contains the difference of two matrices A, B
func Subtract(A, B mat.Matrix) mat.Matrix {
	rows, cols := A.Dims()
	matrix := mat.NewDense(rows, cols, nil)
	matrix.Sub(A, B)

	return matrix
}

// MultiplyElems returns a matrix that contains the elemental product of two matrices A, B
func MultiplyElems(A, B mat.Matrix) mat.Matrix {
	rows, cols := A.Dims()
	matrix := mat.NewDense(rows, cols, nil)
	matrix.MulElem(A, B)

	return matrix
}

// Map returns the result of a map onto a matrix A
func Map(fn func(i, j int, v float64) float64, A mat.Matrix) mat.Matrix {
	rows, cols := A.Dims()
	matrix := mat.NewDense(rows, cols, nil)
	matrix.Apply(fn, A)

	return matrix
}

// Update returns a *Dense from a matrix A
func Update(A mat.Matrix) *mat.Dense {
	rows, cols := A.Dims()
	matrix := mat.NewDense(rows, cols, nil)
	matrix.Copy(A)

	return matrix
}

// Print will print out a matrix in formatted form
func Print(A mat.Matrix) {
	fmt.Printf("\n%v\n\n", mat.Formatted(A, mat.Prefix(""), mat.Excerpt(0)))
}

// Shuffle returns two shuffled matrices, provide target and input to retain mapping
func Shuffle(A, B mat.Matrix) (mat.Matrix, mat.Matrix) {
	// Get matrix dims and create the matrix that is returned
	rowsA, colsA := A.Dims()
	input := mat.NewDense(rowsA, colsA, nil)
	input.Copy(A)

	// Get matrix dims and create the matrix that is returned
	rowsB, colsB := B.Dims()
	target := mat.NewDense(rowsB, colsB, nil)
	target.Copy(B)

	// Create buffers
	w, x := make([]float64, colsA), make([]float64, colsA)
	y, z := make([]float64, colsB), make([]float64, colsB)

	// Make it random everytime this function is called
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(rowsA, func(i, j int) {
		if i != j {
			// Deep copy the []float64 since RawRowView() returns a struct pointer
			copy(w, input.RawRowView(i))
			copy(x, input.RawRowView(j))

			copy(y, target.RawRowView(i))
			copy(z, target.RawRowView(j))

			// Swap A matrix
			input.SetRow(i, x)
			input.SetRow(j, w)

			// Swap B matrix
			target.SetRow(i, z)
			target.SetRow(j, y)
		}
	})

	return input, target
}

// ==================================================

// KaimingInitialization returns a kaiming initialized matrix
func KaimingInitialization(rows, cols int) *mat.Dense {
	length := rows * cols
	data := make([]float64, length)
	sigma := math.Sqrt(2.0 / float64(length))
	for i := range data {
		data[i] = rand.NormFloat64() * sigma
	}

	return mat.NewDense(rows, cols, data)
}

// ==================================================

// Sigmoid used in Map(), is an activation function
func Sigmoid(i, j int, v float64) float64 {
	return 1.0 / (1 + math.Exp(-v))
}

// SigmoidDerivative used in Map(), is the derivative of the Sigmoid function, useful for backpropogation
func SigmoidDerivative(i, j int, v float64) float64 {
	// Note: v is expected to have already been returned by Sigmoid()
	return v * (1.0 - v)
}
