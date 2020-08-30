import HelloSwift
import TensorFlow
import Numerics

var h = HelloSwift()
print("\(h.text) \(h.num)")

var m: Tensor<Float> = [[1, 2], [3, 4]]
print(m)
print(m.transposed())
print(Float.pi * m)
print(TensorFlow._Raw.matrixInverse(m))
print(TensorFlow._Raw.matMul(m, m))

var (s, u, v) = m.svd()
print(s)
print(u!)
print(v!)

// Complex numbers tensors not supported yet?
// var (e, v_): (Tensor<Float>, Tensor<Float>) = TensorFlow._Raw.eig(m)
// print(e)
// print(v_)

// var q: Tensor<Float> = TensorFlow._Raw.complex(real: Tensor<Float>(1.0), imag: Tensor<Float>(2.0))
// print(q)

var w = Complex(1, 2)
print(w)
