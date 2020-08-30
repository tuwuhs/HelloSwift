import HelloSwift
import Numerics
import SwiftFusion
import TensorFlow

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
// print(e)f
// print(v_)print

// var q: Tensor<Float> = TensorFlow._Raw.complex(real: Tensor<Float>(1.0), imag: Tensor<Float>(2.0))
// print(q)

var w = Complex(1, 2)
print(w)

var p1 = Pose2(Rot2(90.0 * Double.pi / 180.0), Vector2(1.0, 2.0))
var p2 = Pose2(Rot2(60.0 * Double.pi / 180.0), Vector2(-2.0, 1.0))
print(p1)
print(p2)
print(p1 * p2)

print("\nHello SwiftFusion")

var x = VariableAssignments()
let pose1Id = x.store(Pose2(0.5, 0.0, 0.2))
let pose2Id = x.store(Pose2(2.3, 0.1, -0.2))
let pose3Id = x.store(Pose2(4.1, 0.1, .pi / 2))
let pose4Id = x.store(Pose2(4.0, 2.0, .pi))
let pose5Id = x.store(Pose2(2.1, 2.1, -.pi / 2))
let pose6Id = x.store(Pose2(2.1, 2.1, -.pi / 2))
let pose7Id = x.store(Pose2(2.1, 2.1, -.pi / 2))

var graph = FactorGraph()
graph.store(PriorFactor(pose1Id, Pose2(0, 0, 0)))
graph.store(BetweenFactor(pose1Id, pose2Id, Pose2(2, 0, 0)))
graph.store(BetweenFactor(pose2Id, pose3Id, Pose2(2, 0, 0)))
graph.store(BetweenFactor(pose3Id, pose4Id, Pose2(0, 0, .pi / 2)))
graph.store(BetweenFactor(pose4Id, pose5Id, Pose2(2, 0, .pi / 2)))
graph.store(BetweenFactor(pose5Id, pose6Id, Pose2(2, 0, .pi / 2)))
graph.store(BetweenFactor(pose6Id, pose7Id, Pose2(2, 0, .pi / 2)))

print()
print(x[pose1Id])
print(x[pose2Id])
print(x[pose3Id])
print(x[pose4Id])
print(x[pose5Id])
print(x[pose6Id])
print(x[pose7Id])

// for _ in 0..<3 {
//     let linearized = graph.linearized(at: x)
//     var dx = x.tangentVectorZeros
//     var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 400)
//     optimizer.optimize(gfg: linearized, initial: &dx)
//     x.move(along: dx)
// }

var optimizer = LM(precision: 1e-5, max_iteration: 500)
try? optimizer.optimize(graph: graph, initial: &x)

print()
print(x[pose1Id])
print(x[pose2Id])
print(x[pose3Id])
print(x[pose4Id])
print(x[pose5Id])
print(x[pose6Id])
print(x[pose7Id])
