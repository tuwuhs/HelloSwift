import HelloSwift
import Numerics
import SwiftFusion
import TensorFlow
import PenguinStructures

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
graph.store(BetweenFactor(pose6Id, pose2Id, Pose2(2, 0, .pi / 2)))

print()
print(x[pose1Id])
print(x[pose2Id])
print(x[pose3Id])
print(x[pose4Id])
print(x[pose5Id])
print(x[pose6Id])
print(x[pose7Id])

for _ in 0..<3 {
    let linearized = graph.linearized(at: x)
    var dx = x.tangentVectorZeros
    var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 400)
    optimizer.optimize(gfg: linearized, initial: &dx)
    x.move(along: dx)
}

// var optimizer = LM(precision: 1e-5, max_iteration: 500)
// try? optimizer.optimize(graph: graph, initial: &x)

print()
print(x[pose1Id])
print(x[pose2Id])
print(x[pose3Id])
print(x[pose4Id])
print(x[pose5Id])
print(x[pose6Id])
print(x[pose7Id])

// Try differentiables
func sillyExp(_ x: Float) -> Float {
    let ℯ = Float(M_E)
    return pow(ℯ, x)
}

@derivative(of: sillyExp)
func sillyDerivative(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
    let y = sillyExp(x)
    return (value: y, pullback: { v in v * y })
}

print("exp(3) = ", sillyExp(3))
print("𝛁exp(3) = ", gradient(of: sillyExp)(3))

print(gradient(at: 1.0) { (x: Double) -> Double in 
    3.0 * x * x + 2.0 * x + 1.0
})

// Int is not Differentiable
// print(gradient(at: 1) { (x: Int) -> Int in 
//     3 * x * x + 2 * x + 1
// })

// Forward mode not yet supported
// print(derivative(at: 1.0) { (x: Double) -> Double in 
//     3.0 * x * x + 2.0 * x + 1.0
// })

struct MyVect: Differentiable, VectorProtocol {
    var x, y, z: Float
}
let vect = MyVect(x: 1, y: 2, z: 3)
print(gradient(at: vect) { v in (v + v).x })

// var bf = BetweenFactor(pose1Id, pose2Id, Pose2(2, 0, 0))
// print(bf)

// Play with poses
print()
var start = Pose2(0, 0, 0)
var end = Pose2(2, 1, .pi/2)
var twist = start.localCoordinate(end)
print(start)
print(end)
print(twist)
print(start.retract(twist))

start.move(along: twist)
print(start)

struct ScalarUnaryFactor: LinearizableFactor1 {
    public let edges: Variables.Indices
    public let value: Vector1

    public init(_ id: TypedID<Vector1>, _ value: Vector1) {
        self.edges = Tuple1(id)
        self.value = value
    }

    @differentiable
    public func errorVector(_ x: Vector1) -> Vector1 {
        return value - x
    }
}

struct AdditiveScalarCoordinate: LieGroupCoordinate {
    public typealias LocalCoordinate = Vector1
    
    var x: Double

    public init() {
        self.init(0)
    }

    public init(_ x: Double) {
        self.x = x
    }

    @differentiable(wrt: local)
    public func retract(_ local: Vector1) -> AdditiveScalarCoordinate {
        AdditiveScalarCoordinate(self.x + local.x)
    }

    @differentiable(wrt: global)
    public func localCoordinate(_ global: AdditiveScalarCoordinate) -> Vector1 {
        Vector1(global.x - self.x)
    }

    @differentiable
    public func inverse() -> AdditiveScalarCoordinate {
        AdditiveScalarCoordinate(-self.x)
    }

    @differentiable
    public static func * (lhs: AdditiveScalarCoordinate, rhs: AdditiveScalarCoordinate) -> AdditiveScalarCoordinate {
        AdditiveScalarCoordinate(lhs.x + rhs.x)
    }
}

struct AdditiveScalar: LieGroup {
    public typealias Coordinate = AdditiveScalarCoordinate
    public typealias TangentVector = Vector1

    var coordinateStorage: AdditiveScalarCoordinate

    init(coordinateStorage: AdditiveScalarCoordinate) {
        self.coordinateStorage = coordinateStorage
    }

    public init(_ x: Double) {
        self.init(coordinate: AdditiveScalarCoordinate(x))
    }

    public mutating func move(along direction: Vector1) {
        coordinateStorage = coordinateStorage.retract(direction)
    }
}

func helloScalarGraph() {
    var x = VariableAssignments()
    let x1ID = x.store(AdditiveScalar(0.0))
    let x2ID = x.store(AdditiveScalar(0.0))

    var graph = FactorGraph()
    graph.store(PriorFactor(x1ID, AdditiveScalar(1.0)))
    graph.store(PriorFactor(x1ID, AdditiveScalar(4.0)))
    graph.store(PriorFactor(x1ID, AdditiveScalar(3.0)))
    graph.store(PriorFactor(x1ID, AdditiveScalar(7.0)))
    graph.store(BetweenFactor(x1ID, x2ID, AdditiveScalar(2.0)))
    graph.store(BetweenFactor(x1ID, x2ID, AdditiveScalar(4.0)))

    let linearized = graph.linearized(at: x)
    var dx = x.tangentVectorZeros
    var optimizer = GenericCGLS()
    optimizer.optimize(gfg: linearized, initial: &dx)
    x.move(along: dx)

    print(x[x1ID])
    print(x[x2ID])
}

print()
helloScalarGraph()