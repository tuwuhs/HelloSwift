// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "HelloSwift",
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(name: "HelloSwift", targets: ["HelloSwift"]),
        .executable(name: "Hello", targets: ["Hello"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-numerics", from: "0.0.5"),
        .package(url: "https://github.com/borglab/SwiftFusion.git", .branch("master")),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "HelloSwift",
            dependencies: []),
        .target(
            name: "Hello", 
            dependencies: [
                "HelloSwift",
                .product(name: "Numerics", package: "swift-numerics"),
                .product(name: "SwiftFusion", package: "SwiftFusion"),
            ]),
        .testTarget(
            name: "HelloSwiftTests",
            dependencies: ["HelloSwift"]),
    ]
)
