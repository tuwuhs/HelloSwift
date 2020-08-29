    import XCTest
    @testable import HelloSwift

    final class HelloSwiftTests: XCTestCase {
        func testExample() {
            // This is an example of a functional test case.
            // Use XCTAssert and related functions to verify your tests produce the correct
            // results.
            XCTAssertEqual(HelloSwift().text, "Hello, World!")
        }

        static var allTests = [
            ("testExample", testExample),
        ]
    }
