import * as utils from "../src/components/utils"

describe("Test string conversions", () => {
    test("Converting a string to a list of lists", () => {
        const str = "[[1, 2, 3], [2, 3, 4]]"
        const result = utils.stringToListList(str)

        // Because fuck Javascript
        expect(result.toString()).toBe([[1, 2, 3], [2, 3, 4]].toString())
    })
})