import * as utils from "../src/components/utils"

describe("Converting a string of colours to a Typescript array of colours", () => {
    test("Splitting a list into a list of lists of tuples", () => {
        const inp = "[[(2, 2, 4), (4, 4, 6)], [(1, 3, 4), (1, 4, 6)]]"
        const expected = ["[(2, 2, 4), (4, 4, 6)]", "[(1, 3, 4), (1, 4, 6)]"]
        const actual = utils.string2List(inp)

        // Because fuck Javascript
        expect(actual?.toString()).toBe(expected.toString())
    })

    test("Parse a list containing Python tuples into a list of list of numbers", () => {
        const input = "[(2, 2, 4), (4, 4, 6)]"
        const expected = ["(2, 2, 4)", "(4, 4, 6)"]
        const actual = utils.tupleList2List(input)

        // Of course, because Javascript
        expect(actual?.toString()).toBe(expected.toString())
    })

    test("Parse a Python tuple into a Javascript list", () => {
        const input = "(2, 2, 4)"
        const expected = [2, 2, 4]
        const actual = utils.tuple2List(input)

        // Javascript shouldn't have existed
        expect(actual?.toString()).toBe(expected.toString())
    })

    test("Parse data from backend into array containing CSS styles", () => {
        const inp = "[[(2, 2, 4), (4, 4, 6)], [(1, 3, 4), (1, 4, 6)]]"
        const expected = [["background-colour: rgb(2, 2, 4)", "background-colour: rgb(4, 4, 6)"], ["background-colour: rgb(1, 3, 4)", "background-colour: rgb(1, 4, 6)"]]
        const actual = utils.parseBackendData(inp)

        expect(actual?.toString()).toBe(expected.toString())
    })    
})