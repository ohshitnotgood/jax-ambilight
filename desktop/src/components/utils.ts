/**
 * Returns each element in a list, given that each element is a list on it's own. 
 * 
 * The input is taken as a string.
 * 
 * For instance, an input of "[[(2, 2, 4), (4, 4, 6)], [(1, 3, 4), (1, 4, 6)]]"
 * returns the elements ["[(2, 2, 4), (4, 4, 6)]", "[(1, 3, 4), (1, 4, 6)]"] as a Typescript array.
 */
export function string2List(inp: string) {
    return inp.match(/\[[^\[\]]*\]/g);
}


/**
 * Assume that the input is "[(1, 2, 3), (3, 4, 5)]", the output will then be
 * ["(1, 2, 3)", "(3, 4, 5)"]
 * @param inp String containing a list of Python tuples 
 * @returns Each tuple in a list, with the input being a string
 */
export function tupleList2List(inp: string) {
    return inp.match(/\(\s*-?\d+,\s*-?\d+,\s*-?\d+\s*\)/g)
}

/**
 * Assume that the input is "(1, 2, 3)".
 * The output then would be ["1", "2", "3"]
 * @param inp String containing Python tuples
 * @returns Javascript list
 */
export function tuple2List(inp: string) {
    inp = inp.replaceAll("(", "")
    inp = inp.replaceAll(")", "")
    return inp.match(/\d+/g)
}

export function parseBackendData(string: string): string[][] {
    const a = string2List(string)
    let out_a: string[][] = []
    
    a?.forEach((each_a) => {
        const b = tupleList2List(each_a)
        let out_b: string[] = []
        b?.forEach((each_b) => {
            const c = tuple2List(each_b)
            let out_c: string = `background-colour: rgb(${c![0]}, ${c![1]}, ${c![2]})`
            out_b.push(out_c)
        })
        out_a.push(out_b)
    })
    return out_a
}