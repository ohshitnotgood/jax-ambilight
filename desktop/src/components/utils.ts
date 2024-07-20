export function stringToListList(str: string): number[][] {
    const result = str.match(/\[[^\[\]]*\]/g);
    let out: number[][] = []
    result?.forEach((each) => {
        each = each.replace("[", "")
        each = each.replace("]", "")
        const r = each.split(",").map((e) => {return parseInt(e)})
        out.push(r)

    })
    return out
}