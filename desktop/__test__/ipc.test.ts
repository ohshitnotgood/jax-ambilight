import { exec } from "node:child_process"
import { sendMessage, connect } from "../src/components/ipc"


describe("IPC over Unix sockets", () => {
    test("Run simple terminal command within NodeJS", () => {
        exec("echo hello world", (err, stdout, stderr) => {
            expect(stdout).toBe("hello world\n")
        })
    })

    // test("Establishing basic connection with Python process", () => {
    //     const socket = connect()
    //     console.log(socket.address)
    // })
})