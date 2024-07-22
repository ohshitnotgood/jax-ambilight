import { exec, spawn } from "node:child_process"
import *  as IPCClient from "../src/components/ipc"

describe("Test IPC from NodeJS", () => {
    beforeAll(() => {
        IPCClient.createIPCClient()
    })

    test("Connecting to a IPC server", async () => {
        IPCClient.onMessageReceived((data: any) => {
            expect(data).toBe("hello world")
        })
    })
})
