import { exec, spawn } from "node:child_process"
import *  as IPCClient from "../src/components/ipc"

describe("Test IPC from NodeJS", () => {
    beforeAll(() => {
        IPCClient.createIPCClient()
    })

    test("Run a Python script from NodeJS", async () => {
        IPCClient.onMessageReceived((data: any) => {
            expect(data).toBe("hello world")
        })
    })
})