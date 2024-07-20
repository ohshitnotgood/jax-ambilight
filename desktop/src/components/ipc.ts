import net from "node:net"

const path = "/tmp/jr.sock"
export let client: net.Socket

export function connect() {
    let message: string = "something"
    
    net.connect({
        path: path,
        onread: {
            buffer: Buffer.alloc(4 * 1024),
            callback: function (nread: any, buf: any) {
                message = buf.toString('utf8', 0, nread);
                return true
            }
        }
    })
    return message
}

export function createIPCClient(path="/tmp/jr.sock") {
    client = net.createConnection(path)
    client = client.setEncoding("utf-8")
}


export function sendMessage(message: string) {
    client.write(message)
}


export function onMessageReceived(callback: (data: any) => any) {
    client.on("data", (data: any) => {
        callback(data)
    })
}
