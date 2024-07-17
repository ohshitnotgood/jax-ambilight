import net from "node:net"

const path = "/tmp/jr.sock"

export function connect() {
    return net.createConnection(path)
}


export function sendMessage() {
    // TODO: Implement
}


export function onMessageReceived() {
    // TODO: implement
}

export function onConnection() {
    // TODO: implement
}