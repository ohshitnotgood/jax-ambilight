use std::os::unix::net::UnixStream;
use std::io::prelude::*;

fn main() -> std::io::Result<()> {
    let mut client = UnixStream::connect("/tmp/jr.sock").unwrap();
    client.write_all(b"kill_srvr")?;
    println!("Send message");
    let mut buff = [0; 1024];
    let count = client.read(&mut buff).unwrap();
    let response = String::from_utf8(buff[..count].to_vec()).unwrap();
    println!("{response}");
    Ok(())
}