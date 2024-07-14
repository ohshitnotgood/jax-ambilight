# JAX Ambilight

This project rewrites the swiftly-lights project to bring cross-platform compatibility using Python and JAX.

## About `swiftly-lights`
`swiftly-lights` was a project to bring Ambilight on Apple computers using a strip of LED lights. The goal was to record the screen with ScreenCaptureKit, and use Metal to calculate the average colour across several zones throughout the screen.


## About `jax-ambilight`
The goal of `jax-ambilight` is to rewrite the project using Python and JAX to achieve maximum performance while still maintaining cross-platform compatibility.

`jax` currently has a Metal plugin to run on Apple Silicon and has experimental support for AMD GPUs. Primary development will be done on Nvidia GPUs.


# Python backend
Project currently uses the `mss` library to capture the screen and then uses `jax` to calculate the average colour in each of those zones. The idea would be to allow the end-user to choose how many zones they would want.

# Kotlin frontend
Desktop app writtin using Kotlin Multiplatform that communicates to the backend over Unix sockets. Will work on an alternative IPC on Windows.