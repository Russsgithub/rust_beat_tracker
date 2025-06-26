# Real-Time BPM and Beat Phase Tracker in Rust

This is a real-time beat tracking system implemented in Rust. It uses your microphone input to estimate musical tempo (BPM) and beat phase using FFT, mel spectral flux, and autocorrelation.

## Features

- 🎧 Real-time audio input via `cpal`
- ⚡ FFT-based spectral flux computation using `rustfft`
- 🔥 Mel frequency band filtering for perceptual accuracy
- 🧠 Autocorrelation and parabolic interpolation for tempo estimation
- 🕒 Estimates both **tempo (BPM)** and **beat phase 0-100%)**
- 🧹 Includes smoothing and harmonic correction for stable BPM
- 🛠 Written entirely in safe, multithreaded Rust

## How It Works

1. **Microphone Input**: Audio frames are captured in real-time using [`cpal`](https://docs.rs/cpal).
2. **FFT Analysis**: Each frame is windowed and passed through a fast Fourier transform using [`rustfft`](https://docs.rs/rustfft).
3. **Mel Filtering**: The power spectrum is projected onto 40 mel bands, approximating human hearing.
4. **Spectral Flux**: Positive energy changes in mel bands are accumulated to detect rhythmic transients.
5. **Tempo Estimation**: Autocorrelation on the flux curve detects periodicities corresponding to tempo.
6. **Refinements**:
   - Parabolic interpolation for sub-frame lag accuracy.
   - Harmonic correction to choose half/double tempo when appropriate.
   - Smoothing via exponential moving average and median filtering.
7. **Phase Tracking**: Calculates beat position within the estimated tempo period.

## Requirements

- Rust (stable)
- A working microphone input device

## Run
- cargo run --release

## Notes
This is a work in progress vibe coding experiment, and should be viewed as such.
Bpm estimation seems mostly accurate but does lose accuracy at high bpm rates.

Further improvements are needed.
