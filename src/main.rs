use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use std::io;
use std::io::Write;


// Your constants here
const SAMPLE_RATE: usize = 44100;
const FFT_SIZE: usize = 1024;  // Must be power of two
const HOP_SIZE: usize = 512;   // 50% overlap
const MIN_BPM: f32 = 40.0;
const MAX_BPM: f32 = 200.0;

fn mel_filters(
    sample_rate: usize,
    fft_size: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let hz_to_mel = |hz: f32| -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    };
    let mel_to_hz = |mel: f32| -> f32 {
        700.0 * (10f32.powf(mel / 2595.0) - 1.0)
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let bin_freq = |freq: f32| -> usize {
        ((freq / sample_rate as f32) * (fft_size as f32)) as usize
    };

    let mut filters = Vec::with_capacity(n_mels);
    for _ in 0..n_mels {
        filters.push(vec![0.0; fft_size / 2]);
    }

    for m in 1..n_mels + 1 {
        let f_m_minus = bin_freq(hz_points[m - 1]).min(fft_size / 2 - 1);
        let f_m = bin_freq(hz_points[m]).min(fft_size / 2 - 1);
        let f_m_plus = bin_freq(hz_points[m + 1]).min(fft_size / 2 - 1);

        for k in f_m_minus..f_m {
            filters[m - 1][k] = (k - f_m_minus) as f32 / (f_m - f_m_minus) as f32;
        }
        for k in f_m..f_m_plus {
            filters[m - 1][k] = (f_m_plus - k) as f32 / (f_m_plus - f_m) as f32;
        }
    }

    filters
}


// BeatTracker struct as you already have
struct BeatTracker {
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
    prev_magnitude: Vec<f32>,
    spectral_flux: Vec<f32>,
    tempo_period_samples: Option<usize>,
    phase_samples: usize,
    mel_filters: Vec<Vec<f32>>,
    bpm_history: VecDeque<f32>,
    bpm_smoothed: Option<f32>,
    smoothing_alpha: f32,
}

impl BeatTracker {
    fn new() -> Self {
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (FFT_SIZE as f32)).cos())
            .collect();

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        let mel_filters = mel_filters(SAMPLE_RATE, FFT_SIZE, 40, 20.0, 8000.0);

        Self {
            fft,
            window,
            prev_magnitude: vec![0.0; FFT_SIZE / 2],
            spectral_flux: Vec::new(),
            tempo_period_samples: None,
            phase_samples: 0,
            mel_filters,
            bpm_history: VecDeque::with_capacity(10),
            bpm_smoothed: None,
            smoothing_alpha: 0.8,
        }
    }

    fn process_frame(&mut self, input: &[f32]) -> Option<(f32, f32)> {
        assert_eq!(input.len(), FFT_SIZE);

        let mut buffer: Vec<Complex<f32>> = input
            .iter()
            .zip(self.window.iter())
            .map(|(x, w)| Complex::new(x * w, 0.0))
            .collect();

        self.fft.process(&mut buffer);

        let spectrum: Vec<f32> = buffer[..FFT_SIZE / 2].iter().map(|c| c.norm()).collect();

        let mel_bands: Vec<f32> = self.mel_filters
            .iter()
            .map(|filter| {
                filter.iter().zip(spectrum.iter()).map(|(w, s)| w * s).sum()
            })
            .collect();


        let flux = mel_bands
            .iter()
            .zip(self.prev_magnitude.iter())
            .map(|(curr, prev)| (curr - prev).max(0.0))
            .sum::<f32>();
        
        self.prev_magnitude = mel_bands;

        self.spectral_flux.push(flux);

        if self.spectral_flux.len() > SAMPLE_RATE / HOP_SIZE * 10 {
            self.spectral_flux.remove(0);
        }

        if self.spectral_flux.len() > SAMPLE_RATE / HOP_SIZE * 2 {
            if let Some(bpm) = self.estimate_bpm() {
                if bpm.is_finite() && bpm > 1.0 {
                    let beat_period_samples = (60.0 / bpm * SAMPLE_RATE as f32) as usize;
                    if beat_period_samples > 0 {
                        self.tempo_period_samples = Some(beat_period_samples);
                        self.phase_samples = (self.phase_samples + HOP_SIZE) % beat_period_samples;
                        let phase = self.phase_samples as f32 / beat_period_samples as f32;
                        return Some((bpm, phase));
                    }
                }
            }

        }

        None
    }

    /// Refines lag estimate using parabolic interpolation
    fn parabolic_interpolation(&self, corr: &[f32], lag: usize) -> f32 {
        if lag == 0 || lag + 1 >= corr.len() {
            return lag as f32;
        }
        let y0 = corr[lag - 1];
        let y1 = corr[lag];
        let y2 = corr[lag + 1];
        let denom = 2.0 * (y0 - 2.0 * y1 + y2);
        if denom.abs() < 1e-6 {
            return lag as f32;
        }
        lag as f32 + (y0 - y2) / denom
    }



    fn estimate_bpm(&mut self) -> Option<f32> {
        // Use continuous flux instead of binary onset curve
        let onset_curve = self.spectral_flux.clone();
    
        // Autocorrelation over lag range
        let max_lag = SAMPLE_RATE / HOP_SIZE * 60 / MIN_BPM as usize;
        let min_lag = SAMPLE_RATE / HOP_SIZE * 60 / MAX_BPM as usize;
        let max_lag = max_lag.min(onset_curve.len() / 2);
    
        let mut best_lag = 0;
        let mut best_corr = 0.0;
        let mut corrs = vec![0.0; max_lag + 1];
    
        for lag in min_lag..=max_lag {
            let mut sum = 0.0;
            for i in 0..(onset_curve.len() - lag) {
                sum += onset_curve[i] * onset_curve[i + lag];
            }
            corrs[lag] = sum;
            if sum > best_corr {
                best_corr = sum;
                best_lag = lag;
            }
        }
    
        if best_lag == 0 {
            return None;
        }
    
        let mut refined_lag = self.parabolic_interpolation(&corrs, best_lag);

        // Check for higher BPM harmonics (half or third lag)
        for factor in [2, 3] {
            let alt_lag = best_lag / factor;
            if alt_lag > 1 && alt_lag + 1 < corrs.len() {
                let alt_lag_refined = self.parabolic_interpolation(&corrs, alt_lag);
                let alt_corr = corrs[alt_lag];
                if alt_corr > 0.8 * corrs[best_lag] {
                    refined_lag = alt_lag_refined;
                    break;
                }
            }
        }

        // Check if half the lag (double BPM) has a strong peak too
        let half_lag = best_lag / 2;
        if half_lag > 0 && half_lag + 1 < corrs.len() {
            let half_lag_refined = self.parabolic_interpolation(&corrs, half_lag);
            if corrs[half_lag] > 0.95 * corrs[best_lag] && corrs[half_lag] > 0.5 {
                // If the subharmonic is strong enough, use it
                refined_lag = half_lag_refined;
            }
        }

        // Optional: Try a faster lag and see if it's close enough
        let alt_lag = if refined_lag >= 1.5 { refined_lag / 2.0 } else { refined_lag };
        let alt_lag_idx = alt_lag.round() as usize;
        let refined_lag_idx = refined_lag.round() as usize;

        if alt_lag_idx < corrs.len() && refined_lag_idx < corrs.len() && corrs[alt_lag_idx] > 0.9 * corrs[refined_lag_idx] {
            refined_lag = alt_lag;
        }
        
        let bpm = 60.0 / (refined_lag * HOP_SIZE as f32 / SAMPLE_RATE as f32);


        fn snap_bpm(bpm: f32) -> f32 {
            let candidates = (60..=180).step_by(1).map(|x| x as f32);
            for cand in candidates {
                if (bpm - cand).abs() < 0.9 {
                    return cand;
                }
            }
            bpm
        }

        let bpm = snap_bpm(bpm);

        if bpm.is_finite() {
            self.bpm_history.push_back(bpm);
            if self.bpm_history.len() > 10 {
                self.bpm_history.pop_front();
            }
        
            let mut sorted: Vec<_> = self.bpm_history.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_bpm = sorted[sorted.len() / 2];

            let smoothed = if let Some(prev) = self.bpm_smoothed {
                self.smoothing_alpha * median_bpm + (1.0 - self.smoothing_alpha) * prev
            } else {
                median_bpm
            };

            self.bpm_smoothed = Some(smoothed);
        
            return Some(median_bpm);
        }
        

        if bpm.is_finite() && bpm >= MIN_BPM && bpm <= MAX_BPM {
            Some(bpm)
        } else {
            None
        }
    }


}

fn main() -> Result<(), anyhow::Error> {
    let host = cpal::default_host();

    let device = host
        .default_input_device()
        .expect("No input device available");

    println!("Input device: {}", device.name()?);

    // Use input config with 44100 Hz and f32 samples if available
    let config = device.default_input_config()?.config();

    println!("Input config: {:?}", config);

    // Shared buffer with mutex to accumulate samples from audio callback
    // Using VecDeque for efficient push/pop front
    let audio_buffer = Arc::new(Mutex::new(VecDeque::<f32>::new()));

    // Shared tracker state protected by Mutex
    let tracker = Arc::new(Mutex::new(BeatTracker::new()));

    // Clone handles for audio callback closure
    let audio_buffer_cb = audio_buffer.clone();

    // Build input stream
    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Audio callback: push incoming samples into buffer
            let mut buffer = audio_buffer_cb.lock().unwrap();

            // Push all samples
            buffer.extend(data);

            // Optional: keep buffer size manageable
            if buffer.len() > SAMPLE_RATE * 30 {
                // Drop oldest samples if too large (>30 sec)
                let to_drop = buffer.len() - SAMPLE_RATE * 30;
                for _ in 0..to_drop {
                    buffer.pop_front();
                }
            }
        },
        move |err| {
            eprintln!("Stream error: {}", err);
        },
        None,
    )?;

    stream.play()?;

    println!("Listening to mic input... press Ctrl+C to quit.");

    // Main processing loop: check buffer periodically for enough samples
    loop {
        // Sleep a bit to avoid busy wait
        std::thread::sleep(std::time::Duration::from_millis(10));

        let mut buffer = audio_buffer.lock().unwrap();
        let mut tracker = tracker.lock().unwrap();

        // Process frames with hop size while enough samples are available
        while buffer.len() >= FFT_SIZE {
            // Copy FFT_SIZE samples into a Vec<f32>
            let frame: Vec<f32> = buffer.iter().cloned().take(FFT_SIZE).collect();

            let rms = (frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
            
            if rms < 0.01 {
                for _ in 0..HOP_SIZE {
                    buffer.pop_front();
                }
                continue;
            }

            if let Some((bpm, phase)) = tracker.process_frame(&frame) {
                print!("\rEstimated BPM: {:.2}, Beat Phase: {:.2}%     ", bpm, phase * 100.0);
                io::stdout().flush().unwrap();
            }

            // Remove HOP_SIZE samples (50% overlap)
            for _ in 0..HOP_SIZE {
                buffer.pop_front();
            }
        }
    }
}
