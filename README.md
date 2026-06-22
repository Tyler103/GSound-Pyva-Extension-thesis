# GSound-Pyva Extension — Multi-Floor Gunshot Acoustic Simulation

M.S. Computer Science thesis project — Florida Institute of Technology, Tyler Ton.

Simulates gunshot acoustic propagation across two floors using physics-based ray tracing and structural acoustics.

## Tools
- **GSound** — geometric ray tracing for room impulse responses
- **pyva** — frequency-dependent floor slab transmission loss
- **py_auralizer** — Ambisonic IR generation from ray data

## How It Works
1. Ray trace gunshot in Room 1 → identify floor-hitting rays
2. Compute floor impact points → create virtual sources on Room 2 ceiling
3. Apply pyva TL (drywall 12mm) per frequency band
4. Ray trace Room 2 from virtual sources → generate Ambisonic IR
5. Convolve both IRs with anechoic gunshot recording

## Results
- Floor-hitting rays: 755 / 5,898 (14.2% of energy)
- Virtual sources: 50
- Room 2 level: ~23% of Room 1 (-12.7 dB)

## Usage
```bash
python3 Test_with_pyva.py
```


