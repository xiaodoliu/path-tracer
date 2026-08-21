# CUDA Path Tracer

## Quick Start: Build, Test, Render, and Assemble

Run all commands from the repository root:

```bash
cd /path/to/path-tracer
```

Replace `/path/to/path-tracer` with the location where you cloned the
repository. If your terminal is already in the repository root, skip this
command.

### 1. Build with CUDA and C++20

```bash
mkdir -p build/frames

nvcc -std=c++20 -O2 -I. \
  src/main.cu \
  -o build/main
```

### 2. Benchmark one fast frame

```bash
time ./build/main \
  --fast \
  --clouds \
  --cloud-count 3 \
  --cloud-density 0.70 \
  --lightning \
  --no-thunder \
  --frames 1 \
  --fps 30
```

`--fast` is equivalent to:

```text
--width 640 --samples 9 --max-depth 6
```

It changes only render resolution and path-tracing quality. It does not change
FPS, cloud movement, cloud density, lightning, or rain. Options written after
`--fast` override individual preset values.

Convert the first frame to PNG for inspection:

```bash
ffmpeg -y \
  -i build/frames/frame_0000.ppm \
  build/frame_0000.png
```

### 3. Render a fast 40-second storm

At 30 FPS, 1,200 frames produce a 40-second video:

```bash
./build/main \
  --fast \
  --clouds \
  --cloud-count 3 \
  --cloud-speed 3.0 \
  --cloud-density 0.70 \
  --lightning \
  --frames 1200 \
  --fps 30 \
  --lightning-first 0.8 \
  --lightning-interval 6.0
```

When the render finishes normally, it writes frames `0000` through `1199` and
then generates `build/thunder.wav`.

### 4. Render at 1280x720 with reduced sampling

This is slower than `--fast` but higher resolution:

```bash
./build/main \
  --width 1280 \
  --samples 16 \
  --max-depth 8 \
  --clouds \
  --cloud-count 3 \
  --cloud-speed 3.0 \
  --cloud-density 0.70 \
  --lightning \
  --frames 1200 \
  --fps 30 \
  --lightning-first 0.8 \
  --lightning-interval 6.0
```

### 5. Assemble all frames and thunder into MP4

Use the same frame rate and frame count that were passed to the renderer:

```bash
ffmpeg -y \
  -framerate 30 \
  -start_number 0 \
  -i build/frames/frame_%04d.ppm \
  -i build/thunder.wav \
  -frames:v 1200 \
  -map 0:v:0 \
  -map 1:a:0 \
  -c:v libx264 \
  -preset slow \
  -crf 18 \
  -pix_fmt yuv420p \
  -c:a aac \
  -b:a 192k \
  -shortest \
  build/thunderstorm_40s.mp4
```

### 6. Assemble an interrupted render

If the last message is `frame 826 done`, frames `0000` through `0826` normally
exist, giving 827 frames total. Confirm the last file first:

```bash
ls -l build/frames/frame_0826.ppm
```

Then encode exactly those frames without audio:

```bash
ffmpeg -y \
  -framerate 30 \
  -start_number 0 \
  -i build/frames/frame_%04d.ppm \
  -frames:v 827 \
  -map 0:v:0 \
  -an \
  -c:v libx264 \
  -preset slow \
  -crf 18 \
  -pix_fmt yuv420p \
  build/storm_partial.mp4
```

Replace `827` with the actual number of contiguous frames. Limiting
`-frames:v` prevents stale files from an earlier render from being appended.
An interrupted render does not regenerate `thunder.wav`, so an existing WAV
may be stale or only a fraction of a second long.

## Render Quality and Performance

The default remains the high-quality 1280-wide, 200-sample, depth-30 render.
Use the fast preset for animation previews:

```bash
./build/main --fast --clouds --lightning --frames 300 --fps 30
```

The fast preset uses a 640-pixel width, 9 samples per pixel, and a maximum path
depth of 6. Each value can also be controlled independently; put overrides
after `--fast` when combining them:

```bash
./build/main --fast --width 1280 --samples 16 --max-depth 8 \
  --clouds --cloud-count 3 --lightning --frames 300 --fps 30
```

Useful controls:

```text
--fast             Preview preset: width 640, 9 samples, depth 6
--width N          Output width; height follows the 16:9 aspect ratio
--samples N        Exact path-tracing samples per pixel
--max-depth N      Maximum path bounces
--cloud-count N    Number of independently moving cloud banks
```

Frames use binary P6 PPM encoding. FFmpeg reads these with the same
`frame_%04d.ppm` input pattern as the previous ASCII PPM files.

## Render Video

The post-processing animation path traces the scene once, then applies the
animated rain effect to the cached image for every requested frame.

## Path-Traced Moving Clouds

Enable volumetric clouds moving from right to left:

```bash
./build/main --clouds --frames 60 --fps 30
```

The clouds affect light transport and cast moving shadows, so the complete
scene is path-traced again for every frame. Start with a short preview before
requesting a long animation. Cloud movement and optical density are adjustable:

```bash
./build/main --clouds --frames 60 --fps 30 \
  --cloud-speed 1.5 --cloud-density 0.45
```

Several independently varied cloud banks are distributed across the sky at
frame zero. Each bank moves left, then recycles to the right with a new size,
height, depth, and procedural density pattern, so long animations do not run
out of clouds. Adjust the right-side recycle position with `--cloud-start-x`
(more negative places newly entering clouds farther to the right):

```bash
./build/main --clouds --frames 300 --fps 30 \
  --cloud-speed 1.5 --cloud-start-x -27
```

Higher density makes clouds and shadows more opaque and can increase volume
noise. Without `--clouds`, the fast cached rain-only animation is still used.

Lightning events recur for the full animation. Consecutive strikes are shuffled
across the left, middle, and right of the image; some occur in front of the
mountains while others are naturally occluded behind the range.

To inspect cloud shape and moving ground shadows without rain distortion:

```bash
./build/main --clouds --no-rain --frames 60 --fps 30
```

## Lightning and Thunder

Lightning uses deterministic multi-pulse events, a path-traced flash light, and
a visible branching emissive bolt. Thunder is generated as a 48 kHz mono WAV
after the frames finish rendering.

Render a ten-second storm sequence at 30 FPS:

```bash
./build/main --clouds --lightning --frames 300 --fps 30 \
  --cloud-speed 3.0 --cloud-start-x -27 \
  --lightning-first 0.8 --lightning-interval 6.0 \
  --lightning-intensity 450 --thunder-delay 1.0
```

This writes the image sequence and `build/thunder.wav`. Combine both into an
MP4 with:

```bash
ffmpeg -y \
  -framerate 30 -start_number 0 -i build/frames/frame_%04d.ppm \
  -i build/thunder.wav \
  -frames:v 300 -c:v libx264 -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 192k -shortest build/thunderstorm.mp4
```

Useful controls:

```text
--lightning-first N       First strike time in seconds
--lightning-interval N    Average seconds between strikes
--lightning-intensity N   Flash and bolt brightness
--lightning-seed N        Reproducible strike timing and shape
--thunder-delay N         Seconds between flash and thunder
--thunder-volume N        Procedural thunder volume
--no-thunder              Render lightning without generating audio
```

Render a three-second animation at 30 FPS:

```bash
./build/main --frames 90 --fps 30
```

The renderer writes frames to:

```bash
build/frames/frame_0000.ppm
build/frames/frame_0001.ppm
build/frames/frame_0002.ppm
...
```

After rendering the frames, create an MP4 video with:

```bash
ffmpeg -framerate 30 -i build/frames/frame_%04d.ppm -frames:v 90 \
  -c:v libx264 -pix_fmt yuv420p build/output.mp4
```

The FFmpeg `-framerate` value must match `--fps`, and `-frames:v` should
match `--frames`. Limiting the frame count prevents older frames in the output
directory from being appended accidentally.

Use `--time-scale` to change only the rain animation speed:

```bash
./build/main --frames 90 --fps 30 --time-scale 0.5
```

Resume at a later frame without resetting the procedural rain phase:

```bash
./build/main --frames 90 --fps 30 --start-frame 90
```

## Quiet Laptop Rendering on Pop!_OS / ASUS

CUDA path tracing can make laptop fans loud, especially on high-performance laptop GPUs.

On ASUS laptops with `asusctl`, switch to the Quiet profile before preview rendering:

```bash
asusctl profile set Quiet
```

Check the active profile with:

```bash
asusctl profile get
```

The important line is:

```text
Active profile: Quiet
```

Then run the renderer normally.

You can monitor GPU usage, temperature, and power draw with:

```bash
watch -n 1 nvidia-smi
```

To switch back to balanced mode later:

```bash
asusctl profile set Balanced
```

For maximum performance rendering:

```bash
asusctl profile set Performance
```

Note: on some laptop GPUs, manually setting the NVIDIA power limit with `nvidia-smi -pl` may not be supported. In that case, use the ASUS profile mode instead.
