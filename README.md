# Path Tracer

## Render Video

After rendering frames to `build/frames/frame_0000.ppm`, `frame_0001.ppm`, etc., create an MP4 with:

```bash
ffmpeg -framerate 30 -i build/frames/frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p build/output.mp4
```
