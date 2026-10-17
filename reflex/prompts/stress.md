You are planning a robustness test for a tiny on-device image classifier trained by Reflex.
Pick 5–6 perturbations that are REALISTIC for this deployment (look at the spec and the
sample frames: lighting, camera, motion, clutter), with parameters, and say why each one
matters here. Prefer perturbations that are likely to happen in the live demo.

Menu (kind → params, ranges):
- brightness      {"factor": 0.2–2.0}          multiply pixel values (0.5 = half as bright)
- contrast        {"factor": 0.2–2.0}
- gaussian_blur   {"sigma": 0.3–8}             out of focus
- motion_blur     {"length": 3–40, "angle": 0–180}   fast movement, px
- noise           {"std": 1–60}                sensor noise, 0–255 scale
- occlusion_patch {"size": 0.05–0.6}           random grey square, fraction of the short side
- rotation        {"degrees": -45–45}          camera tilt
- jpeg_quality    {"quality": 5–95}            compression artefacts
- crop_shift      {"shift": 0–0.3}             camera moved: crop + shift, fraction of size
- color_shift     {"hue": -90–90, "saturation": 0–2}   white balance / different lighting colour

Return ONLY a JSON object:

```json
{"perturbations": [
  {"kind": "brightness", "params": {"factor": 0.5}, "why": "venue lighting is dimmer than the recording"}
]}
```
