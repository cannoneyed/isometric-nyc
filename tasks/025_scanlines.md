# Scanlines

in src/isometric_nyc/app, can we add "scanlines" to an overlay canvas on top of
the deck.gl tile viewer?

I asked gemini for a shader, and this is what it suggested

```
// --- UNIFORMS (Variables you send from your game code) ---
uniform sampler2D screen_texture; // The image behind the shader
uniform vec2 resolution;          // The screen resolution (e.g., 1920.0, 1080.0)
uniform float scanline_count;     // Total scanlines (e.g., 1080.0 or 240.0 for retro)
uniform float scanline_opacity;   // How strong the effect is (0.0 to 1.0)

// --- FRAGMENT SHADER ---
void main() {
    // 1. Get the current pixel coordinate (0.0 to 1.0)
    vec2 uv = gl_FragCoord.xy / resolution.xy;

    // 2. Sample the game color at this pixel
    // (In Godot/Unity, this might be 'texture(SCREEN_TEXTURE, uv)')
    vec4 color = texture(screen_texture, uv);

    // 3. Generate the Scanline Pattern
    // We use a Sine wave based on the Y coordinate.
    // +1.0 and * 0.5 shifts the wave from [-1, 1] to [0, 1]
    float scanline = sin(uv.y * scanline_count * 3.14159 * 2.0);
    scanline = (scanline + 1.0) * 0.5;

    // 4. Apply Intensity
    // We mix white (1.0) with the scanline value based on opacity
    // This darkens the "troughs" of the sine wave
    float pattern = 1.0 - (scanline_opacity * (1.0 - scanline));

    // 5. Output Result
    gl_FragColor = vec4(color.rgb * pattern, color.a);
}
```
