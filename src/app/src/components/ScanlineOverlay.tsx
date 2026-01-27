import { useMemo } from "react";

interface ScanlineOverlayProps {
  enabled?: boolean;
  scanlineCount?: number;
  scanlineOpacity?: number;
}

export function ScanlineOverlay({
  enabled = true,
  scanlineCount = 480,
  scanlineOpacity = 0.15,
}: ScanlineOverlayProps) {
  // Calculate scanline height based on count
  // Higher count = thinner lines
  const lineHeight = useMemo(() => {
    // Target roughly the specified number of scanlines across a typical screen
    // Each "scanline" is a dark line + a gap
    return Math.max(1, Math.round(1080 / scanlineCount));
  }, [scanlineCount]);

  if (!enabled) return null;

  return (
    <div
      className="scanline-overlay"
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 10,
        // CSS-based scanlines using repeating gradient
        background: `repeating-linear-gradient(
          to bottom,
          transparent 0px,
          transparent ${lineHeight}px,
          rgba(0, 0, 0, ${scanlineOpacity}) ${lineHeight}px,
          rgba(0, 0, 0, ${scanlineOpacity}) ${lineHeight * 2}px
        )`,
        // Optional: add a subtle CRT curvature vignette
        maskImage:
          "radial-gradient(ellipse 120% 120% at 50% 50%, black 60%, transparent 100%)",
        WebkitMaskImage:
          "radial-gradient(ellipse 120% 120% at 50% 50%, black 60%, transparent 100%)",
      }}
    />
  );
}
