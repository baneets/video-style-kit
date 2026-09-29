import React from "react";
import { CalculateMetadataFunction, Composition } from "remotion";
import { Overlay, OverlayProps } from "./Overlay";

const DEMO: OverlayProps = {
  width: 1080,
  height: 1920,
  fps: 30,
  durationInFrames: 90,
  captions: {
    style: { font: "Montserrat", weight: 800, size: 0.05, case: "upper", color: "#FFFFFF", stroke_color: "#000000",
      stroke_width: 0.08, shadow: 0.5, y: 0.7, mode: "row", highlight: { color: "#FFE14D" },
      enter: { kind: "pop", duration_s: 0.15, easing: "spring" } },
    rows: [{ start: 0, end: 1.5, words: [{ text: "this", start: 0, end: 0.4 }, { text: "is", start: 0.4, end: 0.7 }, { text: "your", start: 0.7, end: 1.0 }] },
      { start: 1.5, end: 3, words: [{ text: "style", start: 1.5, end: 2.2 }, { text: "kit", start: 2.2, end: 3 }] }],
  },
  texts: [{ text: "Hook title", start: 0, end: 2.5, y: 0.22,
    style: { font: "Anton", size: 0.06, color: "#111111", box: { color: "#FFFFFF", opacity: 1 }, enter: { kind: "slide_up", duration_s: 0.3 } } }],
};

// Size, rate and length come from the props file the kit writes, so one composition serves every style.
const metadata: CalculateMetadataFunction<OverlayProps> = ({ props }) => ({
  width: props.width, height: props.height, fps: props.fps, durationInFrames: Math.max(1, props.durationInFrames),
});

export const Root: React.FC = () => (
  <Composition id="Overlay" component={Overlay} width={1080} height={1920} fps={30} durationInFrames={90}
    defaultProps={DEMO} calculateMetadata={metadata} />
);
