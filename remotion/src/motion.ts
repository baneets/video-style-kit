// Enter / exit motion for any text element, driven by the numbers a style scan measured.
import { Easing, interpolate, spring } from "remotion";

export type Anim = {
  kind: string; // cut | fade | pop | scale | slide_up | slide_down | slide_left | slide_right | wipe | typewriter
  duration_s?: number;
  easing?: string; // linear | ease_in | ease_out | ease_in_out | ease_out_expo | spring
  distance?: number; // slide distance as a fraction of frame height (slide_up/down) or width (left/right)
  from_scale?: number; // pop / scale start
};

export type Pose = { opacity: number; scale: number; dx: number; dy: number; reveal: number; chars: number };

const EASE: Record<string, (t: number) => number> = {
  linear: Easing.linear,
  ease_in: Easing.in(Easing.cubic),
  ease_out: Easing.out(Easing.cubic),
  ease_in_out: Easing.inOut(Easing.cubic),
  ease_out_expo: Easing.out(Easing.exp),
  smooth: Easing.bezier(0.16, 1, 0.3, 1),
};

const progress = (f: number, frames: number, easing: string | undefined, fps: number): number => {
  if (frames <= 0) return 1;
  if (easing === "spring" || easing === "spring_overshoot") {
    // damping 12 overshoots ~8 %, the common "pop" look
    return spring({ frame: f, fps, config: { damping: 12, stiffness: 180, mass: 0.6 }, durationInFrames: frames });
  }
  const e = EASE[easing ?? "ease_out"] ?? EASE.ease_out;
  return interpolate(f, [0, frames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: e });
};

const REST: Pose = { opacity: 1, scale: 1, dx: 0, dy: 0, reveal: 1, chars: 1 };

/** How far into its entry (p 0..1) an element is, as a pose. Exits use the same shapes run backwards. */
const shape = (a: Anim, p: number, w: number, h: number): Pose => {
  const q = Math.min(1, Math.max(0, p));
  const fadeFast = Math.min(1, q * 3);
  switch (a.kind) {
    case "fade":
      return { ...REST, opacity: p };
    case "pop": {
      const s0 = a.from_scale ?? 0.6;
      return { ...REST, opacity: fadeFast, scale: s0 + (1 - s0) * p };
    }
    case "scale": {
      const s0 = a.from_scale ?? 0.85;
      return { ...REST, opacity: fadeFast, scale: s0 + (1 - s0) * p };
    }
    case "slide_up":
      return { ...REST, opacity: fadeFast, dy: (1 - p) * (a.distance ?? 0.04) * h };
    case "slide_down":
      return { ...REST, opacity: fadeFast, dy: -(1 - p) * (a.distance ?? 0.04) * h };
    case "slide_left":
      return { ...REST, opacity: fadeFast, dx: (1 - p) * (a.distance ?? 0.08) * w };
    case "slide_right":
      return { ...REST, opacity: fadeFast, dx: -(1 - p) * (a.distance ?? 0.08) * w };
    case "wipe":
      return { ...REST, reveal: q };
    case "typewriter":
      return { ...REST, chars: q };
    default:
      return REST;
  }
};

export const pose = (
  frame: number, startF: number, endF: number, fps: number, w: number, h: number, enter?: Anim, exit?: Anim,
): Pose => {
  const inF = enter && enter.kind !== "cut" && enter.kind !== "none" ? Math.max(1, Math.round((enter.duration_s ?? 0.2) * fps)) : 0;
  const outF = exit && exit.kind !== "cut" && exit.kind !== "none" ? Math.max(1, Math.round((exit.duration_s ?? 0.2) * fps)) : 0;
  const life = endF - startF;
  const f = frame - startF;
  if (inF && f < inF) return shape(enter!, progress(f, inF, enter!.easing, fps), w, h);
  if (outF && life - f <= outF) {
    const back = progress(life - f, outF, exit!.easing === "spring" ? "ease_in" : exit!.easing, fps);
    return shape(exit!, back, w, h);
  }
  return REST;
};
