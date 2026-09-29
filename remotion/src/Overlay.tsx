import React, { useEffect, useState } from "react";
import { AbsoluteFill, continueRender, delayRender, useCurrentFrame, useVideoConfig } from "remotion";
import { loadFonts } from "./fonts";
import { Anim, pose } from "./motion";

export type Box = { color: string; opacity?: number; padding?: number; radius?: number };
export type TextStyle = {
  font: string;
  weight?: number;
  size: number; // font size as a fraction of frame height
  case?: "upper" | "lower" | "title" | "as-is";
  color: string;
  stroke_color?: string | null;
  stroke_width?: number; // fraction of font size
  shadow?: number; // 0..1 drop-shadow strength
  box?: Box | null;
  letter_spacing?: number; // em
  line_height?: number;
  max_width?: number; // fraction of frame width
  align?: "center" | "left" | "right";
  enter?: Anim;
  exit?: Anim;
};
export type Word = { text: string; start: number; end: number };
export type Row = { start: number; end: number; words: Word[]; y?: number };
export type CaptionStyle = TextStyle & {
  y: number;
  mode?: "row" | "reveal";
  lines?: number; // rows shown at once (a "page"); words_per_line words on each
  words_per_line?: number;
  highlight?: { color?: string | null; box_color?: string | null; scale?: number } | null;
};
export type TextItem = { text: string; start: number; end: number; y: number; x?: number; style: TextStyle };
export type OverlayProps = {
  width: number;
  height: number;
  fps: number;
  durationInFrames: number;
  captions?: { style: CaptionStyle; rows: Row[] } | null;
  texts?: TextItem[];
};

const applyCase = (s: string, c?: string) =>
  c === "upper" ? s.toUpperCase() : c === "lower" ? s.toLowerCase()
    : c === "title" ? s.replace(/\w\S*/g, (t) => t[0].toUpperCase() + t.slice(1).toLowerCase()) : s;

const rgba = (hex: string, a = 1) => {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h.slice(0, 6), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};

/** A solid outline from 16 offset shadows (renders the same in every Chrome, unlike -webkit-text-stroke). */
const outline = (px: number, color: string, shadow: number, fontPx: number) => {
  const parts: string[] = [];
  if (px > 0) {
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2;
      parts.push(`${(Math.cos(a) * px).toFixed(2)}px ${(Math.sin(a) * px).toFixed(2)}px 0 ${color}`);
    }
  }
  if (shadow > 0) parts.push(`0 ${(fontPx * 0.06).toFixed(1)}px ${(fontPx * 0.18).toFixed(1)}px rgba(0,0,0,${Math.min(1, shadow)})`);
  return parts.join(", ") || "none";
};

const Block: React.FC<{
  style: TextStyle; fonts: Record<string, string>; y: number; x?: number; startF: number; endF: number;
  children: (chars: number) => React.ReactNode;
}> = ({ style, fonts, y, x, startF, endF, children }) => {
  const frame = useCurrentFrame();
  const { width, height, fps } = useVideoConfig();
  const p = pose(frame, startF, endF, fps, width, height, style.enter, style.exit);
  const fontPx = style.size * height;
  const box = style.box;
  const clip = p.reveal < 1 ? `inset(0 ${(1 - p.reveal) * 100}% 0 0)` : undefined;
  return (
    <div
      style={{
        position: "absolute",
        left: x !== undefined ? x * width : 0,
        width: x !== undefined ? undefined : width,
        top: y * height,
        display: "flex",
        justifyContent: x !== undefined ? "flex-start" : "center",
        transform: `translate(${p.dx}px, calc(-50% + ${p.dy}px))`,
        opacity: p.opacity,
      }}
    >
      <div
        style={{
          maxWidth: (style.max_width ?? 0.86) * width,
          fontFamily: fonts[style.font] ?? style.font,
          fontWeight: style.weight ?? 700,
          fontSize: fontPx,
          lineHeight: style.line_height ?? 1.12,
          letterSpacing: `${style.letter_spacing ?? 0}em`,
          color: style.color,
          textAlign: style.align ?? "center",
          textShadow: outline((style.stroke_width ?? 0) * fontPx, style.stroke_color ?? "#000", style.shadow ?? 0, fontPx),
          background: box ? rgba(box.color, box.opacity ?? 1) : undefined,
          padding: box ? `${(box.padding ?? 0.22) * fontPx}px ${(box.padding ?? 0.22) * fontPx * 1.6}px` : undefined,
          borderRadius: box ? (box.radius ?? 0.2) * fontPx : undefined,
          transform: `scale(${p.scale})`,
          transformOrigin: "50% 50%",
          clipPath: clip,
          whiteSpace: "pre-wrap",
          overflowWrap: "break-word",
        }}
      >
        {children(p.chars)}
      </div>
    </div>
  );
};

export const Overlay: React.FC<OverlayProps> = ({ captions, texts }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const [fonts, setFonts] = useState<Record<string, string> | null>(null);
  const [handle] = useState(() => delayRender("loading fonts"));
  useEffect(() => {
    const names = [...(texts ?? []).map((t) => t.style.font), ...(captions ? [captions.style.font] : [])];
    loadFonts(names)
      .then((f) => { setFonts(f); continueRender(handle); })
      .catch((e) => { console.error(e); setFonts({}); continueRender(handle); });
  }, [captions, texts, handle]);
  if (!fonts) return null;
  const t = frame / fps;
  const F = (s: number) => Math.round(s * fps);

  const row = captions?.rows.find((r) => F(r.start) <= frame && frame < F(r.end));
  const cs = captions?.style;
  return (
    <AbsoluteFill style={{ backgroundColor: "transparent" }}>
      {(texts ?? []).filter((x) => F(x.start) <= frame && frame < F(x.end)).map((x, i) => (
        <Block key={`t${i}`} style={x.style} fonts={fonts} y={x.y} x={x.x} startF={F(x.start)} endF={F(x.end)}>
          {(chars) => {
            const s = applyCase(x.text, x.style.case);
            return chars < 1 ? s.slice(0, Math.ceil(s.length * chars)) : s;
          }}
        </Block>
      ))}
      {row && cs ? (
        <Block key={`r${row.start}`} style={cs} fonts={fonts} y={row.y ?? cs.y} startF={F(row.start)} endF={F(row.end)}>
          {() =>
            row.words.map((w, i) => {
              const next = row.words[i + 1];
              const active = t >= w.start && (next ? t < next.start : true);
              const hidden = cs.mode === "reveal" && t < w.start;
              const hl = active && cs.highlight ? cs.highlight : null;
              return (
                <React.Fragment key={i}>
                  <span
                    style={{
                      visibility: hidden ? "hidden" : "visible",
                      color: hl?.color ?? undefined,
                      background: hl?.box_color ? rgba(hl.box_color, 1) : undefined,
                      borderRadius: hl?.box_color ? "0.18em" : undefined,
                      padding: hl?.box_color ? "0 0.12em" : undefined,
                      display: "inline-block",
                      transform: hl?.scale ? `scale(${hl.scale})` : undefined,
                    }}
                  >
                    {applyCase(w.text, cs.case)}
                  </span>
                  {i < row.words.length - 1
                    ? (cs.lines ?? 1) > 1 && cs.words_per_line && (i + 1) % cs.words_per_line === 0 ? <br /> : " "
                    : ""}
                </React.Fragment>
              );
            })
          }
        </Block>
      ) : null}
    </AbsoluteFill>
  );
};
