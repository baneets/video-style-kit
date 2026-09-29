// Fonts a style can name. Google fonts are fetched at render time (needs internet); any other name is used as an
// installed system font (e.g. "Helvetica Neue", "Arial Black").
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Loader = () => Promise<any>; // each font module types its own weights/subsets; the call below is the same for all

export const GOOGLE_FONTS: Record<string, Loader> = {
  Montserrat: () => import("@remotion/google-fonts/Montserrat"),
  Inter: () => import("@remotion/google-fonts/Inter"),
  Poppins: () => import("@remotion/google-fonts/Poppins"),
  Anton: () => import("@remotion/google-fonts/Anton"),
  "Bebas Neue": () => import("@remotion/google-fonts/BebasNeue"),
  Oswald: () => import("@remotion/google-fonts/Oswald"),
  Roboto: () => import("@remotion/google-fonts/Roboto"),
  "Archivo Black": () => import("@remotion/google-fonts/ArchivoBlack"),
  "League Spartan": () => import("@remotion/google-fonts/LeagueSpartan"),
  "DM Sans": () => import("@remotion/google-fonts/DMSans"),
  "Playfair Display": () => import("@remotion/google-fonts/PlayfairDisplay"),
  Lora: () => import("@remotion/google-fonts/Lora"),
  "Permanent Marker": () => import("@remotion/google-fonts/PermanentMarker"),
  "Barlow Condensed": () => import("@remotion/google-fonts/BarlowCondensed"),
  Figtree: () => import("@remotion/google-fonts/Figtree"),
  Manrope: () => import("@remotion/google-fonts/Manrope"),
  "Space Grotesk": () => import("@remotion/google-fonts/SpaceGrotesk"),
  "Instrument Serif": () => import("@remotion/google-fonts/InstrumentSerif"),
  Kanit: () => import("@remotion/google-fonts/Kanit"),
  Rubik: () => import("@remotion/google-fonts/Rubik"),
  "Plus Jakarta Sans": () => import("@remotion/google-fonts/PlusJakartaSans"),
  Outfit: () => import("@remotion/google-fonts/Outfit"),
  Bangers: () => import("@remotion/google-fonts/Bangers"),
  Lato: () => import("@remotion/google-fonts/Lato"),
  "Open Sans": () => import("@remotion/google-fonts/OpenSans"),
  Raleway: () => import("@remotion/google-fonts/Raleway"),
  "Work Sans": () => import("@remotion/google-fonts/WorkSans"),
  Nunito: () => import("@remotion/google-fonts/Nunito"),
  Merriweather: () => import("@remotion/google-fonts/Merriweather"),
};

/** Load every Google font in `names`; resolves to a map name -> CSS font-family. Unknown names pass through. */
export const loadFonts = async (names: string[]): Promise<Record<string, string>> => {
  const out: Record<string, string> = {};
  for (const n of Array.from(new Set(names))) {
    const loader = GOOGLE_FONTS[n];
    if (!loader) {
      out[n] = `"${n}", "Helvetica Neue", Arial, sans-serif`;
      continue;
    }
    const mod = await loader();
    const f = mod.loadFont("normal", { subsets: ["latin"] });
    await f.waitUntilDone();
    out[n] = `"${f.fontFamily}", sans-serif`;
  }
  return out;
};
