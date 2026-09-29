import { Config } from "@remotion/cli/config";

// Overlays are rendered with an alpha channel: PNG frames into ProRes 4444.
Config.setVideoImageFormat("png");
Config.setOverwriteOutput(true);
