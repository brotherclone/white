"use client";

import { useEffect, useRef } from "react";
import Script from "next/script";

interface Props {
  url: string;
}


export default function MidiPlayer({ url }: Props) {
  const playerRef = useRef<HTMLElement & { start?: () => void; stop?: () => void }>(null);

  // Auto-play when the player element mounts / url changes
  useEffect(() => {
    const el = playerRef.current;
    if (!el) return;
    // Give the custom element a moment to upgrade after the script loads
    const t = setTimeout(() => el.start?.(), 300);
    return () => {
      clearTimeout(t);
      el.stop?.();
    };
  }, [url]);

  return (
    <>
      <Script
        src="https://cdn.jsdelivr.net/combine/npm/tone@14,npm/@magenta/music@1.23.1/es6/core.js,npm/html-midi-player@1.5.0"
        strategy="lazyOnload"
      />
      <div className="flex items-center gap-3">
        <midi-player
          ref={playerRef as React.RefObject<HTMLElement>}
          src={url}
          sound-font="https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus"
          className="w-full"
        />
      </div>
    </>
  );
}
