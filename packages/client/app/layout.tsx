// web/app/layout.tsx
//
// Replace the existing layout. Removes Geist fonts, adds Typekit.

import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";

export const metadata: Metadata = {
  title: "white — candidate browser",
  description: "The Earthly Frames · Rainbow Table IX · generation pipeline",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full antialiased" suppressHydrationWarning>
      <head>
        <Script
          src="https://use.typekit.net/nig1rii.js"
          strategy="beforeInteractive"
        />
        <Script id="typekit-load" strategy="beforeInteractive">
          {`try{Typekit.load({async:true});}catch(e){}`}
        </Script>
      </head>
      <body className="min-h-full flex flex-col bg-zinc-950 text-zinc-200">{children}</body>
    </html>
  );
}
