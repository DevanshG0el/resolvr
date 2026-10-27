import type { Metadata, Viewport } from "next";
import { Noto_Sans_Devanagari, Noto_Sans_Oriya, Noto_Sans_Tamil, Outfit } from "next/font/google";
import Nav from "@/components/Nav";
import "./globals.css";

const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit" });
// Indic fallbacks: Source 2/3 records are often written in Devanagari, Tamil or Odia
const deva = Noto_Sans_Devanagari({ subsets: ["devanagari"], variable: "--font-deva", weight: ["400", "600"] });
const tamil = Noto_Sans_Tamil({ subsets: ["tamil"], variable: "--font-tamil", weight: ["400", "600"] });
const odia = Noto_Sans_Oriya({ subsets: ["oriya"], variable: "--font-odia", weight: ["400", "600"] });

export const metadata: Metadata = {
  title: "Resolvr · Business Entity Resolution",
  description: "Match noisy business records across sources and countries with a hybrid ML pipeline.",
  manifest: "/manifest.webmanifest",
  icons: { icon: "/icon-192.png", apple: "/icon-192.png" },
  appleWebApp: { capable: true, title: "Resolvr", statusBarStyle: "black-translucent" },
};

export const viewport: Viewport = { themeColor: "#05031a", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${outfit.variable} ${deva.variable} ${tamil.variable} ${odia.variable}`}>
      <body className="min-h-dvh font-sans">
        <Nav />
        {children}
        <footer className="border-t border-white/10 px-4 py-8 text-center text-xs text-white/40 sm:px-8">
          Resolvr · Business Entity Resolution mini project · ML Challenge 2026 data (US, India, France)
        </footer>
      </body>
    </html>
  );
}
