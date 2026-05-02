"use client";

import Link from "next/link";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200 flex flex-col items-center justify-center gap-8 font-mono">
      <h1 className="text-2xl font-bold text-white tracking-tight">white</h1>
      <div className="flex flex-col gap-4 w-64">
        <Link
          href="/songs"
          className="block text-center px-6 py-4 rounded-lg bg-zinc-900 border border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800 transition-colors text-zinc-200 font-semibold tracking-wide"
        >
          Generation
        </Link>
        <Link
          href="/board"
          className="block text-center px-6 py-4 rounded-lg bg-zinc-900 border border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800 transition-colors text-zinc-200 font-semibold tracking-wide"
        >
          Composition Board
        </Link>
      </div>
    </div>
  );
}
