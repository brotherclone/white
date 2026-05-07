"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import { startGenerate, getGenerateStatus } from "@/lib/api";

type Toast = { kind: "success" | "error"; message: string };

export default function AgentRunPage() {
  const [generating, setGenerating] = useState(false);
  const [toast, setToast] = useState<Toast | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const showToast = (kind: Toast["kind"], message: string) => {
    setToast({ kind, message });
    setTimeout(() => setToast(null), 6000);
  };

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      await startGenerate();
    } catch (e) {
      setGenerating(false);
      showToast("error", e instanceof Error ? e.message : "Failed to start generate");
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const status = await getGenerateStatus();
        if (status.status === "done") {
          clearInterval(pollRef.current!);
          pollRef.current = null;
          setGenerating(false);
          showToast("success", "Generation complete — return to Songs to see new proposals");
        } else if (status.status === "error") {
          clearInterval(pollRef.current!);
          pollRef.current = null;
          setGenerating(false);
          showToast("error", status.error ?? "Generation failed");
        }
      } catch {
        // transient poll failure — keep trying
      }
    }, 5000);
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200 p-6 font-mono">
      <div className="mb-6">
        <Link
          href="/"
          className="text-xs font-sans text-zinc-500 hover:text-zinc-300 transition-colors"
        >
          ← Songs
        </Link>
      </div>

      <h1 className="text-xl font-bold text-white tracking-tight mb-1">Run Agent</h1>
      <p className="text-zinc-500 text-xs font-sans mb-8">
        Generate new song proposals via the LangChain workflow
      </p>

      {toast && (
        <div
          className={`rounded p-3 mb-6 text-sm font-sans border ${
            toast.kind === "success"
              ? "bg-green-900/40 border-green-700 text-green-300"
              : "bg-red-900/40 border-red-700 text-red-300"
          }`}
        >
          {toast.message}
        </div>
      )}

      <button
        onClick={handleGenerate}
        disabled={generating}
        className="flex items-center gap-2 px-4 py-2 text-sm font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-200 hover:bg-zinc-700 hover:border-zinc-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {generating ? (
          <>
            <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
            Generating…
          </>
        ) : (
          "Generate New Song"
        )}
      </button>
    </div>
  );
}
