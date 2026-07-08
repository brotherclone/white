"use client";

import { useEffect, useRef, useState } from "react";
import { Command } from "cmdk";

export interface ComboboxOption<T extends string> {
  value: T;
  label: string;
  count?: number;
  /** Extra terms (e.g. song titles under a thread) that should also match this option. */
  keywords?: string[];
  /** Dimmed secondary text shown under the label — use to disambiguate duplicate labels. */
  secondary?: string;
}

interface ComboboxProps<T extends string> {
  options: ComboboxOption<T>[];
  value: T;
  onChange: (value: T) => void;
  placeholder?: string;
  triggerLabel: string;
  emptyMessage?: string;
  className?: string;
}

export function Combobox<T extends string>({
  options,
  value,
  onChange,
  placeholder = "Search…",
  triggerLabel,
  emptyMessage = "No matches",
  className = "",
}: ComboboxProps<T>) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClickOutside = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onClickOutside);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onClickOutside);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className={`relative ${className}`}>
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="px-2.5 py-1 text-[10px] font-sans border border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-zinc-500 hover:text-zinc-100 rounded transition-colors flex items-center gap-1.5"
      >
        {triggerLabel}
        <span className="text-zinc-600">▾</span>
      </button>
      {open && (
        <div className="absolute z-20 mt-1 w-72 rounded border border-zinc-700 bg-zinc-900 shadow-xl overflow-hidden">
          <Command className="flex flex-col" loop>
            <Command.Input
              autoFocus
              placeholder={placeholder}
              className="w-full px-2.5 py-2 text-xs font-sans bg-zinc-950 border-b border-zinc-800 text-zinc-200 placeholder:text-zinc-600 focus:outline-none"
            />
            <Command.List className="max-h-64 overflow-y-auto p-1">
              <Command.Empty className="px-2.5 py-2 text-[11px] text-zinc-600 font-sans">
                {emptyMessage}
              </Command.Empty>
              {options.map(opt => (
                <Command.Item
                  key={opt.value}
                  value={opt.label}
                  keywords={opt.keywords}
                  onSelect={() => {
                    onChange(opt.value);
                    setOpen(false);
                  }}
                  className={`px-2.5 py-1.5 text-xs font-sans rounded cursor-pointer flex items-center justify-between gap-2 data-[selected=true]:bg-zinc-800 ${
                    opt.value === value ? "text-white" : "text-zinc-300"
                  }`}
                >
                  <span className="flex flex-col min-w-0">
                    <span className="truncate">{opt.label}</span>
                    {opt.secondary && (
                      <span className="truncate text-[10px] text-zinc-600">{opt.secondary}</span>
                    )}
                  </span>
                  {opt.count !== undefined && (
                    <span className="text-zinc-600 shrink-0">{opt.count}</span>
                  )}
                </Command.Item>
              ))}
            </Command.List>
          </Command>
        </div>
      )}
    </div>
  );
}
