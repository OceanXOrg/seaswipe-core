import { useEffect } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import type { PatchSource } from "@/api/client";

interface FrameLightboxProps {
  filename: string;
  fetchSource: () => Promise<PatchSource>;
  /** Stable query key for react-query caching. */
  queryKey: readonly unknown[];
  subtitle?: string;
  onClose: () => void;
  onPrev: (() => void) | null;
  onNext: (() => void) | null;
}

export default function FrameLightbox({
  filename,
  fetchSource,
  queryKey,
  subtitle,
  onClose,
  onPrev,
  onNext,
}: FrameLightboxProps) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && onPrev) onPrev();
      if (e.key === "ArrowRight" && onNext) onNext();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose, onPrev, onNext]);

  const { data, isLoading, isError, error } = useQuery({
    queryKey,
    queryFn: fetchSource,
    staleTime: 5 * 60 * 1000,
  });

  // The source frame is only available when the original frames folder is
  // present (recorded in the preprocessing manifest). If not, fall back to
  // showing the patch crop itself.
  const frameUrl = data?.frame_url ?? null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
      onClick={onClose}
    >
      <div
        className="relative flex max-h-[95vh] max-w-[95vw] flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute -right-2 -top-2 z-10 flex h-8 w-8 items-center justify-center rounded-full bg-grey-300 text-white hover:bg-grey-200"
          aria-label="Close"
        >
          x
        </button>

        {isLoading && (
          <div className="flex h-64 w-[60vw] items-center justify-center text-sm text-gray-300">
            Loading frame…
          </div>
        )}

        {isError && (
          <div className="flex h-64 w-[60vw] items-center justify-center p-4 text-center text-sm text-red-400">
            Failed to load source frame: {(error as Error).message}
          </div>
        )}

        {data && frameUrl && (
          <div
            className="relative"
            style={{
              aspectRatio: `${data.frame_width} / ${data.frame_height}`,
              maxHeight: "85vh",
              maxWidth: "90vw",
            }}
          >
            <img
              src={frameUrl}
              alt={data.frame_filename}
              className="h-full w-full object-contain"
            />
            <div
              className="pointer-events-none absolute border-2 border-red-500 shadow-[0_0_0_1px_rgba(0,0,0,0.6)]"
              style={{
                left: `${(data.bbox[0] / data.frame_width) * 100}%`,
                top: `${(data.bbox[1] / data.frame_height) * 100}%`,
                width: `${(data.bbox[2] / data.frame_width) * 100}%`,
                height: `${(data.bbox[3] / data.frame_height) * 100}%`,
              }}
            />
          </div>
        )}

        {data && !frameUrl && (
          <div className="flex h-64 w-[60vw] items-center justify-center p-4 text-center text-sm text-gray-300">
            Source frame not available (the original frames folder was not found
            for this dataset).
          </div>
        )}

        <div className="mt-2 flex items-center justify-between gap-4">
          <div className="min-w-0">
            <p className="truncate text-sm text-gray-300">
              {data?.frame_filename ?? filename}
            </p>
            {subtitle && (
              <p className="truncate text-xs text-gray-400">{subtitle}</p>
            )}
          </div>
          <div className="flex shrink-0 gap-2">
            <button
              onClick={onPrev ?? undefined}
              disabled={!onPrev}
              className="rounded bg-blue-100 px-3 py-1 text-sm text-white disabled:opacity-40"
            >
              Prev
            </button>
            <button
              onClick={onNext ?? undefined}
              disabled={!onNext}
              className="rounded bg-blue-100 px-3 py-1 text-sm text-white disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
