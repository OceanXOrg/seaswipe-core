import { useEffect, useMemo, useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { analysisApi } from "@/api/client";
import PatchGrid from "@/components/PatchGrid";
import FrameLightbox from "@/components/FrameLightbox";

export default function ResultsTab() {
  const [threshold, setThreshold] = useState<number | null>(null);
  const [thresholdTouched, setThresholdTouched] = useState(false);
  const [page, setPage] = useState(0);
  const [sort, setSort] = useState<"desc" | "asc">("desc");
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);
  const PAGE_SIZE = 24;

  useEffect(() => {
    setPage(0);
  }, [threshold, sort]);

  const { data: histogram } = useQuery({
    queryKey: ["histogram"],
    queryFn: () => analysisApi.histogram(),
  });

  // Pull the suggested threshold from the F1 sweep over evaluation buckets.
  // Falls back to 1.0 when evaluation hasn't been run.
  const { data: evalStatus } = useQuery({
    queryKey: ["evalStatus"],
    queryFn: () => analysisApi.evalStatus(),
  });

  const suggestedThreshold = useMemo(() => {
    const buckets = evalStatus?.results?.buckets;
    if (!buckets || buckets.length === 0) return null;
    const estPos = buckets.map((b) => b.p_pos * b.size);
    const estNeg = buckets.map((b) => (1 - b.p_pos) * b.size);
    const totalPos = estPos.reduce((a, b) => a + b, 0);
    if (totalPos === 0) return null;
    let bestF1 = -1;
    let bestThr = buckets[0].score_min;
    for (let b = 0; b < buckets.length; b++) {
      let tp = 0;
      let fp = 0;
      for (let i = b; i < buckets.length; i++) {
        tp += estPos[i];
        fp += estNeg[i];
      }
      const fn = totalPos - tp;
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      if (f1 > bestF1) {
        bestF1 = f1;
        bestThr = buckets[b].score_min;
      }
    }
    return bestThr;
  }, [evalStatus]);

  useEffect(() => {
    if (thresholdTouched) return;
    if (threshold !== null) return;
    if (evalStatus === undefined) return; // still loading
    setThreshold(suggestedThreshold ?? 1.0);
  }, [evalStatus, suggestedThreshold, threshold, thresholdTouched]);

  const effectiveThreshold = threshold ?? 1.0;

  const { data: positives, isFetching: positivesFetching } = useQuery({
    queryKey: ["positives", effectiveThreshold, page, sort],
    queryFn: () => analysisApi.positives(effectiveThreshold, page, PAGE_SIZE, sort),
    placeholderData: keepPreviousData,
    enabled: threshold !== null,
  });

  const predictedPositives = positives?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(predictedPositives / PAGE_SIZE));

  return (
    <div className="space-y-6">
      {/* Threshold control */}
      <div className="flex flex-wrap items-center gap-4 rounded-lg bg-white p-4 shadow-sm">
        <label className="text-sm font-medium text-grey-300">Confidence threshold:</label>
        <input
          type="range"
          min={histogram ? Math.floor(histogram.min * 10) / 10 : -5}
          max={histogram ? Math.ceil(histogram.max * 10) / 10 : 5}
          step={0.1}
          value={effectiveThreshold}
          onChange={(e) => {
            setThreshold(Number(e.target.value));
            setThresholdTouched(true);
          }}
          className="w-48"
          aria-label="Confidence threshold"
        />
        <span className="font-mono text-sm text-grey-300">{effectiveThreshold.toFixed(1)}</span>
        <span className="text-xs text-grey-200">({predictedPositives} predicted positives)</span>
        {suggestedThreshold !== null && (
          <span className="text-xs text-grey-200">
            &middot; suggested by F1: {suggestedThreshold.toFixed(2)}
            {thresholdTouched && Math.abs(effectiveThreshold - suggestedThreshold) > 1e-6 && (
              <button
                onClick={() => {
                  setThreshold(suggestedThreshold);
                  setThresholdTouched(false);
                }}
                className="ml-2 text-blue-100 hover:underline"
              >
                reset
              </button>
            )}
          </span>
        )}
      </div>

      {/* Predicted Positives */}
      <section>
        {!histogram ? (
          <div className="rounded-lg bg-white p-4 text-sm text-grey-200 shadow-sm">
            Train a classifier first to see predictions.
          </div>
        ) : predictedPositives === 0 ? (
          <div className="rounded-lg bg-white p-4 text-sm text-grey-200 shadow-sm">
            No predictions above threshold {effectiveThreshold.toFixed(1)}. Try lowering it.
          </div>
        ) : positives && positives.patches.length > 0 ? (
          <PatchGrid
            imageUrl={(f) => analysisApi.patchUrl(f)}
            images={positives.patches}
            title={`Predicted Positives (${predictedPositives})`}
            headerExtra={
              <div className="flex items-center gap-3 text-xs text-grey-200">
                <label className="flex items-center gap-1.5">
                  <span>Sort:</span>
                  <select
                    value={sort}
                    onChange={(e) => setSort(e.target.value as "desc" | "asc")}
                    className="rounded border border-grey-100 bg-white px-1.5 py-1 text-xs text-grey-300"
                    aria-label="Sort positives by confidence"
                  >
                    <option value="desc">Highest confidence first</option>
                    <option value="asc">Lowest confidence first</option>
                  </select>
                </label>
                {predictedPositives > PAGE_SIZE && (
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setPage((p) => Math.max(0, p - 1))}
                      disabled={page === 0 || positivesFetching}
                      className="rounded border border-grey-100 px-2 py-1 hover:bg-gray-50 disabled:opacity-40"
                      aria-label="Previous page"
                    >
                      &larr; Prev
                    </button>
                    <span className="font-mono">
                      Page {page + 1} / {totalPages}
                    </span>
                    <button
                      onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                      disabled={page >= totalPages - 1 || positivesFetching}
                      className="rounded border border-grey-100 px-2 py-1 hover:bg-gray-50 disabled:opacity-40"
                      aria-label="Next page"
                    >
                      Next &rarr;
                    </button>
                  </div>
                )}
              </div>
            }
            onImageClick={(img) => {
              const idx = positives.patches.findIndex((p) => p.filename === img.filename);
              if (idx >= 0) setLightboxIdx(idx);
            }}
          />
        ) : (
          <div className="rounded-lg bg-white p-4 text-sm text-grey-200 shadow-sm">
            Loading patches...
          </div>
        )}
      </section>

      {/* Export */}
      <div className="flex flex-wrap gap-3">
        <a
          href={`/api/predictions/export?threshold=${effectiveThreshold}`}
          className="rounded bg-blue-100 px-4 py-2 text-sm font-medium text-white hover:bg-blue-200"
          download
        >
          Export Positive Predictions as CSV
        </a>
        <a
          href="/api/export"
          className="rounded border border-blue-100 px-4 py-2 text-sm font-medium text-blue-100 hover:bg-blue-50"
          download
        >
          Export Full Analysis (ZIP)
        </a>
      </div>

      {lightboxIdx !== null && positives && positives.patches[lightboxIdx] && (
        <FrameLightbox
          filename={positives.patches[lightboxIdx].filename}
          queryKey={["patch-source", positives.patches[lightboxIdx].filename]}
          fetchSource={() => analysisApi.patchSource(positives.patches[lightboxIdx].filename)}
          subtitle={`confidence ${positives.patches[lightboxIdx].logit.toFixed(2)}`}
          onClose={() => setLightboxIdx(null)}
          onPrev={lightboxIdx > 0 ? () => setLightboxIdx(lightboxIdx - 1) : null}
          onNext={
            lightboxIdx < positives.patches.length - 1
              ? () => setLightboxIdx(lightboxIdx + 1)
              : null
          }
        />
      )}
    </div>
  );
}
