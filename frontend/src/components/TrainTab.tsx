import { useState, useCallback, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { analysisApi } from "@/api/client";
import PatchGrid, { type PatchGridImage } from "@/components/PatchGrid";
import FrameLightbox from "@/components/FrameLightbox";

export default function TrainTab() {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState("");
  const [targetLogit, setTargetLogit] = useState(0);
  const [savedLabels, setSavedLabels] = useState<Record<string, number | null>>({});
  const [currentImages, setCurrentImages] = useState<
    { filename: string; logit?: number; similarity?: number }[]
  >([]);
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);
  const [reviewLightboxIdx, setReviewLightboxIdx] = useState<number | null>(null);

  // Active-learning sample size K (default 10). 1/4 is shown as dissimilar in
  // the seed search, 3/4 as nearest neighbours.
  const SAMPLE_SIZE_KEY = "seaswipe.train.sampleSize";
  const [sampleSize, setSampleSize] = useState<number>(() => {
    const stored = Number(localStorage.getItem(SAMPLE_SIZE_KEY));
    return Number.isFinite(stored) && stored > 0 ? stored : 10;
  });
  useEffect(() => {
    localStorage.setItem(SAMPLE_SIZE_KEY, String(sampleSize));
  }, [sampleSize]);
  const nDissimilar = Math.round(sampleSize * 0.25);

  const [randomPatches, setRandomPatches] = useState<string[]>([]);
  const [selectedQueryPatch, setSelectedQueryPatch] = useState<string | null>(null);
  const [showingSearchResults, setShowingSearchResults] = useState(false);
  const [reviewing, setReviewing] = useState(false);

  const { data: stats } = useQuery({
    queryKey: ["stats"],
    queryFn: () => analysisApi.stats(),
  });

  const { data: histogram } = useQuery({
    queryKey: ["histogram"],
    queryFn: () => analysisApi.histogram(),
    enabled: stats?.has_model ?? false,
  });

  const { data: labelsData } = useQuery({
    queryKey: ["labels"],
    queryFn: () => analysisApi.labels(),
    enabled: reviewing,
  });

  const [reviewLabels, setReviewLabels] = useState<Record<string, number | null>>({});
  const [reviewFilenames, setReviewFilenames] = useState<string[]>([]);

  const seededReviewRef = useRef(false);
  useEffect(() => {
    if (!reviewing) {
      seededReviewRef.current = false;
      return;
    }
    if (labelsData && !seededReviewRef.current) {
      seededReviewRef.current = true;
      setReviewFilenames(Object.keys(labelsData.labels));
      setReviewLabels(labelsData.labels as Record<string, number | null>);
    }
  }, [reviewing, labelsData]);

  const randomPatchesMutation = useMutation({
    mutationFn: () => analysisApi.randomPatches(sampleSize),
    onSuccess: (data) => {
      setRandomPatches(data.filenames);
      setSelectedQueryPatch(null);
      if (showingSearchResults) {
        setCurrentImages([]);
        setShowingSearchResults(false);
      }
    },
  });

  const searchMutation = useMutation({
    mutationFn: (q: string) =>
      analysisApi.search(q, sampleSize - nDissimilar, nDissimilar),
    onSuccess: (data) => {
      setCurrentImages(
        data.neighbors.map((n) => ({
          filename: n.filename,
          similarity: n.similarity,
        })),
      );
      setSavedLabels({});
      setShowingSearchResults(true);
    },
  });

  const annotateMutation = useMutation({
    mutationFn: ({ filename, label }: { filename: string; label: number | null }) =>
      analysisApi.annotate({ [filename]: label }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      queryClient.invalidateQueries({ queryKey: ["labels"] });
    },
  });

  const annotateBulkMutation = useMutation({
    mutationFn: (labels: Record<string, number | null>) => analysisApi.annotate(labels),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      queryClient.invalidateQueries({ queryKey: ["labels"] });
    },
  });

  const trainMutation = useMutation({
    mutationFn: () => analysisApi.train(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      queryClient.invalidateQueries({ queryKey: ["histogram"] });
      queryClient.invalidateQueries({ queryKey: ["evalStatus"] });
    },
  });

  const sampleMutation = useMutation({
    mutationFn: () => analysisApi.sample(targetLogit, sampleSize),
    onSuccess: (data) => {
      setCurrentImages(
        data.images.map((img) => ({ filename: img.filename, logit: img.logit })),
      );
      setSavedLabels({});
      setShowingSearchResults(false);
    },
  });

  const handleReviewLabel = useCallback((filename: string, label: number | null) => {
    setReviewLabels((prev) => ({ ...prev, [filename]: label }));
    annotateMutation.mutate({ filename, label });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleReviewLabelAll = useCallback(
    (label: number) => {
      const payload: Record<string, number | null> = {};
      for (const f of reviewFilenames) payload[f] = label;
      setReviewLabels((prev) => ({ ...prev, ...payload }));
      annotateBulkMutation.mutate(payload);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [reviewFilenames],
  );

  const handleLabel = useCallback((filename: string, label: number | null) => {
    setSavedLabels((prev) => ({ ...prev, [filename]: label }));
    annotateMutation.mutate({ filename, label });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleLabelAll = useCallback(
    (label: number) => {
      const payload: Record<string, number | null> = {};
      for (const img of currentImages) payload[img.filename] = label;
      setSavedLabels((prev) => ({ ...prev, ...payload }));
      annotateBulkMutation.mutate(payload);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [currentImages],
  );

  const handleHistogramClick = (data: { bin_start: number; bin_end: number }) => {
    if (data) {
      setTargetLogit(Math.round(((data.bin_start + data.bin_end) / 2) * 100) / 100);
    }
  };

  const handleSelectQueryPatch = (filename: string) => {
    setSelectedQueryPatch(filename);
    searchMutation.mutate(filename);
  };

  const handleBackToSelection = () => {
    setCurrentImages([]);
    setShowingSearchResults(false);
    setSelectedQueryPatch(null);
  };

  return (
    <div className="space-y-6">
      {/* Stats bar */}
      <div className="flex flex-wrap items-center gap-4 rounded-lg bg-white p-4 shadow-sm">
        <Stat label="Total patches" value={stats?.n_images ?? 0} />
        <Stat label="Labeled" value={stats?.n_labeled ?? 0} />
        <Stat label="Positive" value={stats?.n_pos ?? 0} />
        <Stat label="Negative" value={stats?.n_neg ?? 0} />
        <Stat label="Model" value={stats?.has_model ? "Trained" : "Not trained"} />
        <div className="ml-auto flex gap-2">
          {(stats?.n_labeled ?? 0) > 1 && (
            <button
              onClick={() => setReviewing((r) => !r)}
              className="rounded border border-blue-100 px-3 py-1.5 text-sm font-medium text-blue-100 hover:bg-blue-50"
            >
              {reviewing ? "Exit Label Review" : "Review Labels"}
            </button>
          )}
          <button
            onClick={() => trainMutation.mutate()}
            disabled={trainMutation.isPending || (stats?.n_labeled ?? 0) === 0}
            className="rounded bg-blue-100 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-200 disabled:opacity-50"
          >
            {trainMutation.isPending ? "Training..." : "Train Classifier"}
          </button>
        </div>
      </div>

      {trainMutation.isSuccess && (
        <div className="rounded bg-green-100/10 p-3 text-sm text-green-100">
          Model trained ({trainMutation.data.n_pos} pos, {trainMutation.data.n_neg} neg)
        </div>
      )}
      {trainMutation.isError && (
        <div className="rounded bg-red-50 p-3 text-sm text-red-600">
          {(trainMutation.error as Error).message}
        </div>
      )}

      {/* Review labels mode */}
      {reviewing && (
        <PatchGrid
          imageUrl={(f) => analysisApi.patchUrl(f)}
          images={reviewFilenames.map((f) => ({ filename: f }))}
          labels={reviewLabels}
          onLabel={handleReviewLabel}
          onLabelAll={handleReviewLabelAll}
          title={`Review labels (${reviewFilenames.length} patches)`}
          description="Reviewed labels are autosaved"
          onImageClick={(img: PatchGridImage) => {
            const idx = reviewFilenames.indexOf(img.filename);
            if (idx >= 0) setReviewLightboxIdx(idx);
          }}
        />
      )}

      {reviewLightboxIdx !== null && reviewFilenames[reviewLightboxIdx] && (
        <FrameLightbox
          filename={reviewFilenames[reviewLightboxIdx]}
          queryKey={["patch-source", reviewFilenames[reviewLightboxIdx]]}
          fetchSource={() =>
            analysisApi.patchSource(reviewFilenames[reviewLightboxIdx])
          }
          onClose={() => setReviewLightboxIdx(null)}
          onPrev={reviewLightboxIdx > 0 ? () => setReviewLightboxIdx(reviewLightboxIdx - 1) : null}
          onNext={
            reviewLightboxIdx < reviewFilenames.length - 1
              ? () => setReviewLightboxIdx(reviewLightboxIdx + 1)
              : null
          }
        />
      )}

      {/* Similarity search — only before a model has been trained */}
      {!reviewing && !stats?.has_model && (
        <div className="rounded-lg bg-white p-4 shadow-sm">
          <h3 className="mb-2 text-sm font-semibold text-grey-300">
            Find a seed patch
          </h3>
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1.5">
              <label className="text-xs text-grey-200">Sample size K:</label>
              <input
                type="number"
                min={1}
                value={sampleSize}
                onChange={(e) => setSampleSize(Math.max(1, Number(e.target.value)))}
                className="w-16 rounded border border-grey-100 px-2 py-1 text-sm"
              />
            </div>
            <button
              onClick={() => randomPatchesMutation.mutate()}
              disabled={randomPatchesMutation.isPending}
              className="rounded bg-blue-100 px-3 py-1 text-sm font-medium text-white hover:bg-blue-200 disabled:opacity-50"
            >
              {randomPatches.length > 0 ? "Re-sample" : "Sample random patches"}
            </button>
          </div>

          <div className="flex gap-2">
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Or search by filename..."
              className="grow rounded border border-grey-100 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-100"
              onKeyDown={(e) => {
                if (e.key === "Enter" && searchQuery) searchMutation.mutate(searchQuery);
              }}
            />
            <button
              onClick={() => searchMutation.mutate(searchQuery)}
              disabled={searchMutation.isPending || !searchQuery}
              className="rounded bg-blue-100 px-4 py-2 text-sm font-medium text-white hover:bg-blue-200 disabled:opacity-50"
            >
              Search
            </button>
          </div>
          {searchMutation.isError && (
            <p className="mt-1 text-sm text-red-600">
              {(searchMutation.error as Error).message}
            </p>
          )}
        </div>
      )}

      {/* Random patch selection grid */}
      {!reviewing && !stats?.has_model && randomPatches.length > 0 && !showingSearchResults && (
        <PatchGrid
          imageUrl={(f) => analysisApi.patchUrl(f)}
          images={randomPatches.map((f) => ({ filename: f }))}
          onSelect={handleSelectQueryPatch}
          selectedFilename={selectedQueryPatch ?? undefined}
        />
      )}

      {!reviewing && !stats?.has_model && showingSearchResults && randomPatches.length > 0 && (
        <button
          onClick={handleBackToSelection}
          className="text-sm text-blue-100 hover:underline"
        >
          &larr; Back to patch selection
        </button>
      )}

      {/* Histogram + sample (Entropy Sampling at the boundary) */}
      {!reviewing && histogram && (
        <div className="rounded-lg bg-white p-4 shadow-sm">
          <h3 className="mb-2 text-sm font-semibold text-grey-300">
            Prediction Distribution
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart
              data={histogram.bins}
              onClick={(e) => {
                if (e?.activePayload?.[0]?.payload) {
                  handleHistogramClick(e.activePayload[0].payload);
                }
              }}
            >
              <XAxis dataKey="bin_start" tickFormatter={(v) => v.toFixed(1)} fontSize={11} />
              <YAxis fontSize={11} />
              <Tooltip
                formatter={(value: number) => [value, "Count"]}
                labelFormatter={(label) => `Logit: ${Number(label).toFixed(2)}`}
              />
              <ReferenceLine x={0} stroke="#e94560" strokeDasharray="3 3" />
              <ReferenceLine x={1} stroke="#4caf50" strokeDasharray="3 3" />
              <Bar dataKey="count" fill="#0072CE" />
            </BarChart>
          </ResponsiveContainer>

          <div className="mt-3 flex flex-wrap items-center gap-3">
            <label className="text-sm text-grey-200">Target logit:</label>
            <input
              type="number"
              step="0.1"
              value={targetLogit}
              onChange={(e) => setTargetLogit(Number(e.target.value))}
              className="w-24 rounded border border-grey-100 px-2 py-1 text-sm"
            />
            <label className="text-sm text-grey-200">Sample size K:</label>
            <input
              type="number"
              min={1}
              value={sampleSize}
              onChange={(e) => setSampleSize(Math.max(1, Number(e.target.value)))}
              className="w-20 rounded border border-grey-100 px-2 py-1 text-sm"
            />
            <button
              onClick={() => sampleMutation.mutate()}
              disabled={sampleMutation.isPending}
              className="rounded bg-blue-100 px-3 py-1 text-sm font-medium text-white hover:bg-blue-200 disabled:opacity-50"
            >
              Sample
            </button>
            <span className="text-xs text-grey-200">
              Target logit 0 = Entropy Sampling (decision boundary)
            </span>
          </div>
        </div>
      )}

      {/* Patch grid (search results or logit samples) */}
      {!reviewing && currentImages.length > 0 && (
        <PatchGrid
          imageUrl={(f) => analysisApi.patchUrl(f)}
          images={currentImages}
          labels={savedLabels}
          onLabel={handleLabel}
          onLabelAll={handleLabelAll}
          title={`Annotate ${currentImages.length} Patches`}
          onImageClick={(img: PatchGridImage) => {
            const idx = currentImages.findIndex((c) => c.filename === img.filename);
            if (idx >= 0) setLightboxIdx(idx);
          }}
        />
      )}

      {lightboxIdx !== null && currentImages[lightboxIdx] && (
        <FrameLightbox
          filename={currentImages[lightboxIdx].filename}
          queryKey={["patch-source", currentImages[lightboxIdx].filename]}
          fetchSource={() => analysisApi.patchSource(currentImages[lightboxIdx].filename)}
          subtitle={
            currentImages[lightboxIdx].logit !== undefined
              ? `logit: ${currentImages[lightboxIdx].logit!.toFixed(2)}`
              : currentImages[lightboxIdx].similarity !== undefined
                ? `similarity: ${currentImages[lightboxIdx].similarity!.toFixed(2)}`
                : undefined
          }
          onClose={() => setLightboxIdx(null)}
          onPrev={lightboxIdx > 0 ? () => setLightboxIdx(lightboxIdx - 1) : null}
          onNext={
            lightboxIdx < currentImages.length - 1
              ? () => setLightboxIdx(lightboxIdx + 1)
              : null
          }
        />
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="text-center">
      <div className="text-lg font-bold text-grey-300">{value}</div>
      <div className="text-xs text-grey-200">{label}</div>
    </div>
  );
}
