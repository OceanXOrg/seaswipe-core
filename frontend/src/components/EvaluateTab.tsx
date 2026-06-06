import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { analysisApi } from "@/api/client";
import PatchGrid, { type PatchGridImage } from "@/components/PatchGrid";
import FrameLightbox from "@/components/FrameLightbox";

export default function EvaluateTab() {
  const queryClient = useQueryClient();
  const [currentBucket, setCurrentBucket] = useState(0);
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);
  const [reviewingEval, setReviewingEval] = useState(false);

  const { data: evalStatus } = useQuery({
    queryKey: ["evalStatus"],
    queryFn: () => analysisApi.evalStatus(),
  });

  const { data: bucketData } = useQuery({
    queryKey: ["evalBucket", currentBucket],
    queryFn: () => analysisApi.evalBucket(currentBucket),
    enabled: evalStatus?.active ?? false,
  });

  const evalStartMutation = useMutation({
    mutationFn: () => analysisApi.evalStart(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["evalStatus"] });
      setCurrentBucket(0);
    },
  });

  const evalAnnotateMutation = useMutation({
    mutationFn: (labels: Record<string, number | null>) =>
      analysisApi.evalAnnotate(labels),
    onMutate: async (labels) => {
      const key = ["evalBucket", currentBucket];
      await queryClient.cancelQueries({ queryKey: key });
      queryClient.setQueryData(key, (old: typeof bucketData) =>
        old ? { ...old, labels: { ...old.labels, ...labels } } : old,
      );
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["evalStatus"] });
      queryClient.invalidateQueries({ queryKey: ["evalBucket", currentBucket] });
    },
  });

  const evalComputeMutation = useMutation({
    mutationFn: () => analysisApi.evalCompute(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["evalStatus"] });
      setReviewingEval(false);
    },
  });

  // ── ROC curve data ────────────────────────────────────────────────────
  const rocCurveData = (() => {
    const buckets = evalStatus?.results?.buckets;
    if (!buckets || buckets.length === 0) return [];
    const estPos = buckets.map((b) => b.p_pos * b.size);
    const estNeg = buckets.map((b) => (1 - b.p_pos) * b.size);
    const totalPos = estPos.reduce((a, b) => a + b, 0);
    const totalNeg = estNeg.reduce((a, b) => a + b, 0);
    if (totalPos === 0 || totalNeg === 0) return [];
    const points = [{ fpr: 0, tpr: 0, threshold: "+∞" }];
    let cumPos = 0;
    let cumNeg = 0;
    for (let i = buckets.length - 1; i >= 0; i--) {
      cumPos += estPos[i];
      cumNeg += estNeg[i];
      points.push({
        fpr: cumNeg / totalNeg,
        tpr: cumPos / totalPos,
        threshold: buckets[i].score_min.toFixed(3),
      });
    }
    return points;
  })();

  const prfCurve = evalStatus?.results?.prf_curve ?? [];
  const bestThreshold = evalStatus?.results?.best_threshold ?? null;
  const bestPoint =
    prfCurve.length > 0 ? prfCurve.reduce((a, b) => (b.f1 > a.f1 ? b : a)) : null;
  const countError = evalStatus?.results?.count_error ?? null;

  return (
    <div className="space-y-6">
      {/* ROC-AUC Section */}
      <section className="rounded-lg bg-white p-4 shadow-sm">
        <h3 className="mb-2 text-lg font-semibold text-grey-300">ROC-AUC Evaluation</h3>
        <div className="mb-4 space-y-2 text-sm text-grey-200">
          <p>
            <span className="font-medium">What it measures.</span> For every patch
            the model produces a score — higher means "more likely to be the target
            taxon." ROC-AUC is the probability that a real positive patch gets a
            higher score than a real negative one. <span className="font-medium">1.0</span>{" "}
            means the model always ranks the target above non-targets,{" "}
            <span className="font-medium">0.5</span> means it's guessing.
          </p>
          <p>
            <span className="font-medium">How we compute it.</span> We sort all the
            model's scores and split them into 6 buckets (lowest-scoring → highest-
            scoring). You label 40 random patches from each bucket as positive or
            negative (240 patches total); the same labels also power the
            precision/recall curve and the Count Error Estimation below — no second
            annotation round needed.
          </p>
        </div>

        {!evalStatus?.active ? (
          <button
            onClick={() => evalStartMutation.mutate()}
            disabled={evalStartMutation.isPending}
            className="rounded bg-blue-100 px-4 py-2 text-sm font-medium text-white hover:bg-blue-200 disabled:opacity-50"
          >
            Start Evaluation
          </button>
        ) : (
          <div className="space-y-4">
            {(!evalStatus.results || reviewingEval) && (
              <div className="flex flex-wrap gap-2">
                {evalStatus.progress?.map((p) => (
                  <button
                    key={p.bucket}
                    onClick={() => setCurrentBucket(p.bucket)}
                    className={`rounded px-3 py-1 text-xs font-medium ${
                      currentBucket === p.bucket
                        ? "bg-blue-100 text-white"
                        : p.labeled === p.total
                          ? "bg-green-100 text-white"
                          : "bg-gray-200 text-grey-300"
                    }`}
                  >
                    Bucket {p.bucket} ({p.labeled}/{p.total})
                  </button>
                ))}
              </div>
            )}

            {(!evalStatus.results || reviewingEval) && bucketData && (
              <div>
                <p className="mb-2 text-sm text-grey-200">
                  Bucket {bucketData.bucket}: score range [
                  {bucketData.score_min.toFixed(2)}, {bucketData.score_max.toFixed(2)}]
                  — {bucketData.size} total patches
                </p>
                <PatchGrid
                  imageUrl={(f) => analysisApi.patchUrl(f)}
                  images={bucketData.images.map((f) => ({ filename: f }))}
                  labels={bucketData.labels}
                  onLabel={(fn, label) => evalAnnotateMutation.mutate({ [fn]: label })}
                  onLabelAll={(label) => {
                    const payload: Record<string, number | null> = {};
                    for (const f of bucketData.images) payload[f] = label;
                    evalAnnotateMutation.mutate(payload);
                  }}
                  onImageClick={(img: PatchGridImage) => {
                    const idx = bucketData.images.indexOf(img.filename);
                    if (idx >= 0) setLightboxIdx(idx);
                  }}
                />

                {lightboxIdx !== null && bucketData.images[lightboxIdx] && (
                  <FrameLightbox
                    filename={bucketData.images[lightboxIdx]}
                    queryKey={["patch-source", bucketData.images[lightboxIdx]]}
                    fetchSource={() =>
                      analysisApi.patchSource(bucketData.images[lightboxIdx])
                    }
                    onClose={() => setLightboxIdx(null)}
                    onPrev={lightboxIdx > 0 ? () => setLightboxIdx(lightboxIdx - 1) : null}
                    onNext={
                      lightboxIdx < bucketData.images.length - 1
                        ? () => setLightboxIdx(lightboxIdx + 1)
                        : null
                    }
                  />
                )}
              </div>
            )}

            <div className="flex gap-2">
              {(!evalStatus.results || reviewingEval) && (
                <button
                  onClick={() => evalComputeMutation.mutate()}
                  disabled={evalComputeMutation.isPending}
                  className="rounded bg-blue-200 px-4 py-2 text-sm font-medium text-white hover:bg-blue-300 disabled:opacity-50"
                >
                  {evalComputeMutation.isPending
                    ? "Computing..."
                    : reviewingEval
                      ? "Re-compute"
                      : "Compute Evaluation"}
                </button>
              )}
              {evalStatus.results && !reviewingEval && (
                <button
                  onClick={() => setReviewingEval(true)}
                  className="rounded border border-blue-100 px-4 py-2 text-sm font-medium text-blue-100 hover:bg-blue-50"
                >
                  Review Labels
                </button>
              )}
              {evalStatus.results && reviewingEval && (
                <button
                  onClick={() => setReviewingEval(false)}
                  className="rounded border border-blue-100 px-4 py-2 text-sm font-medium text-blue-100 hover:bg-blue-50"
                >
                  Exit Label Review
                </button>
              )}
            </div>

            {evalStatus.results && (
              <div className="rounded bg-blue-50/30 p-4">
                <p className="mb-20 text-lg font-bold text-grey-300">
                  ROC-AUC: {evalStatus.results.roc_auc.toFixed(4)}
                </p>
                {rocCurveData.length > 0 && (
                  <ResponsiveContainer width="100%" height={360}>
                    <LineChart
                      data={rocCurveData}
                      margin={{ top: 10, right: 20, bottom: 30, left: 30 }}
                    >
                      <XAxis
                        dataKey="fpr"
                        type="number"
                        domain={[0, 1]}
                        label={{ value: "False Positive Rate", position: "bottom" }}
                        tickFormatter={(v: number) => v.toFixed(1)}
                        fontSize={11}
                      />
                      <YAxis
                        dataKey="tpr"
                        type="number"
                        domain={[0, 1]}
                        label={{
                          value: "True Positive Rate",
                          angle: -90,
                          position: "insideLeft",
                          style: { textAnchor: "middle" },
                          offset: 10,
                        }}
                        tickFormatter={(v: number) => v.toFixed(1)}
                        fontSize={11}
                      />
                      <Tooltip
                        formatter={(v: number) => v.toFixed(3)}
                        labelFormatter={(_, payload) => {
                          const p = payload?.[0]?.payload;
                          return p ? `threshold ≥ ${p.threshold}` : "";
                        }}
                      />
                      <ReferenceLine
                        segment={[
                          { x: 0, y: 0 },
                          { x: 1, y: 1 },
                        ]}
                        stroke="#ccc"
                        strokeDasharray="3 3"
                        label={{
                          value: "random guess",
                          fill: "#9ca3af",
                          fontSize: 11,
                          position: "middle",
                        }}
                      />
                      <Line type="linear" dataKey="tpr" stroke="#0072CE" strokeWidth={2} dot />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            )}
          </div>
        )}
      </section>

      {/* PRF Section */}
      <section className="rounded-lg bg-white p-4 shadow-sm">
        <h3 className="mb-2 text-lg font-semibold text-grey-300">
          Precision / Recall / F1 vs Threshold
        </h3>
        <div className="mb-4 space-y-2 text-sm text-grey-200">
          <p>
            <span className="font-medium">What it shows.</span> Precision is the
            fraction of flagged patches that are real; recall is the fraction of real
            targets the model flags. F1 is their harmonic mean. The threshold that
            maximises F1 is suggested as the operating point and used by Count Error
            Estimation below.
          </p>
          <p>
            <span className="font-medium">How we compute it.</span> Using the same
            240 bucket annotations, we estimate the positive rate P(+|b) in each
            bucket and apportion it across the score distribution to get estimated
            TP / FP / FN at every candidate threshold. Method follows Navine et al.
            (2024).
          </p>
        </div>

        {!evalStatus?.results ? (
          <p className="text-sm text-grey-200">
            Compute the evaluation first to see the precision/recall/F1 plot.
          </p>
        ) : prfCurve.length === 0 ? (
          <p className="text-sm text-grey-200">
            Not enough labels to compute precision/recall — make sure each bucket has
            at least one positive label.
          </p>
        ) : (
          <div className="rounded bg-blue-50/30 p-4">
            {bestPoint && (
              <p className="mb-4 text-sm text-grey-300">
                <span className="font-bold">
                  Suggested threshold: {bestPoint.threshold.toFixed(2)}
                </span>{" "}
                — best F1 = {bestPoint.f1.toFixed(3)} (precision{" "}
                {bestPoint.precision.toFixed(3)}, recall {bestPoint.recall.toFixed(3)}).
              </p>
            )}
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={prfCurve} margin={{ top: 10, right: 20, bottom: 30, left: 30 }}>
                <XAxis
                  dataKey="threshold"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  label={{ value: "Confidence threshold (logit)", position: "bottom" }}
                  tickFormatter={(v: number) => v.toFixed(1)}
                  fontSize={11}
                />
                <YAxis
                  type="number"
                  domain={[0, 1]}
                  label={{
                    value: "Score",
                    angle: -90,
                    position: "insideLeft",
                    style: { textAnchor: "middle" },
                    offset: 10,
                  }}
                  tickFormatter={(v: number) => v.toFixed(1)}
                  fontSize={11}
                />
                <Tooltip
                  formatter={(v: number) => v.toFixed(3)}
                  labelFormatter={(v: number) => `threshold = ${Number(v).toFixed(3)}`}
                />
                {bestPoint && (
                  <ReferenceLine
                    x={bestPoint.threshold}
                    stroke="#10b981"
                    strokeDasharray="4 4"
                    label={{ value: "best", fill: "#10b981", fontSize: 11, position: "top" }}
                  />
                )}
                <Line type="monotone" dataKey="precision" stroke="#9ca3af" strokeWidth={1.5} dot={false} name="Precision" />
                <Line type="monotone" dataKey="recall" stroke="#fbbf24" strokeWidth={1.5} dot={false} name="Recall" />
                <Line type="monotone" dataKey="f1" stroke="#0072CE" strokeWidth={2} dot={false} name="F1" />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-2 flex justify-center gap-4 text-xs text-grey-200">
              <span className="flex items-center gap-1">
                <span className="inline-block h-0.5 w-4 bg-[#0072CE]" /> F1
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-0.5 w-4 bg-[#9ca3af]" /> Precision
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-0.5 w-4 bg-[#fbbf24]" /> Recall
              </span>
            </div>
          </div>
        )}
      </section>

      {/* Count Error Estimation Section */}
      <section className="rounded-lg bg-white p-4 shadow-sm">
        <h3 className="mb-2 text-lg font-semibold text-grey-300">
          Count Error Estimation
        </h3>
        <div className="mb-4 space-y-2 text-sm text-grey-200">
          <p>
            <span className="font-medium">What it tells you.</span> Of the organisms
            in this dataset, how many did the model{" "}
            <span className="font-medium text-green-600">correctly find</span>, how
            many did it <span className="font-medium text-red-500">miss</span>, and
            how many of its detections are actually{" "}
            <span className="font-medium text-amber-500">wrong</span>. Numbers are
            estimated counts over the whole dataset, not just the patches you
            annotated.
          </p>
          <p>
            <span className="font-medium">How we compute it.</span> The model flags
            every patch with logit ≥{" "}
            <span className="font-mono font-bold">
              {bestThreshold !== null && bestThreshold !== undefined
                ? bestThreshold.toFixed(2)
                : "—"}
            </span>{" "}
            (the F1-optimal threshold). <em>False positives</em> = the fraction of
            your 240 annotations above that threshold that are negative, extrapolated
            to the full above-threshold population. <em>False negatives</em> = within
            each bucket, the estimated fraction of positives multiplied by the
            patches below threshold (Navine et al. 2024).
          </p>
        </div>

        {!evalStatus?.results ? (
          <p className="text-sm text-grey-200">
            Compute the evaluation first to see the count error estimates.
          </p>
        ) : !countError ? (
          <p className="text-sm text-grey-200">Not enough labels to estimate count error.</p>
        ) : (
          (() => {
            const fmt = (n: number | null | undefined) =>
              n === null || n === undefined || Number.isNaN(n)
                ? "—"
                : Math.round(n).toLocaleString();
            const pct = (n: number | null | undefined) =>
              n === null || n === undefined || Number.isNaN(n) ? "—" : `${n.toFixed(1)}%`;

            const tp = countError.est_true_positives ?? 0;
            const fp = countError.est_false_positives ?? 0;
            const fn = countError.est_false_negatives;
            const total = countError.est_total_organisms ?? tp + fp + fn;
            const safeTotal = total > 0 ? total : 1;

            const cards = [
              {
                key: "tp",
                label: "Correctly classified",
                sublabel: "True positives — organisms the model found",
                count: countError.est_true_positives,
                pctVal: countError.tp_pct,
                color: "#059669",
                bg: "rgba(16, 185, 129, 0.12)",
                icon: "✓",
              },
              {
                key: "fp",
                label: "Incorrectly classified",
                sublabel: "False positives — detections that are not the target",
                count: countError.est_false_positives,
                pctVal: countError.fp_pct,
                color: "#d97706",
                bg: "rgba(251, 191, 36, 0.18)",
                icon: "!",
              },
              {
                key: "fn",
                label: "Missed",
                sublabel: "False negatives — organisms the model did not flag",
                count: countError.est_false_negatives,
                pctVal: countError.fn_pct,
                color: "#dc2626",
                bg: "rgba(239, 68, 68, 0.12)",
                icon: "✗",
              },
            ] as const;

            const fpTpUnknown =
              countError.est_true_positives === null ||
              countError.est_false_positives === null;

            return (
              <div className="rounded bg-blue-50/30 p-4">
                <p className="mb-4 text-sm text-grey-300">
                  Using the F1-optimal threshold{" "}
                  <span className="font-mono font-bold">{countError.threshold.toFixed(2)}</span>.
                  Estimated population: <span className="font-bold">{fmt(total)}</span>{" "}
                  organisms across the dataset.
                </p>

                {fpTpUnknown && (
                  <p className="mb-4 rounded bg-amber-50 px-3 py-2 text-xs text-amber-700">
                    No annotated patches landed above the suggested threshold, so
                    true/false positives cannot be estimated. Missed organisms (false
                    negatives) are still shown — they don't depend on the
                    above-threshold sample.
                  </p>
                )}

                <div className="mb-6 flex h-6 overflow-hidden rounded border border-grey-100 shadow-inner">
                  {cards.map((c) => {
                    const v = c.count ?? 0;
                    const w = Number.isNaN(v) ? 0 : (v / safeTotal) * 100;
                    return (
                      <div
                        key={c.key}
                        style={{ width: `${w}%`, background: c.color }}
                        title={`${c.label}: ${fmt(c.count)} (${pct(c.pctVal)})`}
                      />
                    );
                  })}
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  {cards.map((c) => (
                    <div key={c.key} className="rounded-lg p-4" style={{ background: c.bg }}>
                      <div className="flex items-center justify-between">
                        <span
                          className="flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold text-white"
                          style={{ background: c.color }}
                        >
                          {c.icon}
                        </span>
                        <span className="text-xs font-medium" style={{ color: c.color }}>
                          {pct(c.pctVal)}
                        </span>
                      </div>
                      <div className="mt-2 text-3xl font-bold" style={{ color: c.color }}>
                        {fmt(c.count)}
                      </div>
                      <div className="mt-1 text-sm font-medium text-grey-300">{c.label}</div>
                      <div className="mt-1 text-xs text-grey-200">{c.sublabel}</div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 text-xs text-grey-200">
                  Based on {countError.n_annotated_above_threshold} of your 240
                  annotated patches with logit ≥ {countError.threshold.toFixed(2)} (
                  {countError.n_annotated_above_false_positive} marked negative,{" "}
                  {countError.n_annotated_above_true_positive} marked positive) —
                  observed false-positive rate{" "}
                  {countError.observed_fpr === null
                    ? "—"
                    : `${(countError.observed_fpr * 100).toFixed(1)}%`}
                  , extrapolated to the {fmt(countError.n_above_threshold)} patches
                  above threshold.
                </div>
              </div>
            );
          })()
        )}
      </section>
    </div>
  );
}
