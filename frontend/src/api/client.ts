/**
 * API client — typed fetch wrappers for the SeaSwipe analysis backend.
 *
 * The backend serves a single active analysis (selected by taxon), so none of
 * these calls take an analysis id.
 */

const BASE = "/api";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// ── Types ──────────────────────────────────────────────────────────────────

export interface AnalysisSummary {
  taxon: string;
  slug: string;
  run_id: string;
  n_pos: number;
  n_neg: number;
  has_model: boolean;
  evaluated: boolean;
}

export interface Current {
  active: boolean;
  taxon?: string;
  slug?: string;
  run_id?: string;
  has_model?: boolean;
}

export interface SearchResult {
  query: string;
  neighbors: { index: number; filename: string; similarity: number }[];
}

export interface TrainResult {
  n_pos: number;
  n_neg: number;
}

export interface HistogramData {
  bins: { bin_start: number; bin_end: number; count: number }[];
  min: number;
  max: number;
  mean: number;
  n_positive: number;
  n_negative: number;
}

export interface SampleResult {
  images: { filename: string; logit: number }[];
  target_logit: number;
}

export interface AnnotateResult {
  n_pos: number;
  n_neg: number;
  n_total: number;
}

export interface Stats {
  n_images: number;
  n_pos: number;
  n_neg: number;
  n_labeled: number;
  has_model: boolean;
}

export interface PrfPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  est_tp: number;
  est_fp: number;
  est_fn: number;
}

export interface CountError {
  threshold: number;
  n_above_threshold: number;
  n_annotated_above_threshold: number;
  n_annotated_above_false_positive: number;
  n_annotated_above_true_positive: number;
  observed_fpr: number | null;
  est_true_positives: number | null;
  est_false_positives: number | null;
  est_false_negatives: number;
  est_total_organisms: number | null;
  tp_pct: number | null;
  fp_pct: number | null;
  fn_pct: number | null;
}

export interface EvalStatus {
  active: boolean;
  n_buckets?: number;
  progress?: { bucket: number; labeled: number; total: number }[];
  total_labeled?: number;
  total_samples?: number;
  results?: {
    roc_auc: number;
    buckets: {
      bucket: number;
      size: number;
      score_min: number;
      score_max: number;
      n_sampled: number;
      n_pos: number;
      n_neg: number;
      p_pos: number;
    }[];
    prf_curve?: PrfPoint[];
    best_threshold?: number | null;
    count_error?: CountError | null;
  };
}

export interface BucketData {
  bucket: number;
  n_buckets: number;
  size: number;
  score_min: number;
  score_max: number;
  images: string[];
  labels: Record<string, number>;
}

export interface PositivesData {
  total: number;
  page: number;
  page_size: number;
  threshold: number;
  patches: { filename: string; logit: number }[];
}

export interface PatchSource {
  frame_filename: string;
  frame_url: string | null;
  bbox: [number, number, number, number]; // [x, y, w, h]
  frame_width: number;
  frame_height: number;
}

// ── API ────────────────────────────────────────────────────────────────────

export const analysisApi = {
  listAnalyses: () =>
    request<{ analyses: AnalysisSummary[] }>("/analyses"),
  select: (taxon: string) =>
    request<{ taxon: string; slug: string; run_id: string }>("/select", {
      method: "POST",
      body: JSON.stringify({ taxon }),
    }),
  current: () => request<Current>("/current"),
  defaults: () =>
    request<{
      initialisation_nearest_neighbours: number;
      initialisation_n_dissimilar: number;
      default_sample_size: number;
    }>("/defaults"),
  stats: () => request<Stats>("/stats"),

  // Search & annotate
  randomPatches: (n = 20) =>
    request<{ filenames: string[] }>(`/random-patches?n=${n}`),
  search: (query: string, k = 15, n_dissimilar = 5) =>
    request<SearchResult>("/search", {
      method: "POST",
      body: JSON.stringify({ query, k, n_dissimilar }),
    }),
  labels: () => request<{ labels: Record<string, number> }>("/labels"),
  annotate: (labels: Record<string, number | null>) =>
    request<AnnotateResult>("/annotate", {
      method: "POST",
      body: JSON.stringify({ labels }),
    }),

  // Train & sample
  train: () => request<TrainResult>("/train", { method: "POST" }),
  histogram: () => request<HistogramData>("/histogram"),
  sample: (target_logit = 0, n = 20) =>
    request<SampleResult>("/sample", {
      method: "POST",
      body: JSON.stringify({ target_logit, n }),
    }),

  // Evaluation
  evalStart: (n_buckets = 6, k = 40) =>
    request("/eval/start", {
      method: "POST",
      body: JSON.stringify({ n_buckets, k }),
    }),
  evalBucket: (bucketIdx: number) =>
    request<BucketData>(`/eval/bucket/${bucketIdx}`),
  evalAnnotate: (labels: Record<string, number | null>) =>
    request("/eval/annotate", {
      method: "POST",
      body: JSON.stringify({ labels }),
    }),
  evalCompute: () => request("/eval/compute", { method: "POST" }),
  evalStatus: () => request<EvalStatus>("/eval/status"),

  // Predicted positives
  positives: (
    threshold = 1.0,
    page = 0,
    pageSize = 24,
    sort: "desc" | "asc" = "desc",
  ) =>
    request<PositivesData>(
      `/positives?threshold=${threshold}&page=${page}&page_size=${pageSize}&sort=${sort}`,
    ),

  // Patch image URL helper + source frame
  patchUrl: (filename: string) =>
    `${BASE}/patches/${encodeURIComponent(filename)}`,
  patchSource: (filename: string) =>
    request<PatchSource>(`/patches/${encodeURIComponent(filename)}/source`),
};
