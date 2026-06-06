import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { analysisApi } from "@/api/client";

export default function TaxonGate({
  onSelected,
}: {
  onSelected: (taxon: string) => void;
}) {
  const [name, setName] = useState("");

  const { data } = useQuery({
    queryKey: ["analyses"],
    queryFn: () => analysisApi.listAnalyses(),
  });

  const selectMutation = useMutation({
    mutationFn: (taxon: string) => analysisApi.select(taxon),
    onSuccess: (res) => onSelected(res.taxon),
  });

  const existing = data?.analyses ?? [];

  return (
    <div className="mx-auto max-w-xl py-16">
      <h1 className="text-2xl font-bold text-grey-300">SeaSwipe</h1>
      <p className="mt-1 text-sm text-grey-200">
        Build a classifier for a single taxon, then evaluate it and
        export its detections. One analysis per taxon.
      </p>

      <div className="mt-8 rounded-lg bg-white p-6 shadow-sm">
        <label className="block text-sm font-semibold text-grey-300">
          Taxon name
        </label>
        <p className="mb-2 text-xs text-grey-200">
          e.g. "Sea pen", "Bamboo coral". This names the analysis and its output
          folder.
        </p>
        <div className="flex gap-2">
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter a taxon name..."
            className="grow rounded border border-grey-100 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-100"
            onKeyDown={(e) => {
              if (e.key === "Enter" && name.trim()) selectMutation.mutate(name.trim());
            }}
          />
          <button
            onClick={() => selectMutation.mutate(name.trim())}
            disabled={!name.trim() || selectMutation.isPending}
            className="rounded bg-blue-100 px-4 py-2 text-sm font-medium text-white hover:bg-blue-200 disabled:opacity-50"
          >
            {selectMutation.isPending ? "Starting..." : "Start analysis"}
          </button>
        </div>
        {selectMutation.isError && (
          <p className="mt-2 text-sm text-red-600">
            {(selectMutation.error as Error).message}
          </p>
        )}
      </div>

      {existing.length > 0 && (
        <div className="mt-6">
          <h2 className="mb-2 text-sm font-semibold text-grey-300">
            Resume an existing analysis
          </h2>
          <div className="space-y-2">
            {existing.map((a) => (
              <button
                key={a.slug}
                onClick={() => selectMutation.mutate(a.taxon)}
                className="flex w-full items-center justify-between rounded-lg border border-grey-100 bg-white px-4 py-3 text-left hover:border-blue-100 hover:bg-blue-50/40"
              >
                <span className="font-medium text-grey-300">{a.taxon}</span>
                <span className="text-xs text-grey-200">
                  {a.n_pos} pos / {a.n_neg} neg
                  {a.has_model ? " · trained" : ""}
                  {a.evaluated ? " · evaluated" : ""}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
