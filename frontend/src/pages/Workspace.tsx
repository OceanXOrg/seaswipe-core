import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { analysisApi } from "@/api/client";
import SpeciesGate from "@/components/SpeciesGate";
import TrainTab from "@/components/TrainTab";
import EvaluateTab from "@/components/EvaluateTab";
import ResultsTab from "@/components/ResultsTab";

const TABS = ["Train", "Evaluate", "Results"] as const;
type Tab = (typeof TABS)[number];

export default function Workspace() {
  const queryClient = useQueryClient();
  const [species, setSpecies] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("Train");

  // On load, resume whatever analysis the server currently has active (if any).
  const { data: current, isLoading } = useQuery({
    queryKey: ["current"],
    queryFn: () => analysisApi.current(),
  });

  useEffect(() => {
    if (current?.active && current.species && species === null) {
      setSpecies(current.species);
    }
  }, [current, species]);

  const handleSelected = (s: string) => {
    // The server's active analysis changed — drop all cached per-analysis data
    // so the tabs refetch for the newly selected species.
    queryClient.removeQueries();
    setActiveTab("Train");
    setSpecies(s);
  };

  if (isLoading) {
    return (
      <div className="mx-auto max-w-xl py-16">
        <div className="h-8 w-48 animate-pulse rounded bg-gray-200" />
      </div>
    );
  }

  if (species === null) {
    return <SpeciesGate onSelected={handleSelected} />;
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-6" key={species}>
      <div className="mb-4 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-grey-300">{species}</h1>
          <p className="text-sm text-grey-200">Agile-modelling analysis</p>
        </div>
        <button
          onClick={() => {
            queryClient.removeQueries();
            setSpecies(null);
          }}
          className="rounded border border-grey-100 px-3 py-1.5 text-sm font-medium text-grey-200 hover:bg-gray-100"
        >
          Change species
        </button>
      </div>

      {/* Tab bar */}
      <div className="mb-6 flex border-b border-gray-200">
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium ${
              activeTab === tab
                ? "border-b-2 border-blue-100 text-blue-100"
                : "text-grey-200 hover:text-grey-300"
            }`}
            role="tab"
            aria-selected={activeTab === tab}
          >
            {tab}
          </button>
        ))}
      </div>

      {activeTab === "Train" && <TrainTab />}
      {activeTab === "Evaluate" && <EvaluateTab />}
      {activeTab === "Results" && <ResultsTab />}
    </div>
  );
}
