import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import "@/index.css";
import Workspace from "@/pages/Workspace";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <div className="flex min-h-screen flex-col">
        <div className="flex-1">
          <Workspace />
        </div>
        <footer className="border-t border-gray-200 py-4 text-center text-sm text-grey-200">
          © OceanX 2026
        </footer>
      </div>
    </QueryClientProvider>
  </React.StrictMode>,
);
