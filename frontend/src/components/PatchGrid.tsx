import type { ReactNode } from "react";

export interface PatchGridImage {
  filename: string;
  logit?: number;
  similarity?: number;
}

interface PatchGridProps {
  imageUrl: (filename: string) => string;
  images: PatchGridImage[];
  labels?: Record<string, number | null>;
  onLabel?: (filename: string, label: number | null) => void;
  /** Bulk-annotate all images. Only rendered when annotating (onLabel set). */
  onLabelAll?: (label: number) => void;
  onSelect?: (filename: string) => void;
  selectedFilename?: string;
  title?: string;
  /** Optional short description shown below the title. */
  description?: string;
  /** Optional content rendered to the right of the title (e.g. controls). */
  headerExtra?: ReactNode;
  /** Click handler on the image (used for lightboxes). Ignored in select mode. */
  onImageClick?: (img: PatchGridImage) => void;
}

export default function PatchGrid({
  imageUrl,
  images,
  labels = {},
  onLabel,
  onLabelAll,
  onSelect,
  selectedFilename,
  title,
  description,
  headerExtra,
  onImageClick,
}: PatchGridProps) {
  const isSelectMode = !!onSelect;
  const heading =
    title ??
    (isSelectMode
      ? `Select a patch (${images.length} patches)`
      : `Annotate ${images.length} patches`);

  return (
    <div className="rounded-lg bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-grey-300">{heading}</h3>
        <div className="flex items-center gap-2">
          {!isSelectMode && onLabel && onLabelAll && images.length > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-grey-200">Mark all:</span>
              <button
                type="button"
                onClick={() => onLabelAll(1)}
                className="flex h-7 items-center gap-1 rounded-full bg-white px-2 text-xs font-medium text-green-100 shadow ring-1 ring-green-100 hover:bg-green-100 hover:text-white"
                aria-label="Mark all as positive"
              >
                <span className="text-sm font-bold">&#10003;</span>
                Positive
              </button>
              <button
                type="button"
                onClick={() => onLabelAll(0)}
                className="flex h-7 items-center gap-1 rounded-full bg-white px-2 text-xs font-medium text-red-500 shadow ring-1 ring-red-500 hover:bg-red-500 hover:text-white"
                aria-label="Mark all as negative"
              >
                <span className="text-sm font-bold">&#10005;</span>
                Negative
              </button>
            </div>
          )}
          {headerExtra}
        </div>
      </div>
      {description && (
        <p className="mt-1 mb-3 text-xs text-grey-200">{description}</p>
      )}
      {!description && <div className="mb-3" />}
      <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
        {images.map((img) => {
          const label = labels[img.filename];
          const isSelected = selectedFilename === img.filename;
          const ringClass = isSelectMode
            ? isSelected
              ? "ring-2 ring-blue-100"
              : ""
            : label === 1
              ? "ring-2 ring-green-100"
              : label === 0
                ? "ring-2 ring-red-500"
                : "";

          const imageClickable = isSelectMode || !!onImageClick;
          const handleImageClick = isSelectMode
            ? () => onSelect!(img.filename)
            : onImageClick
              ? () => onImageClick(img)
              : undefined;

          return (
            <div
              key={img.filename}
              className={`group relative rounded-lg ${
                isSelectMode
                  ? "cursor-pointer hover:ring-2 hover:ring-blue-100/50"
                  : ""
              } ${ringClass}`}
              onClick={isSelectMode ? handleImageClick : undefined}
            >
              <div className="overflow-hidden rounded-lg bg-gray-100">
                <img
                  src={imageUrl(img.filename)}
                  alt={img.filename}
                  className={`aspect-square w-full object-cover ${
                    imageClickable && !isSelectMode ? "cursor-pointer" : ""
                  }`}
                  onClick={!isSelectMode ? handleImageClick : undefined}
                  loading="lazy"
                />
              </div>

              {/* Label buttons — only in annotation mode */}
              {!isSelectMode && onLabel && (
                <div className="absolute right-1.5 top-1.5 flex gap-1.5">
                  <button
                    onClick={() => onLabel(img.filename, label === 1 ? null : 1)}
                    className={`flex h-8 w-8 items-center justify-center rounded-full text-base font-bold shadow ${
                      label === 1
                        ? "bg-green-100 text-white"
                        : "bg-white/80 text-green-100 hover:bg-green-100 hover:text-white"
                    }`}
                    aria-label={`Mark ${img.filename} as positive`}
                  >
                    &#10003;
                  </button>
                  <button
                    onClick={() => onLabel(img.filename, label === 0 ? null : 0)}
                    className={`flex h-8 w-8 items-center justify-center rounded-full text-base font-bold shadow ${
                      label === 0
                        ? "bg-red-500 text-white"
                        : "bg-white/80 text-red-500 hover:bg-red-500 hover:text-white"
                    }`}
                    aria-label={`Mark ${img.filename} as negative`}
                  >
                    &#10005;
                  </button>
                </div>
              )}

              {img.logit !== undefined && (
                <div className="pointer-events-none absolute bottom-1.5 right-1.5 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] leading-tight text-white shadow">
                  {img.logit.toFixed(2)}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
