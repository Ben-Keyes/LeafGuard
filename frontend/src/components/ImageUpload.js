import { useRef, useState } from "react";
// This is where the magic happens!
export default function ImageUpload() {
  const fileInputRef = useRef(null);

  // What the user picked
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  // What the server returned
  const [prediction, setPrediction] = useState(null);

  // UI state
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  // URL for Render build
  const PREDICT_URL = "/api/predict";

  const clearResultState = () => {
    setPrediction(null);
    setErrorMessage("");
  };

  const setImageFromFile = (file) => {
    if (!file) return;

    clearResultState();
    setSelectedImage(file);

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files?.[0];
    setImageFromFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    setImageFromFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const resetAll = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setPrediction(null);
    setErrorMessage("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const uploadAndPredict = async () => {
    if (!selectedImage) {
      alert("Please choose an image first.");
      return;
    }

    clearResultState();
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);

      const response = await fetch(PREDICT_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(text || `Request failed (${response.status})`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.error("Upload error:", err);
      setErrorMessage(err?.message || "Something went wrong while uploading.");
    } finally {
      setIsLoading(false);
    }
  };

  // Handles collecting response even if in different cases (snake_case and camelCase)
  const predictedClass =
    prediction?.predicted_class ?? prediction?.predictedClass ?? null;
  const className = prediction?.class_name ?? prediction?.className ?? null;
  const confidence =
    prediction?.confidence ??
    prediction?.confidence_score ??
    prediction?.confidenceScore ??
    null;

  // Low confidence thresholding (< 0.65)
  const threshold = prediction?.threshold ?? 0.65;
  const isConfident =
    prediction?.is_confident ??
    prediction?.isConfident ??
    (confidence == null ? null : confidence >= threshold);

  return (
    <div className="relative min-h-[calc(100vh-4rem)] px-4 py-10 overflow-hidden">
  {/* Background image layer */}
  <div
    className="absolute inset-0 bg-center bg-cover opacity-50"
    style={{ backgroundImage: "url('/assets/leaves-bg.png')" }}
  />

  {/* soft color overlay */}
  <div className="absolute inset-0 bg-gradient-to-b from-emerald-50 via-white to-emerald-50 opacity-80" />

  {/* Actual content */}
  <div className="relative z-10">

      <div className="mx-auto w-full max-w-5xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight text-emerald-950">
            LeafGuard
          </h1>
          <p className="mt-2 text-emerald-900/70">
            Upload a photo of a leaf to predict the most likely disease.
            If the model provides confidence, it’ll show up here too.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          {/* Upload panel */}
          <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-emerald-100">
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              className={[
                "rounded-2xl border-2 border-dashed p-6 transition",
                isDragging
                  ? "border-emerald-400 bg-emerald-50"
                  : "border-emerald-100 hover:border-emerald-200",
              ].join(" ")}
            >
              <div className="flex flex-col items-center justify-center gap-3 text-center">
                <div className="rounded-full bg-emerald-100 px-4 py-2 text-sm text-emerald-900">
                  Drag and drop an image here
                </div>
                <div className="text-sm text-emerald-900/60">or</div>

                <div className="flex flex-wrap items-center justify-center gap-3">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="rounded-xl bg-emerald-900 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-800"
                  >
                    Choose image
                  </button>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileInputChange}
                    className="hidden"
                  />

                  <button
                    onClick={uploadAndPredict}
                    disabled={!selectedImage || isLoading}
                    className={[
                      "rounded-xl px-4 py-2 text-sm font-medium transition",
                      !selectedImage || isLoading
                        ? "cursor-not-allowed bg-emerald-100 text-emerald-900/50"
                        : "bg-emerald-600 text-white hover:bg-emerald-700",
                    ].join(" ")}
                  >
                    {isLoading ? "Analyzing…" : "Upload & predict"}
                  </button>

                  <button
                    type="button"
                    onClick={resetAll}
                    disabled={!selectedImage && !prediction && !errorMessage}
                    className={[
                      "rounded-xl border px-4 py-2 text-sm font-medium transition",
                      !selectedImage && !prediction && !errorMessage
                        ? "cursor-not-allowed border-emerald-100 text-emerald-900/30"
                        : "border-emerald-200 text-emerald-950 hover:bg-emerald-50",
                    ].join(" ")}
                  >
                    Reset
                  </button>
                </div>

                <p className="text-xs text-emerald-900/60">
                  Tip: close-up leaf, good lighting, minimal background works
                  best.
                </p>
              </div>

              {/* File + preview */}
              <div className="mt-6 grid gap-4 sm:grid-cols-2">
                <div className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-3">
                  <div className="text-xs text-emerald-900/60">
                    Selected image
                  </div>
                  <div className="mt-1 truncate text-sm font-medium text-emerald-950">
                    {selectedImage ? selectedImage.name : "None yet"}
                  </div>
                  {selectedImage && (
                    <div className="mt-1 text-xs text-emerald-900/60">
                      {(selectedImage.size / (1024 * 1024)).toFixed(2)} MB
                    </div>
                  )}
                </div>

                <div className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-3">
                  <div className="text-xs text-emerald-900/60">Preview</div>
                  <div className="mt-2 aspect-video overflow-hidden rounded-lg border border-emerald-100 bg-white">
                    {previewUrl ? (
                      <img
                        src={previewUrl}
                        alt="Selected leaf preview"
                        className="h-full w-full object-contain"
                      />
                    ) : (
                      <div className="flex h-full items-center justify-center text-xs text-emerald-900/50">
                        No image selected
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Error */}
            {errorMessage && (
              <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4">
                <div className="text-sm font-semibold text-red-800">
                  Couldn’t analyze that image
                </div>
                <div className="mt-1 text-sm text-red-700">{errorMessage}</div>
              </div>
            )}
          </div>

          {/* Results panel */}
          <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-emerald-100">
            <h2 className="text-lg font-semibold text-emerald-950">Result</h2>
            <p className="mt-1 text-sm text-emerald-900/70">
              Your prediction will show here after you upload an image.
            </p>

            {!prediction && !isLoading && (
              <div className="mt-6 rounded-xl border border-emerald-100 bg-emerald-50/50 p-4 text-sm text-emerald-900/70">
                No result yet — choose an image and click{" "}
                <span className="font-medium">Upload & predict</span>.
              </div>
            )}

            {isLoading && (
              <div className="mt-6 rounded-xl border border-emerald-100 bg-emerald-50/50 p-4 text-sm text-emerald-900/70">
                Thinking… this can take a few seconds on local CPU.
              </div>
            )}

            {prediction && (
              <div className="mt-6 space-y-4">
                <div className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-4">
                  <div className="text-xs text-emerald-900/60">
                    Predicted disease
                  </div>
                  <div className="mt-1 text-base font-semibold text-emerald-950">
                    {className ?? "—"}
                  </div>

                  {/* confidence threshold warning */}
                  {isConfident === false && (
                    <div className="mt-3 rounded-lg border border-yellow-200 bg-yellow-50 px-3 py-2 text-sm text-yellow-900">
                      Low confidence — try a closer photo with better lighting and less background.
                    </div>
                  )}

                  <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                    <div className="rounded-lg bg-white p-3 ring-1 ring-emerald-100">
                      <div className="text-xs text-emerald-900/60">
                        Class index
                      </div>
                      <div className="mt-1 font-medium text-emerald-950">
                        {predictedClass ?? "—"}
                      </div>
                    </div>

                    {/* Confidence metric with badge and bar */}
                    <div className="rounded-lg bg-white p-3 ring-1 ring-emerald-100">
                      <div className="text-xs text-emerald-900/60">
                        Confidence
                      </div>

                      {confidence == null ? (
                        <div className="mt-1 font-medium text-emerald-950">—</div>
                      ) : (
                        <div className="mt-2">
                          <div className="flex items-center justify-between">
                            <div className="font-medium text-emerald-950">
                              {(confidence * 100).toFixed(1)}%
                            </div>

                            <span
                              className={[
                                "rounded-full px-2 py-0.5 text-xs font-medium",
                                confidence >= 0.85
                                  ? "bg-emerald-100 text-emerald-900"
                                  : confidence >= threshold
                                  ? "bg-amber-100 text-amber-900"
                                  : "bg-red-100 text-red-900",
                              ].join(" ")}
                            >
                              {confidence >= 0.85
                                ? "High"
                                : confidence >= threshold
                                ? "Medium"
                                : "Low"}
                            </span>
                          </div>

                          <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-emerald-50 ring-1 ring-emerald-100">
                            <div
                              className="h-full rounded-full bg-emerald-600"
                              style={{
                                width: `${
                                  Math.max(0, Math.min(1, confidence)) * 100
                                }%`,
                              }}
                            />
                          </div>

                          {confidence < threshold && (
                            <p className="mt-2 text-xs text-emerald-900/70">
                              Low confidence — try a clearer photo (good lighting, leaf fills the frame).
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* The top 3 predictions*/}
                  {prediction?.top3 && Array.isArray(prediction.top3) && (
                    <div className="mt-4 rounded-xl border border-emerald-100 bg-white p-3">
                      <div className="text-xs font-semibold text-emerald-900">
                        Top 3 predictions
                      </div>

                      <div className="mt-2 space-y-2">
                        {prediction.top3.map((p, i) => (
                          <div
                            key={i}
                            className="flex items-center justify-between rounded-lg bg-emerald-50/50 px-3 py-2 text-sm"
                          >
                            <div className="font-medium text-emerald-950">
                              #{i + 1}: {p.class_name}
                            </div>
                            <div className="text-emerald-900/70">
                              {(p.confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Raw JSON*/}
                <details className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-4">
                  <summary className="cursor-pointer text-xs font-semibold text-emerald-900">
                    Show raw response
                  </summary>
                  <pre className="mt-3 overflow-auto text-xs text-emerald-950">
                    {JSON.stringify(prediction, null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </div>
        </div>

        <p className="mt-8 text-center text-xs text-emerald-900/40">
          Local demo • React + Tailwind • Backend served via Docker :)
        </p>
      </div>
    </div>
    </div>
  );
}
