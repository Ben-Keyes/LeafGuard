import ImageUpload from "./components/ImageUpload";

function App() {
  return (
    <div className="relative min-h-screen bg-emerald-50">
      {/* Leaf bacground */}
      <div
        className="pointer-events-none fixed inset-0 z-0 opacity-45"
        style={{
          backgroundImage: "url(/leaves-bg.png)",
          backgroundRepeat: "repeat",
          backgroundSize: "520px auto",
          backgroundPosition: "center",
        }}
      />

      {/* App content */}
      <div className="relative z-10">
        <ImageUpload />
      </div>
    </div>
  );
}

export default App;
