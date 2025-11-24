export default function Header({ mode, setMode }) {
  
  const getButtonClasses = (buttonMode) => {
    const baseClasses = "py-2 px-4 rounded-lg font-medium transition-all duration-150";
    if (mode === buttonMode) {
      return `${baseClasses} bg-blue-600 text-white shadow-md`;
    } else {
      return `${baseClasses} bg-gray-200 text-gray-700 hover:bg-gray-300`;
    }
  };

  return (
    <header className="bg-white text-gray-800 p-4 shadow-sm border-b border-gray-200 flex justify-between items-center">
    
      <h1 className="font-semibold text-xl text-gray-700">
        Geospatial Risk Mapping of NCDs
      </h1>

  
      <div className="flex space-x-2">
        <button
          onClick={() => setMode('heatmap')}
          className={getButtonClasses('heatmap')}
        >
          Population Risk
        </button>
        <button
          onClick={() => setMode('gnn')}
          className={getButtonClasses('gnn')}
        >
          Spatial Analysis
        </button>

        <button
          onClick={() => setMode('personal')}
          className={getButtonClasses('personal')}
        >
          Personal Risk
        </button>
      </div>
    </header>
  );
}