import React, { useState } from "react";
import MapView from "./components/MapVire";
import SidebarForm from "./components/Sidebar";
import districtCoords from "./data/district_coords.json";
import Header from "./components/header";
import './App.css';

function App() {
  const [heatmapData, setHeatmapData] = useState(null);
  const [personalPrediction, setPersonalPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // --- NEW: State to control which form is visible ---
  // 'heatmap' or 'personal'
  const [mode, setMode] = useState('heatmap'); // Default to heatmap

  const handleHeatmapSubmit = async (disease) => {
    setIsLoading(true);
    setHeatmapData(null); 
    setPersonalPrediction(null);
    try {
      const response = await fetch(`http://127.0.0.1:5000/get_risk_heatmap?disease=${disease}`);
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Heatmap generation failed.");
      }
      const data = await response.json(); 
      setHeatmapData(data);
    } catch (error) {
      console.error("Error fetching heatmap:", error);
      alert(`Error: ${error.message}`);
    }
    setIsLoading(false);
  };

  const handlePersonalSubmit = async (formData) => {
    setIsLoading(true);
    setPersonalPrediction(null);
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Prediction failed.");
      }
      const data = await response.json();
      let coords = districtCoords[data.district];
      if (!coords) {
         const normKey = Object.keys(districtCoords).find(k => 
          k.split(' (')[0].trim().toUpperCase() === data.district.split(' (')[0].trim().toUpperCase()
        );
        if(normKey) {
            coords = districtCoords[normKey];
        } else {
            throw new Error(`Coordinates not found for district: ${data.district}`);
        }
      }
      setPersonalPrediction({
        lat: coords[0],
        lon: coords[1],
        district: data.district,
        disease: data.predicted_class,
      });
    } catch (error) {
      console.error("Error fetching prediction:", error);
      alert(`Error: ${error.message}`);
    }
    setIsLoading(false);
  };

  // --- NEW: Dummy function for the neighbor analysis ---
  const handleNeighborAnalysis = () => {
    if (!heatmapData) {
      alert("Please generate a heatmap first.");
      return;
    }
    alert("Neighbor Analysis feature coming soon!\nThis will analyze the currently displayed heatmap.");
    // You would add your backend call here
  };

  return (
    // Main container uses a COLUMN layout
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", backgroundColor: "#fff" }}>
      {/* 1. Header is at the top, passing mode state */}
      <Header mode={mode} setMode={setMode} />

      {/* 2. Main content area uses a ROW layout */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

        {/* 2a. MapView fills the available space */}
        <div style={{ flex: 1, height: "100%" }}>
          <MapView
            heatmapData={heatmapData}
            personalPrediction={personalPrediction}
          />
        </div>

        {/* 2b. Sidebar has a fixed width and is now plain white */}
        <div style={{ width: "350px", padding: "1.5rem", overflowY: "auto", borderLeft: "1px solid #e5e7eb", background: "#ffffff" }}>
          <SidebarForm
            // Pass the current mode
            mode={mode}
            onHeatmapSubmit={handleHeatmapSubmit}
            onPersonalSubmit={handlePersonalSubmit}
            // Pass the new dummy handler
            onNeighborAnalysis={handleNeighborAnalysis}
            isLoading={isLoading}
          />
        </div>

      </div>
    </div>
  );
}

export default App;