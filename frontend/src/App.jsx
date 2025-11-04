import React, { useState } from "react"; // No useEffect needed
import MapView from "./components/MapVire";
import SidebarForm from "./components/Sidebar";
import districtCoords from "./data/district_coords.json";
import Header from "./components/header";
import './App.css';

function App() {
  const [heatmapData, setHeatmapData] = useState(null);
  const [personalPrediction, setPersonalPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // This is the handler for the HEATMAP button
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

  // This is the handler for the PERSONAL button
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
        if (normKey) {
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

  // --- THIS IS THE NEW LAYOUT ---
  return (
    // Main container uses a COLUMN layout
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* 1. Header is at the top */}
      <Header />

      {/* 2. Main content area uses a ROW layout and fills the remaining space */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

        {/* 2a. MapView fills the available space */}
        <div style={{ flex: 1, height: "100%" }}>
          <MapView
            heatmapData={heatmapData}
            personalPrediction={personalPrediction}
          />
        </div>

        {/* 2b. Sidebar has a fixed width and will scroll if content is too long */}
        <div style={{ width: "350px", padding: "1rem", overflowY: "auto", borderLeft: "1px solid #ccc", background: "#f9f9f9" }}>
          <SidebarForm
            onHeatmapSubmit={handleHeatmapSubmit}
            onPersonalSubmit={handlePersonalSubmit}
            isLoading={isLoading}
          />
        </div>

      </div>
    </div>
  );
}

export default App;
