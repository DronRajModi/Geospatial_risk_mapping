import React, { useState } from "react";
import MapView from "./components/MapVire";
import SidebarForm from "./components/Sidebar";
import districtCoords from "./data/district_coords.json";
import stateCoords from "./data/state_coords.json"; 
import Header from "./components/header";
import './App.css';

function App() {
  const [heatmapData, setHeatmapData] = useState(null);
  const [personalPrediction, setPersonalPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState('heatmap');
  

  // This holds the list of GNN neighbors to highlight
  const [neighborList, setNeighborList] = useState(null); 
  // This holds the [lat, lon] for the map to zoom to
  const [mapZoomTarget, setMapZoomTarget] = useState(null); 
 

  const handleHeatmapSubmit = async (disease) => {
    setIsLoading(true);
    setHeatmapData(null); 
    setPersonalPrediction(null);
    setNeighborList(null); 
    setMapZoomTarget(null); 
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
    setNeighborList(null); 
    setMapZoomTarget(null);
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
      const districtName = formData.District; 
      let coords = districtCoords[districtName];
      if (!coords) {
        const normKey = Object.keys(districtCoords).find(k => 
          k.split(' (')[0].trim().toUpperCase() === districtName.split(' (')[0].trim().toUpperCase()
        );
        if(normKey) {
            coords = districtCoords[normKey];
        } else {
            throw new Error(`Coordinates not found for district: ${districtName}`);
        }
      }
      setPersonalPrediction({
        lat: coords[0],
        lon: coords[1],
        district: districtName,
        disease: data.main_prediction.disease, 
        results: data, 
      });
      setMapZoomTarget([coords[0], coords[1]]);

    } catch (error) {
      console.error("Error fetching prediction:", error);
      alert(`Error: ${error.message}`);
    }
    setIsLoading(false);
  };

  const handleClearPrediction = () => {
    setPersonalPrediction(null);
    setMapZoomTarget(null); // Clear zoom
  };


  const handleNeighborAnalysis = async (state, district) => {
    if (!district) {
  
      setMapZoomTarget(stateCoords[state]);
      setNeighborList(null);
      return;
    }

    setIsLoading(true);
    setNeighborList(null);
    try {
      // 1. Fetch neighbors from our new endpoint
      const response = await fetch(`http://127.0.0.1:5000/get_neighbors?district=${district}`);
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Neighbor analysis failed.");
      }
      const data = await response.json();
      
      // 2. Set the neighbor list for highlighting
      setNeighborList(data.neighbors);

      // 3. Find coords for the selected district and zoom
      let coords = districtCoords[district];
      if (!coords) {
        const normKey = Object.keys(districtCoords).find(k => 
          k.split(' (')[0].trim().toUpperCase() === district.split(' (')[0].trim().toUpperCase()
        );
        if(normKey) { coords = districtCoords[normKey]; }
      }
      if (coords) {
        setMapZoomTarget([coords[0], coords[1]]);
      } else {
        // Fallback to state zoom if district coords not found
        setMapZoomTarget(stateCoords[state]);
      }

    } catch (error) {
      console.error("Error fetching neighbors:", error);
      alert(`Error: ${error.message}`);
    }
    setIsLoading(false);
  };
 

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", backgroundColor: "#fff" }}>
      <Header mode={mode} setMode={setMode} />
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <div style={{ flex: 1, height: "100%" }}>
          <MapView
            heatmapData={heatmapData}
            personalPrediction={personalPrediction}
            neighborList={neighborList}
            mapZoomTarget={mapZoomTarget}
         
          />
        </div>
        <div style={{ width: "350px", padding: "1.5rem", overflowY: "auto", borderLeft: "1px solid #e5e7eb", background: "#ffffff" }}>
          <SidebarForm
            mode={mode}
            onHeatmapSubmit={handleHeatmapSubmit}
            onPersonalSubmit={handlePersonalSubmit}
            isLoading={isLoading}
            personalPrediction={personalPrediction}
            onClearPrediction={handleClearPrediction}
            onNeighborAnalysis={handleNeighborAnalysis}
          
          />
        </div>
      </div>
    </div>
  );
}

export default App;