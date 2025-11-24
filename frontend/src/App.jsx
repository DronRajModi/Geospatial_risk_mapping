import React, { useState, useEffect } from "react";
import MapView from "../src/components/MapVire";
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
  
  const [neighborList, setNeighborList] = useState(null); 
  const [mapZoomTarget, setMapZoomTarget] = useState(null); 
  const [highlightedDistrict, setHighlightedDistrict] = useState(null);
  const [currentDiseaseLabel, setCurrentDiseaseLabel] = useState("");

  // RESET MAP WHEN SWITCHING TABS
  useEffect(() => {
    setNeighborList(null);
    setHighlightedDistrict(null);
    setPersonalPrediction(null);
    // Reset zoom to center of India
    setMapZoomTarget([20.5937, 78.9629]); 
  }, [mode]);


  const handleManualZoom = (state, district) => {
    if (district) {

      let coords = districtCoords[district];
      if (!coords) {
        const normKey = Object.keys(districtCoords).find(k => 
          k.split(' (')[0].trim().toUpperCase() === district.split(' (')[0].trim().toUpperCase()
        );
        if(normKey) coords = districtCoords[normKey];
      }

      if (coords) {
        setMapZoomTarget([coords[0], coords[1]]);
      }
    } 
    else if (state) {
      const coords = stateCoords[state];
      if (coords) {
        setMapZoomTarget([coords[0], coords[1]]);
      }
    }
  };

  const handleHeatmapSubmit = async (disease) => {
    setIsLoading(true);
    setHeatmapData(null); 
    setPersonalPrediction(null);
    setHighlightedDistrict(null);
    setNeighborList(null); 
    setMapZoomTarget([20.5937, 78.9629]); 

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
    setHighlightedDistrict(null);
    setNeighborList(null); 
    
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
        if(normKey) coords = districtCoords[normKey];
      }

      setPersonalPrediction({
        lat: coords ? coords[0] : 20.5937,
        lon: coords ? coords[1] : 78.9629,
        district: districtName,
        disease: data.main_prediction.disease, 
        results: data, 
      });
      
      if(coords) setMapZoomTarget([coords[0], coords[1]]);

    } catch (error) {
      console.error("Error fetching prediction:", error);
      alert(`Error: ${error.message}`);
    }
    setIsLoading(false);
  };

  const handleClearPrediction = () => {
    setPersonalPrediction(null);
    setMapZoomTarget([20.5937, 78.9629]); 
    setHighlightedDistrict(null);
  };

  const handleNeighborAnalysis = async (state, district, disease) => {
    if (!district) {
      handleManualZoom(state, null);
      setNeighborList(null);
      setHighlightedDistrict(null);
      return;
    }

    setIsLoading(true);
    setNeighborList(null);
    setHighlightedDistrict(district); 
    setCurrentDiseaseLabel(disease); 
    handleManualZoom(state, district);

    try {
      const response = await fetch(`http://127.0.0.1:5000/get_neighbors?district=${district}`);
      if (!response.ok) throw new Error("Neighbor analysis failed.");
      const data = await response.json();
      setNeighborList(data.neighbors);
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
            gnnNeighbors={neighborList}           
            highlightedDistrict={highlightedDistrict} 
            mapZoomTarget={mapZoomTarget}
            selectedDisease={currentDiseaseLabel} 
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
            onZoom={handleManualZoom} 
            heatmapData={heatmapData} 
          />
        </div>
      </div>
    </div>
  );
}

export default App;