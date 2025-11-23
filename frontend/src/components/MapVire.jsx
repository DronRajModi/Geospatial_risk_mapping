import React, { useEffect, useRef, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap, GeoJSON } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import iconUrl from "leaflet/dist/images/marker-icon.png";
import iconShadow from "leaflet/dist/images/marker-shadow.png";

let DefaultIcon = L.icon({
  iconUrl,
  shadowUrl: iconShadow,
});
L.Marker.prototype.options.icon = DefaultIcon;

// Helper to handle zooming
function MapFlyTo({ position }) {
  const map = useMap();
  useEffect(() => {
    if (position) {
      // Zoom in if it's a specific district (GN/Prediction), Zoom out if it's India (Heatmap)
      const zoomLevel = position[0] === 20.5937 ? 5 : 8; 
      map.flyTo(position, zoomLevel);
    }
  }, [position, map]);
  return null;
}

function ChoroplethLayer({ heatmapData, gnnNeighbors, highlightedDistrict }) {
  const geoJsonRef = useRef(null);
  const [geoJson, setGeoJson] = useState(null);

  useEffect(() => {
    fetch("/india_districts.geojson")
      .then((response) => response.json())
      .then((data) => setGeoJson(data))
      .catch((error) => console.error("Error loading GeoJSON:", error));
  }, []);

  const normalizeName = (name) => {
    if (!name) return "";
    return name.split(' (')[0].trim().toUpperCase();
  };

  function getColor(percentage) {
    if (percentage === null || percentage === undefined) return "#DDDDDD"; 
    const r = Math.min(255, Math.round(percentage * 5));
    const g = Math.min(255, Math.round(255 - percentage * 5));
    return `rgb(${r},${g},0)`;
  }

  function getRiskForDistrict(feature) {
    if (!feature.properties.district || !heatmapData) return null;
    const geoJsonName = normalizeName(feature.properties.district);
    const risk = heatmapData[geoJsonName];
    return risk !== undefined ? risk : null;
  }

  function styleFeature(feature) {
    if (!feature.properties.district) return { fillOpacity: 0, weight: 0 };

    const districtNameNorm = normalizeName(feature.properties.district);
    const selectedNorm = normalizeName(highlightedDistrict);
    
    const isNeighbor = gnnNeighbors && gnnNeighbors.some(n => normalizeName(n) === districtNameNorm);
    const isSelected = highlightedDistrict && districtNameNorm === selectedNorm;

    // 1. Blue Selection
    if (isSelected) {
      return { fillColor: "#2563EB", weight: 2, opacity: 1, color: "white", fillOpacity: 1 };
    }
    // 2. Red Neighbors
    if (isNeighbor) {
      return { fillColor: "#DC2626", weight: 2, opacity: 1, color: "#7F1D1D", fillOpacity: 0.8 };
    }
    // 3. Heatmap
    if (heatmapData) {
      const risk = getRiskForDistrict(feature);
      return { 
        fillColor: getColor(risk), 
        weight: 0.5, 
        opacity: 1, 
        color: 'white', 
        dashArray: '3', 
        fillOpacity: (risk === null ? 0.2 : 0.7) 
      };
    }
    // 4. Default Grey
    return { fillColor: "#DDDDDD", weight: 0.5, opacity: 1, color: 'white', dashArray: '3', fillOpacity: 0.2 };
  }

  function onEachFeature(feature, layer) {
    if (!feature.properties.district) return;
    const districtName = feature.properties.district;
    const districtNameNorm = normalizeName(districtName);
    
    let popupContent = `<b>${districtName}</b>`;
    
    if (heatmapData) {
       const risk = getRiskForDistrict(feature);
       if (risk !== null) popupContent += `<br/>Risk: <b>${risk}%</b>`;
    }
    layer.bindPopup(popupContent);
  }

  if (!geoJson) return null;

  // --- THE CRITICAL FIX IS HERE ---
  // We create a unique key string. Whenever this string changes, 
  // React destroys the old GeoJSON layer and builds a new one with the correct colors.
  const layerKey = `layer-${heatmapData ? 'heat' : 'noheat'}-${highlightedDistrict || 'none'}-${gnnNeighbors ? gnnNeighbors.length : 0}`;

  return (
    <GeoJSON
      key={layerKey} 
      data={geoJson}
      style={styleFeature}
      onEachFeature={onEachFeature}
    />
  );
}

const MapView = ({ heatmapData, personalPrediction, gnnNeighbors, highlightedDistrict, mapZoomTarget }) => {
  const indiaCenter = [20.5937, 78.9629];
  const markerPosition = personalPrediction ? [personalPrediction.lat, personalPrediction.lon] : null;
  const flyToPosition = mapZoomTarget || markerPosition;

  return (
    <div className="w-full h-full">
      <MapContainer center={indiaCenter} zoom={5} style={{ width: "100%", height: "100%" }}>
        <TileLayer url="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png" />
        
        <ChoroplethLayer 
            heatmapData={heatmapData} 
            gnnNeighbors={gnnNeighbors}
            highlightedDistrict={highlightedDistrict}
        />

        {personalPrediction && markerPosition && (
          <Marker position={markerPosition}>
            <Popup>
              <b>{personalPrediction.district}</b><br />Predicted: <b>{personalPrediction.disease}</b>
            </Popup>
          </Marker>
        )}

        <MapFlyTo position={flyToPosition} />
      </MapContainer>
    </div>
  );
};

export default MapView;