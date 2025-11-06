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

function MapFlyTo({ position }) {
  const map = useMap();
  useEffect(() => {
    if (position) {
      map.flyTo(position, 10); 
    }
  }, [position, map]);
  return null;
}

function ChoroplethLayer({ heatmapData }) {
  const geoJsonRef = useRef(null);
  const [geoJson, setGeoJson] = useState(null); 

  // Fetch the GeoJSON shapes from the /public/ folder
  useEffect(() => {
    fetch("/india_districts.geojson") 
      .then(response => response.json())
      .then(data => setGeoJson(data))
      .catch(error => console.error("Error loading GeoJSON:", error));
  }, []); 

  function getColor(percentage) {
    if (percentage === null || percentage === undefined) return "#DDDDDD"; // Grey for no data
    const r = Math.min(255, Math.round(percentage * 5)); 
    const g = Math.min(255, Math.round(255 - (percentage * 5)));
    return `rgb(${r},${g},0)`;
  }

  // --- NAME-MATCHING FIX ---
  function getRiskForDistrict(feature) {
    if (!feature.properties.district || !heatmapData) {
      return null;
    }
    
    const geoJsonName = feature.properties.district.split(' (')[0].trim().toUpperCase();
    

    const risk = heatmapData[geoJsonName];
    
    return risk !== undefined ? risk : null;
  }

  function styleFeature(feature) {
    if (!feature.properties.district) {
      return { fillOpacity: 0, weight: 0, opacity: 0 }; 
    }
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

  function onEachFeature(feature, layer) {
    if (!feature.properties.district) return; 

    const districtName = feature.properties.district; 
    const risk = getRiskForDistrict(feature);
    
    let popupContent = `<b>${districtName}</b><br/>No risk data available`;
    if (risk !== null) {
      popupContent = `<b>${districtName}</b><br/>Risk: <b>${risk}%</b>`;
    }
    layer.bindPopup(popupContent);
  }

  useEffect(() => {
    if (geoJsonRef.current && geoJson) {
      geoJsonRef.current.clearLayers().addData(geoJson);
    }
  }, [heatmapData, geoJson]); 


  if (!heatmapData || !geoJson) {
    return null; 
  }

  return (
    <GeoJSON 
      ref={geoJsonRef}
      data={geoJson} 
      style={styleFeature}
      onEachFeature={onEachFeature}
    />
  );
}


//  4. Main Map Component 
const MapView = ({ heatmapData, personalPrediction }) => {
  const indiaCenter = [20.5937, 78.9629];
  const markerPosition = personalPrediction ? [personalPrediction.lat, personalPrediction.lon] : null;

  return (
    <div className="w-full h-full">
      <MapContainer
        center={indiaCenter}
        zoom={5}
        style={{ width: "100%", height: "100%" }} 
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png"
        />

        {/*  HEATMAP LAYER */}

        {heatmapData && <ChoroplethLayer heatmapData={heatmapData} />}

        {/*  PERSONAL MARKER LAYER */}
        {personalPrediction && markerPosition && (
          <Marker position={markerPosition}>
            <Popup>
              <b>{personalPrediction.district} (Your Prediction)</b>
              <br />
              Predicted Disease: <b>{personalPrediction.disease}</b>
            </Popup>
          </Marker>
        )}

        <MapFlyTo position={markerPosition} />

      </MapContainer>
    </div>
  );
};

export default MapView;