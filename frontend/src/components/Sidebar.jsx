import React, { useState, useEffect } from "react";
import indiaData from "../data/state.json"; 
import stateCoords from "../data/state_coords.json"; 
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function SidebarForm({ 
  mode, 
  onHeatmapSubmit, 
  onPersonalSubmit, 
  isLoading,
  personalPrediction,
  onClearPrediction,
  onNeighborAnalysis,
  onZoom, 
  heatmapData 
}) {
  return (
    <div className="flex flex-col h-full">
      <div className="flex grow overflow-y-auto pr-2">
        
     
        {mode === 'heatmap' && (
          <PopulationRiskForm 
            onSubmit={onHeatmapSubmit} 
            isLoading={isLoading} 
            onZoom={onZoom}
            heatmapData={heatmapData} 
          />
        )}

      
        {mode === 'gnn' && (
          <NeighborAnalysisForm 
            onNeighborAnalysis={onNeighborAnalysis}
            isLoading={isLoading}
            onZoom={onZoom}
          />
        )}

     
        {mode === 'personal' && (
          personalPrediction ? (
            <PersonalResults 
              prediction={personalPrediction}
              onClear={onClearPrediction}
            />
          ) : (
            <PersonalForm 
              onSubmit={onPersonalSubmit} 
              isLoading={isLoading} 
            />
          )
        )}
      </div>
    </div>
  );
}


function PopulationRiskForm({ onSubmit, isLoading, onZoom, heatmapData }) {
  const [disease, setDisease] = useState('CVD');
  const [selectedState, setSelectedState] = useState("");
  const [selectedDistrict, setSelectedDistrict] = useState("");

  const stateNames = Object.keys(stateCoords); 
  const districtNames = (selectedState && indiaData[selectedState]) ? indiaData[selectedState] : [];

  const handleStateChange = (e) => {
    const newState = e.target.value;
    setSelectedState(newState);
    setSelectedDistrict("");
    if (onZoom) onZoom(newState, null); 
  };

  const handleDistrictChange = (e) => {
    const newDistrict = e.target.value;
    setSelectedDistrict(newDistrict);
    if (onZoom) onZoom(selectedState, newDistrict); 
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(disease);
  };

  
  let specificRisk = null;
  if (heatmapData && selectedDistrict) {
    const normalizedDist = selectedDistrict.toUpperCase();
    if (heatmapData[normalizedDist] !== undefined) {
      specificRisk = heatmapData[normalizedDist];
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold mb-4 text-gray-800">Population Risk</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-gray-600 font-medium mb-1">Select Disease</label>
            <select name="disease" value={disease} onChange={(e) => setDisease(e.target.value)} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white text-black focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option value="CVD">Cardiovascular Disease</option>
              <option value="Liver_Cancer">Liver Cancer</option>
              <option value="Breast_Cancer">Breast Cancer</option>
              <option value="Stroke">Stroke</option>
              <option value="Lung_Cancer">Lung Cancer</option>
            </select>
          </div>
          <button type="submit" className="w-full bg-green-600 text-white py-2 rounded-lg hover:bg-green-700 disabled:bg-gray-400 font-medium shadow-md" disabled={isLoading}>
            {isLoading ? "Generating..." : "Show Risk Heatmap"}
          </button>
        </form>
      </div>

    
      <div className="border-t pt-4">
        <h3 className="text-lg font-medium text-gray-700 mb-2">Explore Region</h3>
        <div className="space-y-3">
          <select value={selectedState} onChange={handleStateChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white text-black">
            <option value="">Select State to Zoom</option>
            {stateNames.map((state) => (<option key={state} value={state}>{state}</option>))}
          </select>
          <select value={selectedDistrict} onChange={handleDistrictChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white text-black" disabled={!selectedState}>
            <option value="">Select District to Zoom</option>
            {districtNames.map((dist) => (<option key={dist} value={dist}>{dist}</option>))}
          </select>
        </div>
      </div>

    
      {selectedDistrict && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-gray-600">Average Risk for</p>
          <p className="font-bold text-gray-800 text-lg">{selectedDistrict}</p>
          <div className="mt-2 text-3xl font-bold text-blue-700">
            {specificRisk !== null ? `${specificRisk}%` : "No Data"}
          </div>
          <p className="text-xs text-gray-500 mt-1">Based on synthetic population analysis</p>
        </div>
      )}
    </div>
  );
}


function NeighborAnalysisForm({ onNeighborAnalysis, isLoading, onZoom }) {
  const [selectedState, setSelectedState] = useState("");
  const [selectedDistrict, setSelectedDistrict] = useState("");
  // const [disease, setDisease] = useState('CVD'); 

  const stateNames = Object.keys(stateCoords); 
  const districtNames = (selectedState && indiaData[selectedState]) ? indiaData[selectedState] : [];

  const handleStateChange = (e) => {
    const newState = e.target.value;
    setSelectedState(newState);
    setSelectedDistrict("");
    if (onZoom) onZoom(newState, null);
  };

  const handleDistrictChange = (e) => {
    const newDistrict = e.target.value;
    setSelectedDistrict(newDistrict);
    if (onZoom) onZoom(selectedState, newDistrict);
  };

  const handleAnalyze = () => {
    if (!selectedDistrict) return alert("Select a district");
    onNeighborAnalysis(selectedState, selectedDistrict);
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">Spatial Neighbors (GNN)</h2>
      <p className="text-sm text-gray-600 mb-4">
        Find districts that are "structurally similar" (Socio-Economic Twins) to the selected district using Graph Neural Networks.
      </p>
      
      <div className="space-y-3">
        <select value={selectedState} onChange={handleStateChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white text-black">
          <option value="">Select State</option>
          {stateNames.map((state) => (<option key={state} value={state}>{state}</option>))}
        </select>
        <select value={selectedDistrict} onChange={handleDistrictChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white text-black" disabled={!selectedState}>
          <option value="">Select District</option>
          {districtNames.map((dist) => (<option key={dist} value={dist}>{dist}</option>))}
        </select>
      </div>

      <button onClick={handleAnalyze} className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 font-medium shadow-md" disabled={isLoading || !selectedDistrict}>
        {isLoading ? "Analyzing..." : "Find Similar Districts"}
      </button>
    </div>
  );
}


function PersonalForm({ onSubmit, isLoading }) {
  const [formData, setFormData] = useState({
    state: "", District: "", Age: "", Gender: "",
    Tobacco_Use: "No", Alcohol_Use: "No", Hypertension: "No", Diabetes: "No",
    Obese: "", Cholesterol: "", Sleep_Hours: "", Urban_or_Rural: "Urban",
  });
  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name === 'state') {
      setFormData({ ...formData, state: value, District: "" });
    } else {
      setFormData({ ...formData, [name]: value });
    }
  };
  const handleSubmit = (e) => {
    e.preventDefault();
    const requiredKeys = [
      'District', 'Age', 'Gender', 'Tobacco_Use', 'Alcohol_Use', 
      'Hypertension', 'Diabetes', 'Obese', 'Cholesterol', 
      'Sleep_Hours', 'Urban_or_Rural'
    ];
    for (const key of requiredKeys) {
      if (!formData[key] || formData[key] === "") {
        let friendlyKey = key.replace(/_/g, ' ');
        if (key === 'Obese' || key === 'Cholesterol') friendlyKey += ' (%)';
        if (key === 'Sleep_Hours') friendlyKey += ' (hours)';
        alert(`Please fill in all fields. '${friendlyKey}' is missing.`);
        return;
      }
    }
    const { state, ...backendData } = formData;
    onSubmit(backendData); 
  };
  const stateNames = Object.keys(indiaData);
  const stateDistricts = formData.state ? indiaData[formData.state] : [];
  return (
    <div>
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">
        Check Your Personal Risk
      </h2>
      <form onSubmit={handleSubmit} className="space-y-4">
 
        <div><label className="block text-gray-600 font-medium mb-1">State</label><select name="state" value={formData.state} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"><option value="">Select State</option>{stateNames.map((state) => ( <option key={state} value={state}>{state}</option> ))}</select></div>
        <div><label className="block text-gray-600 font-medium mb-1">District</label><select name="District" value={formData.District} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500" disabled={!formData.state}><option value="">Select District</option>{stateDistricts.map((dist) => ( <option key={dist} value={dist}>{dist}</option> ))}</select></div>
        <div><label className="block text-gray-600 font-medium mb-1">Age</label><input type="number" name="Age" value={formData.Age} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Enter age (e.g., 55)" min="18" max="100" /></div>
        <div><label className="block text-gray-600 font-medium mb-1">Gender</label><select name="Gender" value={formData.Gender} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"><option value="">Select Gender</option><option value="Male">Male</option><option value="Female">Female</option><option value="Other">Other</option></select></div>
        <div><label className="block text-gray-600 font-medium mb-1">Tobacco Use</label><select name="Tobacco_Use" value={formData.Tobacco_Use} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"><option value="No">No</option><option value="Yes">Yes</option></select></div>
        <div><label className="block text-gray-600 font-medium mb-1">Alcohol Use</label><select name="Alcohol_Use" value={formData.Alcohol_Use} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"><option value="No">No</option><option value="Yes">Yes</option></select></div>
        <div><label className="block text-gray-600 font-medium mb-1">Hypertension</label><select name="Hypertension" value={formData.Hypertension} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"><option value="No">No</option><option value="Yes">Yes</option></select></div>
        <div><label className="block text-gray-600 font-medium mb-1">Diabetes</label><select name="Diabetes" value={formData.Diabetes} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"><option value="No">No</option><option value="Yes">Yes</option></select></div>
        <div><label className="block text-gray-600 font-medium mb-1">Obese (%)</label><input type="number" name="Obese" value={formData.Obese} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Enter % (e.g., 28.5)" step="0.1" /></div>
        <div><label className="block text-gray-600 font-medium mb-1">Cholesterol (%)</label><input type="number" name="Cholesterol" value={formData.Cholesterol} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Enter % (e.g., 45.0)" step="0.1" /></div>
        <div><label className="block text-gray-600 font-medium mb-1">Sleep Hours (avg. per night)</label><input type="number" name="Sleep_Hours" value={formData.Sleep_Hours} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Enter hours (e.g., 6.5)" step="0.5" /></div>
        <div><label className="block text-gray-600 font-medium mb-1">Area Type</label><select name="Urban_or_Rural" value={formData.Urban_or_Rural} onChange={handleChange} className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"><option value="Urban">Urban</option><option value="Rural">Rural</option></select></div>
        <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium shadow-md" disabled={isLoading}>
          {isLoading ? "Analyzing..." : "Show My Prediction"}
        </button>
      </form>
    </div>
  );
}



function PersonalResults({ prediction, onClear }) {
  const { results, district } = prediction;
  const { 
    main_prediction, 
    age_risk_profile, 
    // spatial_neighbors, 
    top_risk_factors, 
    lifestyle_tips 
  } = results;

  // Chart data logic
  const diseaseColor = main_prediction.disease === 'CVD' ? 'text-red-600' :
                       main_prediction.disease === 'Stroke' ? 'text-red-600' :
                       main_prediction.disease === 'Liver_Cancer' ? 'text-orange-600' :
                       main_prediction.disease === 'Breast_Cancer' ? 'text-pink-600' :
                       'text-gray-800';
  const chartData = {
    labels: age_risk_profile.labels,
    datasets: [ {
        label: `Risk of ${main_prediction.disease}`,
        data: age_risk_profile.scores,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        tension: 0.1
      }, ],
  };
  const chartOptions = {
    responsive: true,
    plugins: { legend: { display: false }, title: { display: true, text: 'Your Risk vs. Age' } },
    scales: { y: { title: { display: true, text: 'Risk (%)' } }, x: { title: { display: true, text: 'Age' } } }
  };
  const factors = Object.entries(top_risk_factors)
    .sort(([,a], [,b]) => b - a)
    .map(([name, value]) => ({ name, value: (value * 100).toFixed(1) }));
  const tips = Object.entries(lifestyle_tips);

  return (
    <div className="space-y-6">
      <button onClick={onClear} className="text-blue-600 hover:text-blue-800 font-medium">
        &larr; Back to Form
      </button>
      <h2 className="text-2xl font-semibold text-gray-800">
        Your Personal Risk Report
      </h2>
      <p className="text-sm text-gray-500 -mt-4">For: {district}</p>

      {/* 1. Main Prediction */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
        <div className="text-sm font-medium text-gray-600">PRIMARY RISK</div>
        <div className={`text-3xl font-bold my-1 ${diseaseColor}`}>
          {main_prediction.disease.replace('_', ' ')}
        </div>
        <div className="text-lg font-medium text-gray-700">
          Confidence: {main_prediction.confidence}
        </div>
      </div>

      {/* 2. Age Risk Chart */}
      <div className="p-4 border rounded-lg shadow-sm">
        <Line options={chartOptions} data={chartData} />
      </div>

      {/* 3. Top Risk Factors */}
      <div className="p-4 border rounded-lg shadow-sm">
        <h3 className="text-lg font-semibold text-gray-700 mb-3">Top 5 Risk Factors</h3>
        <ul className="space-y-2">
          {factors.map(factor => (
            <li key={factor.name} className="flex justify-between items-center">
              <span className="text-gray-600">{factor.name}</span>
              <span className="font-medium text-gray-800">{factor.value}%</span>
            </li>
          ))}
        </ul>
      </div>

      {/* 4. Lifestyle Tips */}
      {tips.length > 0 && (
        <div className="p-4 border rounded-lg shadow-sm bg-gray-50">
          <h3 className="text-lg font-semibold text-gray-700 mb-3">Lifestyle Analysis</h3>
          <ul className="space-y-2">
            {tips.map(([tip, result]) => (
              <li key={tip} className="text-gray-700">
                <span className="font-medium">{tip}:</span> {result}
              </li>
            ))}
          </ul>
        </div>
      )}
      
    </div>
  );
}