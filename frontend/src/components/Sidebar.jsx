import React, { useState } from "react";
import indiaData from "../data/state.json"; 
import stateCoords from "../data/state_coords.json"; // Import for state dropdowns

// --- Chart.js Imports ---
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend
);
// --- End Chart.js Imports ---


export default function SidebarForm({ 
  mode, 
  onHeatmapSubmit, 
  onPersonalSubmit, 
  isLoading,
  personalPrediction,
  onClearPrediction,
  onNeighborAnalysis // This prop comes from App.jsx
}) {
  return (
    <div className="flex flex-col h-full">
      <div className="flex-grow overflow-y-auto pr-2">
        
        {/* --- THIS IS THE CRITICAL LOGIC ---
          * This logic checks if you are in 'personal' mode AND
          * if 'personalPrediction' data exists. If it does,
          * it shows the results. Otherwise, it shows the form.
          * Your current file is probably missing this.
          * ---
        */}
        {mode === 'heatmap' ? (
          <HeatmapForm 
            onSubmit={onHeatmapSubmit} 
            isLoading={isLoading} 
            onNeighborAnalysis={onNeighborAnalysis}
          />
        ) : personalPrediction ? (
          <PersonalResults 
            prediction={personalPrediction}
            onClear={onClearPrediction}
          />
        ) : (
          <PersonalForm 
            onSubmit={onPersonalSubmit} 
            isLoading={isLoading} 
          />
        )}
        {/* --- END OF CRITICAL LOGIC --- */}
      </div>
    </div>
  );
}


// --- Sub-component for the Heatmap Form (With Neighbor Analysis) ---
function HeatmapForm({ onSubmit, isLoading, onNeighborAnalysis }) {
  const [disease, setDisease] = useState('CVD');
  const [selectedState, setSelectedState] = useState("");
  const [selectedDistrict, setSelectedDistrict] = useState("");

  // Use stateCoords for the state list, indiaData for the district list
  const stateNames = Object.keys(stateCoords); 
  const districtNames = selectedState ? indiaData[selectedState] : []; 

  const handleStateChange = (e) => {
    setSelectedState(e.target.value);
    setSelectedDistrict("");
    onNeighborAnalysis(e.target.value, null); // Zoom to state
  };
  
  const handleDistrictChange = (e) => {
    setSelectedDistrict(e.target.value);
  };
  
  const handleAnalyzeClick = () => {
    if (!selectedDistrict) {
      alert("Please select a district to analyze.");
      return;
    }
    onNeighborAnalysis(selectedState, selectedDistrict);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(disease);
  };

  return (
    <div className="space-y-6">
      {/* Section 1: Generate Heatmap */}
      <div>
        <h2 className="text-2xl font-semibold mb-4 text-gray-800">
          View Population Risk
        </h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-gray-600 font-medium mb-1">Select Disease</label>
            <select
              name="disease"
              value={disease}
              onChange={(e) => setDisease(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="CVD">Cardiovascular Disease</option>
              <option value="Liver_Cancer">Liver Cancer</option>
              <option value="Breast_Cancer">Breast Cancer</option>
              <option value="Stroke">Stroke</option>
              <option value="Lung_Cancer">Lung Cancer</option>
            </select>
          </div>
          <button
            type="submit"
            className="w-full bg-green-600 text-white py-2 rounded-lg hover:bg-green-700 disabled:bg-gray-400 font-medium shadow-md"
            disabled={isLoading}
          >
            {isLoading ? "Generating..." : "Show Risk Heatmap"}
          </button>
        </form>
      </div>

      {/* Section 2: Analyze Neighbors */}
      <div className="border-t pt-6">
        <h2 className="text-2xl font-semibold mb-4 text-gray-800">
          Analyze Spatial Neighbors
        </h2>
        <div className="space-y-4">
          <div>
            <label className="block text-gray-600 font-medium mb-1">Select State</label>
            <select
              name="state"
              value={selectedState}
              onChange={handleStateChange}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select State to Zoom</option>
              {stateNames.map((state) => ( <option key={state} value={state}>{state}</option> ))}
            </select>
          </div>
          <div>
            <label className="block text-gray-600 font-medium mb-1">Select District</label>
            <select
              name="district"
              value={selectedDistrict}
              onChange={handleDistrictChange}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={!selectedState}
            >
              <option value="">Select District to Analyze</option>
              {districtNames.map((dist) => ( <option key={dist} value={dist}>{dist}</option> ))}
            </select>
          </div>
          <button
            type="button"
            onClick={handleAnalyzeClick}
            className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium shadow-md"
            disabled={isLoading || !selectedDistrict}
          >
            {isLoading ? "Analyzing..." : "Highlight GNN Neighbors"}
          </button>
        </div>
      </div>
    </div>
  );
}


// --- Sub-component for the Personal Prediction Form (Unchanged) ---
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
        {/* All form fields... */}
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

// --- PersonalResults Component (With Neighbors REMOVED) ---
// This is the component that shows your results.
function PersonalResults({ prediction, onClear }) {
  const { results, district } = prediction;
  const { 
    main_prediction, 
    age_risk_profile, 
    // spatial_neighbors, // <-- This is GONE, as you requested
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
      
      {/* 5. Spatial Neighbors section is now GONE */}
    </div>
  );
}