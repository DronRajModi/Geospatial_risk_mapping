import React, { useState } from "react";
import indiaData from "../data/state.json";

// This component receives the 'mode' from App.js
export default function SidebarForm({ 
  mode, 
  onHeatmapSubmit, 
  onPersonalSubmit, 
  onNeighborAnalysis, 
  isLoading 
}) {
  return (
    <div className="flex flex-col h-full">
      
      {/* 1. Main form area */}
      <div className="flex-grow">
        {/* Conditionally render the correct form based on the mode */}
        {mode === 'heatmap' ? (
          <HeatmapForm 
            onSubmit={onHeatmapSubmit} 
            isLoading={isLoading} 
          />
        ) : (
          <PersonalForm 
            onSubmit={onPersonalSubmit} 
            isLoading={isLoading} 
          />
        )}
      </div>

      {/* 2. NEW: Tools section at the bottom */}
    
    </div>
  );
}


// --- Sub-component for the Heatmap Form ---
function HeatmapForm({ onSubmit, isLoading }) {
  const [disease, setDisease] = useState('CVD');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(disease);
  };

  return (
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
  );
}


// --- Sub-component for the Personal Prediction Form ---
function PersonalForm({ onSubmit, isLoading }) {
  const [formData, setFormData] = useState({
    state: "",
    district: "",
    age: "",
    sex: "",
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.district || !formData.age || !formData.sex) {
      alert("Please fill in District, Age, and Sex for a personal prediction.");
      return;
    }
    onSubmit(formData); 
  };

  const stateNames = Object.keys(indiaData);
  const stateDistricts = formData.state ? indiaData[formData.state] : [];

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">
        Check Your Personal Risk
      </h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* State Dropdown */}
        <div>
          <label className="block text-gray-600 font-medium mb-1">State</label>
          <select
            name="state"
            value={formData.state}
            onChange={handleChange}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Select State</option>
            {stateNames.map((state) => (
              <option key={state} value={state}>{state}</option>
            ))}
          </select>
        </div>

        {/* District Dropdown */}
        <div>
          <label className="block text-gray-600 font-medium mb-1">District</label>
          <select
            name="district"
            value={formData.district}
            onChange={handleChange}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={!formData.state}
          >
            <option value="">Select District</option>
            {stateDistricts.map((dist) => (
              <option key={dist} value={dist}>{dist}</option>
            ))}
          </select>
        </div>

        {/* Age Input */}
        <div>
          <label className="block text-gray-600 font-medium mb-1">Age</label>
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleChange}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter age"
          />
        </div>

        {/* Sex Selection */}
        <div>
          <label className="block text-gray-600 font-medium mb-1">Sex</label>
          <select
            name="sex"
            value={formData.sex}
            onChange={handleChange}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Select Sex</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>
        </div>
        
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium shadow-md"
          disabled={isLoading}
        >
          {isLoading ? "Analyzing..." : "Show My Prediction"}
        </button>
      </form>
    </div>
  );
}