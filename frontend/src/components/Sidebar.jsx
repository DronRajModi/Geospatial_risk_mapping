import React, { useState } from "react";
import indiaData from "../data/state.json";

export default function SidebarForm({ onHeatmapSubmit, onPersonalSubmit, isLoading }) {

  const [formData, setFormData] = useState({
    state: "",
    district: "",
    // Set the default to a disease that is in your list
    disease: "CVD",
    age: "",
    sex: "",
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // This function is for the HEATMAP button
  const handleHeatmap = (e) => {
    e.preventDefault();
    if (!formData.disease) {
      alert("Please select a disease to generate the heatmap.");
      return;
    }
    onHeatmapSubmit(formData.disease);
  };

  // This function is for the PERSONAL PREDICTION button
  const handlePersonal = (e) => {
    e.preventDefault();
    if (!formData.district || !formData.age || !formData.sex) {
      alert("Please fill in District, Age, and Sex for a personal prediction.");
      return;
    }
    onPersonalSubmit(formData);
  };

  const stateNames = Object.keys(indiaData);
  const stateDistricts = indiaData[formData.state] || [];

  return (
    <div>
      {/* --- Section 1: Heatmap --- */}
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">
        1. View Population Risk
      </h2>
      <form onSubmit={handleHeatmap} className="space-y-4 p-4 border rounded bg-gray-50">
        <div>
          <label className="block text-gray-600 font-medium mb-1">Select Disease</label>
          <select
            name="disease"
            value={formData.disease}
            onChange={handleChange}
            className="w-full border rounded px-3 py-2"
          >
            {/* These 'value' strings must EXACTLY match your training data */}
            <option value="CVD">Cardiovascular Disease</option>
            <option value="Liver_Cancer">Liver Cancer</option>
            <option value="Breast_Cancer">Breast Cancer</option>
            <option value="Stroke">Stroke</option>
            <option value ="Lung_Cancer">Lung Cancer</option>
           
          </select>
        </div>
        <button
          type="submit"
          className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700 disabled:bg-gray-400"
          disabled={isLoading}
        >
          {isLoading ? "Generating..." : "Show Risk Heatmap"}
        </button>
      </form>

      <hr className="my-6" />

      {/* --- Section 2: Personal Prediction --- */}
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">
     
      </h2>
      <form onSubmit={handlePersonal} className="space-y-4">
        {/* State Dropdown */}
        <div>
          <label className="block text-gray-600 font-medium mb-1">State</label>
          <select
            name="state"
            value={formData.state}
            onChange={handleChange}
            className="w-full border rounded px-3 py-2"
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
            className="w-full border rounded px-3 py-2"
            dot disabled={!formData.state}
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
            className="w-full border rounded px-3 py-2"
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
            className="w-full border rounded px-3 py-2"
          >
            <option value="">Select Sex</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>
        </div>

        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:bg-gray-400"
          disabled={isLoading}
        >
          {isLoading ? "Analyzing..." : "Show My Prediction"}
        </button>
      </form>
    </div>
  );
}