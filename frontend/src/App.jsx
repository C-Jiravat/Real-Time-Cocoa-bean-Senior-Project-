import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';
import { predictImage } from './api';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelType, setModelType] = useState('hybrid'); // 'yolo' or 'hybrid'
  const [error, setError] = useState(null);

  const handleImageUpload = (uploadedFile) => {
    setFile(uploadedFile);
    setPreview(URL.createObjectURL(uploadedFile));
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    try {
      const data = await predictImage(file, modelType);
      setResult(data);
    } catch (err) {
      setError('Failed to process image. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Cocoa Bean Analyzer</h1>
        <p>AI-Powered Quality & Color Classification</p>
      </header>

      <main className="main-content">
        <div className="control-panel glass-panel">
          <h2>Configuration</h2>

          <div className="model-selector">
            <label>Select Model:</label>
            <div className="radio-group">
              <label className={`radio-label ${modelType === 'yolo' ? 'active' : ''}`}>
                <input
                  type="radio"
                  value="yolo"
                  checked={modelType === 'yolo'}
                  onChange={(e) => setModelType(e.target.value)}
                />
                YOLOv11 Only
              </label>
              <label className={`radio-label ${modelType === 'hybrid' ? 'active' : ''}`}>
                <input
                  type="radio"
                  value="hybrid"
                  checked={modelType === 'hybrid'}
                  onChange={(e) => setModelType(e.target.value)}
                />
                YOLOv11 + ConvNeXt
              </label>
            </div>
          </div>

          <div className="upload-section">
            <ImageUpload onImageUpload={handleImageUpload} />
          </div>

          {file && (
            <button
              className="predict-button"
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? 'Analyzing...' : 'Analyze Bean'}
            </button>
          )}

          {error && <div className="error-message">{error}</div>}
        </div>

        <div className="result-panel glass-panel">
          {preview ? (
            <ResultDisplay
              imageSrc={preview}
              detections={result?.detections}
              summary={result?.summary || {}}
              grade={result?.grade}
              modelType={modelType}
            />
          ) : (
            <div className="placeholder-text">
              Upload an image to see results here.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
