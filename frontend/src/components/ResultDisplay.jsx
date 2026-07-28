import React, { useEffect, useRef } from 'react';

const ResultDisplay = ({ imageSrc, detections, summary, grade, modelType }) => {
    const canvasRef = useRef(null);
    const imgRef = useRef(null);

    useEffect(() => {
        if (!imageSrc || !detections || !imgRef.current) return;

        const img = imgRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Wait for image to load
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            drawDetections(ctx, detections);
        };

        // If image is already loaded (e.g. from cache)
        if (img.complete) {
            canvas.width = img.width;
            canvas.height = img.height;
            drawDetections(ctx, detections);
        }

    }, [imageSrc, detections]);

    const drawDetections = (ctx, detections) => {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        detections.forEach((det) => {
            const [x1, y1, x2, y2] = det.bbox;
            const width = x2 - x1;
            const height = y2 - y1;

            // Color based on class name (simple hash or predefined)
            let color = '#00FF00';
            if (det.class_name.includes('Purple')) color = '#800080';
            if (det.class_name.includes('Brown')) color = '#A52A2A';
            if (det.class_name.includes('Moldy')) color = '#FF0000';
            if (det.class_name.includes('Sprouted')) color = '#FFFF00';
            if (det.class_name.includes('Slaty')) color = '#808080';

            ctx.strokeStyle = color;
            ctx.lineWidth = 4;
            ctx.strokeRect(x1, y1, width, height);

            ctx.fillStyle = color;
            ctx.font = '20px Arial';
            ctx.fillText(`${det.class_name} (${(det.confidence * 100).toFixed(1)}%)`, x1, y1 - 10);
        });
    };

    return (
        <div className="result-container">
            <div className="image-wrapper" style={{ position: 'relative', display: 'inline-block' }}>
                <img ref={imgRef} src={imageSrc} alt="Uploaded" style={{ maxWidth: '100%', display: 'block' }} />
                <canvas
                    ref={canvasRef}
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        pointerEvents: 'none',
                    }}
                />
            </div>

            <div className="stats-panel">
                <h3>Analysis Results</h3>
                <p><strong>Model:</strong> {modelType === 'yolo' ? 'YOLOv11 Only' : 'YOLOv11 + ConvNeXt'}</p>
                {grade && <p className="grade"><strong>Grade:</strong> {grade}</p>}

                <h4>Summary</h4>
                <ul>
                    {Object.entries(summary).map(([key, count]) => (
                        <li key={key}>{key}: {count}</li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default ResultDisplay;
