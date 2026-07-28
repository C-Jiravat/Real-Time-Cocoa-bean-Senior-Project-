import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const predictImage = async (file, modelType) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelType);

    try {
        const response = await axios.post(`${API_URL}/predict`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error predicting image:', error);
        throw error;
    }
};
