
from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts potential DDoS attacks based on network traffic data.

    Returns:
        JSON response containing the predicted attack label ('Normal' or 'DDoS').

    Raises:
        ValueError: If invalid data format is received.
    """

    # Get data from the request
    try:
        data = request.get_json()
        if not data:
            raise ValueError('Empty JSON data received.')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400  # Bad Request

    # Preprocess data (assuming you have a pre-trained model expecting specific features)
    def preprocess_data(data):
        """
        Preprocesses incoming network traffic data.

        Args:
            data: A dictionary containing network traffic features.

        Returns:
            A numpy array representing the preprocessed data suitable for the model.
        """

        # Extract relevant features (replace with your actual feature names)
        packet_size = data.get('Min Packet Length')
        arrival_time = data.get('URG Flag Count')
        source_ip = data.get('SYN Flag Count')  # Consider anonymizing IP addresses
        destination_ip = data.get('Fwd Packet Length Max')
        protocol = data.get('Fwd Packet Length Min')
        total_backward_packets = data.get('Total Backward Packets')
        average_packet_size = data.get('Average Packet Size')
        ack_flag_count = data.get('ACK Flag Count')
        inbound = data.get('Inbound')
        init_win_bytes_fwd = data.get('Init Win bytes forward')

        # Feature scaling or normalization (consider based on your model's requirements)
        # ... (add your feature scaling/normalization logic here)

        # Combine features into a single array
        preprocessed_data = np.array([packet_size, arrival_time, source_ip, destination_ip,
                                     protocol, total_backward_packets, average_packet_size,
                                     ack_flag_count, inbound, init_win_bytes_fwd])

        return preprocessed_data

    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Make prediction using your trained model (replace with your model call)
    prediction = my_model.predict(preprocessed_data.reshape(1, -1))  # Reshape for single sample
    predicted_label = np.argmax(prediction)  # Get the class index

    # Map index to label (assuming classes are 0 for 'Normal' and 1 for 'DDoS')
    attack_label = 'Normal' if predicted_label == 0 else 'DDoS'

    # Return prediction as JSON response
    return jsonify({'prediction': attack_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
