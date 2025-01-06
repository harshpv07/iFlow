from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/send_query', methods=['GET', 'POST'])
def send_query():
    data = request.get_json()
    user_input = data.get('query')
    
    if not user_input:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # Create temporary script file
        script_content = data.get('script')
        if script_content:
            temp_script_path = "temp_script.cmd"
            with open(temp_script_path, "w") as f:
                f.write(script_content)

            # Execute the script
            result = subprocess.run(
                [temp_script_path],
                capture_output=True,
                text=True,
                shell=True
            )

            # Clean up
            os.remove(temp_script_path)

            if result.returncode == 0:
                return jsonify({
                    'success': True,
                    'output': result.stdout
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.stderr
                })

        return jsonify({'error': 'No script provided'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
